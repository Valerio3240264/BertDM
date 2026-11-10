"""Shared configuration-driven entry point for torchrun and standalone workers."""

import json
from pathlib import Path

import torch

from .config import DecisionConfig
from .distributed import initialize_distributed, finish_distributed
from .download import download_cosmos, download_wikihop
from .pipeline import DecisionPipeline
from .training import (TrainConfig, Trainer, seed_everything, resolve_precision, smoke_test,
                       resolve_checkpoint, evaluate, _atomic_json)
from .training_data import prepare_split, check_split_paths


def _paths(config):
  output, cache = Path(config["output_dir"]).resolve(), Path(config["cache_dir"]).resolve()
  if Path("/kaggle/input").exists():
    working = Path("/kaggle/working").resolve()
    if not output.is_relative_to(working) or not cache.is_relative_to(working):
      raise ValueError("Kaggle outputs and writable caches must be under /kaggle/working")
  return output, cache


def _pipeline(config, *, cached_only=False):
  if config.get("resume_from"):
    pipeline = DecisionPipeline.from_pretrained(resolve_checkpoint(config["resume_from"]))
    requested = DecisionConfig(**config["model"])
    for key in ("revision", "max_length", "head_width", "dropout", "context_truncation"):
      if getattr(pipeline.model.config, key) != getattr(requested, key):
        raise ValueError(f"Resume model setting changed: {key}")
    return pipeline
  if (Path(config["output_dir"]) / "latest.json").exists():
    raise ValueError("Output contains a previous run; set resume_from or choose a fresh output_dir")
  return DecisionPipeline.from_checkpoint(
      DecisionConfig(**config["model"]),
      local_files_only=cached_only or not config.get("allow_downloads", True),
      cache_dir=str(Path(config["output_dir"]) / "hf-cache" / "hub"))


def _sources(config, output):
  sources = {}
  for name, paths in config["datasets"].items():
    if name not in ("cosmos", "wikihop") or set(paths) != {"train", "validation"}:
      raise ValueError("Expected cosmos/wikihop with explicit train/validation keys")
    if paths["train"] is None and paths["validation"] is None:
      if not config.get("allow_downloads", True):
        raise ValueError(f"Supply local official split paths for offline {name}")
      downloader = download_cosmos if name == "cosmos" else download_wikihop
      sources[name] = {split: str(path) for split, path in downloader(output / "data" / name).items()}
    elif paths["train"] is None or paths["validation"] is None:
      raise ValueError(f"Supply both split paths for {name}")
    else:
      sources[name] = {split: str(Path(path).resolve()) for split, path in paths.items()}
  check_split_paths(sources)
  return sources


def run_training(config):
  """All workers call this function with the same JSON-serializable configuration."""
  runtime = initialize_distributed(config.get("device", "auto"))
  try:
    output, cache = runtime.local_call(lambda: _paths(config))
    training = TrainConfig(**config["training"])
    seed_everything(training.seed, training.deterministic)
    holder = {}
    # Warm the Hub cache once. No model or encoded dataset is broadcast as an object.
    runtime.primary_call(lambda: holder.update(pipeline=_pipeline(config)))
    pipeline = runtime.local_call(lambda: holder["pipeline"] if runtime.primary else _pipeline(config, cached_only=True))
    sources = runtime.primary_call(lambda: _sources(config, output))
    def warm_preprocessing():
      for name, paths in sources.items():
        for split, path in paths.items():
          prepare_split(pipeline, name, split, path, cache, read_cache_dirs=config.get("read_cache_dirs", ()))
    runtime.primary_call(warm_preprocessing)
    def load_preprocessing():
      datasets, summaries = {"train": {}, "validation": {}}, {"train": {}, "validation": {}}
      for name, paths in sources.items():
        for split, path in paths.items():
          rows, report = prepare_split(pipeline, name, split, path, cache,
                                       read_cache_dirs=config.get("read_cache_dirs", ()))
          datasets[split][name], summaries[split][name] = rows, report
      return datasets, summaries
    datasets, summaries = runtime.local_call(load_preprocessing)
    runtime.primary_call(lambda: _atomic_json(output / "dataset-summaries.json", summaries))
    precision = runtime.local_call(lambda: resolve_precision(training.precision, runtime.device))
    precisions = runtime.gather(precision)
    runtime.check(len(set(precisions)) == 1, "Resolved precision must agree on every rank")
    # Smoke before DDP attaches reducer hooks. It performs no optimizer update.
    smoke = runtime.local_call(lambda: smoke_test(pipeline, runtime.device, precision,
                                                 training.gradient_checkpointing, training.initial_scale))
    smokes = runtime.gather(dict(smoke, rank=runtime.rank))
    runtime.primary_call(lambda: print(json.dumps({"kind": "smoke", "ranks": smokes}), flush=True))
    trainer = Trainer(pipeline, datasets, summaries, training, output, runtime.device)
    # The model initialization is synchronized; stochastic training streams differ by rank.
    seed_everything(training.seed + runtime.rank, training.deterministic)
    if config.get("resume_from"):
      trainer.resume(config["resume_from"])
    result = trainer.fit(stop_after_updates=config.get("stop_after_updates"))
    metrics = evaluate(pipeline, datasets["validation"], runtime.device, precision,
                       training.eval_batch_size, training.choice_count_bounds)
    for name, report in summaries["validation"].items():
      metrics[name].update(coverage=report["coverage"], rejected=report["rejected"])
    result.update(metrics=metrics, world_size=runtime.world_size)
    runtime.primary_call(lambda: _atomic_json(output / "final-metrics.json", metrics))
    runtime.primary_call(lambda: _atomic_json(output / "result.json", result))
    return result
  finally:
    finish_distributed()
