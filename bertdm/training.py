"""Joint fine-tuning, separate validation, and optimizer-boundary resume support."""

import gc
import json
import math
import os
import platform
import random
import shutil
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .download import sha256
from .pipeline import DecisionPipeline
from .training_data import digest_json, epoch_plan, materialize_example, training_batch


MEMORY_HELP = ("Reduce max_length (512 -> 256 -> 128) and micro_batch_size to 1; increase "
               "gradient_accumulation to retain the effective batch size. Enable gradient_checkpointing "
               "and fp16/bf16 autocast. Keep output_attentions/output_hidden_states disabled. "
               "Preserve math SDPA and the asymmetric mask. AdamW state also consumes memory.")


@dataclass(frozen=True)
class TrainConfig:
  epochs: int = 2
  micro_batch_size: int = 1
  eval_batch_size: int = 1
  gradient_accumulation: int = 8
  encoder_lr: float = 2e-5
  head_lr: float = 1e-4
  weight_decay: float = 0.01
  warmup_ratio: float = 0.06
  max_grad_norm: float = 1.0
  seed: int = 17
  precision: str = "auto"
  gradient_checkpointing: bool = True
  deterministic: bool = True
  initial_scale: float = 1024.0
  samples_per_epoch: int | None = None
  dataset_weights: dict | None = None
  shuffle_choices: bool = True
  choice_count_bounds: tuple = (4, 8, 16, 32)
  selection_dataset: str = "macro"
  selection_metric: str = "accuracy"
  log_every_updates: int = 10
  save_every_updates: int = 100
  allow_validation_rejections: bool = False
  allow_training_rejections: bool = False
  prune_unreferenced_checkpoints: bool = True

  def __post_init__(self):
    for field in ("epochs", "micro_batch_size", "eval_batch_size", "gradient_accumulation",
                  "log_every_updates", "save_every_updates"):
      if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
        raise ValueError(f"{field} must be a positive integer")
    for field in ("encoder_lr", "head_lr", "max_grad_norm", "initial_scale"):
      if not math.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
        raise ValueError(f"{field} must be finite and positive")
    if not math.isfinite(self.weight_decay) or self.weight_decay < 0 or not 0 <= self.warmup_ratio < 1:
      raise ValueError("Invalid weight decay or warmup ratio")
    if self.precision not in ("auto", "fp32", "fp16", "bf16"):
      raise ValueError("precision must be auto/fp32/fp16/bf16")
    if self.samples_per_epoch is not None and (type(self.samples_per_epoch) is not int or self.samples_per_epoch <= 0):
      raise ValueError("samples_per_epoch must be positive or None")
    if self.selection_metric not in ("accuracy", "nll"):
      raise ValueError("selection_metric must be accuracy or nll")
    if (list(self.choice_count_bounds) != sorted(set(self.choice_count_bounds))
        or any(type(b) is not int or b < 2 for b in self.choice_count_bounds)):
      raise ValueError("choice_count_bounds must be increasing integer boundaries >= 2")


def seed_everything(seed, deterministic=True):
  os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  torch.use_deterministic_algorithms(deterministic)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = deterministic
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False


def random_state():
  return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_random_state(state):
  random.setstate(state["python"])
  np.random.set_state(state["numpy"])
  torch.set_rng_state(state["torch"])
  if state["cuda"] is not None and torch.cuda.is_available():
    if len(state["cuda"]) != torch.cuda.device_count():
      raise ValueError("CUDA device count differs from the saved random state")
    torch.cuda.set_rng_state_all(state["cuda"])


def resolve_precision(requested, device):
  device = torch.device(device)
  if device.type != "cuda":
    if requested not in ("auto", "fp32"):
      raise ValueError("Use auto/fp32 for CPU training; GPU AMP requires CUDA")
    return "fp32"
  if requested == "auto":
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
  if requested == "bf16" and not torch.cuda.is_bf16_supported():
    raise ValueError("This GPU does not support bf16; use auto or fp16")
  return requested


def autocast_context(precision, device):
  if precision == "fp32":
    return nullcontext()
  return torch.autocast(torch.device(device).type,
                        dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def environment_info():
  import transformers
  return {"python": platform.python_version(), "platform": platform.system(),
          "torch": str(torch.__version__), "transformers": transformers.__version__,
          "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _to_device(batch, device):
  return {key: value.to(device) for key, value in batch.items()}


def smoke_test(pipeline, device, precision, gradient_checkpointing=True, initial_scale=1024.0):
  """Small forward/backward, with no optimizer update or persistent RNG change."""
  device = torch.device(device)
  model = pipeline.model.to(device)
  previous_mode = model.training
  previous_checkpointing = model.backbone.gradient_checkpointing
  rng = random_state()
  try:
    model.backbone.gradient_checkpointing = gradient_checkpointing
    model.train()
    # Short two-choice input; use real tokenizer and the same model/collator/mask.
    example = {"context": "a b", "question": "a ?",
               "choices": ["a", "b"], "label": 0}
    batch = _to_device(pipeline.collate([pipeline.encode(example)]), device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16",
                                  init_scale=initial_scale)
    with autocast_context(precision, device):
      output = model(**batch)
    if not torch.isfinite(output.loss):
      raise FloatingPointError("Smoke-test loss is nonfinite")
    scaler.scale(output.loss).backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
      raise FloatingPointError("Smoke-test gradients are missing/nonfinite; lower initial_scale for fp16")
    probabilities = output.probabilities().detach().cpu().tolist()
    return {"loss": float(output.loss.detach()), "probabilities": probabilities,
            "precision": precision, "gradient_checkpointing": gradient_checkpointing}
  except torch.cuda.OutOfMemoryError as exc:
    raise RuntimeError(MEMORY_HELP) from exc
  finally:
    model.zero_grad(set_to_none=True)
    model.train(previous_mode)
    model.backbone.gradient_checkpointing = previous_checkpointing
    restore_random_state(rng)
    gc.collect()
    if device.type == "cuda":
      torch.cuda.empty_cache()


def _range_name(count, bounds):
  lower = 2
  for upper in bounds:
    if count <= upper:
      return f"{lower}-{upper}"
    lower = upper + 1
  return f"{lower}+"


@torch.inference_mode()
def evaluate(pipeline, validation, device, precision, batch_size=1, bounds=(4, 8, 16, 32)):
  model = pipeline.model
  was_training = model.training
  model.eval()
  metrics = {}
  try:
    for dataset, rows in sorted(validation.items()):
      if not rows:
        raise ValueError(f"Empty validation dataset: {dataset}")
      totals = {"count": 0, "correct": 0, "nll_sum": 0.0}
      bins = {_range_name(n, bounds): {"count": 0, "correct": 0, "nll_sum": 0.0}
              for n in (2, *[b + 1 for b in bounds])}
      for start in range(0, len(rows), batch_size):
        batch = _to_device(pipeline.collate([materialize_example(r) for r in rows[start:start + batch_size]]), device)
        with autocast_context(precision, device):
          output = model(**batch)
        losses = F.cross_entropy(output.logits.float(), batch["labels"], reduction="none")
        if not torch.isfinite(losses).all():
          raise FloatingPointError(f"Nonfinite validation NLL in {dataset}")
        correct = output.logits.argmax(-1) == batch["labels"]
        counts = batch["valid_choice_mask"].sum(-1)
        for n, ok, loss in zip(counts.tolist(), correct.tolist(), losses.tolist()):
          for bucket in (totals, bins[_range_name(n, bounds)]):
            bucket["count"] += 1
            bucket["correct"] += int(ok)
            bucket["nll_sum"] += loss
      def finish(bucket):
        n = bucket["count"]
        return {"count": n, "accuracy": bucket["correct"] / n if n else None,
                "nll": bucket["nll_sum"] / n if n else None}
      metrics[dataset] = dict(finish(totals), by_choice_count={key: finish(value) for key, value in bins.items()})
    metrics["macro"] = {name: sum(m[name] for m in metrics.values()) / len(metrics)
                         for name in ("accuracy", "nll")}
    return metrics
  finally:
    model.train(was_training)


def resolve_checkpoint(path):
  path = Path(path)
  if path.is_dir() and (path / "latest.json").is_file():
    path = path / "latest.json"
  if path.suffix == ".json":
    pointer = json.loads(path.read_text(encoding="utf-8"))
    path = path.parent / pointer["checkpoint"]
  if not (path / "training_state.pt").is_file() or not (path / "checkpoint.json").is_file():
    raise ValueError(f"No complete resumable checkpoint at {path}")
  return path.resolve()


def _atomic_json(path, value):
  path = Path(path)
  temporary = path.with_suffix(".tmp")
  temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
  temporary.replace(path)


class Trainer:
  def __init__(self, pipeline, datasets, summaries, config, output_dir, device):
    self.pipeline, self.datasets, self.summaries, self.config = pipeline, datasets, summaries, config
    self.output_dir, self.device = Path(output_dir), torch.device(device)
    if set(datasets) != {"train", "validation"} or set(datasets["train"]) != set(datasets["validation"]):
      raise ValueError("Supply matching dataset names in separate train and validation mappings")
    self.weights = config.dataset_weights or {name: 1.0 for name in datasets["train"]}
    self.samples = config.samples_per_epoch or sum(map(len, datasets["train"].values()))
    epoch_plan(datasets["train"], self.weights, 1, config.seed, 0)  # Validate weights.
    for split, reports in summaries.items():
      for name, report in reports.items():
        if report["retained"] == 0:
          raise ValueError(f"{name}/{split} retained no examples")
        allow = config.allow_training_rejections if split == "train" else config.allow_validation_rejections
        if report["rejected"] and not allow:
          raise ValueError(f"{name}/{split} rejected {report['rejected']} rows. Inspect the ledger, increase "
                           "the budget if possible, or explicitly allow rejections and report coverage.")
    if config.selection_dataset not in (*datasets["validation"], "macro"):
      raise ValueError("selection_dataset must be macro or a validation dataset name")
    self.precision = resolve_precision(config.precision, self.device)
    self.device_identity = {"type": self.device.type,
                            "index": (self.device.index or 0) if self.device.type == "cuda" else None}
    self.model = pipeline.model.to(self.device)
    self.model.backbone.gradient_checkpointing = config.gradient_checkpointing
    groups = []
    for component in ("backbone", "head"):
      for decay in (True, False):
        parameters = [p for _, p in getattr(self.model, component).named_parameters()
                      if p.requires_grad and (p.ndim >= 2) == decay]
        groups.append({"params": parameters, "lr": config.encoder_lr if component == "backbone" else config.head_lr,
                       "weight_decay": config.weight_decay if decay else 0.0,
                       "name": component + ("_decay" if decay else "_no_decay")})
    self.optimizer = torch.optim.AdamW(groups)
    updates = math.ceil(self.samples / (config.micro_batch_size * config.gradient_accumulation))
    self.total_updates = updates * config.epochs
    warmup = int(self.total_updates * config.warmup_ratio)
    def schedule(step):
      if step < warmup:
        return step / max(1, warmup)
      return max(0.0, (self.total_updates - step) / max(1, self.total_updates - warmup))
    self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
    self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda" and self.precision == "fp16",
                                      init_scale=config.initial_scale)
    self.state = {"epoch": 0, "sample_offset": 0, "global_step": 0, "samples_seen": 0,
                  "best_score": None, "best_checkpoint": None, "elapsed_seconds": 0.0}
    self.signature = digest_json({"training": asdict(config), "model": pipeline.model.config.to_dict(),
                                  "data": {s: {n: r["cache_key"] for n, r in v.items()} for s, v in summaries.items()},
                                  "trainer_code": sha256(__file__),
                                  "model_code": {name: sha256(Path(__file__).parent / name) for name in (
                                      "model.py", "masking.py", "pipeline.py", "config.py", "_vendor/neobert.py",
                                      "_vendor/rotary.py", "_vendor/swiglu.py")}})
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self._resume_required = (self.output_dir / "latest.json").is_file()
    if (self.output_dir / "run.json").is_file():
      previous = json.loads((self.output_dir / "run.json").read_text(encoding="utf-8"))
      if previous["signature"] != self.signature:
        raise ValueError("Output directory belongs to a different run; choose a new output directory")
    _atomic_json(self.output_dir / "run.json", {"training": asdict(config), "model": self.model.config.to_dict(),
                                               "precision": self.precision, "environment": environment_info(),
                                               "summaries": summaries, "signature": self.signature,
                                               "effective_batch_size": config.micro_batch_size * config.gradient_accumulation,
                                               "samples_per_epoch": self.samples, "total_updates": self.total_updates})

  def _log(self, record):
    with (self.output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
      stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)

  def resume(self, path):
    path = resolve_checkpoint(path)
    # Only load training_state.pt produced by this project and trusted by you.
    payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if payload["signature"] != self.signature:
      raise ValueError("Resume configuration/data/code differs. Restore the original settings; use a bundle "
                       "without optimizer state for an intentional new fine-tuning run.")
    if (payload["precision"] != self.precision or payload["environment"] != environment_info()
        or payload["device"] != self.device_identity):
      raise ValueError("Resume runtime/precision differs; exact continuation requires the same environment")
    # Also load model weights here, so callers cannot accidentally resume state into fresh weights.
    from safetensors.torch import load_model
    load_model(self.model, str(path / "model.safetensors"), strict=True, device=str(self.device))
    self.optimizer.load_state_dict(payload["optimizer"])
    self.scheduler.load_state_dict(payload["scheduler"])
    self.scaler.load_state_dict(payload["scaler"])
    self.state = payload["progress"]
    history = self.output_dir / "history.jsonl"
    if history.is_file():
      # An interrupted same-directory run may have logged unsaved updates.
      # Replay them from the saved boundary without retaining stale duplicate steps.
      records = []
      for line in history.read_text(encoding="utf-8").splitlines():
        try:
          record = json.loads(line)
        except json.JSONDecodeError:
          continue  # A crash may leave an incomplete final log record.
        if record.get("step", 0) <= self.state["global_step"]:
          records.append(record)
      temporary = history.with_suffix(".tmp")
      temporary.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
      temporary.replace(history)
    # Preserve the previous best when writing into a new /kaggle/working directory.
    old_root = path.parent.parent
    old_best = self.state["best_checkpoint"]
    if old_best and not (self.output_dir / old_best).exists():
      source = old_root / old_best
      if not source.is_dir():
        raise ValueError("Previous best checkpoint is missing; persist the whole output directory")
      destination = self.output_dir / old_best
      destination.parent.mkdir(parents=True, exist_ok=True)
      shutil.copytree(source, destination)
      _atomic_json(self.output_dir / "best.json", {"checkpoint": old_best, "score": self.state["best_score"]})
    restore_random_state(payload["random"])
    self._resume_required = False
    self._log({"kind": "resume", "step": self.state["global_step"],
               "epoch": self.state["epoch"], "sample_offset": self.state["sample_offset"]})
    print(f"Resumed optimizer step {self.state['global_step']}, epoch {self.state['epoch']}, "
          f"next sample {self.state['sample_offset']}")

  def save_checkpoint(self, *, is_best=False):
    relative = f"checkpoints/step-{self.state['global_step']:08d}"
    folder = self.output_dir / relative
    if is_best:
      self.state["best_checkpoint"] = relative
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "checkpoint.json").is_file():
      self.pipeline.save_pretrained(folder)
    payload = {"signature": self.signature, "progress": dict(self.state), "random": random_state(),
               "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
               "scaler": self.scaler.state_dict(), "precision": self.precision,
               "environment": environment_info(), "device": self.device_identity}
    temporary = folder / "training_state.tmp"
    torch.save(payload, temporary)
    temporary.replace(folder / "training_state.pt")
    _atomic_json(folder / "checkpoint.json", {"signature": self.signature, "step": self.state["global_step"]})
    _atomic_json(self.output_dir / "latest.json", {"checkpoint": relative})
    if is_best:
      _atomic_json(self.output_dir / "best.json", {"checkpoint": relative, "score": self.state["best_score"]})
    if self.config.prune_unreferenced_checkpoints:
      keep = {relative, self.state["best_checkpoint"]}
      root = (self.output_dir / "checkpoints").resolve()
      for candidate in root.glob("step-*"):
        resolved = candidate.resolve()
        # Only remove our own completed, unreferenced child directories.
        if (resolved.parent == root and candidate.is_dir() and (candidate / "checkpoint.json").is_file()
            and f"checkpoints/{candidate.name}" not in keep):
          shutil.rmtree(resolved)
    return folder

  def _validate_and_save(self):
    started = time.perf_counter()
    metrics = evaluate(self.pipeline, self.datasets["validation"], self.device, self.precision,
                       self.config.eval_batch_size, self.config.choice_count_bounds)
    for name, report in self.summaries["validation"].items():
      metrics[name]["coverage"] = report["coverage"]
      metrics[name]["rejected"] = report["rejected"]
    score = metrics[self.config.selection_dataset][self.config.selection_metric]
    old = self.state["best_score"]
    improved = old is None or (score > old if self.config.selection_metric == "accuracy" else score < old)
    if improved:
      self.state["best_score"] = score
    self._log({"kind": "validation", "step": self.state["global_step"], "metrics": metrics,
               "selection": f"{self.config.selection_dataset}/{self.config.selection_metric}",
               "is_best": improved, "seconds": time.perf_counter() - started})
    self.save_checkpoint(is_best=improved)
    return metrics

  def fit(self, *, stop_after_updates=None):
    """Save/resume at completed optimizer steps; no half-accumulated gradients saved."""
    config = self.config
    if self._resume_required:
      raise ValueError("Existing latest checkpoint found; resume it or choose a new output directory")
    if stop_after_updates is not None and stop_after_updates <= self.state["global_step"]:
      raise ValueError("stop_after_updates must exceed the current optimizer step")
    try:
      while self.state["epoch"] < config.epochs:
        epoch = self.state["epoch"]
        plan = epoch_plan(self.datasets["train"], self.weights, self.samples, config.seed, epoch)
        self._log({"kind": "sampling", "epoch": epoch, "weights": self.weights,
                   "planned_examples": dict(Counter(name for name, _, _ in plan)),
                   "next_sample": self.state["sample_offset"]})
        while self.state["sample_offset"] < self.samples:
          started = time.perf_counter()
          offset = self.state["sample_offset"]
          chunk = plan[offset:offset + config.micro_batch_size * config.gradient_accumulation]
          self.model.train()
          self.optimizer.zero_grad(set_to_none=True)
          loss_sum = 0.0
          used_lrs = {group["name"]: group["lr"] for group in self.optimizer.param_groups}
          for start in range(0, len(chunk), config.micro_batch_size):
            draws = chunk[start:start + config.micro_batch_size]
            batch = _to_device(training_batch(self.pipeline, self.datasets["train"], draws, config.shuffle_choices), self.device)
            with autocast_context(self.precision, self.device):
              output = self.model(**batch)
              # Weight partial microbatches by example count, including the final accumulation window.
              loss = output.loss * (len(draws) / len(chunk))
            if not torch.isfinite(output.loss):
              raise FloatingPointError(f"Nonfinite training loss at step {self.state['global_step']}; draws={draws}")
            loss_sum += float(output.loss.detach()) * len(draws)
            self.scaler.scale(loss).backward()
          self.scaler.unscale_(self.optimizer)
          norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
          self.scaler.step(self.optimizer)
          self.scaler.update()
          self.scheduler.step()
          self.optimizer.zero_grad(set_to_none=True)
          self.state["sample_offset"] += len(chunk)
          self.state["samples_seen"] += len(chunk)
          self.state["global_step"] += 1
          seconds = time.perf_counter() - started
          self.state["elapsed_seconds"] += seconds
          record = {"kind": "train", "epoch": epoch, "step": self.state["global_step"],
                    "sample_offset": self.state["sample_offset"], "loss": loss_sum / len(chunk),
                    "learning_rates": used_lrs, "grad_norm": float(norm), "scale": self.scaler.get_scale(),
                    "examples": len(chunk), "datasets": dict(Counter(n for n, _, _ in chunk)),
                    "seconds": seconds, "examples_per_second": len(chunk) / seconds}
          # Persist every update; print at a configurable cadence.
          with (self.output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
          if self.state["global_step"] % config.log_every_updates == 0:
            print(json.dumps(record), flush=True)
          stop = stop_after_updates is not None and self.state["global_step"] >= stop_after_updates
          if self.state["global_step"] % config.save_every_updates == 0 and not stop:
            self.save_checkpoint()
          if stop:
            break
        complete_epoch = self.state["sample_offset"] == self.samples
        if complete_epoch:
          self.state["epoch"] += 1
          self.state["sample_offset"] = 0
        metrics = self._validate_and_save()
        if stop_after_updates is not None and self.state["global_step"] >= stop_after_updates:
          return {"status": "complete" if self.state["epoch"] == config.epochs else "stopped_at_update_limit",
                  "progress": dict(self.state), "metrics": metrics}
      return {"status": "complete", "progress": dict(self.state)}
    except torch.cuda.OutOfMemoryError as exc:
      self.optimizer.zero_grad(set_to_none=True)
      raise RuntimeError(MEMORY_HELP + " Resume from the last completed checkpoint.") from exc
