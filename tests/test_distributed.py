"""Real multiprocess Gloo tests; no GPU or network downloads are required."""

import copy
import json
import random
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from bertdm import DecisionPipeline
from bertdm.distributed import DistributedRuntime
from bertdm.masking import build_attention_bias
from bertdm.training import TrainConfig, Trainer, evaluate, resolve_checkpoint, seed_everything
from bertdm.training_data import compact_example, materialize_example, partition_window


@pytest.mark.parametrize("size,batch,world", [(1, 1, 2), (5, 2, 2), (7, 1, 3), (8, 2, 2)])
def test_distributed_window_keeps_each_real_draw_once(size, batch, world):
  draws = [("cosmos", i, i) for i in range(size)]
  partitions = [partition_window(draws, rank, world, batch) for rank in range(world)]
  assert len({len(p) for p in partitions}) == 1
  real = [draw[1] for steps in partitions for rows, weights in steps
          for draw, active in zip(rows, weights) if active]
  assert sorted(real) == list(range(size))
  assert all(len(rows) == batch for steps in partitions for rows, _ in steps)


def _distributed_worker(rank, world, folder):
  folder = Path(folder)
  torch.set_num_threads(1)
  dist.init_process_group("gloo", init_method=(folder / "store").resolve().as_uri(),
                          rank=rank, world_size=world, timeout=timedelta(seconds=90))
  try:
    data = json.loads((folder / "data.json").read_text())
    datasets, reports = data["datasets"], data["reports"]
    runtime = DistributedRuntime.current("cpu")
    config = TrainConfig(epochs=1, micro_batch_size=1, gradient_accumulation=2,
                         samples_per_epoch=5, warmup_ratio=0.0, precision="fp32",
                         log_every_updates=100, save_every_updates=100)
    def create(name, cfg=config, dropout=0.0):
      pipeline = DecisionPipeline.from_pretrained(folder / "initial")
      pipeline.model.config = replace(pipeline.model.config, dropout=dropout)
      pipeline.model.head.dropout.p = dropout
      trainer = Trainer(pipeline, datasets, reports, cfg, folder / name, "cpu")
      seed_everything(77 + rank)
      return trainer
    baseline = create("ddp-baseline")
    # Check every layer/head with the actual DDP-owned backbone and masks.
    batch = baseline.pipeline.collate([materialize_example(datasets["train"]["wikihop"][0])])
    output = baseline.raw_model(**batch, output_attentions=True)
    forbidden = torch.isneginf(build_attention_bias(batch["choice_group_ids"], batch["attention_mask"]))
    for weights in output.attentions:
      assert (weights[forbidden.expand_as(weights)] == 0).all()
    del output
    baseline.fit()
    metrics = evaluate(baseline.pipeline, datasets["validation"], "cpu", "fp32")
    assert metrics["cosmos"]["count"] == 1  # rank 1 has an empty validation shard
    assert metrics["wikihop"]["count"] == 3
    for name, parameter in baseline.raw_model.named_parameters():
      copies = runtime.gather(parameter.detach().tolist())
      assert copies[0] == copies[1], name

    cfg = replace(config, epochs=2)
    whole = create("ddp-whole", cfg, dropout=0.15)
    whole.fit()
    expected_rng = (random.random(), float(np.random.rand()), torch.rand(3))
    partial = create("ddp-partial", cfg, dropout=0.15)
    partial.fit(stop_after_updates=1)
    checkpoint = resolve_checkpoint(folder / "ddp-partial")
    resumed_pipeline = DecisionPipeline.from_pretrained(checkpoint)
    resumed = Trainer(resumed_pipeline, datasets, reports, cfg, folder / "ddp-resumed", "cpu")
    resumed.resume(checkpoint)
    resumed.fit()
    actual_rng = (random.random(), float(np.random.rand()), torch.rand(3))
    assert expected_rng[:2] == actual_rng[:2]
    torch.testing.assert_close(expected_rng[2], actual_rng[2], rtol=0, atol=0)
    assert resumed.state["global_step"] == whole.state["global_step"] == 4
    for name, parameter in resumed.raw_model.named_parameters():
      torch.testing.assert_close(parameter, dict(whole.raw_model.named_parameters())[name], rtol=0, atol=0)

    failed = create("ddp-nonfinite")
    if rank == 1:
      with torch.no_grad():
        failed.raw_model.head.gate.weight.fill_(float("nan"))
    try:
      failed.fit()
    except FloatingPointError as exc:
      assert "Nonfinite training loss" in str(exc)
    else:
      raise AssertionError("Every rank must abort when any rank has a nonfinite loss")
    try:
      runtime.primary_call(lambda: (_ for _ in ()).throw(OSError("simulated checkpoint I/O failure")))
    except RuntimeError as exc:
      assert "simulated checkpoint I/O failure" in str(exc)
    else:
      raise AssertionError("Every rank must receive the primary I/O error")
    (folder / f"rank-{rank}-passed.json").write_text(json.dumps({"metrics": metrics, "step": resumed.state["global_step"]}))
  finally:
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_process_training_metrics_and_exact_resume(pipeline, examples, tmp_path):
  # Include multiple choice-count bins and a final window with only one real row.
  extra = dict(examples[0], choices=["alpha", "beta", "gamma", "delta", "first"], label=3)
  rows = [compact_example(pipeline.encode(e)) for e in [*examples, extra]]
  datasets = {"train": {"cosmos": rows[:2], "wikihop": rows},
              "validation": {"cosmos": rows[:1], "wikihop": rows}}
  reports = {s: {n: {"retained": len(r), "rejected": 0, "coverage": 1.0,
                     "cache_key": s + n} for n, r in values.items()} for s, values in datasets.items()}
  pipeline.save_pretrained(tmp_path / "initial")
  (tmp_path / "data.json").write_text(json.dumps({"datasets": datasets, "reports": reports}))
  processes = mp.spawn(_distributed_worker, args=(2, str(tmp_path)), nprocs=2, join=False)
  try:
    deadline = time.monotonic() + 180
    while not processes.join(timeout=1):
      if time.monotonic() > deadline:
        pytest.fail("Two-process DDP test exceeded 180 seconds")
  finally:
    for process in processes.processes:
      if process.is_alive():
        process.terminate()
      process.join(timeout=10)
  assert all((tmp_path / f"rank-{rank}-passed.json").is_file() for rank in (0, 1))
  config = TrainConfig(epochs=1, micro_batch_size=2, gradient_accumulation=2,
                       samples_per_epoch=5, warmup_ratio=0.0, precision="fp32",
                       log_every_updates=100, save_every_updates=100)
  single = Trainer(pipeline, datasets, reports, config, tmp_path / "single", "cpu")
  single.fit()
  distributed = DecisionPipeline.from_pretrained(resolve_checkpoint(tmp_path / "ddp-baseline"))
  for name, parameter in pipeline.model.named_parameters():
    torch.testing.assert_close(parameter, dict(distributed.model.named_parameters())[name], rtol=2e-4, atol=1e-7)
  expected = evaluate(pipeline, datasets["validation"], "cpu", "fp32")
  actual = json.loads((tmp_path / "rank-0-passed.json").read_text())["metrics"]
  for name in ("cosmos", "wikihop"):
    assert actual[name]["accuracy"] == expected[name]["accuracy"]
    assert actual[name]["nll"] == pytest.approx(expected[name]["nll"], abs=2e-6)
  checkpoint = resolve_checkpoint(tmp_path / "ddp-resumed")
  payload = torch.load(checkpoint / "training_state.pt", weights_only=False)
  assert payload["world_size"] == 2
  assert [state["rank"] for state in payload["ranks"]] == [0, 1]
  assert not torch.equal(payload["ranks"][0]["random"]["torch"], payload["ranks"][1]["random"]["torch"])
  with pytest.raises(ValueError, match="world size"):
    single.resume(checkpoint)
  records = [json.loads(line) for line in (tmp_path / "ddp-baseline" / "history.jsonl").read_text().splitlines()]
  updates = [r for r in records if r["kind"] == "train"]
  assert len(updates) == 2 and sum(r["examples"] for r in updates) == 5
