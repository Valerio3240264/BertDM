import copy
import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from bertdm import DecisionPipeline
from bertdm.training import Trainer, TrainConfig, evaluate, resolve_checkpoint, seed_everything, smoke_test
from bertdm.training_data import (check_split_paths, compact_example, epoch_plan, materialize_example,
                                  permute_choices, prepare_split, training_batch)


def training_inputs(pipeline, examples):
  rows = [compact_example(pipeline.encode(e)) for e in examples]
  datasets = {split: {name: copy.deepcopy(rows) for name in ("cosmos", "wikihop")}
              for split in ("train", "validation")}
  summaries = {split: {name: {"retained": len(rows), "rejected": 0, "coverage": 1.0,
                              "cache_key": split + name} for name in datasets[split]} for split in datasets}
  return datasets, summaries


def small_config(**kwargs):
  return TrainConfig(epochs=2, micro_batch_size=2, gradient_accumulation=2, samples_per_epoch=9,
                     warmup_ratio=0.2, precision="fp32", save_every_updates=100,
                     log_every_updates=100, **kwargs)


def test_choice_shuffle_preserves_full_blocks_and_label(pipeline, examples):
  original = compact_example(pipeline.encode(examples[0]))
  permuted = permute_choices(original, [2, 0, 1])
  assert permuted["labels"] == 2
  for new, old in enumerate([2, 0, 1], 1):
    old_ids = [t for t, g in zip(original["input_ids"], original["choice_group_ids"]) if g == old + 1]
    new_ids = [t for t, g in zip(permuted["input_ids"], permuted["choice_group_ids"]) if g == new]
    assert old_ids == new_ids
  assert len(original["input_ids"]) == len(permuted["input_ids"])
  assert permuted["position_ids"] == list(range(len(original["input_ids"])))
  assert permuted["input_ids"][:sum(permuted["context_mask"])] == original["input_ids"][:sum(permuted["context_mask"])]
  with pytest.raises(ValueError):
    permute_choices(original, [0, 0, 1])


def test_sampling_is_dataset_balanced_deterministic_and_train_only(pipeline, examples):
  data, _ = training_inputs(pipeline, examples)
  data["train"]["wikihop"] *= 20
  plan = epoch_plan(data["train"], {"cosmos": 1, "wikihop": 3}, 10000, 4, 0)
  fraction = sum(n == "wikihop" for n, _, _ in plan) / len(plan)
  assert 0.72 < fraction < 0.78
  assert plan == epoch_plan(data["train"], {"cosmos": 1, "wikihop": 3}, 10000, 4, 0)
  assert plan != epoch_plan(data["train"], {"cosmos": 1, "wikihop": 3}, 10000, 4, 1)
  before = copy.deepcopy(data["validation"])
  training_batch(pipeline, data["train"], plan[:4], True)
  assert before == data["validation"]
  with pytest.raises(ValueError, match="validation"):
    check_split_paths({"cosmos": {"train": "same", "validation": "same"}})


def test_cache_and_rejection_summary_invalidation(pipeline, examples, tmp_path):
  source = tmp_path / "train.json"
  raw = [{"supports": [e["context"]], "query": e["question"], "candidates": e["choices"],
          "answer": e["choices"][e["label"]]} for e in examples]
  source.write_text(json.dumps(raw + [{"supports": []}]), encoding="utf-8")
  rows, report = prepare_split(pipeline, "wikihop", "train", source, tmp_path / "cache")
  assert report["total"] == 3 and report["retained"] == 2 and report["rejected"] == 1
  assert report["token_lengths"]["max"] <= 48
  assert sum(report["choice_counts"].values()) == 2
  again, cached = prepare_split(pipeline, "wikihop", "train", source, tmp_path / "new-cache",
                                read_cache_dirs=[tmp_path / "cache"])
  assert cached["cache_hit"] and again == rows
  assert cached["cache_key"] == report["cache_key"]
  ledger = open(report["cache_dir"] + "/rejections.jsonl", encoding="utf-8").read()
  assert '"index": 2' in ledger
  pipeline.encode.max_length = 47
  _, changed = prepare_split(pipeline, "wikihop", "train", source, tmp_path / "cache")
  assert changed["cache_key"] != report["cache_key"]
  source.write_text(json.dumps(raw), encoding="utf-8")
  _, changed_source = prepare_split(pipeline, "wikihop", "train", source, tmp_path / "cache")
  assert changed_source["cache_key"] != changed["cache_key"]


def test_checkpointed_forward_backward_preserves_mask_and_gradients(pipeline, batch):
  model = pipeline.model.train()
  plain = model(**batch, output_attentions=True)
  plain.loss.backward()
  gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
  model.zero_grad(set_to_none=True)
  model.backbone.gradient_checkpointing_enable()
  checkpointed = model(**batch, output_attentions=True)
  checkpointed.loss.backward()
  torch.testing.assert_close(plain.logits, checkpointed.logits, rtol=0, atol=0)
  for old, new in zip(plain.attentions, checkpointed.attentions):
    torch.testing.assert_close(old, new, rtol=0, atol=0)
  for name, parameter in model.named_parameters():
    torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)


def test_smoke_does_not_update_weights_or_rng(pipeline):
  seed_everything(7)
  state = torch.get_rng_state().clone()
  weights = {n: p.clone() for n, p in pipeline.model.named_parameters()}
  report = smoke_test(pipeline, "cpu", "fp32")
  assert np.isfinite(report["loss"])
  assert torch.equal(state, torch.get_rng_state())
  for n, p in pipeline.model.named_parameters():
    assert p.grad is None
    torch.testing.assert_close(p, weights[n], rtol=0, atol=0)


def test_metrics_are_separate_deterministic_and_correct(pipeline, examples):
  data, _ = training_inputs(pipeline, examples)
  before = copy.deepcopy(data)
  metrics = evaluate(pipeline, data["validation"], "cpu", "fp32", batch_size=2)
  assert set(metrics) == {"cosmos", "wikihop", "macro"}
  batch = pipeline.collate([materialize_example(r) for r in data["validation"]["cosmos"]])
  output = pipeline.model.eval()(**batch)
  assert metrics["cosmos"]["nll"] == pytest.approx(float(output.loss.detach()), abs=1e-7)
  assert sum(v["count"] for v in metrics["cosmos"]["by_choice_count"].values()) == 2
  assert before == data
  assert metrics == evaluate(pipeline, data["validation"], "cpu", "fp32", batch_size=2)


def test_partial_accumulation_matches_larger_batch(pipeline, examples, tmp_path):
  data, reports = training_inputs(pipeline, examples)
  base = copy.deepcopy(pipeline)
  config = small_config()
  first = Trainer(pipeline, data, reports, replace(config, micro_batch_size=1, gradient_accumulation=4), tmp_path / "micro", "cpu")
  second = Trainer(base, data, reports, replace(config, micro_batch_size=4, gradient_accumulation=1), tmp_path / "large", "cpu")
  first.fit()
  second.fit()
  assert first.state["samples_seen"] == 18
  for n, p in pipeline.model.named_parameters():
    torch.testing.assert_close(p, dict(base.model.named_parameters())[n], rtol=1e-4, atol=1e-7)


def test_resume_matches_uninterrupted_including_dropout_and_rng(pipeline, examples, tmp_path):
  pipeline.model.head.dropout.p = 0.15
  # Persist dropout configuration as well as the live module.
  pipeline.model.config = replace(pipeline.model.config, dropout=0.15)
  data, reports = training_inputs(pipeline, examples)
  fresh = copy.deepcopy(pipeline)
  config = small_config()
  seed_everything(123)
  uninterrupted = Trainer(pipeline, data, reports, config, tmp_path / "whole", "cpu")
  uninterrupted.fit()
  expected_random = (random.random(), np.random.rand(), torch.rand(1))
  seed_everything(123)
  interrupted = Trainer(fresh, data, reports, config, tmp_path / "partial", "cpu")
  result = interrupted.fit(stop_after_updates=1)
  assert result["status"] == "stopped_at_update_limit"
  checkpoint = resolve_checkpoint(tmp_path / "partial")
  loaded = DecisionPipeline.from_pretrained(checkpoint)
  with (tmp_path / "partial" / "history.jsonl").open("a", encoding="utf-8") as stream:
    stream.write('{"kind":"train","step":999}\n{"interrupted":')
  same_directory = Trainer(copy.deepcopy(loaded), data, reports, config, tmp_path / "partial", "cpu")
  with pytest.raises(ValueError, match="resume"):
    same_directory.fit()
  same_directory.resume(checkpoint)
  assert all(json.loads(line).get("step", 0) <= 1 for line in
             (tmp_path / "partial" / "history.jsonl").read_text().splitlines())
  resumed = Trainer(loaded, data, reports, config, tmp_path / "resumed", "cpu")
  resumed.resume(checkpoint)
  resumed.fit()
  actual_random = (random.random(), np.random.rand(), torch.rand(1))
  assert actual_random[:2] == expected_random[:2]
  torch.testing.assert_close(actual_random[2], expected_random[2], rtol=0, atol=0)
  assert resumed.state["global_step"] == uninterrupted.state["global_step"] == 6
  assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
  for n, p in loaded.model.named_parameters():
    torch.testing.assert_close(p, dict(pipeline.model.named_parameters())[n], rtol=0, atol=0)
  assert resolve_checkpoint(tmp_path / "resumed" / "best.json").is_dir()
  assert resolve_checkpoint(tmp_path / "resumed" / "latest.json").is_dir()
  assert len(list((tmp_path / "resumed" / "checkpoints").glob("step-*"))) <= 2
  incompatible = Trainer(copy.deepcopy(loaded), data, reports, replace(config, encoder_lr=1e-3), tmp_path / "bad", "cpu")
  with pytest.raises(ValueError, match="differs"):
    incompatible.resume(checkpoint)


def test_rejections_require_explicit_policy(pipeline, examples, tmp_path):
  data, reports = training_inputs(pipeline, examples)
  reports["validation"]["wikihop"]["rejected"] = 1
  with pytest.raises(ValueError, match="rejected"):
    Trainer(pipeline, data, reports, small_config(), tmp_path, "cpu")


def test_nonfinite_training_loss_aborts(pipeline, examples, tmp_path):
  data, reports = training_inputs(pipeline, examples)
  with torch.no_grad():
    pipeline.model.head.gate.weight.fill_(float("nan"))
  trainer = Trainer(pipeline, data, reports, small_config(), tmp_path, "cpu")
  with pytest.raises(FloatingPointError, match="Nonfinite training loss"):
    trainer.fit()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_fp16_trainer_scaler_and_resume(pipeline, examples, tmp_path):
  data, reports = training_inputs(pipeline, examples)
  config = replace(small_config(), precision="fp16", epochs=1, samples_per_epoch=5)
  trainer = Trainer(pipeline, data, reports, config, tmp_path / "gpu", "cuda")
  trainer.fit(stop_after_updates=1)
  checkpoint = resolve_checkpoint(tmp_path / "gpu")
  loaded = DecisionPipeline.from_pretrained(checkpoint)
  resumed = Trainer(loaded, data, reports, config, tmp_path / "gpu-resumed", "cuda")
  resumed.resume(checkpoint)
  assert resumed.scaler.state_dict() == trainer.scaler.state_dict()
  result = resumed.fit()
  assert result["status"] == "complete"
  assert resumed.state["global_step"] == 2


def test_notebook_is_clean_and_imports_modules():
  import ast
  from pathlib import Path

  notebook = json.loads((Path(__file__).parents[1] / "train.ipynb").read_text(encoding="utf-8"))
  cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
  assert sum("parameters" in c.get("metadata", {}).get("tags", []) for c in cells) == 1
  for cell in cells:
    assert cell["execution_count"] is None and cell["outputs"] == []
    ast.parse("".join(cell["source"]))
  source = "\n".join("".join(c["source"]) for c in cells)
  assert "from bertdm.training import" in source
  assert "class DecisionModel" not in source
