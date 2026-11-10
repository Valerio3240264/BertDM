"""Execute actual notebook cells without requiring a local Jupyter server."""

import copy
import importlib.metadata
import json
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("num_processes", [1, 2])
def test_notebook_top_to_bottom_and_resume(pipeline, examples, tmp_path, monkeypatch, num_processes):
  pytest.importorskip("IPython")
  root = Path(__file__).parents[1]
  for line in (root / "requirements-kaggle.txt").read_text().splitlines():
    if not line or line.startswith("#"):
      continue
    name, version = line.split("==")
    try:
      installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      installed = None
    if installed != version:
      pytest.skip("Notebook execution needs the pinned Kaggle requirements")
  notebook = json.loads((root / "train.ipynb").read_text(encoding="utf-8"))
  config_cell = next(c for c in notebook["cells"] if "parameters" in c.get("metadata", {}).get("tags", []))
  namespace = {}
  exec("".join(config_cell["source"]), namespace)
  cfg = namespace["CFG"]
  checkpoint = tmp_path / "input" / "checkpoint"
  pipeline.model.backbone.save_pretrained(checkpoint, safe_serialization=True)
  pipeline.tokenizer.save_pretrained(checkpoint)
  sources = {}
  for name in ("cosmos", "wikihop"):
    sources[name] = {}
    for split in ("train", "validation"):
      if name == "cosmos":
        raw = [{"context": e["context"], "question": e["question"], "label": e["label"],
                **{f"answer{i}": choice for i, choice in enumerate(
                    e["choices"] + (["gamma"] if len(e["choices"]) == 3 else ["first", "third"]))}}
               for e in examples]
      else:
        raw = [{"supports": [e["context"]], "query": e["question"], "candidates": e["choices"],
                "answer": e["choices"][e["label"]]} for e in examples]
      path = tmp_path / "input" / f"{name}-{split}.json"
      path.write_text(json.dumps(raw), encoding="utf-8")
      sources[name][split] = str(path)
  source_bytes = {path: Path(path).read_bytes() for splits in sources.values() for path in splits.values()}
  cfg.update(repo_dir=str(root), output_dir=str(tmp_path / "working" / "run1"),
             cache_dir=str(tmp_path / "working" / "cache"), allow_downloads=False,
             install_dependencies=False, device="cpu", stop_after_updates=1, datasets=sources)
  cfg["distributed"] = {"num_processes": num_processes}
  cfg["model"].update(checkpoint=str(checkpoint), max_length=48, head_width=17, dropout=0.0)
  cfg["training"].update(epochs=1, micro_batch_size=1, gradient_accumulation=2,
                         samples_per_epoch=5, log_every_updates=1, save_every_updates=2, precision="fp32")
  monkeypatch.setattr(sys, "path", list(sys.path))
  monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
  for key in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TOKENIZERS_PARALLELISM",
              "CUBLAS_WORKSPACE_CONFIG", "PYTHONDONTWRITEBYTECODE"):
    monkeypatch.setenv(key, __import__("os").environ.get(key, ""))
  for segment in (1, 2):
    if segment == 2:
      cfg.update(resume_from=str(tmp_path / "working" / "run1" / "latest.json"),
                 output_dir=str(tmp_path / "working" / "run2"), stop_after_updates=None)
    namespace = {}
    for index, cell in enumerate(notebook["cells"]):
      if cell["cell_type"] != "code":
        continue
      source = "CFG = " + repr(copy.deepcopy(cfg)) if cell is config_cell else "".join(cell["source"])
      exec(compile(source, f"train.ipynb:cell{index}", "exec"), namespace)
    answer = json.loads((Path(cfg["output_dir"]) / "inference-example.json").read_text())
    assert sum(answer["probabilities"]) == pytest.approx(1.0)
    assert answer["selected_choice"] == answer["choices"][answer["selected_index"]]
    assert namespace["result"]["status"] == ("stopped_at_update_limit" if segment == 1 else "complete")
  assert all(Path(path).read_bytes() == before for path, before in source_bytes.items())
