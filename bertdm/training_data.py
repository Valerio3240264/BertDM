"""Auditable preprocessing cache and deterministic training-only augmentation."""

import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import tokenizers
import transformers

from .data import ExampleError, normalize_cosmos, normalize_wikihop, read_records
from .download import sha256


def digest_json(value):
  return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preprocessing_identity(pipeline, source, dataset, split):
  tokenizer = pipeline.tokenizer
  if not tokenizer.is_fast:
    raise ValueError("Training cache requires a fast tokenizer with serializable content")
  code = {name: sha256(Path(__file__).with_name(name))
          for name in ("data.py", "tokenization.py", "training_data.py")}
  backend = json.loads(tokenizer.backend_tokenizer.to_str())
  backend["padding"], backend["truncation"] = None, None
  return {
      "format": 1, "dataset": dataset, "split": split, "source_sha256": sha256(source),
      "revision": pipeline.model.config.revision,
      "tokenizer": digest_json(backend),
      "special_tokens": tokenizer.special_tokens_map, "special_ids": tokenizer.all_special_ids,
      "tokenizer_class": type(tokenizer).__name__, "transformers": transformers.__version__,
      "tokenizers": tokenizers.__version__, "max_length": pipeline.encode.max_length,
      "supported_length": pipeline.model.backbone.config.max_length,
      "context_truncation": pipeline.encode.context_truncation, "code": code,
  }


def compact_example(encoded):
  return {key: encoded[key] for key in ("input_ids", "choice_group_ids", "choice_content_mask",
                                        "special_tokens_mask", "labels")}


def materialize_example(encoded):
  """Rebuild redundant metadata just before using the existing collator."""
  groups = encoded["choice_group_ids"]
  length = len(groups)
  return dict(encoded, attention_mask=[True] * length, position_ids=list(range(length)),
              context_mask=[g == 0 for g in groups], valid_choice_mask=[True] * max(groups))


def permute_choices(encoded, order):
  """Reorder complete choice blocks, remap ownership/label, keep shared prefix."""
  groups = encoded["choice_group_ids"]
  count = max(groups)
  if sorted(order) != list(range(count)):
    raise ValueError("order must be a permutation of every choice slot")
  indices = [i for i, g in enumerate(groups) if g == 0]
  new_groups = [0] * len(indices)
  for new_slot, old_slot in enumerate(order, 1):
    block = [i for i, g in enumerate(groups) if g == old_slot + 1]
    indices.extend(block)
    new_groups.extend([new_slot] * len(block))
  result = {key: [encoded[key][i] for i in indices]
            for key in ("input_ids", "choice_content_mask", "special_tokens_mask")}
  result["choice_group_ids"] = new_groups
  result["labels"] = order.index(encoded["labels"])
  return materialize_example(result)


def _summary(rows, total, rejections):
  lengths = np.asarray([len(row["input_ids"]) for row in rows])
  stats = {} if not len(lengths) else {
      "min": int(lengths.min()), "max": int(lengths.max()), "mean": float(lengths.mean()),
      **{f"p{p}": float(np.percentile(lengths, p)) for p in (50, 90, 95, 99)}}
  return {"total": total, "retained": len(rows), "rejected": len(rejections),
          "coverage": len(rows) / total if total else 0.0, "token_lengths": stats,
          "choice_counts": dict(sorted(Counter(max(r["choice_group_ids"]) for r in rows).items())),
          "rejection_reasons": dict(Counter(r["reason"] for r in rejections))}


def prepare_split(pipeline, dataset, split, source, cache_dir, *, read_cache_dirs=()):
  """Record every rejected row; training policy is checked after summaries print.

  Disk cache contains compact encoded records, a rejection ledger, and an identity
  including tokenizer bytes, revision, source content, settings and preprocessing code.
  Read-only cache roots can be supplied from /kaggle/input.
  """
  if dataset not in ("cosmos", "wikihop") or split not in ("train", "validation"):
    raise ValueError("Expected cosmos/wikihop and train/validation")
  identity = preprocessing_identity(pipeline, source, dataset, split)
  key = digest_json(identity)
  for root in (*read_cache_dirs, cache_dir):
    folder = Path(root) / key
    if (folder / "manifest.json").is_file():
      manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
      if manifest["identity"] != identity:
        raise ValueError(f"Cache identity mismatch: {folder}")
      with gzip.open(folder / "examples.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
      if len(rows) != manifest["summary"]["retained"]:
        raise ValueError(f"Incomplete preprocessing cache: {folder}")
      return rows, dict(manifest["summary"], cache_key=key, cache_dir=str(folder), cache_hit=True)
  folder = Path(cache_dir) / key
  folder.mkdir(parents=True, exist_ok=True)
  normalize = {"cosmos": normalize_cosmos, "wikihop": normalize_wikihop}[dataset]
  rows, rejections = [], []
  total = 0
  for index, raw in enumerate(read_records(source)):
    total += 1
    try:
      rows.append(compact_example(pipeline.encode(normalize(raw))))
    except ExampleError as exc:
      rejections.append({"index": index, "id": raw.get("id") if isinstance(raw, dict) else None,
                         "reason": type(exc).__name__ + ": " + str(exc)})
  summary = _summary(rows, total, rejections)
  with gzip.open(folder / "examples.jsonl.gz", "wt", encoding="utf-8") as stream:
    for row in rows:
      stream.write(json.dumps(row, separators=(",", ":")) + "\n")
  with (folder / "rejections.jsonl").open("w", encoding="utf-8") as stream:
    for row in rejections:
      stream.write(json.dumps(row) + "\n")
  manifest = {"identity": identity, "source": str(source), "summary": summary}
  temporary = folder / "manifest.tmp"
  temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  temporary.replace(folder / "manifest.json")
  return rows, dict(summary, cache_key=key, cache_dir=str(folder), cache_hit=False)


def check_split_paths(sources):
  """Reject any source path reused across datasets or train/validation splits."""
  paths = [Path(path).resolve() for splits in sources.values() for path in splits.values()]
  if len(paths) != len(set(paths)):
    raise ValueError("Dataset/split source files must be distinct; validation cannot be a training source")


def epoch_plan(train, weights, samples, seed, epoch):
  """Dataset probability is explicit; rows are uniform within that dataset."""
  names = sorted(train)
  if set(names) != set(weights) or any(not train[name] for name in names):
    raise ValueError("Sampling weights must match nonempty training datasets")
  if any(not np.isfinite(w) or w <= 0 for w in weights.values()):
    raise ValueError("Dataset sampling weights must be finite and positive")
  rng = random.Random(seed + epoch * 1000003)
  return [(name, rng.randrange(len(train[name])), rng.randrange(2**63))
          for name in rng.choices(names, weights=[weights[n] for n in names], k=samples)]


def training_batch(pipeline, train, draws, shuffle_choices):
  rows = []
  for name, index, seed in draws:
    encoded = train[name][index]
    if shuffle_choices:
      order = list(range(max(encoded["choice_group_ids"])))
      random.Random(seed).shuffle(order)
      rows.append(permute_choices(encoded, order))
    else:
      rows.append(materialize_example(encoded))
  return pipeline.collate(rows)


def partition_window(draws, rank, world_size, micro_batch_size):
  """Return equal microstep counts, with zero-weight dummy slots in the tail.

  Positions (not dataset row IDs) are partitioned: repeated draws from weighted
  replacement sampling remain distinct observations. No real draw is dropped.
  """
  if not draws or not 0 <= rank < world_size or micro_batch_size <= 0:
    raise ValueError("Expected a nonempty window and valid rank/batch size")
  width = world_size * micro_batch_size
  result = []
  for start in range(0, len(draws), width):
    positions = range(start + rank * micro_batch_size, start + (rank + 1) * micro_batch_size)
    result.append(([draws[i] if i < len(draws) else draws[0] for i in positions],
                   [i < len(draws) for i in positions]))
  return result
