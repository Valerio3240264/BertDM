"""Script-free local readers preserving the supplied official split membership."""

import csv
import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path


class ExampleError(ValueError):
  """A malformed example, with source location added by readers."""


def _text(value, name):
  if not isinstance(value, str) or not value.strip():
    raise ExampleError(f"{name} must be a nonempty string")
  return value


def validate_example(row, *, require_label=True):
  if not isinstance(row, Mapping):
    raise ExampleError("example must be an object")
  context = _text(row.get("context"), "context")
  question = _text(row.get("question"), "question")
  choices = row.get("choices")
  if not isinstance(choices, (list, tuple)) or len(choices) < 2:
    raise ExampleError("choices must contain at least two strings")
  choices = [_text(c, f"choices[{i}]") for i, c in enumerate(choices)]
  # Official Cosmos includes repeated answer text. Choices are indexed slots;
  # preserve duplicates rather than rewriting or filtering official examples.
  label = row.get("label")
  if label is None:
    if require_label:
      raise ExampleError("label is required for training/validation")
  elif isinstance(label, bool) or not isinstance(label, Integral) or not 0 <= label < len(choices):
    raise ExampleError("label must be a zero-based integer within choices")
  return {"context": context, "question": question, "choices": choices,
          "label": int(label) if label is not None else None}


def normalize_cosmos(row):
  if not isinstance(row, Mapping):
    raise ExampleError("example must be an object")
  label = row.get("label")
  if isinstance(label, str):
    try:
      label = int(label)
    except ValueError as exc:
      raise ExampleError("Cosmos label must be an integer") from exc
  return validate_example({"context": row.get("context"), "question": row.get("question"),
                           "choices": [row.get(f"answer{i}") for i in range(4)], "label": label})


def normalize_wikihop(row):
  if not isinstance(row, Mapping):
    raise ExampleError("example must be an object")
  supports = row.get("supports")
  if not isinstance(supports, (list, tuple)) or not supports:
    raise ExampleError("supports must be a nonempty list of documents")
  supports = [_text(s, f"supports[{i}]") for i, s in enumerate(supports)]
  candidates = row.get("candidates")
  if not isinstance(candidates, (list, tuple)):
    raise ExampleError("candidates must be a list")
  answer = _text(row.get("answer"), "answer")
  if candidates.count(answer) != 1:
    raise ExampleError("answer must match exactly one candidate")
  return validate_example({"context": "\n\n".join(supports), "question": row.get("query"),
                           "choices": list(candidates), "label": candidates.index(answer)})


def read_records(path):
  """Read CSV, JSON arrays, JSONL or native Parquet. No dataset scripts."""
  path = Path(path)
  if path.suffix.lower() == ".parquet":
    try:
      import pyarrow.parquet as pq
    except ImportError as exc:
      raise ImportError(
          "Parquet input requires pip install 'bertdm[data]'") from exc
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
      yield from batch.to_pylist()
    return
  with path.open(encoding="utf-8-sig", newline="") as stream:
    if path.suffix.lower() == ".csv":
      yield from csv.DictReader(stream)
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
      for line_number, line in enumerate(stream, 1):
        if line.strip():
          try:
            yield json.loads(line)
          except json.JSONDecodeError as exc:
            raise ExampleError(
                f"{path}: line {line_number}: invalid JSON") from exc
    elif path.suffix.lower() == ".json":
      records = json.load(stream)
      if not isinstance(records, list):
        raise ExampleError(f"{path}: expected a JSON array")
      yield from records
    else:
      raise ValueError(f"Unsupported dataset format: {path.suffix}")


def load_splits(dataset, *, train_file, validation_file):
  """Normalize each official split separately; never shuffle/repartition here."""
  normalizers = {"cosmos": normalize_cosmos, "wikihop": normalize_wikihop,
                 "normalized": validate_example}
  if dataset not in normalizers:
    raise ValueError(f"Unknown dataset {dataset!r}")
  if Path(train_file).resolve() == Path(validation_file).resolve():
    raise ValueError("Training and validation files must be different")
  result = {}
  for split, path in (("train", train_file), ("validation", validation_file)):
    result[split] = []
    for index, row in enumerate(read_records(path)):
      try:
        result[split].append(normalizers[dataset](row))
      except ExampleError as exc:
        raise ExampleError(
            f"{dataset}/{split} {path}: record {index}: {exc}") from exc
    if not result[split]:
      raise ExampleError(f"{path}: {split} is empty")
  return result
