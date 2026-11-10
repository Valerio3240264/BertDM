"""Download data files, not legacy Hugging Face dataset scripts.

WikiHop support documents are already embedded in the official JSON records.
For offline Kaggle sessions download/stage files before turning Internet off.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


COSMOS_REVISION = "b6eb99cca4e2a51dd28a9a6f562534872d851639"
COSMOS_BASE = f"https://raw.githubusercontent.com/wilburOne/cosmosqa/{COSMOS_REVISION}/data"
WIKIHOP_REVISION = "a6842a0767a929f86efd2aa1d7a18f4a6d954266"
WIKIHOP_BASE = f"https://huggingface.co/datasets/community-datasets/qangaroo/resolve/{WIKIHOP_REVISION}/wikihop"


def sha256(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def download_file(url, destination, *, expected_sha256=None):
  """Stream HTTPS to a temporary sibling and atomically publish on success."""
  if not url.startswith("https://"):
    raise ValueError("Download URL must use HTTPS")
  destination = Path(destination)
  destination.parent.mkdir(parents=True, exist_ok=True)
  if not destination.exists():
    temporary = None
    try:
      with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
        temporary = Path(target.name)
        request = urllib.request.Request(
            url, headers={"User-Agent": "BertDM/0.1"})
        with urllib.request.urlopen(request, timeout=120) as source:
          shutil.copyfileobj(source, target)
      if expected_sha256 and sha256(temporary) != expected_sha256:
        raise ValueError(f"SHA256 mismatch: {url}")
      temporary.replace(destination)
    finally:
      if temporary is not None:
        temporary.unlink(missing_ok=True)
  if expected_sha256 and sha256(destination) != expected_sha256:
    raise ValueError(f"SHA256 mismatch: {destination}")
  return destination


def _manifest(directory, dataset, paths, source):
  manifest = {"dataset": dataset, "source": source,
              "splits": {name: {"file": path.name, "sha256": sha256(path)} for name, path in paths.items()}}
  (Path(directory) / "manifest.json").write_text(json.dumps(manifest,
                                                            indent=2), encoding="utf-8")
  return paths


def download_cosmos(directory):
  directory = Path(directory)
  paths = {split: download_file(f"{COSMOS_BASE}/{file}", directory / file)
           for split, file in (("train", "train.csv"), ("validation", "valid.csv"))}
  return _manifest(directory, "cosmos", paths, COSMOS_BASE)


def download_wikihop(directory):
  """Download pinned native Parquet mirror of the original official splits."""
  directory = Path(directory)
  paths = {split: download_file(f"{WIKIHOP_BASE}/{split}-00000-of-00001.parquet", directory / f"{split}.parquet")
           for split in ("train", "validation")}
  return _manifest(directory, "wikihop", paths, WIKIHOP_BASE)


def prepare_wikihop(archive, directory, *, expected_sha256=None):
  """Use local QAngaroo v1.1 ZIP or an explicit HTTPS archive URL.

  Only original wikihop/train.json and wikihop/dev.json are copied, using
  fixed destination names. No archive member path is extracted to disk.
  """
  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=True)
  source = str(archive)
  if source.startswith("https://"):
    archive = download_file(
        source, directory / "qangaroo_v1.1.zip", expected_sha256=expected_sha256)
  else:
    archive = Path(archive)
    if expected_sha256 and sha256(archive) != expected_sha256:
      raise ValueError("Archive SHA256 mismatch")
  paths = {}
  with zipfile.ZipFile(archive) as zipped:
    for split, filename in (("train", "train.json"), ("validation", "dev.json")):
      members = [m for m in zipped.infolist()
                 if PurePosixPath(m.filename).parts[-2:] == ("wikihop", filename) and not m.is_dir()]
      if len(members) != 1:
        raise ValueError(
            f"Expected exactly one original wikihop/{filename} in QAngaroo archive")
      destination = directory / filename
      with zipped.open(members[0]) as stream, destination.open("wb") as target:
        shutil.copyfileobj(stream, target)
      paths[split] = destination
  return _manifest(directory, "wikihop", paths, source)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("dataset", choices=["cosmos", "wikihop"])
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
      "--archive", help="WikiHop: local QAngaroo ZIP or HTTPS URL")
  parser.add_argument(
      "--sha256", help="Optional expected WikiHop archive hash")
  args = parser.parse_args()
  if args.dataset == "cosmos":
    paths = download_cosmos(args.output)
  else:
    if args.archive:
      paths = prepare_wikihop(args.archive, args.output,
                              expected_sha256=args.sha256)
    else:
      if args.sha256:
        parser.error("--sha256 requires --archive")
      paths = download_wikihop(args.output)
  print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == "__main__":
  main()
