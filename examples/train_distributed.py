"""torchrun --standalone --nproc-per-node=2 examples/train_distributed.py --config run.json"""

import argparse
import json
import os
import sys
from pathlib import Path


def prepare_environment(config):
  output = Path(config["output_dir"]).resolve()
  # Set cache and determinism environment before importing Torch/Transformers.
  os.environ["HF_HOME"] = str(output / "hf-cache")
  os.environ["HF_HUB_CACHE"] = str(output / "hf-cache" / "hub")
  os.environ["HF_HUB_OFFLINE"] = "0" if config.get("allow_downloads", True) else "1"
  os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
  sys.dont_write_bytecode = True
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cpu_worker(rank, world_size, store_uri, config):
  """CPU verification fallback for Windows builds whose torchrun lacks libuv."""
  prepare_environment(config)
  os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
                    LOCAL_WORLD_SIZE=str(world_size))
  import torch
  import torch.distributed as dist
  from datetime import timedelta
  torch.set_num_threads(1)
  dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=world_size,
                          timeout=timedelta(minutes=5))
  try:
    from bertdm.training_run import run_training
    run_training(config)
  finally:
    if dist.is_initialized():
      dist.destroy_process_group()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True, help="JSON configuration from train.ipynb")
  parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
  parser.add_argument("--cpu-workers", type=int, help="CPU/Gloo verification fallback; CUDA uses torchrun")
  args = parser.parse_args()
  config = json.loads(args.config.read_text(encoding="utf-8"))
  prepare_environment(config)
  if args.cpu_workers is not None:
    if config.get("device", "auto") == "auto":
      import torch
      if not torch.cuda.is_available():
        config = dict(config, device="cpu")
    if args.cpu_workers < 2 or config.get("device") != "cpu":
      parser.error("--cpu-workers requires >=2 workers and device=cpu in configuration")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
      parser.error("Do not nest --cpu-workers inside torchrun")
    from bertdm.training_run import _paths
    output, _ = _paths(config)
    output.mkdir(parents=True, exist_ok=True)
    import uuid
    import torch.multiprocessing as mp
    store = output / (".gloo-store-" + uuid.uuid4().hex)
    try:
      mp.spawn(cpu_worker, args=(args.cpu_workers, store.as_uri(), config), nprocs=args.cpu_workers, join=True)
    finally:
      store.unlink(missing_ok=True)
    return
  from bertdm.training_run import run_training
  run_training(config)


if __name__ == "__main__":
  main()
