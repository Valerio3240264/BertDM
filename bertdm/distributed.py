"""Small single-node DDP runtime; single-process calls keep ordinary semantics."""

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedRuntime:
  device: torch.device
  rank: int = 0
  world_size: int = 1

  @classmethod
  def current(cls, device):
    initialized = dist.is_available() and dist.is_initialized()
    return cls(torch.device(device), dist.get_rank() if initialized else 0,
               dist.get_world_size() if initialized else 1)

  @property
  def enabled(self):
    return self.world_size > 1

  @property
  def primary(self):
    return self.rank == 0

  def gather(self, value):
    if not self.enabled:
      return [value]
    values = [None] * self.world_size
    dist.all_gather_object(values, value)
    return values

  def check(self, condition, message, error_type=ValueError):
    """All ranks make the same decision before entering the next collective."""
    flag = torch.tensor(int(bool(condition)), dtype=torch.int32, device=self.device)
    if self.enabled:
      dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
      messages = self.gather(None if bool(condition) else f"rank {self.rank}: {message}")
      raise error_type("; ".join(m for m in messages if m))

  def local_call(self, function):
    """Coordinate errors in local setup/loading stages that contain no collectives."""
    if not self.enabled:
      return function()
    result, problem = None, None
    try:
      result = function()
    except Exception as exc:
      problem = f"rank {self.rank}: {type(exc).__name__}: {exc}"
    problems = self.gather(problem)
    if any(problems):
      raise RuntimeError("; ".join(p for p in problems if p))
    return result

  def primary_call(self, function):
    """Only rank zero performs shared I/O; broadcast its result or failure."""
    if not self.enabled:
      return function()
    response = [None]
    if self.primary:
      try:
        response[0] = (True, function())
      except Exception as exc:
        response[0] = (False, f"Rank-zero operation failed: {type(exc).__name__}: {exc}")
    dist.broadcast_object_list(response, src=0)
    ok, result = response[0]
    if not ok:
      raise RuntimeError(result)
    return result

  def sum(self, tensor):
    if self.enabled:
      dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

  def maximum(self, value):
    tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
    if self.enabled:
      dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def initialize_distributed(requested_device="auto"):
  """torchrun supplies ranks. CUDA uses NCCL; CPU/Gloo supports verification."""
  world = int(os.environ.get("WORLD_SIZE", "1"))
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  if requested_device == "auto":
    requested_device = "cuda" if torch.cuda.is_available() else "cpu"
  requested = torch.device(requested_device)
  if world > 1 and int(os.environ.get("LOCAL_WORLD_SIZE", str(world))) != world:
    raise ValueError("This launcher supports one node with multiple workers")
  if requested.type == "cuda":
    if not torch.cuda.is_available():
      raise ValueError("CUDA requested but unavailable")
    if world > torch.cuda.device_count():
      raise ValueError("Requested more workers than available CUDA devices")
    if world > 1 and requested.index is not None:
      raise ValueError("For DDP use device='cuda' or 'auto'; LOCAL_RANK assigns each GPU")
    device = torch.device("cuda", local_rank if world > 1 else requested.index or 0)
    torch.cuda.set_device(device)
    backend = "nccl"
    if world > 1 and not dist.is_nccl_available():
      raise RuntimeError("CUDA DDP requires a PyTorch build with NCCL (Kaggle/Linux)")
  elif requested.type == "cpu":
    device, backend = requested, "gloo"
  else:
    raise ValueError("Training supports CPU or CUDA")
  if world > 1 and not dist.is_initialized():
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
  return DistributedRuntime.current(device)


def finish_distributed():
  if dist.is_available() and dist.is_initialized():
    dist.destroy_process_group()
