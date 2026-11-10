"""Serializable settings; remote checkpoints must use immutable commit hashes."""

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath


DEFAULT_CHECKPOINT = "chandar-lab/NeoBERT"
DEFAULT_REVISION = "5424c8efeea6491b151d62dee55a752165407430"


def validate_revision(checkpoint, revision):
  local = Path(checkpoint).is_absolute() or PureWindowsPath(
      checkpoint).is_absolute() or Path(checkpoint).is_dir()
  if not local and not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
    raise ValueError(
        "Remote checkpoints require a pinned 40-character commit revision")


@dataclass(frozen=True)
class DecisionConfig:
  checkpoint: str = DEFAULT_CHECKPOINT
  revision: str = DEFAULT_REVISION
  max_length: int = 1024
  head_width: int = 256
  dropout: float = 0.1
  context_truncation: str = "head"
  attention_backend: str = "sdpa_math"
  format_version: int = 1

  def __post_init__(self):
    checkpoint = str(self.checkpoint)
    # Capture local provenance now so offline bundle reload does not depend
    # on the original snapshot directory continuing to exist.
    if Path(checkpoint).is_dir():
      checkpoint = str(Path(checkpoint).resolve())
    object.__setattr__(self, "checkpoint", checkpoint)
    validate_revision(self.checkpoint, self.revision)
    for key in ("max_length", "head_width"):
      value = getattr(self, key)
      if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
      raise ValueError("dropout must be finite and in [0, 1)")
    if self.context_truncation not in ("head", "tail"):
      raise ValueError("context_truncation must be head or tail")
    if self.attention_backend != "sdpa_math":
      raise ValueError("Only sdpa_math supports the enforced backend contract")
    if self.format_version != 1:
      raise ValueError("Unsupported bundle format_version")

  def to_dict(self):
    return asdict(self)
