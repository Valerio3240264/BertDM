"""Portable eager equivalent of xFormers' packed SwiGLU (same state keys).

The checkpoint stores w12 = concatenate([gate, value], dim=0), and w3.
No FFN parameters are removed, reinitialized after loading, or approximated.
"""

from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
  def __init__(self, in_features, hidden_features, out_features, bias=False):
    super().__init__()
    self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
    self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

  def forward(self, x):
    gate, value = self.w12(x).chunk(2, dim=-1)
    return self.w3(F.silu(gate) * value)
