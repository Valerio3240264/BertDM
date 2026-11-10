"""Receiving queries are rows; source keys are columns."""

import math

import torch


def build_attention_bias(choice_group_ids, attention_mask, *, dtype=torch.float32):
  """Return additive [B, 1, L, L] bias, broadcast across *all* heads.

  Group 0 is shared context/question; 1..C are disjoint choices; -1 is padding.
  Padded queries get a self-only sentinel row, then their states are zeroed by
  the encoder. Real queries never see padded keys. No row is entirely -inf.
  """
  if choice_group_ids.ndim != 2 or attention_mask.shape != choice_group_ids.shape:
    raise ValueError(
        "Expected matching [batch, length] groups and attention_mask")
  if choice_group_ids.dtype != torch.long or attention_mask.dtype != torch.bool:
    raise ValueError("groups must be int64 and attention_mask must be bool")
  if not dtype.is_floating_point:
    raise ValueError("attention bias must have a floating dtype")
  active = attention_mask
  if not active.any(-1).all() or not torch.equal(choice_group_ids == -1, ~active):
    raise ValueError(
        "Every sequence needs real tokens; only padding has group -1")
  if (choice_group_ids < -1).any():
    raise ValueError("Invalid negative token group")
  queries = choice_group_ids.unsqueeze(-1)
  keys = choice_group_ids.unsqueeze(-2)
  allowed = (keys == 0) | ((queries > 0) & (queries == keys))
  allowed = allowed & active.unsqueeze(-1) & active.unsqueeze(-2)
  diagonal = torch.eye(
      active.shape[1], dtype=torch.bool, device=active.device).unsqueeze(0)
  allowed = allowed | (~active.unsqueeze(-1) & diagonal)
  return torch.zeros(allowed.shape, device=active.device, dtype=dtype).masked_fill(
      ~allowed, float("-inf")
  ).unsqueeze(1)


def masked_softmax(logits, valid_choice_mask, temperature=1.0):
  if not isinstance(temperature, (float, int)) or not math.isfinite(temperature) or temperature <= 0:
    raise ValueError("temperature must be finite and positive")
  if logits.ndim != 2 or valid_choice_mask.shape != logits.shape or valid_choice_mask.dtype != torch.bool:
    raise ValueError(
        "Expected logits and bool valid_choice_mask with shape [B, C]")
  if not valid_choice_mask.any(-1).all():
    raise ValueError("Each example needs at least one valid choice")
  if not torch.isfinite(logits[valid_choice_mask]).all():
    raise ValueError("Valid logits must be finite")
  # Center before scaling, avoiding positive overflow for very small temperatures.
  masked = logits.float().masked_fill(~valid_choice_mask, float("-inf"))
  centered = masked - masked.max(-1, keepdim=True).values
  # Avoid casting a tiny positive Python float to float32 zero before division.
  scaled = centered.double() if temperature < torch.finfo(
      torch.float32).tiny else centered
  return (scaled / temperature).softmax(-1).float().masked_fill(~valid_choice_mask, 0)
