"""NeoBERT with an additive asymmetric mask and one shared gated scoring head."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput

from ._vendor.neobert import EncoderBlock, NeoBERT, NeoBERTConfig
from ._vendor.rotary import apply_rotary_emb
from .config import DecisionConfig
from .masking import build_attention_bias, masked_softmax


class DecisionEncoderBlock(EncoderBlock):
  """Keep upstream forward/FFN/norm/residual; replace only attention dispatch."""

  def _att_block(self, x, attention_mask, freqs_cis, output_attentions,
                 max_seqlen=None, cu_seqlens=None):
    if cu_seqlens is not None or max_seqlen is not None:
      raise ValueError(
          "Packed/FlashAttention inputs are incompatible with choice isolation")
    batch, length, _ = x.shape
    # Important: upstream interleaves Q/K/V *within each head*, not globally.
    q, k, v = self.qkv(x).view(
        batch, length, self.config.num_attention_heads, 3 * self.config.dim_head
    ).chunk(3, dim=-1)
    q, k = apply_rotary_emb(q, k, freqs_cis)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    weights = None
    if output_attentions:
      # Diagnostic path uses exactly the same additive 0/-inf semantics.
      weights = (q.float() @ k.float().transpose(-1, -2)) / \
          self.config.dim_head ** 0.5
      weights = (weights + attention_mask.float()).softmax(-1)
      attended = (weights @ v.float()).to(x.dtype)
    else:
      # Force math: no automatic FlashAttention, efficient attention, or packing.
      with sdpa_kernel(backends=[SDPBackend.MATH]):
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask.to(q.dtype), dropout_p=0.0, is_causal=False
        )
    attended = attended.transpose(1, 2).reshape(
        batch, length, self.config.hidden_size)
    return self.wo(attended), weights


class DecisionEncoder(NeoBERT):
  block_class = DecisionEncoderBlock
  gradient_checkpointing = False

  def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    if gradient_checkpointing_kwargs not in (None, {"use_reentrant": False}):
      raise ValueError("Only non-reentrant checkpointing is supported")
    self.gradient_checkpointing = True

  def gradient_checkpointing_disable(self):
    self.gradient_checkpointing = False

  def forward(self, input_ids, attention_bias, attention_mask, position_ids=None,
              output_hidden_states=False, output_attentions=False, *, cu_seqlens=None, max_seqlen=None):
    if cu_seqlens is not None or max_seqlen is not None:
      raise ValueError(
          "Sequence packing and FlashAttention are explicitly unsupported")
    length = input_ids.shape[1]
    if length > self.config.max_length:
      raise ValueError("Sequence exceeds checkpoint RoPE limit")
    freqs = self.freqs_cis[:length].unsqueeze(
        0) if position_ids is None else self.freqs_cis[position_ids]
    x = self.encoder(input_ids)
    keep = attention_mask.unsqueeze(-1)
    x = x.masked_fill(~keep, 0)
    hidden, attentions = [], []
    for layer in self.transformer_encoder:
      # Original EncoderBlock.forward executes both residuals and its internal FFN.
      if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
        x, attn = checkpoint(layer, x, attention_bias, freqs, output_attentions,
                             use_reentrant=False, preserve_rng_state=True)
      else:
        x, attn = layer(x, attention_bias, freqs, output_attentions)
      x = x.masked_fill(~keep, 0)
      if output_hidden_states:
        hidden.append(x)
      if output_attentions:
        attentions.append(attn)
    x = self.layer_norm(x).masked_fill(~keep, 0)
    return BaseModelOutput(last_hidden_state=x,
                           hidden_states=tuple(
                               hidden) if output_hidden_states else None,
                           attentions=tuple(attentions) if output_attentions else None)


class SharedScoringHead(nn.Module):
  def __init__(self, hidden_size, width, dropout):
    super().__init__()
    self.gate = nn.Linear(hidden_size, width)
    self.value = nn.Linear(hidden_size, width)
    self.dropout = nn.Dropout(dropout)
    self.out = nn.Linear(width, 1, bias=False)

  def forward(self, pooled):
    h = F.silu(self.gate(pooled)) * self.value(pooled)
    return self.out(self.dropout(h)).squeeze(-1)


@dataclass
class DecisionOutput(ModelOutput):
  loss: torch.Tensor | None = None
  logits: torch.Tensor | None = None
  valid_choice_mask: torch.Tensor | None = None
  pooled_representations: torch.Tensor | None = None
  last_hidden_state: torch.Tensor | None = None
  hidden_states: tuple | None = None
  attentions: tuple | None = None

  def probabilities(self, temperature=1.0):
    return masked_softmax(self.logits, self.valid_choice_mask, temperature)


class DecisionModel(nn.Module):
  def __init__(self, backbone, config=None):
    super().__init__()
    self.config = config or DecisionConfig()
    if not isinstance(backbone, DecisionEncoder):
      raise TypeError(
          "backbone must be a DecisionEncoder with enforced attention")
    if self.config.max_length > backbone.config.max_length:
      raise ValueError(
          "Decision max_length exceeds checkpoint supported limit")
    if backbone.config.dim_head % 2:
      raise ValueError("RoPE requires an even head dimension")
    self.backbone = backbone
    self.head = SharedScoringHead(
        backbone.config.hidden_size, self.config.head_width, self.config.dropout)

  @classmethod
  def from_checkpoint(cls, config=None, *, local_files_only=False, cache_dir=None):
    config = config or DecisionConfig()
    backbone_config = NeoBERTConfig.from_pretrained(
        config.checkpoint, revision=config.revision, local_files_only=local_files_only, cache_dir=cache_dir
    )
    if backbone_config.model_type != "neobert":
      raise ValueError(
          "Only checkpoints using the inspected NeoBERT architecture are supported")
    if config.max_length > backbone_config.max_length:
      raise ValueError(
          "Decision max_length exceeds checkpoint supported limit")
    backbone, info = DecisionEncoder.from_pretrained(
        config.checkpoint, config=backbone_config, revision=config.revision,
        local_files_only=local_files_only, cache_dir=cache_dir,
        use_safetensors=True, output_loading_info=True
    )
    # The published checkpoint also contains an MLM decoder, unused for decisions.
    allowed = {"decoder.weight", "decoder.bias"}
    if (info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs")
            or set(info["unexpected_keys"]) - allowed):
      raise RuntimeError(
          f"Checkpoint is not an exact NeoBERT encoder match: {info}")
    return cls(backbone, config)

  def forward(self, input_ids, attention_mask, context_mask, choice_group_ids,
              choice_content_mask, valid_choice_mask, special_tokens_mask,
              position_ids=None, labels=None, *, return_pooled=False,
              output_hidden_states=False, output_attentions=False,
              cu_seqlens=None, max_seqlen=None):
    if cu_seqlens is not None or max_seqlen is not None:
      raise ValueError(
          "Sequence packing and FlashAttention are explicitly unsupported")
    if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.shape[0] == 0:
      raise ValueError("input_ids must be int64 [B, L]")
    if not 0 < input_ids.shape[1] <= self.config.max_length:
      raise ValueError("Sequence exceeds configured length budget or is empty")
    for mask in (attention_mask, context_mask, choice_content_mask, special_tokens_mask):
      if mask.shape != input_ids.shape or mask.dtype != torch.bool:
        raise ValueError("Token masks must be boolean [B, L]")
    if choice_group_ids.shape != input_ids.shape or choice_group_ids.dtype != torch.long:
      raise ValueError("choice_group_ids must be int64 [B, L]")
    if (valid_choice_mask.ndim != 2 or valid_choice_mask.shape[0] != input_ids.shape[0]
            or valid_choice_mask.shape[1] == 0 or valid_choice_mask.dtype != torch.bool
            or not valid_choice_mask.any(-1).all()):
      raise ValueError(
          "valid_choice_mask must be bool [B, C] with valid slots in each row")
    if not torch.equal(context_mask, (choice_group_ids == 0) & attention_mask):
      raise ValueError("context_mask disagrees with token ownership")
    if not context_mask.any(-1).all():
      raise ValueError("Every example needs shared context/question tokens")
    if (choice_content_mask & ((choice_group_ids <= 0) | ~attention_mask | special_tokens_mask)).any():
      raise ValueError(
          "Choice content cannot contain context, specials or padding")
    count = valid_choice_mask.shape[1]
    if (choice_group_ids > count).any():
      raise ValueError("Choice group exceeds available choice slots")
    slots = (choice_group_ids - 1).clamp_min(0)
    active_groups = choice_group_ids > 0
    if (active_groups & ~valid_choice_mask.gather(1, slots)).any():
      raise ValueError("Tokens assigned to an invalid choice slot")
    expected = torch.arange(
        input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
    if position_ids is None:
      position_ids = expected.masked_fill(~attention_mask, 0)
    elif (position_ids.shape != input_ids.shape or position_ids.dtype != torch.long
          or not torch.equal(position_ids[attention_mask], expected[attention_mask])
          or (position_ids[~attention_mask] != 0).any()):
      raise ValueError(
          "Only sequential positions (padding position 0) are supported")
    pooling = F.one_hot(slots, num_classes=count).transpose(1, 2)
    pooling = pooling * choice_content_mask.unsqueeze(1)
    denominators = pooling.sum(-1)
    if not torch.equal(denominators > 0, valid_choice_mask):
      raise ValueError(
          "Each valid choice needs content tokens; invalid slots must have none")
    if labels is not None:
      if (labels.shape != (input_ids.shape[0],) or labels.dtype != torch.long
              or (labels < 0).any() or (labels >= count).any()):
        raise ValueError("labels must be int64 [B] with in-range values")
      if not valid_choice_mask.gather(1, labels.unsqueeze(1)).all():
        raise ValueError("Label points to an invalid choice slot")
    bias = build_attention_bias(choice_group_ids, attention_mask)
    encoded = self.backbone(input_ids, bias, attention_mask, position_ids,
                            output_hidden_states, output_attentions)
    # Accumulate means in fp32, including when the encoder runs under autocast.
    with torch.autocast(device_type=input_ids.device.type, enabled=False):
      pooled = torch.bmm(pooling.float(), encoded.last_hidden_state.float())
      pooled = pooled / denominators.clamp_min(1).unsqueeze(-1)
    pooled = pooled.to(encoded.last_hidden_state.dtype)
    logits = self.head(pooled).masked_fill(~valid_choice_mask, float("-inf"))
    loss = F.cross_entropy(
        logits.float(), labels) if labels is not None else None
    return DecisionOutput(loss=loss, logits=logits, valid_choice_mask=valid_choice_mask,
                          pooled_representations=pooled if return_pooled else None,
                          last_hidden_state=encoded.last_hidden_state if output_hidden_states else None,
                          hidden_states=encoded.hidden_states, attentions=encoded.attentions)

  def save_pretrained(self, directory, tokenizer):
    """Save a standalone bundle: encoder+head, both configs, and tokenizer."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"decision": self.config.to_dict(
    ), "backbone": self.backbone.config.to_dict()}
    (directory / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_model(self, str(directory / "model.safetensors"))
    tokenizer.save_pretrained(directory / "tokenizer")

  @classmethod
  def from_pretrained(cls, directory, *, device="cpu"):
    """Reload fully offline; original Hub checkpoint/code is not contacted."""
    directory = Path(directory)
    payload = json.loads(
        (directory / "config.json").read_text(encoding="utf-8"))
    config = DecisionConfig(**payload["decision"])
    backbone = DecisionEncoder(NeoBERTConfig.from_dict(payload["backbone"]))
    model = cls(backbone, config)
    load_model(model, str(directory / "model.safetensors"),
               strict=True, device="cpu")
    return model.to(device).eval()
