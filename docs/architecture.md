# Implementation contract

Inspected source: `chandar-lab/NeoBERT` at immutable revision
`5424c8efeea6491b151d62dee55a752165407430`. The actual safetensors header stores
`model.encoder.weight`, `model.transformer_encoder.*.{qkv,wo}.weight`,
`model.transformer_encoder.*.ffn.{w12,w3}.weight`, both RMSNorms per block,
`model.layer_norm.weight`, and an MLM `decoder.{weight,bias}`.

`DecisionEncoder` subclasses the vendored upstream encoder.
`DecisionEncoderBlock` subclasses its block and inherits its original `forward`:

1. `attention_norm(x)` → attention output → residual addition.
2. `ffn_norm(x)` → original packed SwiGLU → residual addition.
3. Final encoder RMSNorm, after all blocks.

The local block factory avoids building or reinitializing a replacement backbone
after pretrained weights load. The decision head is initialized separately;
calling its constructor never calls `post_init()` over the loaded backbone.
Loading fails on any missing/unexpected encoder key or shape mismatch. The
published checkpoint's MLM decoder is the only permitted extra parameter pair.
The default checkpoint has 28 layers, width 768, 12 heads, and a 4096-position
complex RoPE buffer. The buffer is regenerated deterministically on reload.

The QKV projection is reshaped as `[B,L,H,3*D]` then split on the final axis.
Splitting the original projection into three width-sized blocks would corrupt
the pretrained weight interpretation. Queries and keys pass through the original
`apply_rotary_emb` (complex64 multiplication with original float32 frequency
construction), then are transposed to `[B,H,L,D]`.

Upstream's two-dimensional key-padding expansion cannot accept the asymmetric
four-dimensional mask. Its diagnostic eager path multiplies scores by the mask;
its SDPA path converts the mask to bool. Neither is used here. The adapter forwards
an additive `[B,1,L,L]` bias unchanged to math SDPA (broadcast to every head).
The alternate diagnostic calculation adds that bias before softmax. No attention
dropout is introduced; upstream uses zero. Only the new scoring head has dropout.
The optional FlashAttention import is disabled, and packed inputs are rejected
before any layer runs.

Optional training activation checkpointing uses PyTorch's non-reentrant
`checkpoint` around each original block, with the additive bias and RoPE
frequencies passed explicitly and RNG preservation enabled. Recomputed attention
still forces math SDPA. Inference runs blocks directly. Checkpointing changes
activation storage, not the attention graph or pretrained parameter layout.

Shared group 0 comprises the complete question and shared specials as well as
the retained document context. This makes question information accessible to all
choices without letting shared states relay information between choices. Choice
separators belong to their own group so they cannot relay another choice either.
No token is shared across examples. No global CLS pooling is used for scoring.

Each valid choice must have a positive content count. Pooling accumulates in fp32
and returns zero for invalid slots; logits mask those slots to `-inf` before CE or
softmax. Invalid padded logits are expected to be nonfinite; valid logits, loss,
probabilities, hidden states, and gradients must remain finite. A safe padded
query sentinel is an implementation detail; it cannot communicate into any real
token and is removed by zeroing states after every block.

Positions are sequential; reordering choices changes their RoPE distances to
context. Changing a choice's **content at fixed length/positions** cannot affect
shared states or other choices at any layer. Changing its length can change the
context budget and shift subsequent positions before the encoder runs. This is
an expected preprocessing/position effect, not an attention-mask leak.

The length budget reserves each complete choice and all structural specials,
then the full question. Only context is truncated, deterministically from a
configured end. Budgets are checked against `backbone.config.max_length`, not
the BERT tokenizer's legacy nominal length. Labels participate only in validation
and the loss, never in context selection or choice layout.
