# Validation performed

Checked on 2026-11-09 in Windows / Python 3.12.10, PyTorch 2.13.0+cu130,
Transformers 4.48.3, PyArrow 26.0.1, and an NVIDIA RTX 4050 Laptop GPU.

## Automated checks

`python -m pytest -q --basetemp=.cache/pytest-run4 --tb=short`:
**42 passed**, including the CUDA test. The explicit temporary directory keeps
test writes within the implementation environment's sandbox. Normal environments
can use `python -m pytest -q`.

Coverage includes the literal query/key mask matrix; forbidden attention weights
in every head and all three tiny layers; fixed-length choice perturbation and
gradient isolation; context sensitivity; batch/standalone padding invariance;
safe padded queries and zeroed hidden states; independent content mean pooling;
finite gradients through embeddings, each original FFN, and the shared head;
CPU bf16 and CUDA fp16 autocast; forced math SDPA dispatch; probabilities and
temperature; strict checkpoint loading; rejecting missing encoder weights;
offline tokenizer/config/weight reload even without the original snapshot;
official-format readers; variable choices; reserved length budgets; and downloads.

An unrestricted tiny input also matches the original vendored NeoBERT encoder
numerically, checking that the adapter preserves its RoPE, norms, residuals,
QKV interpretation, and FFN computation when the permitted edges agree.

Editable installation, both example CLIs (`--help`), and Python compilation were
checked. The environment reports an unrelated optional Triton/CUDA-toolkit lookup
warning even though CUDA execution succeeds. Transformers 4.48.3 also warns that
NeoBERT's `max_length` overlaps its generation configuration naming; this encoder
does not use generation APIs.

## Real checkpoint and data checks

- Downloaded the pinned public checkpoint
  `5424c8efeea6491b151d62dee55a752165407430` (980,567,608-byte safetensors file).
  Compared **all 170 encoder tensors** exactly to their original safetensors
  values after loading. The retained encoder has 221,665,536 parameters. The
  scoring head is newly initialized; the unused MLM decoder is excluded.
- Ran CPU inference with the full checkpoint, and a real WikiHop validation
  example with 18 choices. Valid probabilities and CE loss were finite.
- Saved/reloaded the full encoder/head/config/tokenizer bundle offline. The same
  real example produced exactly identical probabilities after reload.
- Ran full-model CUDA fp16-autocast forward/backward at batch size 1, 256-token
  budget. Loss/probabilities and **all parameter gradients** were finite.
  Peak PyTorch allocated memory was 2,009,506,304 bytes (about 1.87 GiB), without
  optimizer state. This is a smoke result on this GPU, not a Kaggle capacity claim.
- Normalized and tokenized **all Cosmos QA train/validation rows** from the pinned
  authors' CSVs at a 512-token budget: **25,262 / 2,985**, no malformed records or
  budget rejections. Longest resulting sequences were 341 / 261 tokens.
  Repeated answer text in 10 official training rows was retained.
- Normalized and tokenized **all original WikiHop train/validation rows** from
  the pinned native Parquet mirror at a 512-token budget: **43,738 / 5,129**, no
  malformed records or budget rejections. Maximum choice counts were 79 / 69.
  Long document context was truncated while preserving every complete candidate.

Downloads and smoke artifacts reside in ignored `.cache/`; they are not package
source or trained task checkpoints.

## Remaining limits

No full task fine-tuning, benchmark accuracy, multi-seed evaluation, or execution
inside Kaggle was performed. GPU smoke checks do not establish optimizer memory,
throughput, long-sequence stability, or 4096-token batch capacity. PyTorch versions
other than the recorded runtime were not exercised here.

The WikiHop mirror's full files and split counts were checked, but were not
independently compared byte-for-byte with an original QAngaroo ZIP. Explicit
archive loading is available when that provenance is required. User-supplied
local snapshot/data provenance remains the caller's responsibility.

Sequential positions make choice order and earlier choice lengths affect later
choices' RoPE distances to context. Fixed-position isolation does not imply
permutation invariance. Deterministic prefix/tail truncation can remove relevant
multi-hop evidence without exceeding the budget. Math attention is quadratic;
FlashAttention, packing, arbitrary backbone architectures, and exact optimizer/RNG
training-state restoration are outside the implemented contract.
