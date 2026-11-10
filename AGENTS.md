# BertDM agent guide

Kaggle-oriented multiple-choice model using a pinned NeoBERT encoder. Keep reusable
logic in Python modules and use the same preprocessing for training and inference.

## Layout

- `bertdm/`: configuration, dataset readers/downloads, tokenization, masking, model,
  and the public `DecisionPipeline` interface.
- `bertdm/_vendor/`: inspected NeoBERT source and portable packed SwiGLU. Preserve
  attribution and record upstream-source changes in `NOTICE.md`.
- `examples/`: executable training and inference entry points.
- `tests/`: offline tiny-model tests; CUDA checks skip when unavailable.
- `docs/`: architecture, data provenance, and recorded validation results.

## Required invariants

- Preserve official Cosmos QA and WikiHop train/validation membership, support
  order, and candidate order. Normalize to `{context, question, choices, label}`
  with zero-based labels. Retain repeated Cosmos answer slots.
- Read local files or download native data files; do not depend on legacy Hugging
  Face dataset scripts. Keep source revisions and download hashes documented.
- Reserve complete choices, the full question, and structural special tokens
  before deterministic context truncation. Reject malformed examples and impossible
  budgets explicitly. Never drop choices or use labels to select context.
- Shared context/question tokens read only their shared group. Each choice reads
  the shared group and itself. Enforce this in every head and layer with additive
  `0/-inf` bias: rows are queries, columns are keys.
- Force math SDPA and reject packing. Real queries cannot read padding; padded
  query rows must remain numerically safe and their hidden states must be zeroed.
- Preserve pretrained per-head QKV layout, RoPE, normalization, residuals, and
  internal FFNs. Fail on missing/incompatible encoder weights. Remote checkpoints
  require immutable commit revisions.
- Mean-pool only choice content. Use one shared gated scoring head, mask invalid
  slots, and preserve offline model/config/tokenizer save and reload.
- Keep sequential-position order sensitivity and truncation limits documented.

## Checks

With the project environment active:

```bash
python -m pip install -e ".[data,test]"
python -m pytest -q
python -m compileall -q bertdm examples tests
```

If sandbox temporary-directory access fails, use an unused workspace directory:
`python -m pytest -q --basetemp=.cache/pytest-UNIQUE`.

For behavioral changes, run focused tests and the relevant suite. Check attention
isolation across layers, padding, gradients, probabilities, budgets, and round trips
when affected. Keep routine tests offline; large downloads and full-checkpoint GPU
checks are separate. Report actual checks and limitations without treating smoke
checks as task accuracy or Kaggle capacity measurements.

Keep datasets, weights, environments, caches, and generated outputs in ignored
locations. Update examples and documentation when public behavior changes.
