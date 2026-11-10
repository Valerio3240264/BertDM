# Training notebook validation

`train.ipynb` was added and checked on 2026-11-10. It imports the repository's
`DecisionPipeline`, dataset normalizers, tokenizer/collator, and decision model.
Reusable cache/augmentation and training code lives in `bertdm/training_data.py`
and `bertdm/training.py`. The model gained optional non-reentrant activation
checkpointing around its original blocks; attention still forces math SDPA.

## Notebook workflow

- One parameter cell controls repository/data/checkpoint paths, dependency setup,
  sequence budget, head settings, sampling weights, seed, precision, accumulation,
  learning rates, scheduler, epochs, selection metric, and output/resume paths.
- Read-only inputs and code may reside in `/kaggle/input`. Explicit output/cache
  paths are checked to remain under `/kaggle/working` on Kaggle.
- Pinned data/model-library dependencies are in `requirements-kaggle.txt`.
  Kaggle's CUDA PyTorch wheel is retained; the environment and `pip freeze` are
  saved. Initial missing dependencies, model weights, and data need Internet or
  pre-staged offline files/wheels.
- Cache identity includes source content, tokenizer serialization/special IDs,
  revision, preprocessing versions/code, sequence budget and truncation policy.
  Compact cached records avoid storing redundant token masks in every example.
- Split summaries report original, retained and rejected counts, coverage,
  token-length statistics, and complete choice-count frequencies. Each rejection
  is recorded. Defaults prohibit training after any rejection; explicit overrides
  preserve and report the reduced coverage.
- Training samples datasets with explicit probabilities, uniformly samples rows
  within the selected training dataset, and shuffles complete choice blocks with
  consistent label/group remapping. Validation uses separate, original-order
  records with no augmentation.
- AdamW has encoder/head learning rates and separate decay/no-decay groups.
  Linear warmup/decay, example-weighted partial accumulation windows, gradient
  clipping, CUDA AMP/scaler and block checkpointing are supported.
- Accuracy/NLL are reported separately by dataset and choice-count range.
  Configurable macro or single-dataset accuracy/NLL selects the best checkpoint.
- Relative best/latest pointers reference model/tokenizer bundles plus optimizer,
  scheduler/scaler, RNG and progress state. Resume validates configuration,
  preprocessing, code, runtime, precision and training device. Same-directory
  resume removes logged updates newer than the saved optimizer boundary before
  replaying them. Save the whole output directory between Kaggle sessions.

## Checks actually executed

Environment: Windows, Python 3.12.10, PyTorch 2.13.0+cu130, Transformers 4.48.3,
PyArrow 26.0.1, NVIDIA GeForce RTX 4050 Laptop GPU.

- Notebook format validated with nbformat 5.10.4; all 10 code cells parsed as
  Python. The delivered notebook has 19 cells and no stored execution outputs.
- All notebook code cells executed in order using a tiny local NeoBERT checkpoint
  and synthetic Cosmos/WikiHop-format files. Execution included pinned-dependency
  checks, cache creation, summaries, smoke backward, optimizer updates, separate
  validation, best/latest saving, reload, and choice/probability output.
- The cells executed again with a new output directory and the saved latest
  checkpoint. Resume reused the preprocessing cache, restored training state,
  completed the planned updates, preserved the prior best, and ran inference.
  Input source files remained byte-identical. Tests run cells directly with
  Python `exec`; local Jupyter kernel startup hung in the sandbox and was stopped.
- The complete test suite passed **55 tests**, including baseline model/data
  tests. New tests cover deterministic weighted exposure, complete choice-block
  permutation and labels, cache invalidation/rejections, checkpointed attention
  and gradient parity, smoke-test weight/RNG preservation, separate NLL metrics,
  partial accumulation equivalence, exact CPU resume with dropout and RNG,
  checkpoint retention, rejection policy, nonfinite loss, and CUDA fp16 scaler
  resume. Exact CPU continuation reproduced uninterrupted final parameters.
- The full pretrained 28-layer NeoBERT passed an **11-token, two-choice CUDA fp16
  forward/backward smoke test** with gradient checkpointing and scale 1024.
  Loss and every trainable parameter's gradient were finite. No optimizer update
  was performed in that full-checkpoint smoke test. Peak allocated CUDA memory
  was 2,022,872,064 bytes, dominated by weights/gradients at this short length.
- Python compilation and example/notebook source checks passed. Expected local
  optional Triton-toolkit and upstream `max_length` naming warnings remain.

## Limits

**No full-dataset GPU fine-tuning or Kaggle session execution was performed.**
Synthetic tiny-model optimizer checks and a short full-model backward test do not
establish task accuracy, long-sequence/optimizer memory, throughput, or Kaggle
capacity. The previous full dataset preprocessing scans are recorded separately
in `validation.md`; the new notebook cache was exercised with small fixtures.

Exact continuation is supported at completed optimizer boundaries with the same
runtime/device, code, settings and data; no mid-accumulation gradients are saved.
Changing these requires a new run initialized from a model bundle. Cache files
and Python-serialized training states are trusted local artifacts; only resume
states produced by your own trusted runs. Checkpoint I/O uses several GB for a
full model with AdamW, and periodic saving can be expensive.

Context truncation can remove useful evidence. Sequential positions remain
sensitive to choice order and earlier choice lengths. Dataset-level sampling
uses replacement and controls expected exposure rather than guaranteeing every
unique example appears in each epoch. NLL is uncalibrated mean cross-entropy on
the retained official validation rows, with coverage reported explicitly.
