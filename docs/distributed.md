# Single-node multi-GPU training

Open `train.ipynb` and set `CFG["distributed"]["num_processes"]` to the allocated
GPU count, or `"auto"` for one worker per visible GPU. `1` retains the in-process
single-GPU/CPU workflow. GPU workers require a Linux CUDA PyTorch build with NCCL,
as provided by a suitable Kaggle GPU runtime. This repository supports one node;
it does not implement FSDP, tensor/model parallelism, or elastic world-size changes.

## Launch and batch semantics

The notebook saves its single configuration cell to `notebook-config.json`, releases
its preview model and encoded arrays, and launches `examples/train_distributed.py`:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 examples/train_distributed.py --config /kaggle/working/bertdm-run/notebook-config.json
```

The command assumes the repository is installed or the script is invoked from its
checkout. The script also adds the repository root to Python's import path, so
read-only source attachments work. Set output/cache directories under
`/kaggle/working`, and read-only inputs under `/kaggle/input`.

`LOCAL_RANK` selects a worker's CUDA device. `RANK` identifies it globally and
`WORLD_SIZE` determines the global batch. Use `device="auto"` or `"cuda"` for CUDA
DDP; do not hardcode one GPU index for all workers. Rank zero warms the model/data
caches before other ranks read them. Each worker then loads compact cached records
and holds its own CPU dataset objects and model/optimizer replica.

The global effective batch is:

```
world_size * micro_batch_size * gradient_accumulation
```

For example, one GPU with microbatch 1/accumulation 8 and two GPUs with microbatch
1/accumulation 4 both use global batches of 8. Learning rates are not automatically
scaled. Warmup/decay steps are calculated from global updates. Adding GPUs without
changing accumulation increases effective batch size and reduces updates per epoch.
DDP improves throughput; GPU memory is not pooled. The complete model, gradients,
and ordinary AdamW state must fit on each GPU.

## Preserved model and sampling behavior

`bertdm/distributed.py` provides rank discovery, collective checks, coordinated
setup errors, and primary-only I/O. `bertdm/training_run.py` handles the common
configuration-driven worker workflow. `Trainer` owns an unwrapped `raw_model` for
parameter groups/serialization and a DDP wrapper for training forwards/backwards.

RoPE's deterministic complex-valued buffer is constructed identically on each
worker; DDP buffer broadcasting is disabled, including initialization buffer sync
on the supported interface. Parameter synchronization remains enabled. The model's
per-example asymmetric additive mask, math SDPA, packed-input rejection, original
normalization/FFNs, AMP, and non-reentrant activation checkpointing are preserved.
Different sequence lengths/choice counts on different ranks do not require logit
padding across ranks: DDP synchronizes parameter gradients, not logits.

Every rank regenerates the same dataset-weighted global epoch plan. Each optimizer
window is partitioned across ranks and microsteps, with choice permutation seeds
attached to the original draws. Every real draw is processed once globally, even
though replacement sampling may intentionally select a dataset row multiple times.

The final incomplete window uses zero-weight dummy slots so every rank executes
the same number of forwards/backwards. Dummies are valid encoded examples, excluded
from loss statistics/exposure counts. For each microstep, the differentiable loss
is `world_size / global_real_count * sum(active * per_example_CE)`. DDP's averaging
then yields the true mean gradient over the whole global accumulation window.
There is no further division by accumulation count. Intermediate microsteps wrap
both forward and backward in `DDP.no_sync()`; the last synchronizes gradients.

## Validation, failures, and logging

Validation assigns indices `rank::world_size` for each dataset. Empty shards are
allowed; duplicate padding is never used. The unwrapped model runs local forwards,
so different validation batch counts do not create unmatched DDP calls. Counts,
correct predictions, and NLL sums are globally reduced per dataset and choice bin.
Metrics use the reduced denominators, rather than averages of rank-level means.

All ranks coordinate nonfinite training loss/gradient decisions. Validation
nonfinite checks synchronize after local dataset evaluation. Primary I/O failures
are communicated to peers. Unexpected worker failures/OOM terminate the launcher
job; restart from the last completed checkpoint. There is no rank-local skip policy
that could silently desynchronize optimizer steps.

Only rank zero writes metadata, logs, checkpoint bundles, and best/latest pointers,
or prunes this trainer's unreferenced checkpoint folders. Global training loss,
real-example counts and dataset exposure are logged once. Step timing uses the
maximum rank duration. Dataset summaries retain original validation coverage.

## Checkpoint and resume contract

The new training-state format is version 2. A checkpoint stores common model,
AdamW, scheduler and global progress, plus a list of rank-specific RNG, scaler,
environment and device state. Every rank participates in saving; never call the
collective `Trainer.save_checkpoint()` on rank zero alone.

Resume checks world size, code/data/configuration, precision, rank/device mapping,
and environment. Progress offsets refer to global real draws. Fixed seed/epoch
regenerates the plan, and each rank restores its own stochastic stream. Previous
best checkpoints are copied by rank zero into a new writable output directory.
Keep the complete output directory and relative pointers between Kaggle sessions.

Earlier training-state formats and changed world sizes are not accepted for exact
resume. Their inference bundles remain readable: initialize a new run from the
model weights/tokenizer when changing the optimization regime. Models remain
saved without DDP's `module.` prefix.

## Local checks and remaining limits

Two-process CPU/Gloo tests exercise real DDP, including zero-weight tail slots,
gradient equivalence to a single-process global batch, mask preservation, empty
and uneven validation shards, parameter synchronization, per-rank RNG, exact
dropout resume, and coordinated nonfinite/I/O failure handling.

The notebook is executed with tiny fixtures in both single-process and two-process
modes, including a second resumed session and reload/inference. On Windows, the
CPU-only verification branch uses `--cpu-workers 2` and a FileStore rendezvous
because the installed torchrun build lacks libuv. Kaggle/Linux CUDA continues to
use standard torchrun/NCCL. Local multiprocess checks require localhost sockets;
run outside a sandbox that blocks worker communication.

Actual two-GPU NCCL training and a Kaggle session have not been executed in this
implementation environment, which has one RTX 4050 Laptop GPU and no NCCL build.
CPU multiprocess tests and single-GPU checks cannot establish multi-GPU throughput,
long-sequence capacity, CUDA collective behavior, or full-task accuracy.

Validation recorded on 2026-11-10: the complete suite passed **61 tests** in
32.11 seconds, including the real two-process Gloo tests, both notebook modes with
resumed sessions, and existing single-GPU fp16/scaler checks. Python compilation,
notebook schema/clean-output checks, and `git diff --check` also passed. The initial
sandboxed multiprocess attempt timed out because worker communication was blocked;
the same tests passed outside that sandbox. The Windows notebook verification
used the CPU/FileStore fallback; the Linux torchrun/NCCL launcher remains untested
on actual multiple GPUs here.
