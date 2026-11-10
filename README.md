# BertDM

NeoBERT decision model: one sequence contains shared context/question and every
choice. Each choice reads context and itself through an additive asymmetric mask.
Reusable code lives in `bertdm/*.py`; examples live in `examples/`.

## Kaggle setup

Copy the repository to `/kaggle/working/BertDM`, then run in a notebook:

```python
%cd /kaggle/working/BertDM
%pip install -e '.[data,test]'
```

Python 3.10+ / PyTorch 2.5+ are required. No xFormers, FlashAttention, or legacy
Hugging Face dataset scripts are needed. Default checkpoint:
`chandar-lab/NeoBERT@5424c8efeea6491b151d62dee55a752165407430`.
Compatible NeoBERT checkpoint IDs/paths and immutable revisions are configurable.
For offline Kaggle, attach the checkpoint/tokenizer snapshot and data files as
Kaggle Datasets; pass the local checkpoint path and `local_files_only=True`.

## Data and training

For joint training, open [train.ipynb](train.ipynb). Its single configuration cell
controls paths, balanced dataset sampling, choice-order augmentation, AMP,
accumulation, activation checkpointing, separate validation metrics, and resumable
best/latest checkpoints. Attach the repository alongside the notebook on Kaggle;
see [notebook validation](docs/notebook-validation.md) for executed checks and limits.
`requirements-kaggle.txt` pins notebook dependencies while preserving Kaggle's
installed CUDA PyTorch wheel. Reusable training logic is in
`bertdm/training.py` and `bertdm/training_data.py`.

```bash
python -m bertdm.download cosmos --output /kaggle/working/data/cosmos
python -m bertdm.download wikihop --output /kaggle/working/data/wikihop
python examples/train.py --dataset cosmos --train-file /kaggle/working/data/cosmos/train.csv --validation-file /kaggle/working/data/cosmos/valid.csv --max-length 512 --batch-size 1 --output /kaggle/working/decision
```

Official train/validation membership is preserved. Cosmos downloads the authors'
pinned CSVs; WikiHop downloads pinned native Parquet train/validation files from
a mirror of the original unmasked dataset. Local CSV, JSON/JSONL, Parquet, and
QAngaroo ZIP preparation are supported. WikiHop joins supports in source order,
keeps all candidates, and maps the exact answer to a zero-based candidate index.
See [data sources and downloading](docs/data.md) for paths, hashes, and archive support.

Normalization yields `{context, question, choices, label}`. The deterministic
budget reserves **all complete choices, specials, and the full question before
truncating context**. Set `max_length` within the checkpoint limit (4096 for this
snapshot); configure `context_truncation="head"` or `"tail"`. Malformed rows,
empty tokenized choices, and impossible budgets raise errors with split/index;
records and choices are never silently dropped. Labels never select evidence.

```python
from bertdm import DecisionConfig, DecisionPipeline, load_splits

pipeline = DecisionPipeline.from_checkpoint(DecisionConfig(
    max_length=512, head_width=256, dropout=0.1))
splits = load_splits("wikihop",
    train_file="/kaggle/working/data/wikihop/train.parquet",
    validation_file="/kaggle/working/data/wikihop/validation.parquet")
encoded = pipeline.tokenize_splits(splits)
batch = pipeline.collate(encoded["train"][:2])
output = pipeline.model(**batch, return_pooled=True)
output.loss.backward()
```

## Model and inference

The inspected [upstream NeoBERT implementation](https://huggingface.co/chandar-lab/NeoBERT/blob/5424c8efeea6491b151d62dee55a752165407430/model.py)
is reused with its pretrained QKV layout, RoPE, RMSNorms, residuals, internal
SwiGLUs, and final norm. Only attention dispatch/sequence orchestration change.
Every head/layer uses additive `0/-inf` masking: context reads context; choice i
reads context and choice i. Math SDPA is forced; packing is rejected. Safe padded
query rows avoid NaNs, then their states are zeroed. Dynamic metadata tracks
context, individual choices, content, specials, padding, and valid choice slots.
See [architecture](docs/architecture.md) and [source provenance](bertdm/_vendor/NOTICE.md).

Choice content means feed one shared head:
`h = SiLU(Wg*u + bg) * (Wv*u + bv); score = Wo*Dropout(h)`.
Outputs provide `logits`, `valid_choice_mask`, optional `pooled_representations`,
and optional CE `loss`. `output.probabilities(temperature=1.0)` applies masked
softmax; invalid slots have zero probability and `-inf` logits.

```python
pipeline.save_pretrained("/kaggle/working/decision")
restored = DecisionPipeline.from_pretrained("/kaggle/working/decision", device="cuda")
result = restored.predict([{
    "context": "It rained. Mira had no umbrella.",
    "question": "Why was Mira wet?",
    "choices": ["She walked in the rain.", "The sun dried her."],
}], temperature=1.0)
print(result["probabilities"], result["predictions"])
```

Or run `python examples/infer.py --bundle /kaggle/working/decision`. Bundles include
all encoder/head weights, both configs, and tokenizer, and reload offline with
this package installed. Optimizer/RNG state is separate. Use autocast for mixed
precision while keeping weights/RoPE buffers in fp32. New heads need fine-tuning.

## Checks and limitations

Run `python -m pytest -q`. Tiny tests cover all heads/layers, flow isolation,
padding, pooling, gradients, probabilities, budgets, readers/downloads, strict
pretrained loading, and full offline round trips. [Validation notes](docs/validation.md)
record actual checks and remaining limits.

Sequential positions preserve RoPE but make scores sensitive to choice order and
earlier choice lengths. Isolation holds at fixed lengths/positions. Context
truncation can remove multi-hop evidence. Math attention costs O(L²) per head per
layer; start with short sequences and batch size 1 on Kaggle. Full task training,
accuracy, and Kaggle GPU memory/performance require measurement on the target runtime.
