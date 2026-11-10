# Data sources and tokenization

All readers preserve the supplied train/validation memberships and record order.
There is no random repartitioning, deduplication, support reordering, or choice
selection. `load_splits` accepts separate explicit paths and returns `train` and
`validation` lists of `{context, question, choices, label}`. Labels are zero-based.
Training/validation rows need labels; direct inference can omit `label`.

## Downloads and offline Kaggle

```bash
python -m bertdm.download cosmos --output /kaggle/working/data/cosmos
python -m bertdm.download wikihop --output /kaggle/working/data/wikihop
```

Cosmos uses the authors' raw
[train.csv and valid.csv](https://github.com/wilburOne/cosmosqa/tree/b6eb99cca4e2a51dd28a9a6f562534872d851639/data),
pinned to commit `b6eb99cca4e2a51dd28a9a6f562534872d851639`. It maps `answer0..3`
to four indexed choices. Repeated answer text in 10 official training rows is
retained as separate slots; the numeric supervision is preserved.

WikiHop's default download uses the original **unmasked** WikiHop files in the
[native Parquet distribution](https://huggingface.co/datasets/community-datasets/qangaroo/tree/a6842a0767a929f86efd2aa1d7a18f4a6d954266/wikihop),
pinned to `a6842a0767a929f86efd2aa1d7a18f4a6d954266`. The mirror maps the official
train/dev to train/validation (43,738 / 5,129 rows). Downloading is a direct HTTPS
file transfer; no dataset loader script is executed. Reading Parquet requires the
`data` extra (`pip install 'bertdm[data]'`). The `test` extra also installs PyArrow.

Alternatively, use the original QAngaroo v1.1 ZIP with `--archive`:

```bash
python -m bertdm.download wikihop --archive /kaggle/input/qangaroo/qangaroo_v1.1.zip --output /kaggle/working/data/wikihop-json
```

`--archive` also accepts an explicit current HTTPS ZIP URL, optionally with
`--sha256 HASH`. Only `wikihop/train.json` and `wikihop/dev.json` are copied to
fixed output paths; masked datasets and unrelated archive members are ignored.
Historical archive links can require manual acquisition; see the
[official QAngaroo page](https://qangaroo.cs.ucl.ac.uk/). Support documents are
already inside WikiHop records; no Wikipedia crawling is necessary. Respect the
source datasets' licenses, including QAngaroo's CC BY-SA 3.0 attribution.

Download helpers write manifests with source URLs/revisions and per-split SHA256
hashes. Use them to stage files as Kaggle Datasets for Internet-disabled sessions.
Arbitrary supplied local files remain the caller's provenance responsibility.

## Local formats

CSV, JSON arrays, JSONL/NDJSON, and Parquet are supported. For already normalized
records use `dataset="normalized"`. Official WikiHop JSON and native Parquet use
`query`, `supports`, `candidates`, and `answer`. Supports are joined with blank
lines in source order; candidates keep their original order and length. The
answer must match exactly one candidate string.

```python
from bertdm import load_splits

splits = load_splits("wikihop",
    train_file="/kaggle/working/data/wikihop/train.parquet",
    validation_file="/kaggle/working/data/wikihop/validation.parquet")
```

Malformed fields, empty text/choice lists, ambiguous WikiHop answer matches, and
missing/out-of-range labels raise descriptive errors with source split/index.
Loaders never silently drop records. They cannot certify arbitrary local files
as official or detect all possible contamination in caller-provided files.

## Length budget and metadata

Layout: `[CLS] context [SEP] question [SEP] choice0 [SEP] ... choiceN [SEP]`.
Reserve all complete choices, every separator, CLS, and the complete question
before truncating context. `context_truncation="head"` keeps the beginning;
`"tail"` keeps the end. Labels never influence selection. Choices-only overflow,
question overflow, and empty tokenized choice content are explicit errors.
`tokenize_splits` stops with split/index context on a rejection. No filtering,
choice dropping, answer-dependent evidence retrieval, or auto budget growth is
performed. If implementing a different rejection policy, report its impact on
split counts and validation coverage.

The budget is checked against `backbone.config.max_length` (4096 for the default
snapshot), not the tokenizer's legacy BERT nominal limit. A fixed configured
budget and source text always yield the same layout. Very long WikiHop documents
may lose multi-hop evidence even when complete candidates fit.

Per-token fields are `input_ids`, sequential `position_ids`, `attention_mask`
(true for real tokens), `context_mask` (shared context/question and specials),
`choice_group_ids` (0 shared, 1..C choices, -1 padding), `choice_content_mask`, and
`special_tokens_mask`. A separate `valid_choice_mask` tracks actual choice slots.
Pooling excludes every tokenizer special ID, including special IDs appearing
inside user text. Each valid choice must still have positive content count.
The collator dynamically right-pads sequence lengths and choice counts.
