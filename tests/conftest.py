from bertdm._vendor.neobert import NeoBERTConfig
from bertdm import DecisionConfig, DecisionEncoder, DecisionModel, DecisionPipeline
from transformers import BertTokenizerFast
import torch
import pytest
import os
from pathlib import Path

# Keep all Transformers writes within the workspace and tests entirely offline.
os.environ.setdefault("HF_HOME", str(
    Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture
def tokenizer(tmp_path):
  vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alpha", "beta", "gamma", "delta",
           "context", "question", "first", "second", "third", "choice", "long", "short", "a", "b", "c", "?"]
  path = tmp_path / "vocab.txt"
  path.write_text("\n".join(vocab), encoding="utf-8")
  return BertTokenizerFast(vocab_file=str(path), do_lower_case=True, model_max_length=64)


@pytest.fixture
def pipeline(tokenizer):
  torch.manual_seed(7)
  config = NeoBERTConfig(hidden_size=24, num_hidden_layers=3, num_attention_heads=3,
                         intermediate_size=48, vocab_size=len(tokenizer), max_length=64,
                         norm_eps=1e-5, pad_token_id=tokenizer.pad_token_id)
  model = DecisionModel(DecisionEncoder(config), DecisionConfig(
      max_length=48, head_width=17, dropout=0.0))
  return DecisionPipeline(model, tokenizer)


@pytest.fixture
def examples():
  return [
      {"context": "alpha beta context gamma", "question": "question ?",
       "choices": ["first choice", "second long choice", "third choice"], "label": 1},
      {"context": "short context", "question": "question",
       "choices": ["alpha", "beta choice"], "label": 0},
  ]


@pytest.fixture
def batch(pipeline, examples):
  return pipeline.collate([pipeline.encode(e) for e in examples])
