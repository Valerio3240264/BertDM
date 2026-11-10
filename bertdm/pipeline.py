"""Tokenizer/model bundle used identically for training and inference."""

from pathlib import Path

import torch
from transformers import AutoTokenizer

from .config import DecisionConfig
from .data import ExampleError
from .model import DecisionModel
from .tokenization import DecisionCollator, DecisionTokenizer


class DecisionPipeline:
  def __init__(self, model, tokenizer):
    if len(tokenizer) > model.backbone.config.vocab_size:
      raise ValueError(
          "Tokenizer vocabulary exceeds checkpoint embedding table")
    if tokenizer.pad_token_id != model.backbone.config.pad_token_id:
      raise ValueError("Tokenizer and checkpoint disagree about pad_token_id")
    self.model = model
    self.tokenizer = tokenizer
    self.encode = DecisionTokenizer(tokenizer, max_length=model.config.max_length,
                                    supported_length=model.backbone.config.max_length,
                                    context_truncation=model.config.context_truncation)
    self.collate = DecisionCollator(tokenizer.pad_token_id)

  @classmethod
  def from_checkpoint(cls, config=None, *, local_files_only=False, cache_dir=None):
    config = config or DecisionConfig()
    tokenizer = AutoTokenizer.from_pretrained(
        config.checkpoint, revision=config.revision, local_files_only=local_files_only,
        cache_dir=cache_dir, trust_remote_code=False
    )
    model = DecisionModel.from_checkpoint(
        config, local_files_only=local_files_only, cache_dir=cache_dir)
    return cls(model, tokenizer)

  def tokenize_splits(self, splits):
    result = {}
    for split, examples in splits.items():
      result[split] = []
      for index, example in enumerate(examples):
        try:
          result[split].append(self.encode(example))
        except ExampleError as exc:
          raise ExampleError(f"{split}: record {index}: {exc}") from exc
    return result

  @torch.inference_mode()
  def predict(self, examples, *, temperature=1.0, return_pooled=False):
    device = next(self.model.parameters()).device
    batch = self.collate([self.encode(e) for e in examples])
    batch = {name: value.to(device) for name, value in batch.items()}
    was_training = self.model.training
    self.model.eval()
    try:
      output = self.model(**batch, return_pooled=return_pooled)
      probabilities = output.probabilities(temperature)
      result = {"logits": output.logits, "valid_choice_mask": output.valid_choice_mask,
                "probabilities": probabilities, "predictions": probabilities.argmax(-1)}
      if return_pooled:
        result["pooled_representations"] = output.pooled_representations
      return result
    finally:
      self.model.train(was_training)

  def save_pretrained(self, directory):
    self.model.save_pretrained(directory, self.tokenizer)

  @classmethod
  def from_pretrained(cls, directory, *, device="cpu"):
    model = DecisionModel.from_pretrained(directory, device=device)
    tokenizer = AutoTokenizer.from_pretrained(Path(directory) / "tokenizer", local_files_only=True,
                                              trust_remote_code=False)
    return cls(model, tokenizer)
