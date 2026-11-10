"""One sequence per example, with explicit token ownership and dynamic padding."""

import torch

from .data import ExampleError, validate_example


class LengthBudgetError(ExampleError):
  """Complete question/choices cannot fit; the caller must report or reject it."""


class DecisionTokenizer:
  def __init__(self, tokenizer, *, max_length, supported_length, context_truncation="head"):
    if type(max_length) is not int or not 0 < max_length <= supported_length:
      raise ValueError(
          "max_length must be positive and within checkpoint supported_length")
    if context_truncation not in ("head", "tail"):
      raise ValueError("context_truncation must be head or tail")
    if any(getattr(tokenizer, name) is None for name in ("cls_token_id", "sep_token_id", "pad_token_id")):
      raise ValueError("Tokenizer needs CLS, SEP, and PAD tokens")
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.context_truncation = context_truncation

  def __call__(self, example):
    row = validate_example(example, require_label=False)
    tok = self.tokenizer
    specials = set(tok.all_special_ids)
    # Raw documents may exceed tokenizer.model_max_length; our explicit
    # reservation/truncation below is what bounds the final encoder input.
    def encode(text): return tok.encode(
        text, add_special_tokens=False, truncation=False, verbose=False)
    choices = [encode(text) for text in row["choices"]]
    for i, ids in enumerate(choices):
      if not ids or not any(t not in specials for t in ids):
        raise ExampleError(f"choice {i} has no tokenized content")
    # CLS, two shared SEP tokens, and one SEP per choice are reserved first.
    choices_cost = sum(map(len, choices)) + len(choices) + 3
    if choices_cost > self.max_length:
      raise LengthBudgetError(f"complete choices and special tokens need {choices_cost} tokens; "
                              f"budget is {self.max_length}")
    question = encode(row["question"])
    if not question or not any(t not in specials for t in question):
      raise ExampleError("question has no tokenized content")
    reserved = choices_cost + len(question)
    if reserved > self.max_length:
      raise LengthBudgetError(f"complete question, choices and specials need {reserved} tokens; "
                              f"budget is {self.max_length}")
    context = encode(row["context"])
    available = self.max_length - reserved
    kept = context[:available] if self.context_truncation == "head" else context[-available:] if available else []
    input_ids = [tok.cls_token_id] + kept + \
        [tok.sep_token_id] + question + [tok.sep_token_id]
    groups = [0] * len(input_ids)
    content = [False] * len(input_ids)
    for i, choice in enumerate(choices, 1):
      input_ids.extend(choice + [tok.sep_token_id])
      groups.extend([i] * (len(choice) + 1))
      content.extend([t not in specials for t in choice] + [False])
    result = {
        "input_ids": input_ids,
        "attention_mask": [True] * len(input_ids),
        "position_ids": list(range(len(input_ids))),
        "context_mask": [g == 0 for g in groups],
        "choice_group_ids": groups,  # 0: shared context/question, 1..C: choices, -1: padding
        "choice_content_mask": content,
        "special_tokens_mask": [t in specials for t in input_ids],
        "valid_choice_mask": [True] * len(choices),
    }
    if row["label"] is not None:
      result["labels"] = row["label"]
    return result


class DecisionCollator:
  def __init__(self, pad_token_id):
    self.pad_token_id = pad_token_id

  def __call__(self, examples):
    if not examples:
      raise ValueError("Cannot collate an empty batch")
    labeled = ["labels" in e for e in examples]
    if any(labeled) and not all(labeled):
      raise ValueError("Cannot mix labeled and unlabeled examples")
    length = max(len(e["input_ids"]) for e in examples)
    choices = max(len(e["valid_choice_mask"]) for e in examples)
    defaults = {"input_ids": self.pad_token_id, "attention_mask": False, "position_ids": 0,
                "context_mask": False, "choice_group_ids": -1, "choice_content_mask": False,
                "special_tokens_mask": True, "valid_choice_mask": False}
    bool_keys = {"attention_mask", "context_mask",
                 "choice_content_mask", "special_tokens_mask", "valid_choice_mask"}
    batch = {}
    for key, fill in defaults.items():
      width = choices if key == "valid_choice_mask" else length
      values = [list(e[key]) + [fill] * (width - len(e[key]))
                for e in examples]
      batch[key] = torch.tensor(
          values, dtype=torch.bool if key in bool_keys else torch.long)
    if all(labeled):
      batch["labels"] = torch.tensor([e["labels"]
                                     for e in examples], dtype=torch.long)
    return batch
