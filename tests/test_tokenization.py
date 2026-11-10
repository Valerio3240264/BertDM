import pytest
import torch

from bertdm import DecisionCollator, DecisionTokenizer, ExampleError, LengthBudgetError


def test_complete_choices_reserved_before_context(tokenizer, examples):
  encode = DecisionTokenizer(tokenizer, max_length=19, supported_length=64)
  row = dict(examples[0], context="context " * 100)
  encoded = encode(row)
  assert len(encoded["input_ids"]) == 19
  for index, choice in enumerate(row["choices"], 1):
    owned = [t for t, g, c in zip(encoded["input_ids"], encoded["choice_group_ids"],
                                  encoded["choice_content_mask"]) if g == index and c]
    assert owned == tokenizer.encode(choice, add_special_tokens=False)
  other_label = encode(dict(row, label=0))
  assert {k: v for k, v in encoded.items() if k != "labels"} == {
      k: v for k, v in other_label.items() if k != "labels"}
  assert encoded == encode(row)


def test_exact_budget_zero_context_and_tail(tokenizer):
  row = {"context": "alpha beta gamma delta",
         "question": "question", "choices": ["a", "b"], "label": 0}
  # 3 shared specials + question + 2 content + 2 choice separators = 8.
  encode = DecisionTokenizer(
      tokenizer, max_length=8, supported_length=64, context_truncation="tail")
  result = encode(row)
  assert len(result["input_ids"]) == 8
  assert result["input_ids"][:3] == [tokenizer.cls_token_id, tokenizer.sep_token_id,
                                     tokenizer.convert_tokens_to_ids("question")]
  tail = DecisionTokenizer(tokenizer, max_length=10,
                           supported_length=64, context_truncation="tail")(row)
  assert tail["input_ids"][1:3] == tokenizer.encode(
      "gamma delta", add_special_tokens=False)


def test_budget_rejects_choices_and_question(tokenizer):
  encode = DecisionTokenizer(tokenizer, max_length=10, supported_length=64)
  row = {"context": "context", "question": "question",
         "choices": ["a " * 8, "b"], "label": 0}
  with pytest.raises(LengthBudgetError, match="complete choices"):
    encode(row)
  with pytest.raises(LengthBudgetError, match="complete question"):
    encode(dict(row, choices=["a", "b"], question="question " * 10))
  with pytest.raises(ValueError, match="supported_length"):
    DecisionTokenizer(tokenizer, max_length=65, supported_length=64)


@pytest.mark.parametrize("text", ["[SEP]", "\u200b"])
def test_empty_tokenized_choice_rejected(tokenizer, text):
  encode = DecisionTokenizer(tokenizer, max_length=32, supported_length=64)
  with pytest.raises(ExampleError, match="no tokenized content"):
    encode({"context": "context", "question": "question",
           "choices": [text, "b"], "label": 0})


def test_special_ids_excluded_from_choice_content(tokenizer):
  encode = DecisionTokenizer(tokenizer, max_length=32, supported_length=64)
  result = encode({"context": "context", "question": "question",
                  "choices": ["a [MASK] b", "c"], "label": 0})
  for token, content in zip(result["input_ids"], result["choice_content_mask"]):
    if token in tokenizer.all_special_ids:
      assert not content
  assert sum(result["choice_content_mask"]) == 3


def test_variable_counts_dynamic_padding(pipeline, examples, batch):
  assert batch["valid_choice_mask"].tolist(
  ) == [[True, True, True], [True, True, False]]
  for index, row in enumerate(examples):
    length = len(pipeline.encode(row)["input_ids"])
    assert (batch["choice_group_ids"][index, length:] == -1).all()
    assert not batch["choice_content_mask"][index, length:].any()
    assert not batch["context_mask"][index, length:].any()
    assert batch["special_tokens_mask"][index, length:].all()
    assert torch.equal(batch["position_ids"]
                       [index, :length], torch.arange(length))
  encoded = [pipeline.encode(e) for e in examples]
  del encoded[0]["labels"]
  with pytest.raises(ValueError, match="mix"):
    pipeline.collate(encoded)
  with pytest.raises(ValueError, match="empty"):
    DecisionCollator(0)([])
