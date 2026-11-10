import pytest
import torch
from torch.nn import functional as F

from bertdm import DecisionConfig, DecisionEncoder, DecisionModel, DecisionPipeline, build_attention_bias, masked_softmax
from bertdm._vendor.neobert import NeoBERT, NeoBERTLMHead


def test_mask_orientation_and_safe_padded_queries():
  groups = torch.tensor([[0, 0, 1, 1, 2, 2, -1]])
  active = groups >= 0
  bias = build_attention_bias(groups, active)
  expected = torch.tensor([
      [1, 1, 0, 0, 0, 0, 0],
      [1, 1, 0, 0, 0, 0, 0],
      [1, 1, 1, 1, 0, 0, 0],
      [1, 1, 1, 1, 0, 0, 0],
      [1, 1, 0, 0, 1, 1, 0],
      [1, 1, 0, 0, 1, 1, 0],
      [0, 0, 0, 0, 0, 0, 1],
  ], dtype=torch.bool)
  assert bias.shape == (1, 1, 7, 7)
  assert torch.equal(bias[0, 0] == 0, expected)
  assert torch.isneginf(bias[0, 0][~expected]).all()
  assert torch.isfinite(torch.softmax(bias, -1)).all()


def test_every_head_every_layer_and_sdpa_diagnostic_parity(pipeline, batch):
  model = pipeline.model.eval()
  with torch.no_grad():
    sdpa = model(**batch, return_pooled=True, output_hidden_states=True)
    eager = model(**batch, output_attentions=True)
  torch.testing.assert_close(sdpa.logits, eager.logits)
  bias = build_attention_bias(
      batch["choice_group_ids"], batch["attention_mask"])
  forbidden = torch.isneginf(
      bias).expand(-1, model.backbone.config.num_attention_heads, -1, -1)
  assert len(eager.attentions) == 3
  for weights in eager.attentions:
    assert torch.isfinite(weights).all()
    assert (weights[forbidden] == 0).all()
    torch.testing.assert_close(
        weights.sum(-1), torch.ones_like(weights.sum(-1)))
  # Check the requested content mean independently, excluding SEP/CLS/PAD.
  for b in range(len(batch["input_ids"])):
    for c in range(batch["valid_choice_mask"].shape[1]):
      if batch["valid_choice_mask"][b, c]:
        selected = (batch["choice_group_ids"][b] == c +
                    1) & batch["choice_content_mask"][b]
        expected = sdpa.last_hidden_state[b, selected].mean(0)
        torch.testing.assert_close(sdpa.pooled_representations[b, c], expected)


def test_multilayer_information_flow_isolation_and_context_sensitivity(pipeline, batch):
  model = pipeline.model.eval()
  changed = {k: v.clone() for k, v in batch.items()}
  # Replace choice 0 content in place, preserving all lengths and positions.
  selected = (batch["choice_group_ids"] == 1) & batch["choice_content_mask"]
  changed["input_ids"][selected] = pipeline.tokenizer.convert_tokens_to_ids(
      "delta")
  with torch.no_grad():
    original = model(**batch, return_pooled=True, output_hidden_states=True)
    modified = model(**changed, return_pooled=True, output_hidden_states=True)
  for old, new in zip(original.hidden_states, modified.hidden_states):
    unaffected = batch["context_mask"] | (batch["choice_group_ids"] > 1)
    torch.testing.assert_close(
        old[unaffected], new[unaffected], rtol=0, atol=0)
    assert not torch.equal(old[selected], new[selected])
  torch.testing.assert_close(
      original.logits[:, 1:], modified.logits[:, 1:], rtol=0, atol=0)
  assert not torch.equal(original.logits[:, 0], modified.logits[:, 0])
  changed = {k: v.clone() for k, v in batch.items()}
  changed["input_ids"][:,
                       1] = pipeline.tokenizer.convert_tokens_to_ids("delta")
  with torch.no_grad():
    context_changed = model(**changed, return_pooled=True)
  # Every real choice can receive information from shared context.
  for b, c in batch["valid_choice_mask"].nonzero().tolist():
    assert not torch.equal(
        original.pooled_representations[b, c], context_changed.pooled_representations[b, c])


def test_padding_invariance_and_zero_states(pipeline, examples, batch):
  model = pipeline.model.eval()
  with torch.no_grad():
    together = model(**batch, output_hidden_states=True, return_pooled=True)
    alone = model(**pipeline.collate([pipeline.encode(examples[1])]))
    changed = {k: v.clone() for k, v in batch.items()}
    changed["input_ids"][~batch["attention_mask"]] = 9
    perturbed = model(**changed)
  torch.testing.assert_close(together.logits[1, :2], alone.logits[0])
  torch.testing.assert_close(together.logits, perturbed.logits, rtol=0, atol=0)
  assert torch.isfinite(together.loss)
  assert torch.isneginf(together.logits[1, 2])
  assert (together.last_hidden_state[~batch["attention_mask"]] == 0).all()
  assert (together.pooled_representations[1, 2] == 0).all()
  for hidden in together.hidden_states:
    assert torch.isfinite(hidden).all()
    assert (hidden[~batch["attention_mask"]] == 0).all()


def test_gradients_reach_embeddings_all_blocks_and_shared_head(pipeline, batch):
  model = pipeline.model.train()
  output = model(**batch)
  output.loss.backward()
  for name, parameter in model.named_parameters():
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all(), name
    assert parameter.grad.abs().sum() > 0, name
  # Embeddings used solely for padding have no gradient.
  assert (
      model.backbone.encoder.weight.grad[pipeline.tokenizer.pad_token_id] == 0).all()
  torch.testing.assert_close(
      output.loss, F.cross_entropy(output.logits, batch["labels"]))


def test_forbidden_gradient_paths(pipeline, batch):
  model = pipeline.model.eval()
  embedding = model.backbone.encoder(
      batch["input_ids"]).detach().requires_grad_(True)
  handle = model.backbone.encoder.register_forward_hook(
      lambda module, args, output: embedding)
  try:
    output = model(**batch)
    gradient = torch.autograd.grad(output.logits[0, 0], embedding)[0]
    assert gradient[0, batch["context_mask"][0]].abs().sum() > 0
    assert gradient[0, batch["choice_group_ids"][0] == 1].abs().sum() > 0
    assert (gradient[0, batch["choice_group_ids"][0] > 1] == 0).all()
    assert (gradient[1] == 0).all()
  finally:
    handle.remove()


def test_masked_probabilities_and_temperature(pipeline, batch):
  output = pipeline.model.eval()(**batch)
  probs = output.probabilities()
  assert torch.isfinite(probs).all()
  torch.testing.assert_close(probs.sum(-1), torch.ones(2))
  assert probs[1, 2] == 0
  logits = torch.tensor([[2.0, 4.0, 100.0]])
  mask = torch.tensor([[True, True, False]])
  torch.testing.assert_close(masked_softmax(logits, mask, 2)[
                             0, :2], torch.softmax(torch.tensor([1., 2.]), 0))
  for temperature in (0, -1, float("nan"), float("inf")):
    with pytest.raises(ValueError, match="temperature"):
      masked_softmax(logits, mask, temperature)
  with pytest.raises(ValueError, match="valid choice"):
    masked_softmax(logits, torch.zeros_like(mask))
  assert masked_softmax(logits, mask, 1e-30).tolist() == [[0.0, 1.0, 0.0]]
  assert masked_softmax(logits, mask, 1e-310).tolist() == [[0.0, 1.0, 0.0]]


def test_cpu_autocast_finite_forward_backward(pipeline, batch):
  with torch.autocast("cpu", dtype=torch.bfloat16):
    output = pipeline.model(**batch)
  assert torch.isfinite(output.loss)
  output.loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all()
             for p in pipeline.model.parameters())


def test_sdpa_dispatch_disables_incompatible_backends(pipeline, batch, monkeypatch):
  original = F.scaled_dot_product_attention
  calls = []

  def checked_sdpa(*args, **kwargs):
    assert torch.backends.cuda.math_sdp_enabled()
    assert not torch.backends.cuda.flash_sdp_enabled()
    assert not torch.backends.cuda.mem_efficient_sdp_enabled()
    assert kwargs["attn_mask"].is_floating_point()
    assert torch.isneginf(kwargs["attn_mask"]).any()
    calls.append(True)
    return original(*args, **kwargs)

  monkeypatch.setattr(F, "scaled_dot_product_attention", checked_sdpa)
  pipeline.model(**batch)
  assert len(calls) == pipeline.model.backbone.config.num_hidden_layers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime is unavailable")
def test_cuda_math_attention_autocast_and_padding(pipeline, batch):
  model = pipeline.model.to("cuda")
  batch = {k: v.to("cuda") for k, v in batch.items()}
  with torch.autocast("cuda", dtype=torch.float16):
    output = model(**batch, output_hidden_states=True)
  assert torch.isfinite(output.loss)
  assert torch.isfinite(output.probabilities()).all()
  assert (output.last_hidden_state[~batch["attention_mask"]] == 0).all()
  output.loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all()
             for p in model.parameters())


def test_rejects_invalid_batch_and_packing(pipeline, batch):
  with pytest.raises(ValueError, match="packing"):
    pipeline.model(**batch, cu_seqlens=torch.tensor([0, 10]))
  bad = {k: v.clone() for k, v in batch.items()}
  bad["choice_content_mask"][0, 0] = True
  with pytest.raises(ValueError, match="content"):
    pipeline.model(**bad)
  bad = {k: v.clone() for k, v in batch.items()}
  bad["labels"][1] = 2
  with pytest.raises(ValueError, match="invalid choice"):
    pipeline.model(**bad)
  bad = {k: v.clone() for k, v in batch.items()}
  bad["choice_content_mask"][0, batch["choice_group_ids"][0] == 2] = False
  with pytest.raises(ValueError, match="needs content"):
    pipeline.model(**bad)
  bad = {k: v.clone() for k, v in batch.items()}
  bad["position_ids"][0, 1] = 0
  with pytest.raises(ValueError, match="sequential"):
    pipeline.model(**bad)


def test_pretrained_backbone_math_matches_original(pipeline):
  backbone = pipeline.model.backbone.eval()
  reference = NeoBERT(backbone.config).eval()
  reference.load_state_dict(backbone.state_dict(), strict=True)
  ids = torch.tensor([[2, 5, 6, 7, 3]])
  active = torch.ones_like(ids, dtype=torch.bool)
  with torch.no_grad():
    original = reference(
        input_ids=ids, attention_mask=active).last_hidden_state
    adapted = backbone(ids, torch.zeros(1, 1, 5, 5), active).last_hidden_state
  torch.testing.assert_close(original, adapted, rtol=2e-5, atol=2e-6)
  # Independent packed SwiGLU check against separate gate/value matrix multiplies.
  ffn = backbone.transformer_encoder[0].ffn
  x = torch.randn(2, 4, backbone.config.hidden_size)
  gate_weight, value_weight = ffn.w12.weight.chunk(2, 0)
  expected = F.linear(F.silu(F.linear(x, gate_weight)) *
                      F.linear(x, value_weight), ffn.w3.weight)
  torch.testing.assert_close(ffn(x), expected)


def test_checkpoint_loading_preserves_every_encoder_weight(pipeline, tmp_path):
  # Exercise the published MLM-prefix format through the real HF loader.
  source = NeoBERTLMHead(pipeline.model.backbone.config)
  checkpoint = tmp_path / "upstream"
  source.save_pretrained(checkpoint, safe_serialization=True)
  config = DecisionConfig(checkpoint=str(checkpoint),
                          max_length=48, head_width=17, dropout=0.0)
  loaded = DecisionModel.from_checkpoint(config, local_files_only=True)
  assert set(source.model.state_dict()) == set(loaded.backbone.state_dict())
  for key, value in source.model.state_dict().items():
    torch.testing.assert_close(
        value, loaded.backbone.state_dict()[key], rtol=0, atol=0)


def test_missing_encoder_weights_are_rejected(pipeline, tmp_path):
  from safetensors.torch import save_file

  checkpoint = tmp_path / "broken"
  checkpoint.mkdir()
  source = NeoBERTLMHead(pipeline.model.backbone.config)
  source.config.save_pretrained(checkpoint)
  state = source.state_dict()
  del state["model.transformer_encoder.1.ffn.w12.weight"]
  save_file(state, str(checkpoint / "model.safetensors"),
            metadata={"format": "pt"})
  with pytest.raises(RuntimeError, match="exact NeoBERT encoder match"):
    DecisionModel.from_checkpoint(DecisionConfig(
        checkpoint=str(checkpoint), max_length=48), local_files_only=True)


def test_bundle_roundtrip_includes_tokenizer_settings_and_weights(pipeline, examples, tmp_path, monkeypatch):
  before = pipeline.predict(examples, return_pooled=True)
  directory = tmp_path / "bundle"
  pipeline.save_pretrained(directory)
  assert (directory / "tokenizer" / "tokenizer.json").is_file()
  assert (directory / "config.json").is_file()
  # Ensure offline reload never asks for the original checkpoint.
  monkeypatch.setattr(DecisionModel, "from_checkpoint",
                      lambda *a, **k: pytest.fail("Unexpected Hub access"))
  loaded = DecisionPipeline.from_pretrained(directory)
  after = loaded.predict(examples, return_pooled=True)
  assert loaded.model.config == pipeline.model.config
  for key in before:
    torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
  for key, value in pipeline.model.state_dict().items():
    torch.testing.assert_close(
        value, loaded.model.state_dict()[key], rtol=0, atol=0)
  assert pipeline.encode(examples[0]) == loaded.encode(examples[0])
  # predict restores the caller's training/eval mode and accepts unlabeled rows.
  assert pipeline.model.training
  unlabeled = [dict(examples[1], label=None)]
  assert loaded.predict(unlabeled)["probabilities"].shape == (1, 2)


def test_remote_revision_and_budget_validation(pipeline):
  with pytest.raises(ValueError, match="pinned"):
    DecisionConfig(revision="main")
  with pytest.raises(ValueError, match="supported limit"):
    DecisionModel(pipeline.model.backbone, DecisionConfig(max_length=65))
  with pytest.raises(ValueError, match="backend"):
    DecisionConfig(attention_backend="flash_attention_2")


def test_bundle_reload_does_not_need_original_local_snapshot(pipeline, tmp_path):
  source = tmp_path / "original-snapshot"
  source.mkdir()
  config = DecisionConfig(
      checkpoint=source, revision="local-snapshot", max_length=48)
  model = DecisionModel(pipeline.model.backbone, config)
  bundle = tmp_path / "saved-bundle"
  model.save_pretrained(bundle, pipeline.tokenizer)
  source.rmdir()  # Bundle must not depend on the original path existing.
  loaded = DecisionPipeline.from_pretrained(bundle)
  assert loaded.model.config == config
