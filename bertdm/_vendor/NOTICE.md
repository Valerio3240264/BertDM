# Reused NeoBERT source

`neobert.py` and `rotary.py` were retrieved from
[chandar-lab/NeoBERT](https://huggingface.co/chandar-lab/NeoBERT/tree/5424c8efeea6491b151d62dee55a752165407430),
revision `5424c8efeea6491b151d62dee55a752165407430` (`model.py` renamed `neobert.py`).
The upstream model card declares the code and weights MIT licensed. Copyright
belongs to the upstream authors. Both files also credit the original
[Meta Llama implementation](https://github.com/facebookresearch/llama/blob/main/llama/model.py).
The inspected upstream Hub/GitHub repositories did not contain a separate
LICENSE file; retain this provenance and upstream headers when redistributing.

Local changes to `neobert.py`:

- Import the portable packed SwiGLU equivalent from `swiglu.py` instead of requiring
  xFormers/CUDA binary extensions. The names, shapes, computation, and all FFN weights
  are preserved (`w12`, then `w3`). This follows the documented eager formula in
  [xFormers v0.0.28.post3](https://github.com/facebookresearch/xformers/blob/v0.0.28.post3/xformers/ops/swiglu_op.py).
- Disable the optional FlashAttention import. Packing is rejected by our adapter.
- Add a `block_class` factory hook and use it to instantiate blocks. Our subclass
  inherits the upstream block's forward, norms, FFN, and residual connections.

`rotary.py` is unchanged. All attention dispatch and sequence orchestration changes
are in `bertdm/model.py`. No upstream code is downloaded/executed at model-load time.
