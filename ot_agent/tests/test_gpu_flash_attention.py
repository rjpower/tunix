# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The GPU flash-attention path we added to tunix Qwen3 (model.py) must be
numerically equivalent to the existing materialized attention -- otherwise a
"faithful" long-context run would silently train a different model.

tunix dispatches the flash branch by platform: TPU->splash, GPU->cuDNN, else
->`jax.nn.dot_product_attention(implementation='xla')`. The xla path shares the
exact semantics of the cuDNN path (same API, same is_causal/GQA/scale handling)
but runs on CPU, so we validate the wiring here and validate the actual cuDNN
kernel on GPU in the bigsmoke run. On a non-padded input the model's causal mask
is pure-causal, so flash (is_causal) must equal the materialized causal+pad path
bit-for-bit (up to fp tolerance) at every position.
"""

import jax.numpy as jnp
import numpy as np

import grain.python as grain

from mega_eval.models.qwen3_loader import load_qwen3
from mega_eval.training.common import build_mesh, sft_model_input_fn
from tunix.models.qwen3.model import _flash_attention_backend
from ot_agent.data import _RowSource, _to_columns
from ot_agent.tests._tiny_model import CFG, write_fake_base_model


def _inputs(batch_n, seq, vocab):
  rng = np.random.default_rng(3)
  rows = []
  for _ in range(batch_n):
    toks = rng.integers(0, vocab, size=seq).astype(np.int32)
    loss = np.ones(seq, np.float32)
    pad = np.ones(seq, np.bool_)  # no padding -> the model mask is pure causal
    rows.append((toks, loss, pad))
  ds = grain.MapDataset.source(_RowSource(rows)).batch(batch_n).map(_to_columns)
  return sft_model_input_fn(ds[0])


def test_flash_backend_resolves():
  # Whatever the test host is, dispatch must resolve to a known backend.
  assert _flash_attention_backend() in ("cpu", "gpu", "tpu")


def test_flash_matches_materialized(tmp_path):
  base = tmp_path / "base"
  base.mkdir()
  write_fake_base_model(str(base))

  mesh = build_mesh(tp=1)
  # Same base dir -> identical (deterministic) weights; fp32 to isolate the
  # attention math from bf16 rounding.
  mat = load_qwen3(str(base), mesh=mesh, dtype=jnp.float32, param_dtype=jnp.float32,
                   use_flash_attention=False)
  flash = load_qwen3(str(base), mesh=mesh, dtype=jnp.float32, param_dtype=jnp.float32,
                     use_flash_attention=True)

  inp = _inputs(batch_n=2, seq=16, vocab=CFG["vocab_size"])
  call = lambda m: m(inp["input_tokens"], inp["positions"], None, inp["attention_mask"])[0]
  lm, lf = np.asarray(call(mat)), np.asarray(call(flash))
  assert lm.shape == lf.shape
  assert np.allclose(lm, lf, rtol=1e-4, atol=1e-4), np.abs(lm - lf).max()
