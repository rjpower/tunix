# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The chunked CE only earns its place if it's the SAME loss as tunix's stock
``_default_loss_fn`` -- it just trades the full ``[B,S,V]`` logits for per-chunk
``[B,S/n,V]`` (rematerialized). If the math drifted, a long-context run would
train a subtly different model. So we pin: (1) value+gradient parity vs the stock
loss, and (2) the result is invariant to the chunk count (n=1 == n=5 == n=S).
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import grain.python as grain

from tunix.sft.peft_trainer import _default_loss_fn

from mega_eval.models.qwen3_loader import load_qwen3
from mega_eval.training.common import build_mesh, sft_model_input_fn
from ot_agent.data import _RowSource, _to_columns
from ot_agent.sft import chunked_cross_entropy_loss, make_chunked_cross_entropy_loss
from ot_agent.tests._tiny_model import CFG, write_fake_base_model


def _inputs(batch_n, seq, vocab):
  rng = np.random.default_rng(11)
  rows = []
  for _ in range(batch_n):
    toks = rng.integers(0, vocab, size=seq).astype(np.int32)
    loss = np.zeros(seq, np.float32)
    loss[seq // 3:] = 1.0  # mixed mask -> exercises the masked normalizer
    pad = np.ones(seq, np.bool_)
    rows.append((toks, loss, pad))
  ds = grain.MapDataset.source(_RowSource(rows)).batch(batch_n).map(_to_columns)
  return sft_model_input_fn(ds[0])


def _model(tmp_path):
  base = tmp_path / "base"
  base.mkdir()
  write_fake_base_model(str(base))
  mesh = build_mesh(tp=1)
  return load_qwen3(str(base), mesh=mesh, dtype=jnp.float32, param_dtype=jnp.float32)


def test_chunked_matches_default_value_and_grad(tmp_path):
  model = _model(tmp_path)
  inp = _inputs(batch_n=2, seq=12, vocab=CFG["vocab_size"])

  # The closure factory gives the canonical positional signature that the
  # trainer (and nnx.grad) call as grad_fn(model, **inputs) -- see its docstring.
  chunked = make_chunked_cross_entropy_loss(4)
  ref = float(_default_loss_fn(model, **inp))
  ours = float(chunked(model, **inp))
  assert np.isclose(ref, ours, rtol=1e-5, atol=1e-5), (ref, ours)

  g_ref = jax.tree.leaves(nnx.state(nnx.grad(_default_loss_fn)(model, **inp), nnx.Param))
  g_ours = jax.tree.leaves(nnx.state(nnx.grad(chunked)(model, **inp), nnx.Param))
  assert g_ref and len(g_ref) == len(g_ours)
  for a, b in zip(g_ref, g_ours):
    a, b = np.asarray(a), np.asarray(b)
    assert np.allclose(a, b, rtol=1e-4, atol=1e-5), np.abs(a - b).max()


def test_chunk_count_invariant(tmp_path):
  model = _model(tmp_path)
  inp = _inputs(batch_n=2, seq=12, vocab=CFG["vocab_size"])
  vals = [float(chunked_cross_entropy_loss(model, **inp, n_chunks=n)) for n in (1, 3, 5, 11)]
  for v in vals[1:]:
    assert np.isclose(v, vals[0], rtol=1e-5, atol=1e-5), vals
