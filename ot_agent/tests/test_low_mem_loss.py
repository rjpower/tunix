# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The lean loss is only safe to use on the real 32B run if it's mathematically
identical to tunix's stock ``_default_loss_fn``. The stock loss materializes a
``[B, S, V]`` fp32 one-hot + log_softmax (the 152k-vocab OOM on 4x8 H100); ours
uses ``logit[target] - logsumexp(logits)``. Same NLL, ~1/4 the activation memory.

This pins value AND gradient parity on the real tunix Qwen3 (tiny config, fp32
compute so the only difference under test is the loss math, not bf16 rounding).
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
from ot_agent.sft import low_mem_cross_entropy_loss
from ot_agent.tests._tiny_model import CFG, write_fake_base_model


def _batch_inputs(batch_n, seq, vocab):
  rng = np.random.default_rng(7)
  rows = []
  for _ in range(batch_n):
    toks = rng.integers(0, vocab, size=seq).astype(np.int32)
    loss = np.zeros(seq, np.float32)
    loss[seq // 2:] = 1.0  # train on the back half (assistant-turn analogue)
    pad = np.ones(seq, np.bool_)
    rows.append((toks, loss, pad))
  ds = grain.MapDataset.source(_RowSource(rows)).batch(batch_n).map(_to_columns)
  return sft_model_input_fn(ds[0])


def test_low_mem_loss_matches_default_value_and_grad(tmp_path):
  base = tmp_path / "base"
  base.mkdir()
  write_fake_base_model(str(base))

  mesh = build_mesh(tp=1)
  # fp32 compute: isolate the loss math (no bf16 rounding between the two paths).
  model = load_qwen3(str(base), mesh=mesh, dtype=jnp.float32, param_dtype=jnp.float32)

  inputs = _batch_inputs(batch_n=2, seq=8, vocab=CFG["vocab_size"])

  # Values match.
  ref = float(_default_loss_fn(model, **inputs))
  ours = float(low_mem_cross_entropy_loss(model, **inputs))
  assert np.isclose(ref, ours, rtol=1e-5, atol=1e-5), (ref, ours)

  # Gradients match (the thing that actually trains the model).
  g_ref = nnx.grad(_default_loss_fn)(model, **inputs)
  g_ours = nnx.grad(low_mem_cross_entropy_loss)(model, **inputs)
  ref_leaves = jax.tree.leaves(nnx.state(g_ref, nnx.Param))
  our_leaves = jax.tree.leaves(nnx.state(g_ours, nnx.Param))
  assert ref_leaves, "no gradient leaves -- grad wiring is wrong"
  for a, b in zip(ref_leaves, our_leaves):
    a, b = np.asarray(a), np.asarray(b)
    assert np.allclose(a, b, rtol=1e-4, atol=1e-5), np.abs(a - b).max()
