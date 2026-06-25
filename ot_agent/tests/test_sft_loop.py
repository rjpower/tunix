# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke for the SFT training loop wiring (run_sharded_sft end to end).

The data path is unit-tested (test_data_sharding) and the export is round-trip
tested (test_export_roundtrip); this closes the loop by running the *actual*
``ot_agent.sft.run_sharded_sft`` -- tunix ``PeftTrainer`` + the assistant-turn
loss-mask input fn + clipped AdamW -- for a few steps on the tiny model and a
synthetic pre-encoded dataset. Confirms the dataset column contract, the input
fn, and that an optimizer step actually moves the weights (the loss-mask path
produces a non-zero gradient). Mechanical only -- no learning claim on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import grain.python as grain

from mega_eval.models.qwen3_loader import load_qwen3
from mega_eval.training.common import build_mesh
from ot_agent.data import _RowSource, _to_columns
from ot_agent.sft import run_sharded_sft
from ot_agent.tests._tiny_model import CFG, write_fake_base_model


def _synthetic_dataset(n_rows: int, batch: int, seq: int, vocab: int):
  """Pre-encoded (tokens, loss_mask, pad_mask) rows in the grain contract."""
  rng = np.random.default_rng(1)
  rows = []
  for _ in range(n_rows):
    toks = rng.integers(0, vocab, size=seq).astype(np.int32)
    loss = np.zeros(seq, np.float32)
    loss[seq // 2:] = 1.0  # train on the back half (mimics assistant turn)
    pad = np.ones(seq, np.bool_)
    rows.append((toks, loss, pad))
  return grain.MapDataset.source(_RowSource(rows)).batch(batch).map(_to_columns)


def _params_snapshot(model):
  _, state = nnx.split(model)
  return jax.tree.map(lambda x: np.asarray(jax.device_get(x)).copy(),
                      state.to_pure_dict())


def test_run_sharded_sft_steps_and_updates_weights(tmp_path):
  base = tmp_path / "base"
  base.mkdir()
  write_fake_base_model(str(base))

  mesh = build_mesh(tp=1)  # 1 CPU device -> (fsdp=1, tp=1)
  model = load_qwen3(str(base), mesh=mesh, dtype=jnp.bfloat16, param_dtype=jnp.float32)

  steps, batch, seq = 3, 2, 8
  ds = _synthetic_dataset(n_rows=(steps + 2) * batch, batch=batch, seq=seq,
                          vocab=CFG["vocab_size"])

  before = _params_snapshot(model)
  model = run_sharded_sft(
      model, tokenizer=None, dataset=ds, steps=steps, learning_rate=1e-2,
      mesh=mesh, checkpoint_dir=None,
  )
  after = _params_snapshot(model)

  # At least one trainable matrix moved -> the loss-mask path produced gradient.
  moved = any(
      not np.array_equal(before_leaf, after_leaf)
      for before_leaf, after_leaf in zip(
          jax.tree.leaves(before), jax.tree.leaves(after))
  )
  assert moved, "no parameter changed after SFT steps; loss/grad path is dead"
