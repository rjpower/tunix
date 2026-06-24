# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CPU validation that the HF exporter is the loader's exact inverse.

Builds a tiny *real* (tunix-constructed, non-tied) Qwen3 from fake base
safetensors, loads it with the production ``load_qwen3``, exports it with
``ot_agent.export_hf``, then reloads the exported safetensors and asserts every
tensor is bit-identical to the original. This exercises the whole export path --
nnx param iteration, dotted-key matching, multi-host-safe gather (device_get on
1 process), transform inversion, sharded write, and config copy -- against real
tunix machinery, not a mock. A corrupted inverse (transposed/mis-reshaped
attention weights) would load "successfully" yet fail this round-trip.
"""

import glob
import os

import jax.numpy as jnp
import numpy as np
import safetensors.numpy as safe_np

from mega_eval.models.qwen3_loader import load_qwen3
from ot_agent.export_hf import export_and_mirror
from ot_agent.tests._tiny_model import write_fake_base_model


def test_export_is_loader_inverse(tmp_path):
  base = tmp_path / "base"
  out = tmp_path / "out"
  base.mkdir()
  state = write_fake_base_model(str(base))

  model = load_qwen3(str(base), mesh=None, dtype=jnp.bfloat16, param_dtype=jnp.float32)
  export_and_mirror(model, str(base), str(out))

  exported = {}
  for shard in glob.glob(os.path.join(str(out), "*.safetensors")):
    exported.update(safe_np.load_file(shard))

  assert set(exported) == set(state), (
      f"missing={set(state) - set(exported)} extra={set(exported) - set(state)}")
  for k in state:
    np.testing.assert_array_equal(
        exported[k].astype(np.float32), state[k].astype(np.float32),
        err_msg=f"{k} not bit-identical after loader->export round-trip")
  assert os.path.exists(os.path.join(str(out), "config.json"))
