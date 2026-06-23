# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the pluggable local reshard backend and reshard_pytree.

CPU-only and deterministic. Run with 8 simulated CPU devices::

    XLA_FLAGS="--xla_force_host_platform_device_count=8" JAX_PLATFORMS=cpu \
      .venv/bin/python -m pytest tunix/rl/weight_transfer_test.py -q

The disjoint-mesh reshard test ports PATH1 of scratchpad/spike_reshard.py: it
builds two disjoint sub-meshes (devices[0:4] vs [4:8]), reshards a pytree from
mesh_A to mesh_B, and asserts BOTH value preservation AND that every leaf lands
on mesh_B (``.sharding.mesh == mesh_B`` and physical shards on B's devices).
"""

import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tunix.rl import reshard
from tunix.rl import weight_transfer


def _device_set(mesh):
  """The flat set of jax Device objects backing a mesh."""
  return set(mesh.devices.flatten().tolist())


def _buffer_devices(arr):
  """Devices that this array's addressable shards physically live on."""
  return {shard.device for shard in arr.addressable_shards}


def _build_disjoint_meshes():
  """Builds disjoint mesh_A=devices[0:4], mesh_B=devices[4:8] on one host."""
  devices = jax.devices()
  if len(devices) < 8:
    raise absltest.SkipTest(
        "Disjoint-mesh reshard test needs >=8 devices; run with "
        "XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu."
    )
  mesh_a = Mesh(np.array(devices[0:4]), axis_names=("fsdp",))
  mesh_b = Mesh(np.array(devices[4:8]), axis_names=("fsdp",))
  assert _device_set(mesh_a).isdisjoint(_device_set(mesh_b))
  return mesh_a, mesh_b


def _make_source_pytree(mesh_a):
  """Builds a small pytree on mesh_a: sharded + replicated leaves."""
  rng = np.random.default_rng(0)
  host_arrays = {
      "w": rng.standard_normal((8, 6)).astype(np.float32),  # shards over fsdp
      "b": rng.standard_normal((5,)).astype(np.float32),  # replicated
      "e": rng.standard_normal((12, 3)).astype(np.float32),  # shards over fsdp
  }
  src_shardings = {
      "w": NamedSharding(mesh_a, P("fsdp")),
      "b": NamedSharding(mesh_a, P()),
      "e": NamedSharding(mesh_a, P("fsdp")),
  }

  def _mk(name):
    full = host_arrays[name]
    return jax.make_array_from_callback(
        full.shape, src_shardings[name], lambda idx: full[idx], dtype=full.dtype
    )

  source = {k: _mk(k) for k in host_arrays}
  return source, host_arrays


def _assert_resharded_onto_b(test, out, host_arrays, mesh_b):
  """Asserts value preservation AND that every leaf landed on mesh_b."""
  devs_b = _device_set(mesh_b)
  for key, full in host_arrays.items():
    leaf = out[key]
    # Value preservation, host-local: each addressable shard matches the known
    # deterministic full array on its global index.
    for shard in leaf.addressable_shards:
      np.testing.assert_allclose(
          np.asarray(shard.data), full[shard.index], err_msg=f"value[{key}]"
      )
    # Relocation: the sharding's mesh is exactly mesh_b ...
    test.assertEqual(leaf.sharding.mesh, mesh_b, msg=f"mesh[{key}]")
    # ... and the physical shards live on mesh_b's devices (not routed via A).
    test.assertTrue(
        _buffer_devices(leaf).issubset(devs_b),
        msg=(
            f"phys[{key}] shards {sorted(d.id for d in _buffer_devices(leaf))} "
            f"not subset of mesh_B {sorted(d.id for d in devs_b)}"
        ),
    )


class SelectReshardFnsTest(parameterized.TestCase):
  """Backend selection by capability + explicit override (no global state)."""

  def test_auto_matches_historical_fallback_order(self):
    fns = weight_transfer.select_reshard_fns(
        weight_transfer.LocalReshardBackend.AUTO
    )
    self.assertEqual(
        fns,
        [
            reshard._get_reshard_fn_pathwaysutils,  # pylint: disable=protected-access
            reshard._get_reshard_fn_jax_device_put,  # pylint: disable=protected-access
        ],
    )

  def test_default_arg_is_auto(self):
    self.assertEqual(
        weight_transfer.select_reshard_fns(),
        weight_transfer.select_reshard_fns(
            weight_transfer.LocalReshardBackend.AUTO
        ),
    )

  def test_jax_device_only(self):
    fns = weight_transfer.select_reshard_fns(
        weight_transfer.LocalReshardBackend.JAX_DEVICE
    )
    self.assertEqual(
        fns,
        [reshard._get_reshard_fn_jax_device_put],  # pylint: disable=protected-access
    )

  def test_pathways_only(self):
    fns = weight_transfer.select_reshard_fns(
        weight_transfer.LocalReshardBackend.PATHWAYS
    )
    self.assertEqual(
        fns,
        [reshard._get_reshard_fn_pathwaysutils],  # pylint: disable=protected-access
    )

  def test_returns_fresh_list(self):
    fns = weight_transfer.select_reshard_fns()
    fns.append(None)
    self.assertNotIn(None, weight_transfer.select_reshard_fns())

  def test_capabilities_off_pathways(self):
    caps = weight_transfer.capabilities()
    self.assertFalse(caps["pathways"])
    self.assertEqual(
        caps["backend"], weight_transfer.LocalReshardBackend.JAX_DEVICE.value
    )

  def test_auto_off_pathways_does_not_import_pathwaysutils(self):
    """AUTO must resolve and reshard without requiring pathwaysutils.

    Simulate pathwaysutils being unavailable by blocking its import, then run a
    reshard through the AUTO fallback. It must transparently fall through to
    jax_device.
    """
    mesh_a, mesh_b = _build_disjoint_meshes()
    source, host_arrays = _make_source_pytree(mesh_a)
    target = {
        "w": NamedSharding(mesh_b, P("fsdp")),
        "b": NamedSharding(mesh_b, P()),
        "e": NamedSharding(mesh_b, P("fsdp")),
    }
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def _blocked_import(name, *args, **kwargs):
      if name == "pathwaysutils" or name.startswith("pathwaysutils."):
        raise ImportError("pathwaysutils blocked for test")
      return real_import(name, *args, **kwargs)

    # Drop any cached pathwaysutils so the blocked import is actually exercised.
    saved = {
        k: v for k, v in sys.modules.items() if k.startswith("pathwaysutils")
    }
    for k in saved:
      del sys.modules[k]
    try:
      with mock.patch("builtins.__import__", side_effect=_blocked_import):
        out = reshard.reshard_pytree(
            source,
            target,
            reshard_fns=weight_transfer.select_reshard_fns(
                weight_transfer.LocalReshardBackend.AUTO
            ),
        )
        jax.block_until_ready(out)
    finally:
      sys.modules.update(saved)
    _assert_resharded_onto_b(self, out, host_arrays, mesh_b)

  def test_local_weight_transfer_holder(self):
    holder = weight_transfer.LocalWeightTransfer(
        weight_transfer.LocalReshardBackend.JAX_DEVICE
    )
    self.assertEqual(
        holder.backend, weight_transfer.LocalReshardBackend.JAX_DEVICE
    )
    self.assertEqual(
        holder.reshard_fns,
        [reshard._get_reshard_fn_jax_device_put],  # pylint: disable=protected-access
    )
    # close() is a no-op cleanup hook and must not raise.
    holder.close()


class DisjointMeshReshardTest(parameterized.TestCase):
  """Ports spike_reshard.py PATH1: disjoint-mesh reshard on CPU."""

  @parameterized.named_parameters(
      ("auto", weight_transfer.LocalReshardBackend.AUTO),
      ("jax_device", weight_transfer.LocalReshardBackend.JAX_DEVICE),
  )
  def test_reshard_pytree_relocates_to_disjoint_mesh(self, backend):
    mesh_a, mesh_b = _build_disjoint_meshes()
    source, host_arrays = _make_source_pytree(mesh_a)
    # Source leaves must start on mesh_A.
    for key, leaf in source.items():
      self.assertTrue(
          _buffer_devices(leaf).issubset(_device_set(mesh_a)),
          msg=f"source[{key}] not on mesh_A",
      )
    target = {
        "w": NamedSharding(mesh_b, P("fsdp")),
        "b": NamedSharding(mesh_b, P()),
        "e": NamedSharding(mesh_b, P("fsdp")),
    }
    out = reshard.reshard_pytree(
        source,
        target,
        reshard_fns=weight_transfer.select_reshard_fns(backend),
    )
    jax.block_until_ready(out)
    _assert_resharded_onto_b(self, out, host_arrays, mesh_b)

  def test_default_reshard_fns_none_uses_auto(self):
    """reshard_pytree() with no reshard_fns exercises the lazy-import AUTO path.

    Off-Pathways, AUTO degrades to jax_device, so the disjoint-mesh reshard must
    still land value-correct on mesh_B without passing reshard_fns explicitly.
    """
    mesh_a, mesh_b = _build_disjoint_meshes()
    source, host_arrays = _make_source_pytree(mesh_a)
    target = {
        k: NamedSharding(mesh_b, v.sharding.spec) for k, v in source.items()
    }
    out = reshard.reshard_pytree(source, target)  # reshard_fns defaults to None
    jax.block_until_ready(out)
    _assert_resharded_onto_b(self, out, host_arrays, mesh_b)

  def test_jax_device_backend_explicit(self):
    """JAX_DEVICE selected via an explicit factory list works."""
    mesh_a, mesh_b = _build_disjoint_meshes()
    source, host_arrays = _make_source_pytree(mesh_a)
    target = {
        k: NamedSharding(mesh_b, v.sharding.spec) for k, v in source.items()
    }
    out = reshard.reshard_pytree(
        source,
        target,
        reshard_fns=[
            reshard._get_reshard_fn_jax_device_put  # pylint: disable=protected-access
        ],
    )
    jax.block_until_ready(out)
    _assert_resharded_onto_b(self, out, host_arrays, mesh_b)

  def test_donate_input_does_not_corrupt_output(self):
    """donate_input=True must still produce value-correct output on mesh_B."""
    mesh_a, mesh_b = _build_disjoint_meshes()
    source, host_arrays = _make_source_pytree(mesh_a)
    target = {
        "w": NamedSharding(mesh_b, P("fsdp")),
        "b": NamedSharding(mesh_b, P()),
        "e": NamedSharding(mesh_b, P("fsdp")),
    }
    out = reshard.reshard_pytree(
        source,
        target,
        donate_input=True,
        reshard_fns=weight_transfer.select_reshard_fns(
            weight_transfer.LocalReshardBackend.JAX_DEVICE
        ),
    )
    jax.block_until_ready(out)
    _assert_resharded_onto_b(self, out, host_arrays, mesh_b)


if __name__ == "__main__":
  absltest.main()
