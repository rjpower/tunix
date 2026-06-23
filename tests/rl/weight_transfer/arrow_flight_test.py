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

"""CPU loopback tests for the Arrow Flight weight-transfer transport.

These run with 8 fake CPU devices (forced before JAX is imported) so the
sharded-template path is exercised without accelerators.
"""

import os

# Force 8 fake CPU devices BEFORE importing jax (below / transitively).
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
from jax.sharding import Mesh  # pylint: disable=g-import-not-at-top
from jax.sharding import NamedSharding  # pylint: disable=g-import-not-at-top
from jax.sharding import PartitionSpec  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
from tunix.rl.weight_transfer import arrow_flight  # pylint: disable=g-import-not-at-top
from tunix.rl.weight_transfer import base  # pylint: disable=g-import-not-at-top
from tunix.rl.weight_transfer import coordinator as coordinator_lib  # pylint: disable=g-import-not-at-top


def _config() -> base.WeightTransferConfig:
  # Bind to localhost (single host) and use a single Flight server for
  # deterministic CPU tests.
  return base.WeightTransferConfig(
      mode=base.WeightTransferMode.ARROW_FLIGHT,
      convert_to_bfloat16=True,
      flight_host="localhost",
      num_flight_servers=2,
  )


def _make_pytree(sharding_fn):
  """Builds the test pytree, placing each leaf with ``sharding_fn(spec)``."""
  rng = np.random.default_rng(0)

  def leaf(shape, spec):
    value = jnp.asarray(rng.standard_normal(shape), dtype=jnp.float32)
    return jax.device_put(value, sharding_fn(spec))

  return {
      "embed": leaf((128, 64), PartitionSpec("x", None)),
      "layers": {
          "w": leaf((64, 256), PartitionSpec(None, "x")),
          "b": leaf((256,), PartitionSpec("x")),
      },
      "scalar": leaf((), PartitionSpec()),
  }


class ArrowFlightTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    devices = jax.devices()
    self.assertGreaterEqual(len(devices), 8)
    # Two distinct single-axis CPU meshes -> source and template have the same
    # structure/shapes but different shardings.
    self.src_mesh = Mesh(np.asarray(devices[:4]).reshape(4), ("x",))
    self.dst_mesh = Mesh(np.asarray(devices[4:8]).reshape(4), ("x",))

  def _src_sharding(self, spec):
    return NamedSharding(self.src_mesh, spec)

  def _dst_sharding(self, spec):
    return NamedSharding(self.dst_mesh, spec)

  def _serve_and_receive(self, weight_id, params, template, coordinator):
    server = arrow_flight.ArrowFlightServer(_config(), coordinator=coordinator)
    self.addCleanup(server.cleanup)
    server.serve_weights(weight_id, params)
    client = arrow_flight.ArrowFlightClient(_config(), coordinator=coordinator)
    self.addCleanup(client.cleanup)
    update = client.receive_weights(template)
    return server, client, update

  def test_loopback_roundtrip(self):
    coordinator = coordinator_lib.InProcessCoordinator()
    params = _make_pytree(self._src_sharding)
    # Template: same structure/shapes, DIFFERENT sharding (dst mesh).
    template = _make_pytree(self._dst_sharding)

    _, _, update = self._serve_and_receive(7, params, template, coordinator)

    self.assertIsNotNone(update)
    self.assertEqual(update.weight_id, 7)

    src_leaves = jax.tree_util.tree_leaves(params)
    tmpl_leaves = jax.tree_util.tree_leaves(template)
    out_leaves = jax.tree_util.tree_leaves(update.params)
    self.assertLen(out_leaves, len(src_leaves))
    for src, tmpl, out in zip(src_leaves, tmpl_leaves, out_leaves):
      # Values match within bf16 tolerance.
      np.testing.assert_array_less(
          np.abs(
              np.asarray(out, dtype=np.float32)
              - np.asarray(src, dtype=np.float32)
          ),
          0.05,
      )
      # Restored leaf carries the TEMPLATE sharding, not the source sharding.
      self.assertEqual(out.sharding, tmpl.sharding)

  def test_no_new_weights_returns_none(self):
    coordinator = coordinator_lib.InProcessCoordinator()
    params = _make_pytree(self._src_sharding)
    template = _make_pytree(self._dst_sharding)

    _, client, first = self._serve_and_receive(7, params, template, coordinator)
    self.assertIsNotNone(first)
    # Second poll with no new serve -> None.
    self.assertIsNone(client.receive_weights(template))

  def test_backwards_weight_id_accepted(self):
    coordinator = coordinator_lib.InProcessCoordinator()
    template = _make_pytree(self._dst_sharding)

    server = arrow_flight.ArrowFlightServer(_config(), coordinator=coordinator)
    self.addCleanup(server.cleanup)
    client = arrow_flight.ArrowFlightClient(_config(), coordinator=coordinator)
    self.addCleanup(client.cleanup)

    server.serve_weights(7, _make_pytree(self._src_sharding))
    first = client.receive_weights(template)
    self.assertIsNotNone(first)
    self.assertEqual(first.weight_id, 7)

    # Trainer rolls back to an earlier checkpoint: weight_id 3 < 7 is accepted.
    server.serve_weights(3, _make_pytree(self._src_sharding))
    rolled_back = client.receive_weights(template)
    self.assertIsNotNone(rolled_back)
    self.assertEqual(rolled_back.weight_id, 3)

  def test_large_param_chunked(self):
    coordinator = coordinator_lib.InProcessCoordinator()
    # Shrink the per-record element cap so a modest param spans many batches.
    original = arrow_flight.MAX_ELEMENTS_PER_RECORD
    arrow_flight.MAX_ELEMENTS_PER_RECORD = 1000
    self.addCleanup(setattr, arrow_flight, "MAX_ELEMENTS_PER_RECORD", original)

    rng = np.random.default_rng(1)
    big = jax.device_put(
        jnp.asarray(rng.standard_normal((40, 211)), dtype=jnp.float32),
        self._src_sharding(PartitionSpec("x", None)),
    )
    params = {"big": big}
    template = {
        "big": jax.device_put(
            jnp.zeros((40, 211), dtype=jnp.float32),
            self._dst_sharding(PartitionSpec(None, None)),
        )
    }

    _, _, update = self._serve_and_receive(1, params, template, coordinator)

    self.assertIsNotNone(update)
    # The param's flattened size (40*211 = 8440) exceeds the 1000 cap, so it was
    # chunked; assert the reassembled values still match.
    np.testing.assert_array_less(
        np.abs(
            np.asarray(update.params["big"], dtype=np.float32)
            - np.asarray(big, dtype=np.float32)
        ),
        0.05,
    )
    self.assertEqual(update.params["big"].sharding, template["big"].sharding)


if __name__ == "__main__":
  absltest.main()
