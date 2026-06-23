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

"""GPU device-collective weight transfer (NCCL), single JAX world.

For GPU the strong path is *not* to fold weights to rank 0 and broadcast (what
torch-based stacks like xorl do). Inside one JAX world, a cross-mesh
``jax.device_put`` / reshard lowers to XLA collectives (NCCL all-to-all) in which
**every** GPU sends and receives only its own shards. That is faster than a
rank-0 broadcast and is exactly what the user asked for ("use all GPUs for sync
instead of folding to rank 0"). It is also already correctness-validated:
disjoint-mesh ``reshard_pytree`` PASSes on 8xH100.

So this transport is a thin `WeightTransferServer`/`WeightTransferClient` adapter
over `reshard.reshard_pytree`, giving the benchmark and the RL loop one uniform
interface across transports:

* The server holds the freshly-trained params on the *trainer* mesh (optionally
  bf16-cast on device to halve NCCL bytes) under a ``weight_id``.
* The client reshards them onto its *rollout* template (target mesh/sharding) via
  NCCL collectives.

Because this is a single-controller, single-process JAX program (the trainer and
rollout meshes are disjoint slices of one ``jax.devices()`` pool), the server and
client rendezvous through a process-global registry keyed by the shared
`Coordinator` identity -- no sockets, no second NCCL domain. For genuinely
separate processes/worlds you would need a raw NCCL process group (xorl-style);
that is out of scope here and tracked separately.
"""

import logging
import time
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from tunix.rl import reshard as reshard_lib
from tunix.rl.weight_transfer import base
from tunix.rl.weight_transfer import coordinator as coordinator_lib
from tunix.rl.weight_transfer import state_dict


# Process-global handoff of on-device weight pytrees between an in-world server
# and its clients, keyed by the shared coordinator's id(). The server keeps the
# live jax arrays here (NOT host bytes); the client reads + reshards them. This
# is safe because NCCL transfer is single-process single-controller.
_DEVICE_STORES: dict[int, "_DeviceWeightStore"] = {}


class _DeviceWeightStore:
  """Holds the latest on-device params pytree for one transfer group."""

  def __init__(self):
    self.weight_id: int | None = None
    self.params: PyTree | None = None


def _store_for(
    coordinator: coordinator_lib.Coordinator,
) -> "_DeviceWeightStore":
  store = _DEVICE_STORES.get(id(coordinator))
  if store is None:
    store = _DeviceWeightStore()
    _DEVICE_STORES[id(coordinator)] = store
  return store


def _maybe_cast_bf16(params: PyTree) -> PyTree:
  def f(x):
    if jnp.issubdtype(x.dtype, jnp.floating):
      return x.astype(jnp.bfloat16)
    return x

  return jax.tree.map(f, params)


class NcclWeightServer(base.WeightTransferServer):
  """Serves params on the trainer mesh for in-world NCCL reshard to clients."""

  def __init__(
      self,
      config: base.WeightTransferConfig,
      coordinator: coordinator_lib.Coordinator | None = None,
  ):
    self.config = config
    self.coordinator = coordinator or coordinator_lib.InProcessCoordinator()
    self._store = _store_for(self.coordinator)
    self.metrics = base.ServerMetrics()

  def serve_weights(self, weight_id: int, params: PyTree) -> None:
    self.metrics.total_transfers += 1
    start = time.time()
    try:
      if self.config.convert_to_bfloat16:
        params = _maybe_cast_bf16(params)
      # Keep params resident on device; block so timing is real and downstream
      # reshard sees materialized arrays.
      params = jax.block_until_ready(params)
      flatten_done = time.time()

      self._store.weight_id = weight_id
      self._store.params = params

      keys = state_dict.flat_keys(params)
      self.coordinator.publish(
          base.ServerInfo(
              weight_id=weight_id, server_addresses=[], param_names=keys
          )
      )

      nbytes = _pytree_nbytes(params)
      self.metrics.flatten_time = flatten_done - start
      self.metrics.materialize_time = flatten_done - start
      self.metrics.serve_time = time.time() - start
      self.metrics.transfer_bytes = nbytes
      self.metrics.total_transfer_bytes += nbytes
      self.metrics.param_count = len(keys)
      self.metrics.materialize_mib_per_second = base.mib_per_second(
          nbytes, self.metrics.materialize_time
      )
      self.metrics.successful_transfers += 1
    except Exception:
      self.metrics.failed_transfers += 1
      logging.exception("NCCL serve_weights failed for weight_id %s", weight_id)
      raise

  def cleanup(self) -> None:
    _DEVICE_STORES.pop(id(self.coordinator), None)

  def get_metrics(self) -> base.ServerMetrics:
    return self.metrics


class NcclWeightClient(base.WeightTransferClient):
  """Reshards the server's on-device params onto a rollout template (NCCL)."""

  def __init__(
      self,
      config: base.WeightTransferConfig,
      coordinator: coordinator_lib.Coordinator | None = None,
      reshard_fns: list[Any] | None = None,
  ):
    self.config = config
    self.coordinator = coordinator or coordinator_lib.InProcessCoordinator()
    self._store = _store_for(self.coordinator)
    self._reshard_fns = reshard_fns
    self._last_weight_id: int | None = None
    self.metrics = base.ClientMetrics()

  def receive_weights(
      self, template: PyTree | None
  ) -> base.WeightUpdate | None:
    self.metrics.total_polls += 1
    start = time.time()
    info = self.coordinator.lookup()
    if info is None or info.weight_id is None:
      return None
    if info.weight_id == self._last_weight_id:
      return None
    if template is None:
      raise ValueError("NCCL receive_weights requires a rollout template.")

    poll_done = time.time()
    try:
      source = self._store.params
      if source is None:
        return None
      # The NCCL all-to-all: reshard trainer-mesh params onto the rollout
      # template's mesh/sharding. On GPU this is NCCL collectives using all
      # devices in both meshes.
      resharded = reshard_lib.reshard_pytree(
          source, template, reshard_fns=self._reshard_fns
      )
      resharded = jax.block_until_ready(resharded)
      reshard_done = time.time()

      nbytes = _pytree_nbytes(resharded)
      self.metrics.successful_receives += 1
      self.metrics.receive_bytes = nbytes
      self.metrics.total_receive_bytes += nbytes
      self.metrics.param_count = len(info.param_names)
      self.metrics.poll_time = poll_done - start
      self.metrics.fetch_time = reshard_done - poll_done
      self.metrics.reshard_time = reshard_done - poll_done
      self.metrics.decode_time = 0.0
      self.metrics.fetch_mib_per_second = base.mib_per_second(
          nbytes, self.metrics.fetch_time
      )
      self._last_weight_id = info.weight_id
      return base.WeightUpdate(
          params=resharded, flat_state={}, weight_id=info.weight_id
      )
    except Exception:
      self.metrics.failed_receives += 1
      logging.exception("NCCL receive_weights failed")
      return None

  def cleanup(self) -> None:
    return None

  def get_metrics(self) -> base.ClientMetrics:
    return self.metrics


def _pytree_nbytes(tree: PyTree) -> int:
  total = 0
  for leaf in jax.tree_util.tree_leaves(tree):
    total += int(jnp.size(leaf)) * jnp.dtype(leaf.dtype).itemsize
  return total
