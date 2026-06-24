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

"""Pluggable weight transfer for Tunix RL.

This package is the single home for moving trained weights from the actor
(trainer) onto the rollout (inference) workers. It has two layers:

* **Local reshard** (`local`): move a pytree across meshes *inside one JAX
  program* via ``jax.device_put`` / pathwaysutils. Selected by capability; this
  is what colocated / same-host disaggregated RL uses today. Re-exported here so
  ``from tunix.rl import weight_transfer`` keeps its historical surface.

* **Remote transports** (`base`, `arrow_flight`, `nccl`): a server/client pair
  that moves weights *between hosts/pools* when an in-program cross-host
  ``device_put`` is unavailable or unsafe (e.g. off-Pathways TPU cross-host
  ``device_put`` SIGSEGVs on teardown). Transports:

    - ``ARROW_FLIGHT`` -- host-staged Apache Arrow Flight over gRPC; saturates
      the NIC and is the portable CPU-fallback path (TPU and anywhere).
    - ``NCCL`` -- GPU device-collective transfer; in one JAX world this is a
      sharded cross-mesh reshard that uses *all* GPUs (no rank-0 fold).

The transport modules import heavyweight deps (``pyarrow``) lazily, so importing
this package (which `reshard.py` does) never requires them.
"""

# Local reshard registry (historical public surface; keep stable).
from tunix.rl.weight_transfer.local import (
    JAX_DEVICE_RESHARD_FACTORY,
    PATHWAYS_RESHARD_FACTORY,
    LocalReshardBackend,
    LocalWeightTransfer,
    ReshardFactory,
    capabilities,
    is_pathways_available,
    select_reshard_fns,
)

# Remote-transport interface (lightweight; no pyarrow/torch at import time).
from tunix.rl.weight_transfer.base import (
    ClientMetrics,
    ServerInfo,
    ServerMetrics,
    WeightTransferClient,
    WeightTransferConfig,
    WeightTransferMode,
    WeightTransferServer,
    WeightUpdate,
)

__all__ = [
    # local reshard
    "JAX_DEVICE_RESHARD_FACTORY",
    "PATHWAYS_RESHARD_FACTORY",
    "LocalReshardBackend",
    "LocalWeightTransfer",
    "ReshardFactory",
    "capabilities",
    "is_pathways_available",
    "select_reshard_fns",
    # remote transport
    "ClientMetrics",
    "ServerInfo",
    "ServerMetrics",
    "WeightTransferClient",
    "WeightTransferConfig",
    "WeightTransferMode",
    "WeightTransferServer",
    "WeightUpdate",
    "create_weight_transfer_server",
    "create_weight_transfer_client",
]


def create_weight_transfer_server(config, coordinator=None, **kwargs):
  """Builds a `WeightTransferServer` for `config.mode` (lazy transport import).

  Args:
    config: A `WeightTransferConfig`. ``config.mode`` selects the transport.
    coordinator: Optional rendezvous `Coordinator` (see `coordinator.py`). When
      None, the transport builds an in-process coordinator (single-process use).
    **kwargs: Forwarded to the concrete server (e.g. ``num_servers``).

  Returns:
    A `WeightTransferServer`.

  Raises:
    ValueError: If ``config.mode`` has no server transport.
  """
  if config.mode == WeightTransferMode.ARROW_FLIGHT:
    from tunix.rl.weight_transfer import arrow_flight  # pylint: disable=g-import-not-at-top

    return arrow_flight.ArrowFlightServer(
        config, coordinator=coordinator, **kwargs
    )
  if config.mode == WeightTransferMode.NCCL:
    from tunix.rl.weight_transfer import nccl  # pylint: disable=g-import-not-at-top

    return nccl.NcclWeightServer(config, coordinator=coordinator, **kwargs)
  raise ValueError(
      f"No remote weight-transfer server for mode {config.mode!r}. Local"
      " resharding (JAX_DEVICE) uses LocalWeightTransfer, not a server."
  )


def create_weight_transfer_client(config, coordinator=None, **kwargs):
  """Builds a `WeightTransferClient` for `config.mode` (lazy transport import).

  Args:
    config: A `WeightTransferConfig`. ``config.mode`` selects the transport.
    coordinator: Optional rendezvous `Coordinator`. Must be the *same* object
      (in-process) or address (cross-process) the server publishes to.
    **kwargs: Forwarded to the concrete client.

  Returns:
    A `WeightTransferClient`.

  Raises:
    ValueError: If ``config.mode`` has no client transport.
  """
  if config.mode == WeightTransferMode.ARROW_FLIGHT:
    from tunix.rl.weight_transfer import arrow_flight  # pylint: disable=g-import-not-at-top

    return arrow_flight.ArrowFlightClient(
        config, coordinator=coordinator, **kwargs
    )
  if config.mode == WeightTransferMode.NCCL:
    from tunix.rl.weight_transfer import nccl  # pylint: disable=g-import-not-at-top

    return nccl.NcclWeightClient(config, coordinator=coordinator, **kwargs)
  raise ValueError(
      f"No remote weight-transfer client for mode {config.mode!r}."
  )
