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

"""Transport-agnostic interface for remote weight transfer.

A `WeightTransferServer` lives on the trainer (actor) side and *serves* a fresh
parameter pytree under a monotonically increasing ``weight_id``. A
`WeightTransferClient` lives on each rollout (inference) worker and *receives*
that pytree, resharded onto the worker's local template (so its mesh/sharding is
preserved). Both sides report MiB/s metrics so a benchmark can quote real
numbers.

This mirrors the shape of marin's `WeightTransferServer/Client` but speaks
plain JAX/flax pytrees (see `state_dict.py`) instead of Haliax NamedArrays, and
discovers peers through a pluggable `Coordinator` (see `coordinator.py`) instead
of a Ray actor.
"""

import abc
import dataclasses
import enum
from typing import Any

from jaxtyping import PyTree


@enum.unique
class WeightTransferMode(enum.Enum):
  """Which transport moves weights from trainer to rollout workers.

  Attributes:
    JAX_DEVICE: In-program cross-mesh reshard (``jax.device_put`` /
      pathwaysutils), handled by `local.LocalWeightTransfer` -- not a
      server/client. Listed here so a single config field can select any
      transport. Safe for colocated / same-host disaggregated meshes.
    ARROW_FLIGHT: Host-staged Apache Arrow Flight over gRPC. Device->host->NIC->
      host->device. Saturates the NIC; the portable cross-host path (required on
      off-Pathways TPU, where an in-program cross-host ``device_put`` SIGSEGVs).
    NCCL: GPU device-collective transfer. In one JAX world this is a sharded
      cross-mesh reshard using every GPU (no rank-0 fold).
  """

  JAX_DEVICE = "jax_device"
  ARROW_FLIGHT = "arrow_flight"
  NCCL = "nccl"


@dataclasses.dataclass
class WeightTransferConfig:
  """Configuration for the remote weight-transfer transports.

  Attributes:
    mode: Which `WeightTransferMode` transport to build.
    convert_to_bfloat16: Cast floating params to bf16 before transfer. Halves
      bytes on the wire and the device->host copy; the trainer keeps fp32
      master weights. Integer/bool params are left untouched.
    sync_interval_steps: Serve weights every N trainer steps (used by the loop,
      not the transport itself).
    transfer_timeout: Per-transfer timeout (seconds).
    flight_host: Bind address for the Arrow Flight server(s). ``0.0.0.0`` binds
      all interfaces and advertises a routable host (see `arrow_flight`).
    num_flight_servers: Number of Flight servers to shard params across for
      parallel serving. 0 => auto (``max(1, cpu_count // 4)``).
    coordinator_key: KV key used by the jax-distributed coordinator to publish
      `ServerInfo`. Only relevant cross-process.
    nccl_bucket_mib: Bucket size (MiB) for the NCCL transport's grouped
      transfers.
  """

  mode: WeightTransferMode = WeightTransferMode.ARROW_FLIGHT
  convert_to_bfloat16: bool = True
  sync_interval_steps: int = 1
  transfer_timeout: float = 600.0
  # Only sync the trainer's *own* shards with a jax collective barrier inside
  # serve_weights. Correct ONLY when every process in the JAX world is a
  # training host that calls serve_weights in lockstep (marin's multi-host
  # trainer). MUST be False for disaggregation (trainer + inference in one
  # world, or separate worlds): inference workers never call serve_weights, so
  # a global barrier there would deadlock. Arrow Flight is a network transport;
  # it needs no jax collective by default.
  serve_barrier: bool = False
  # Arrow Flight.
  flight_host: str = "0.0.0.0"
  num_flight_servers: int = 0
  coordinator_key: str = "tunix_weight_transfer/server_info"
  # NCCL.
  nccl_bucket_mib: int = 512


@dataclasses.dataclass
class ServerInfo:
  """Discovery record published by the server, fetched by clients.

  Attributes:
    weight_id: Identifier of the currently served weights (monotonic; may step
      *backwards* if the trainer restores an earlier checkpoint).
    server_addresses: ``grpc://host:port`` of every Flight server (or transport
      endpoint) holding this weight version.
    param_names: Flat state-dict keys available for this weight version.
  """

  weight_id: int | None
  server_addresses: list[str]
  param_names: list[str]


@dataclasses.dataclass
class WeightUpdate:
  """Result of a successful `WeightTransferClient.receive_weights`.

  Attributes:
    params: The received pytree, resharded onto the client's template (None if
      the caller passed ``template=None`` and only wanted the raw flat state).
    flat_state: The raw flat ``{key: host_array}`` state dict as received.
    weight_id: The ``weight_id`` of these weights.
  """

  params: PyTree | None
  flat_state: dict[str, Any]
  weight_id: int


@dataclasses.dataclass
class ServerMetrics:
  """Trainer-side transfer metrics (per last serve + cumulative)."""

  total_transfers: int = 0
  successful_transfers: int = 0
  failed_transfers: int = 0
  transfer_bytes: int = 0
  total_transfer_bytes: int = 0
  param_count: int = 0
  largest_param_bytes: int = 0
  # Phase timings (seconds) of the last serve.
  flatten_time: float = 0.0
  materialize_time: float = 0.0
  serialize_time: float = 0.0
  store_time: float = 0.0
  serve_time: float = 0.0
  # Throughput (MiB/s) of the last serve.
  materialize_mib_per_second: float = 0.0
  serialize_mib_per_second: float = 0.0


@dataclasses.dataclass
class ClientMetrics:
  """Inference-side transfer metrics (per last receive + cumulative)."""

  total_polls: int = 0
  successful_receives: int = 0
  failed_receives: int = 0
  receive_bytes: int = 0
  total_receive_bytes: int = 0
  param_count: int = 0
  largest_param_bytes: int = 0
  # Phase timings (seconds) of the last receive.
  poll_time: float = 0.0
  fetch_time: float = 0.0
  decode_time: float = 0.0
  reshard_time: float = 0.0
  # Throughput (MiB/s) of the last receive.
  fetch_mib_per_second: float = 0.0
  decode_mib_per_second: float = 0.0


class WeightTransferServer(abc.ABC):
  """Trainer-side transport: serves a parameter pytree under a ``weight_id``."""

  @abc.abstractmethod
  def serve_weights(self, weight_id: int, params: PyTree) -> None:
    """Publishes ``params`` as weight version ``weight_id`` for clients to pull.

    Args:
      weight_id: Monotonic identifier for this weight version.
      params: The trainer parameter pytree (flax/nnx leaves).
    """

  @abc.abstractmethod
  def cleanup(self) -> None:
    """Releases transport resources (servers, sockets, process groups)."""

  @abc.abstractmethod
  def get_metrics(self) -> ServerMetrics:
    """Returns the latest `ServerMetrics`."""


class WeightTransferClient(abc.ABC):
  """Inference-side transport: receives a pytree resharded onto a template."""

  @abc.abstractmethod
  def receive_weights(self, template: PyTree | None) -> WeightUpdate | None:
    """Fetches the newest weights, resharded onto ``template``.

    Args:
      template: A pytree with the client's target structure / sharding / dtype.
        Each received leaf is reshaped + resharded to match the matching
        template leaf. If None, only the raw flat host state is returned.

    Returns:
      A `WeightUpdate` if a *new* weight version was available, else None.
    """

  @abc.abstractmethod
  def cleanup(self) -> None:
    """Releases transport resources."""

  @abc.abstractmethod
  def get_metrics(self) -> ClientMetrics:
    """Returns the latest `ClientMetrics`."""


_BYTES_PER_MIB = 1024 * 1024


def mib_per_second(num_bytes: int, seconds: float) -> float:
  """MiB/s for ``num_bytes`` moved in ``seconds`` (0.0 if either is <= 0)."""
  if num_bytes <= 0 or seconds <= 0:
    return 0.0
  return num_bytes / _BYTES_PER_MIB / seconds
