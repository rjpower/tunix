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

"""Apache Arrow Flight weight-transfer transport for Tunix RL.

Ported from marin's ``rl/weight_transfer/arrow_flight.py``, adapted away from
Haliax and Ray:

* Serialization goes through `state_dict.flatten_for_transfer` /
  `state_dict.restore_from_flat` (plain JAX/flax pytrees), not Haliax
  ``state_dict``. The wire format carries, per parameter: the raw flat bytes,
  the numpy dtype string, and a ``(idx, count)`` chunk pair (for arrays larger
  than ``MAX_ELEMENTS_PER_RECORD``). It deliberately does *not* carry shapes:
  the client's template is the source of truth on restore.
* Peer discovery uses a `Coordinator` (see `coordinator.py`) instead of a Ray
  actor. The server publishes a `ServerInfo`; clients poll ``lookup()``.

Like marin, weights are replicated to (and served from) process 0 only, across
``num_servers`` parallel Flight servers. Clients fan a `ThreadPoolExecutor` out
over those servers, picking a server per parameter by ``hash(name) % n`` so the
load is spread. This saturates the NIC and is the portable cross-host path
(required off-Pathways TPU, where an in-program cross-host ``device_put``
SIGSEGVs).
"""

import logging
import math
import os
import socket
import threading
import time
import urllib.request
from collections.abc import Sequence
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from jaxtyping import PyTree

from tunix.rl.weight_transfer import base
from tunix.rl.weight_transfer import coordinator as coordinator_lib
from tunix.rl.weight_transfer import state_dict


logger = logging.getLogger(__name__)


# The maximum number of array elements in a single Arrow RecordBatch. Larger
# arrays are split across multiple RecordBatches to stay under Arrow's 2GB
# per-buffer limit. We assume the largest dtype is 4 bytes (e.g. float32).
MAX_ELEMENTS_PER_RECORD = (2000 * 1000 * 1000) // 4

# Default thread-pool / Flight-server fan-out, derived from CPU count.
_CPU_COUNT = os.cpu_count() or 1
NUM_PARALLEL_SERVERS = max(1, _CPU_COUNT // 4)
NUM_PARALLEL_RECEIVES = max(1, _CPU_COUNT // 4)


# The Arrow record schema for one (chunk of a) parameter. We carry the dtype
# string so bf16 -- which PyArrow can only see as raw uint8 -- round-trips
# exactly, plus an (idx, count) pair so a chunked parameter can be reassembled
# in order. Shapes are intentionally omitted: the client's template supplies the
# shape on restore.
_PARAM_SCHEMA = pa.schema([
    pa.field("data", pa.large_binary()),
    pa.field("dtype", pa.string()),
    pa.field("idx", pa.int64()),
    pa.field("count", pa.int64()),
])


def _resolve_advertise_host() -> str:
  """Resolves a routable host to advertise for the Flight server(s).

  On GCP TPU VMs, queries the metadata server for the internal IP (routable
  within the VPC). Elsewhere, falls back to the hostname when it is not an mDNS
  ``.local`` / ``.localdomain`` name (gRPC's c-ares resolver can't handle
  those), otherwise ``localhost``.
  """
  try:
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/"
        "network-interfaces/0/ip",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=1) as resp:
      ip = resp.read().decode().strip()
      logger.info("Resolved advertise host via GCP metadata: %s", ip)
      return ip
  except Exception:  # pylint: disable=broad-except
    pass

  hostname = socket.gethostname()
  if hostname.endswith(".local") or hostname.endswith(".localdomain"):
    return "localhost"
  return hostname


def _create_binary_array(buffer_data: np.ndarray) -> pa.Array:
  """Builds a single-element Arrow ``large_binary`` array from ``buffer_data``.

  Zero-copy where possible. bfloat16 arrays (from ml_dtypes/jax) aren't
  supported by PyArrow's buffer protocol, so they're viewed as raw ``uint8`` --
  the real dtype travels separately in the record's ``dtype`` column.
  """
  if hasattr(buffer_data, "dtype") and buffer_data.dtype.name == "bfloat16":
    buffer_data = np.ascontiguousarray(buffer_data).view(np.uint8)
  block = pa.py_buffer(buffer_data)
  return pa.Array.from_buffers(
      pa.large_binary(),
      1,  # length
      [None, pa.array([0, len(block)], type=pa.int64()).buffers()[1], block],
  )


def state_dict_to_batches(
    flat: dict[str, np.ndarray], weight_id: int
) -> dict[str, tuple[pa.Schema, Sequence[pa.RecordBatch]]]:
  """Converts a flat state dict to per-parameter Arrow RecordBatches.

  Each value in ``flat`` is a contiguous 1-D host ``np.ndarray`` (as produced by
  `state_dict.flatten_for_transfer`). Arrays larger than
  ``MAX_ELEMENTS_PER_RECORD`` are split across multiple RecordBatches so no
  single Arrow buffer exceeds the 2GB limit.

  Args:
    flat: ``{param_name: 1-D host array}`` to serialize.
    weight_id: Weight version, stamped into the schema metadata.

  Returns:
    ``{param_name: (schema, [batch, ...])}`` for per-parameter Flights.
  """
  schema = _PARAM_SCHEMA.with_metadata({
      "weight_id": str(weight_id),
      "timestamp": str(time.time()),
  })

  result: dict[str, tuple[pa.Schema, Sequence[pa.RecordBatch]]] = {}
  total_bytes = 0
  for name, value in flat.items():
    value = np.ascontiguousarray(value)
    dtype = value.dtype
    total_bytes += value.nbytes

    if value.ndim == 0:
      splits = [value.reshape(1)]
    else:
      assert value.ndim == 1, (
          f"Expected a flattened (1-D) array for parameter {name!r}, got shape"
          f" {value.shape}."
      )
      num_splits = max(1, math.ceil(value.size / MAX_ELEMENTS_PER_RECORD))
      splits = np.array_split(value, num_splits)
    total_parts = len(splits)

    batches = []
    for i, split in enumerate(splits):
      binary_array = _create_binary_array(split)
      batch = pa.RecordBatch.from_arrays(
          [
              binary_array,
              pa.array([str(dtype)], type=pa.string()),
              pa.array([i], type=pa.int64()),
              pa.array([total_parts], type=pa.int64()),
          ],
          schema=schema,
      )
      batches.append(batch)

    result[name] = (schema, batches)

  logger.info(
      "Serialized %d parameters to Arrow, total %.2f MiB",
      len(flat),
      total_bytes / base._BYTES_PER_MIB,  # pylint: disable=protected-access
  )
  return result


def deserialize_arrow_to_array(
    param_name: str, reader: pa.RecordBatchReader
) -> np.ndarray:
  """Reassembles a flat 1-D ``np.ndarray`` from a parameter's RecordBatches.

  Chunks are concatenated as raw bytes and reinterpreted with the dtype carried
  in the record (so bf16 bytes are reinterpreted as bf16). The result is left
  1-D and *not* reshaped: `state_dict.restore_from_flat` reshapes / casts /
  device_puts it onto the client's template leaf.

  Args:
    param_name: Name of the parameter (for logging only).
    reader: Arrow RecordBatch reader for this parameter's Flight stream.

  Returns:
    The parameter's bytes as a flat ``np.ndarray`` with its serialized dtype.
  """
  del param_name  # Only used for debugging.
  parts: list[pa.Scalar] = []
  dtype: str | None = None
  for batch in reader:
    parts.append(batch.column("data")[0])
    if dtype is None:
      dtype = batch.column("dtype")[0].as_py()

  buffer_parts = [
      np.frombuffer(part.as_buffer(), dtype=np.uint8) for part in parts
  ]
  raw = (
      buffer_parts[0]
      if len(buffer_parts) == 1
      else np.concatenate(buffer_parts)
  )
  return raw.view(np.dtype(dtype))


class MarinFlightServer(flight.FlightServerBase):
  """Arrow Flight server that stores and serves flattened weight parameters."""

  def __init__(self, location: str, config: base.WeightTransferConfig):
    super().__init__(location)
    self.config = config
    self._weights_store: dict[
        int, dict[str, tuple[pa.Schema, Sequence[pa.RecordBatch]]]
    ] = {}
    self._latest_weight_id: int | None = None
    self._lock = threading.Lock()
    self._location = location

  def do_put(self, context, descriptor, reader, writer):
    del context, descriptor, reader, writer  # Unused: clients only do_get.

  def do_get(self, context, ticket):
    """Serves one parameter's RecordBatch stream for a ``weight_id/name``."""
    del context
    try:
      ticket_data = ticket.ticket.decode("utf-8")
      if "/" not in ticket_data:
        raise ValueError(
            f"Invalid ticket {ticket_data!r}; expected 'weight_id/param_name'."
        )
      weight_id_str, param_name = ticket_data.split("/", 1)
      weight_id = int(weight_id_str)

      with self._lock:
        if weight_id != self._latest_weight_id:
          logger.debug(
              "Requested weight_id %s stale; serving latest %s",
              weight_id,
              self._latest_weight_id,
          )
          weight_id = self._latest_weight_id
        assert weight_id is not None, "No weights have been published yet."
        schema, batches = self._weights_store[weight_id][param_name]

      return flight.RecordBatchStream(
          pa.RecordBatchReader.from_batches(schema, batches)
      )
    except Exception as e:  # pylint: disable=broad-except
      logger.error("Error in do_get: %s", e)
      raise flight.FlightInternalError(f"Failed to get weights: {e}") from e

  def list_flights(self, context, criteria):
    """Lists every stored ``weight_id/param_name`` as a FlightInfo."""
    del context, criteria
    with self._lock:
      for weight_id, params_dict in self._weights_store.items():
        for param_name, (schema, batches) in params_dict.items():
          ticket_str = f"{weight_id}/{param_name}"
          descriptor = flight.FlightDescriptor.for_command(ticket_str)
          yield flight.FlightInfo(
              schema=schema,
              descriptor=descriptor,
              endpoints=[flight.FlightEndpoint(ticket_str, [self._location])],
              total_records=len(batches),
              total_bytes=sum(batch.nbytes for batch in batches),
          )

  def store_weights(
      self,
      weight_id: int,
      params_dict: dict[str, tuple[pa.Schema, Sequence[pa.RecordBatch]]],
  ) -> None:
    """Replaces the stored weights with ``weight_id``'s parameters."""
    with self._lock:
      self._weights_store.clear()
      self._weights_store[weight_id] = params_dict
      self._latest_weight_id = weight_id

  def get_latest_weight_id(self) -> int | None:
    """Returns the latest stored ``weight_id`` (or None if nothing stored)."""
    with self._lock:
      return self._latest_weight_id


class ArrowFlightServer(base.WeightTransferServer):
  """Serves a parameter pytree over one-or-more Arrow Flight servers.

  Threading model: each `MarinFlightServer` runs ``serve()`` in its own daemon
  thread. Only process 0 stores/serves weights; other processes barrier and
  return from `serve_weights` so the call is collective-safe.
  """

  def __init__(
      self,
      config: base.WeightTransferConfig,
      coordinator: coordinator_lib.Coordinator | None = None,
      num_servers: int | None = None,
  ):
    """Starts the Flight server(s) and resolves the coordinator.

    Args:
      config: Weight-transfer config. ``config.flight_host`` is the bind
        address; ``config.num_flight_servers`` (>0) overrides ``num_servers``.
      coordinator: Rendezvous coordinator the server publishes to. When None, a
        fresh `InProcessCoordinator` is created -- the *same* object must be
        handed to the `ArrowFlightClient` (it is exposed as ``.coordinator``).
      num_servers: Number of parallel Flight servers. Defaults to
        ``config.num_flight_servers`` if >0, else ``NUM_PARALLEL_SERVERS``.
    """
    self.config = config
    if config.num_flight_servers > 0:
      num_servers = config.num_flight_servers
    elif num_servers is None:
      num_servers = NUM_PARALLEL_SERVERS
    self.num_servers = num_servers

    # When no coordinator is supplied, create an in-process one. The caller MUST
    # pass this same object to the client (single-process / test use).
    self.coordinator = (
        coordinator
        if coordinator is not None
        else coordinator_lib.InProcessCoordinator()
    )

    self._flight_servers: list[MarinFlightServer] = []
    self._server_threads: list[threading.Thread] = []
    self._server_addresses: list[str] = []
    self.metrics = base.ServerMetrics()

    advertise_host = (
        config.flight_host
        if config.flight_host != "0.0.0.0"
        else _resolve_advertise_host()
    )
    for i in range(num_servers):
      # Bind on the configured host, port 0 to auto-assign.
      location = f"grpc://{config.flight_host}:0"
      flight_server = MarinFlightServer(location, config)
      address = f"grpc://{advertise_host}:{flight_server.port}"
      self._flight_servers.append(flight_server)
      self._server_addresses.append(address)

      thread = threading.Thread(target=flight_server.serve, daemon=True)
      thread.start()
      self._server_threads.append(thread)
      logger.info("Arrow Flight server %d started at %s", i, address)

  def serve_weights(self, weight_id: int, params: PyTree) -> None:
    """Serializes ``params`` to Arrow, stores on every server, and publishes."""
    self.metrics.total_transfers += 1
    start = time.time()
    try:
      if self.config.serve_barrier:
        _barrier_sync()

      if jax.process_index() != 0:
        if self.config.serve_barrier:
          _barrier_sync()
        return

      keys, flat = state_dict.flatten_for_transfer(
          params, convert_to_bfloat16=self.config.convert_to_bfloat16
      )
      flatten_done = time.time()

      total_bytes, largest_bytes, largest_name = state_dict.summarize(flat)

      params_dict = state_dict_to_batches(flat, weight_id)
      serialize_done = time.time()

      for flight_server in self._flight_servers:
        flight_server.store_weights(weight_id, params_dict)
      store_done = time.time()

      self.coordinator.publish(
          base.ServerInfo(
              weight_id=weight_id,
              server_addresses=list(self._server_addresses),
              param_names=keys,
          )
      )
      publish_done = time.time()

      # `flatten_for_transfer` fuses cast + flatten + device_get, so we report
      # its whole cost as the materialize phase (no separate flatten phase).
      self.metrics.flatten_time = flatten_done - start
      self.metrics.materialize_time = flatten_done - start
      self.metrics.serialize_time = serialize_done - flatten_done
      self.metrics.store_time = store_done - serialize_done
      self.metrics.serve_time = publish_done - start
      self.metrics.transfer_bytes = total_bytes
      self.metrics.total_transfer_bytes += total_bytes
      self.metrics.param_count = len(flat)
      self.metrics.largest_param_bytes = largest_bytes
      self.metrics.materialize_mib_per_second = base.mib_per_second(
          total_bytes, self.metrics.materialize_time
      )
      self.metrics.serialize_mib_per_second = base.mib_per_second(
          total_bytes, self.metrics.serialize_time
      )
      self.metrics.successful_transfers += 1

      logger.info(
          "Served weight_id %s: params=%d bytes=%.2f MiB largest=%s (%.2f MiB)"
          " timings: materialize=%.2fs serialize=%.2fs store=%.2fs",
          weight_id,
          len(flat),
          total_bytes / base._BYTES_PER_MIB,  # pylint: disable=protected-access
          largest_name,
          largest_bytes / base._BYTES_PER_MIB,  # pylint: disable=protected-access
          self.metrics.materialize_time,
          self.metrics.serialize_time,
          self.metrics.store_time,
      )

      if self.config.serve_barrier:
        _barrier_sync()
    except Exception:
      self.metrics.failed_transfers += 1
      logger.exception("Failed to serve weights %s via Arrow Flight", weight_id)
      raise

  def cleanup(self) -> None:
    """Shuts the Flight server(s) down (each shutdown on its own thread)."""
    for flight_server in self._flight_servers:
      logger.debug("Shutting down Arrow Flight server at %s", flight_server)
      threading.Thread(target=flight_server.shutdown, daemon=True).start()

  def get_metrics(self) -> base.ServerMetrics:
    """Returns the latest `ServerMetrics`."""
    return self.metrics


class ArrowFlightClient(base.WeightTransferClient):
  """Pulls weights from Arrow Flight servers, resharded onto a template.

  Parameters are fetched in parallel across servers; a parameter is fetched from
  server ``hash(name) % n`` (matching the server-side fan-out). The raw flat
  ``{name: host array}`` is handed to `state_dict.restore_from_flat`, which
  reshapes / casts / device_puts each leaf onto the *template* leaf's
  shape/dtype/sharding.
  """

  def __init__(
      self,
      config: base.WeightTransferConfig,
      coordinator: coordinator_lib.Coordinator,
  ):
    """Initializes the client.

    Args:
      config: Weight-transfer config (``config.transfer_timeout`` bounds a
        fetch).
      coordinator: The *same* coordinator the server publishes to (for the
        in-process default, the server's ``.coordinator`` object).
    """
    self.config = config
    self.coordinator = coordinator
    # Sentinel distinct from any real weight_id (including a -1 re-init).
    self._last_weight_id: int | None = -2
    self._flight_clients: list[flight.FlightClient] = []
    self._server_addresses: list[str] = []
    self.metrics = base.ClientMetrics()
    self._receive_pool = ThreadPoolExecutor(max_workers=NUM_PARALLEL_RECEIVES)

  def _connect_to_servers(self, addresses: list[str]) -> bool:
    """(Re)connects Flight clients to ``addresses``, reusing them if unchanged."""
    try:
      if set(addresses) != set(self._server_addresses):
        for client in self._flight_clients:
          client.close()
        self._flight_clients = [
            flight.FlightClient(
                addr,
                generic_options=[("grpc.per_message_compression", 0)],
            )
            for addr in addresses
        ]
        self._server_addresses = list(addresses)
        logger.debug("Connected to %d Arrow Flight server(s)", len(addresses))
      return True
    except Exception:  # pylint: disable=broad-except
      logger.warning(
          "Failed to connect to Arrow Flight servers.", exc_info=True
      )
      return False

  def _fetch_param(
      self, weight_id: int, param_name: str
  ) -> tuple[str, np.ndarray]:
    """Fetches one parameter's flat array from its assigned server."""
    ticket = flight.Ticket(f"{weight_id}/{param_name}".encode("utf-8"))
    read_options = pa.ipc.IpcReadOptions(
        ensure_alignment=pa.ipc.Alignment.DataTypeSpecific,
        use_threads=False,
        ensure_native_endian=False,
    )
    call_options = flight.FlightCallOptions(
        read_options=read_options, timeout=self.config.transfer_timeout
    )
    server_id = hash(param_name) % len(self._flight_clients)
    reader = (
        self._flight_clients[server_id]
        .do_get(ticket, options=call_options)
        .to_reader()
    )
    return param_name, deserialize_arrow_to_array(param_name, reader)

  def receive_weights(
      self, template: PyTree | None
  ) -> base.WeightUpdate | None:
    """Fetches the newest weights, resharded onto ``template`` (None if stale)."""
    self.metrics.total_polls += 1
    try:
      start = time.time()
      info = self.coordinator.lookup()
      if info is None:
        logger.info("No Arrow Flight server info available from coordinator.")
        return None

      # We always accept the server's weight_id, even if it is *lower* than the
      # last one we saw: a trainer that crashed and restored from an earlier
      # checkpoint legitimately rolls weights backwards. Only an exact repeat is
      # treated as "no new weights".
      if info.weight_id is None or info.weight_id == self._last_weight_id:
        logger.info("No new weights available from Arrow Flight server.")
        return None

      if not self._connect_to_servers(info.server_addresses):
        return None
      poll_done = time.time()

      flat: dict[str, np.ndarray] = {}
      futures = {
          self._receive_pool.submit(
              self._fetch_param, info.weight_id, name
          ): name
          for name in info.param_names
      }
      for future in as_completed(futures):
        name, array = future.result()
        flat[name] = array
      fetch_done = time.time()

      flat_for_summary = {
          name: np.asarray(value).reshape(-1) for name, value in flat.items()
      }
      receive_bytes, largest_bytes, _ = state_dict.summarize(flat_for_summary)

      if template is not None:
        params = state_dict.restore_from_flat(flat, template)
      else:
        params = None
      decode_done = time.time()

      self.metrics.successful_receives += 1
      self.metrics.receive_bytes = receive_bytes
      self.metrics.total_receive_bytes += receive_bytes
      self.metrics.param_count = len(flat)
      self.metrics.largest_param_bytes = largest_bytes
      self.metrics.poll_time = poll_done - start
      self.metrics.fetch_time = fetch_done - poll_done
      self.metrics.decode_time = decode_done - fetch_done
      self.metrics.reshard_time = decode_done - fetch_done
      self.metrics.fetch_mib_per_second = base.mib_per_second(
          receive_bytes, self.metrics.fetch_time
      )
      self.metrics.decode_mib_per_second = base.mib_per_second(
          receive_bytes, self.metrics.decode_time
      )
      self._last_weight_id = info.weight_id

      logger.info(
          "Received %d params for weight_id %s (bytes=%.2f MiB, poll=%.2fs,"
          " fetch=%.2fs, decode=%.2fs)",
          len(info.param_names),
          info.weight_id,
          receive_bytes / base._BYTES_PER_MIB,  # pylint: disable=protected-access
          self.metrics.poll_time,
          self.metrics.fetch_time,
          self.metrics.decode_time,
      )
      return base.WeightUpdate(
          params=params, flat_state=flat, weight_id=info.weight_id
      )
    except Exception:  # pylint: disable=broad-except
      self.metrics.failed_receives += 1
      logger.error("Failed to receive weights via Arrow Flight", exc_info=True)
      return None

  def cleanup(self) -> None:
    """Closes the Flight clients and the receive thread pool."""
    for client in self._flight_clients:
      try:
        client.close()
      except Exception:  # pylint: disable=broad-except
        logger.debug("Error closing Flight client", exc_info=True)
    self._flight_clients = []
    self._server_addresses = []
    self._receive_pool.shutdown(wait=False, cancel_futures=False)

  def get_metrics(self) -> base.ClientMetrics:
    """Returns the latest `ClientMetrics`."""
    return self.metrics


def _barrier_sync() -> None:
  """Best-effort multi-host barrier (no-op for a single JAX process)."""
  if jax.process_count() <= 1:
    return
  # A trivial collective that all processes participate in, forcing a barrier.
  jax.experimental.multihost_utils.sync_global_devices(
      "tunix_weight_transfer_barrier"
  )
