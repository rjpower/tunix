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

"""Rendezvous coordinators for remote weight transfer.

The server publishes a `ServerInfo` (which hosts hold the current weights and
under what ``weight_id``); clients look it up and decide whether to pull. marin
uses a Ray actor for this; Tunix has no Ray, so we offer two dependency-free
coordinators behind one interface:

* `InProcessCoordinator` -- a shared object guarded by a lock. For a
  single-process benchmark (server thread + N client threads) and for unit
  tests.
* `JaxKvCoordinator` -- publishes the `ServerInfo` as JSON into the
  ``jax.distributed`` coordination KV store, which every process in a multi-host
  JAX world can read without any extra service. This is the cross-host path for
  the colocated "1 trainer + N rollout" topology (one JAX world, disjoint
  meshes).

Both expose ``publish(server_info)`` (server side) and ``lookup() ->
ServerInfo | None`` (client side).
"""

import abc
import json
import logging
import threading

from tunix.rl.weight_transfer.base import ServerInfo


class Coordinator(abc.ABC):
  """Publishes/looks up the current `ServerInfo` for a weight-transfer group."""

  @abc.abstractmethod
  def publish(self, server_info: ServerInfo) -> None:
    """Server side: advertise the current weights + their server addresses."""

  @abc.abstractmethod
  def lookup(self) -> ServerInfo | None:
    """Client side: fetch the latest advertised `ServerInfo` (None if absent)."""


class InProcessCoordinator(Coordinator):
  """Shared-memory coordinator for single-process use (benchmark / tests)."""

  def __init__(self):
    self._lock = threading.Lock()
    self._server_info: ServerInfo | None = None

  def publish(self, server_info: ServerInfo) -> None:
    with self._lock:
      self._server_info = server_info

  def lookup(self) -> ServerInfo | None:
    with self._lock:
      return self._server_info


def _encode(server_info: ServerInfo) -> bytes:
  return json.dumps({
      "weight_id": server_info.weight_id,
      "server_addresses": server_info.server_addresses,
      "param_names": server_info.param_names,
  }).encode("utf-8")


def _decode(payload: bytes | str) -> ServerInfo:
  if isinstance(payload, bytes):
    payload = payload.decode("utf-8")
  obj = json.loads(payload)
  return ServerInfo(
      weight_id=obj["weight_id"],
      server_addresses=list(obj["server_addresses"]),
      param_names=list(obj["param_names"]),
  )


class JaxKvCoordinator(Coordinator):
  """Coordinator backed by the ``jax.distributed`` coordination KV store.

  Every process in a multi-host JAX world (after ``jax.distributed.initialize``)
  shares one coordination service with a key/value store. The server writes
  ``ServerInfo`` under a versioned key; clients read the latest. We version the
  key by ``weight_id`` (plus a ``/latest`` pointer) so a read is wait-free and a
  late client still finds the current weights.

  This intentionally avoids the collective ``broadcast_one_to_all`` (which would
  require all hosts to call in lockstep): the server pushes and clients poll,
  matching the asynchronous serve/receive model.
  """

  def __init__(self, key_prefix: str):
    self._prefix = key_prefix.rstrip("/")
    self._client = self._get_kv_client()

  @staticmethod
  def _get_kv_client():
    """Returns the live jax.distributed KV client, or raises if uninitialized."""
    # The coordination client is created by jax.distributed.initialize().
    from jax._src import distributed  # pylint: disable=g-import-not-at-top

    state = distributed.global_state
    if state is None or state.client is None:
      raise RuntimeError(
          "JaxKvCoordinator requires jax.distributed.initialize() to have run"
          " (no coordination client is attached)."
      )
    return state.client

  def publish(self, server_info: ServerInfo) -> None:
    # The jax KV store is insert-only by default (InsertKeyValue ->
    # ALREADY_EXISTS on the second serve); ``allow_overwrite=True`` makes both
    # the versioned record and the mutable /latest pointer re-publishable so the
    # trainer can serve repeatedly. Write the versioned record first, then
    # advance /latest, so a client reading /latest always finds a written record.
    payload = _encode(server_info).decode("utf-8")
    self._client.key_value_set(
        f"{self._prefix}/v/{server_info.weight_id}",
        payload,
        allow_overwrite=True,
    )
    self._client.key_value_set(
        f"{self._prefix}/latest",
        str(server_info.weight_id),
        allow_overwrite=True,
    )

  def lookup(self) -> ServerInfo | None:
    latest = self._try_get(f"{self._prefix}/latest")
    if latest is None:
      return None
    record = self._try_get(f"{self._prefix}/v/{int(latest)}")
    if record is None:
      return None
    return _decode(record)

  def _try_get(self, key: str) -> str | None:
    """Non-blocking get: returns the value, or None if the key isn't set yet."""
    try:
      # key_value_try_get returns immediately (raises if unset), unlike
      # blocking_key_value_get which would burn the poll on a timeout.
      return self._client.key_value_try_get(key)
    except Exception:  # pylint: disable=broad-except
      logging.debug("KV get miss for %s", key, exc_info=True)
      return None
