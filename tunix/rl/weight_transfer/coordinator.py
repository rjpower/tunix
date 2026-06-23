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
import os
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


def _map_r2_creds_to_aws_env() -> None:
  """Maps Cloudflare R2_* creds to the AWS_* env tensorstore's s3 driver reads.

  Generic: only copies credentials (never an endpoint/region) so this stays a
  plain S3-compatible coordinator. The caller supplies ``endpoint``/``region``
  (e.g. mega_eval passes the marin-na R2 endpoint).
  """
  if "R2_ACCESS_KEY_ID" in os.environ:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ["R2_ACCESS_KEY_ID"])
  if "R2_SECRET_ACCESS_KEY" in os.environ:
    os.environ.setdefault(
        "AWS_SECRET_ACCESS_KEY", os.environ["R2_SECRET_ACCESS_KEY"]
    )


class ObjectStoreCoordinator(Coordinator):
  """Cross-JOB coordinator backed by an S3-compatible object store.

  `JaxKvCoordinator` only rendezvous *within* one ``jax.distributed`` world, so
  two INDEPENDENT iris jobs (a separate trainer job and rollout job, each its own
  JAX world) cannot use it to find each other. This coordinator instead publishes
  the `ServerInfo` to a shared object-store prefix that any job with the same
  bucket creds can read -- the portable cross-job rendezvous for the disaggregated
  separate-jobs RL topology.

  Mirrors `JaxKvCoordinator`'s layout: a versioned record under ``v/<weight_id>``
  plus a mutable ``latest`` pointer, so a read is wait-free and a late-joining
  rollout job still finds the current weights. Writes the versioned record first,
  then advances ``latest``, so a reader of ``latest`` always finds a written
  record. Uses tensorstore's s3 driver (works against AWS S3 and Cloudflare R2).

  Args:
    base_uri: ``s3://bucket/prefix`` rendezvous root (one per RL run).
    endpoint: S3 endpoint URL (e.g. the R2 endpoint). Defaults to
      ``AWS_ENDPOINT_URL``; leave unset for real AWS S3.
    region: S3 region. Defaults to ``AWS_REGION`` or ``auto``.
  """

  def __init__(
      self,
      base_uri: str,
      *,
      endpoint: str | None = None,
      region: str | None = None,
  ):
    if not base_uri.startswith("s3://"):
      raise ValueError(
          f"ObjectStoreCoordinator needs an s3:// base_uri, got {base_uri!r}."
      )
    _map_r2_creds_to_aws_env()
    self._endpoint = endpoint or os.environ.get("AWS_ENDPOINT_URL")
    self._region = region or os.environ.get("AWS_REGION", "auto")
    bucket, _, prefix = base_uri[len("s3://") :].partition("/")
    self._kv = self._open_kvstore(bucket, prefix)

  def _open_kvstore(self, bucket: str, prefix: str):
    import tensorstore as ts  # pylint: disable=g-import-not-at-top

    spec = {
        "driver": "s3",
        "bucket": bucket,
        "path": (prefix.rstrip("/") + "/") if prefix else "",
        "aws_region": self._region,
    }
    if self._endpoint:
      spec["endpoint"] = self._endpoint
    return ts.KvStore.open(spec).result()

  def publish(self, server_info: ServerInfo) -> None:
    self._kv.write(
        f"v/{server_info.weight_id}", _encode(server_info)
    ).result()
    self._kv.write(
        "latest", str(server_info.weight_id).encode("utf-8")
    ).result()

  def lookup(self) -> ServerInfo | None:
    latest = self._try_read("latest")
    if latest is None:
      return None
    record = self._try_read(f"v/{int(latest.decode('utf-8'))}")
    if record is None:
      return None
    return _decode(record)

  def _try_read(self, key: str) -> bytes | None:
    """Non-blocking read: the raw value bytes, or None if the key is absent."""
    try:
      res = self._kv.read(key).result()
      # tensorstore returns state == "missing" (empty value) for an absent key.
      if str(getattr(res, "state", "")) == "missing":
        return None
      value = bytes(res.value)
      return value if value else None
    except Exception:  # pylint: disable=broad-except
      logging.debug("S3 KV read miss for %s", key, exc_info=True)
      return None
