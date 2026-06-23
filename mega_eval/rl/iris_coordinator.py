"""Cross-job weight-transfer rendezvous via the iris endpoint registry.

The disaggregated RL topology runs the trainer and rollout as SEPARATE iris jobs.
iris's endpoint registry is cluster-global: a name registered with an ABSOLUTE
(``/``-prefixed) name bypasses the per-job namespace (see
``iris.client.client.NamespacedEndpointRegistry.register``), so a rollout job can
resolve an endpoint a trainer job registered. This is the native iris mechanism
``iris.runtime.jax_init`` uses for JAX coordinator discovery -- we use it for the
weight-transfer endpoint instead of an object-store coordinator.

Because iris tasks run with ``net=host``, the trainer's auto-bound Arrow Flight
port is reachable at ``IRIS_ADVERTISE_HOST:port`` from any other job, so we just
register those advertised ``grpc://host:port`` addresses here and the
``ArrowFlightClient`` connects directly. Weight-version state (``weight_id`` and,
for M0's templateless client, ``param_names``) rides in the endpoint metadata.

This is a `tunix.rl.weight_transfer.coordinator.Coordinator`, so it drops into
`ArrowFlightServer(coordinator=...)` / `ArrowFlightClient(coordinator=...)`
unchanged. It only works inside an iris job (``iris_ctx()`` raises otherwise).
"""

import json
import logging

from tunix.rl.weight_transfer.base import ServerInfo
from tunix.rl.weight_transfer.coordinator import Coordinator

logger = logging.getLogger(__name__)


class IrisEndpointCoordinator(Coordinator):
  """Coordinator backed by the cluster-global iris endpoint registry.

  Args:
    name: Absolute (``/``-prefixed) endpoint name shared by the trainer and
      rollout jobs of one RL run, e.g. ``/tunix-rl/<run-id>/weights``. The
      leading ``/`` is what makes it cross-job (bypasses the job namespace).
  """

  def __init__(self, name: str):
    if not name.startswith("/"):
      raise ValueError(
          "IrisEndpointCoordinator needs an absolute (/-prefixed) name for "
          f"cross-job rendezvous, got {name!r}."
      )
    from iris.client.client import iris_ctx  # lazy: only valid inside a job.

    self._name = name
    ctx = iris_ctx()
    self._registry = ctx.registry
    self._resolver = ctx.resolver
    self._endpoint_ids: list[str] = []

  def publish(self, server_info: ServerInfo) -> None:
    # Replace any prior registration so weight_id / param_names advance. The
    # registry maps one name -> many endpoints, so we register every Flight
    # address under the same name; resolve() returns them all.
    for endpoint_id in self._endpoint_ids:
      try:
        self._registry.unregister(endpoint_id)
      except Exception:  # pylint: disable=broad-except
        logger.debug("unregister(%s) failed", endpoint_id, exc_info=True)
    self._endpoint_ids = []

    metadata = {
        "weight_id": str(server_info.weight_id),
        "param_names": json.dumps(server_info.param_names),
    }
    for address in server_info.server_addresses:
      endpoint_id = self._registry.register(
          self._name, address, metadata=metadata
      )
      self._endpoint_ids.append(endpoint_id)
    logger.info(
        "Registered %d weight endpoint(s) under %s at weight_id=%s",
        len(self._endpoint_ids),
        self._name,
        server_info.weight_id,
    )

  def lookup(self) -> ServerInfo | None:
    result = self._resolver.resolve(self._name)
    if result.is_empty:
      return None
    endpoints = result.endpoints
    metadata = endpoints[0].metadata or {}
    weight_id = metadata.get("weight_id")
    param_names = metadata.get("param_names")
    return ServerInfo(
        weight_id=int(weight_id) if weight_id is not None else None,
        server_addresses=[e.url for e in endpoints],
        param_names=json.loads(param_names) if param_names else [],
    )
