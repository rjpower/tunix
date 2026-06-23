"""Shared plumbing for the disaggregated (separate-jobs) RL transport.

Both the trainer and rollout jobs use these to (a) pick a cross-job rendezvous
Coordinator -- the iris endpoint registry on a cluster, an object store for local
tests / cross-cluster -- and (b) build the Arrow Flight `WeightTransferConfig` that
advertises a routable host. Factored out of the M0 rendezvous smoke so M1's real
trainer/rollout entrypoints reuse the exact validated path.
"""

import os

import numpy as np

from tunix.rl.weight_transfer import base


def env(name, default=None):
  v = os.environ.get(name)
  return v if v not in (None, "") else default


def coord_mode():
  mode = env("COORD", "auto")
  if mode == "auto":
    return "iris" if env("IRIS_ADVERTISE_HOST") else "s3"
  return mode


def make_coordinator(suffix):
  """A Coordinator for sub-channel ``suffix`` (e.g. weights | done).

  iris  -> IrisEndpointCoordinator("/tunix-rl/<RUN_ID>/<suffix>") (cluster-global).
  s3    -> ObjectStoreCoordinator("<S3_COORD_BASE>/<suffix>")     (local / GPU).
  """
  mode = coord_mode()
  if mode == "iris":
    from mega_eval.rl.iris_coordinator import IrisEndpointCoordinator

    run = env("RUN_ID")
    if not run:
      raise SystemExit("RUN_ID is required for COORD=iris.")
    return IrisEndpointCoordinator(f"/tunix-rl/{run}/{suffix}")
  from tunix.rl.weight_transfer.coordinator import ObjectStoreCoordinator

  base_uri = env("S3_COORD_BASE")
  if not base_uri:
    raise SystemExit("S3_COORD_BASE is required for COORD=s3.")
  return ObjectStoreCoordinator(f"{base_uri.rstrip('/')}/{suffix}")


def flight_config(num_flight_servers=None):
  """Arrow Flight config; binds+advertises the routable host (net=host)."""
  if coord_mode() == "iris":
    flight_host = env("IRIS_ADVERTISE_HOST", "0.0.0.0")
  else:
    flight_host = env("FLIGHT_HOST", "127.0.0.1")
  return base.WeightTransferConfig(
      mode=base.WeightTransferMode.ARROW_FLIGHT,
      convert_to_bfloat16=True,
      flight_host=flight_host,
      num_flight_servers=int(
          num_flight_servers
          if num_flight_servers is not None
          else env("NUM_FLIGHT_SERVERS", "1")
      ),
      serve_barrier=False,  # rollout never calls serve_weights.
  )


def agg_checksum(named):
  """Aggregate order-invariant byte-sum over a {name: array} dict (mod 2^64).

  Commutative => invariant to shape/flattening, so the trainer (over its flat
  transfer dict) and the rollout (over the received flat state) produce the same
  value iff the bytes crossed faithfully -- a cheap cross-job byte-exactness check.
  """
  total = 0
  for arr in named.values():
    a = np.ascontiguousarray(np.asarray(arr))
    total += int(np.frombuffer(a.tobytes(), dtype=np.uint8).sum(dtype=np.uint64))
  return total % (2**64)
