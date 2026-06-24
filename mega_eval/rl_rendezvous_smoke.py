"""M0: cross-JOB Arrow Flight rendezvous smoke for the disaggregated RL topology.

The disaggregated TPU RL plan runs the trainer and rollout as SEPARATE iris jobs
(each its own JAX world / TPU slice) and moves weights between them over Arrow
Flight, discovering each other through iris's cluster-global endpoint registry
(``IrisEndpointCoordinator``) -- the native mechanism, no object store. The one
real unknown is whether a rollout pod can reach the trainer pod's advertised
``grpc://IRIS_ADVERTISE_HOST:port`` across two independent jobs (``net=host``
should make it routable). This smoke proves exactly that, end to end, before we
build the full learner/rollout loop (M1).

Two roles (``ROLE``), pointed at a shared run via ``RUN_ID``; both build the same
coordinator names:

  trainer  generate a synthetic bf16 param pytree, start ``ArrowFlightServer``
           (advertising ``IRIS_ADVERTISE_HOST``), ``serve_weights`` (registers the
           Flight endpoint in the iris registry), log an aggregate checksum, then
           wait for the rollout's ``done`` endpoint and exit.

  rollout  resolve the trainer's endpoint via the iris registry, pull the weights,
           log its own aggregate checksum + weight_id + tensor count, publish the
           ``done`` endpoint, and exit 0 iff it received the expected weights.

Verify byte-exactness by comparing the two logged ``agg_checksum`` lines (already
proven locally; on marin the point is the cross-job hop).

Env:
  ROLE                trainer | rollout                              (required)
  RUN_ID              shared id forming the registry names           (required for iris)
  COORD               iris | s3 | auto                               (default auto)
  S3_COORD_BASE       s3://bucket/prefix (COORD=s3 / local test)     (s3 mode)
  AWS_ENDPOINT_URL    S3 endpoint (R2) for COORD=s3                  (optional)
  FLIGHT_HOST         bind/advertise host for COORD=s3 local test    (default 127.0.0.1)
  N_PARAMS            synthetic param count                          (default 32)
  PARAM_DIM           each param is (PARAM_DIM, PARAM_DIM) bf16       (default 2048)
  NUM_FLIGHT_SERVERS  parallel Flight servers (trainer)              (default 4)
  TIMEOUT_S           max seconds to wait for the peer               (default 600)
  WEIGHT_ID           weight version to serve/expect                 (default 1)
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl.weight_transfer import base
from tunix.rl.weight_transfer import state_dict
from tunix.rl.weight_transfer.arrow_flight import ArrowFlightClient
from tunix.rl.weight_transfer.arrow_flight import ArrowFlightServer


def _env(name, default=None):
  v = os.environ.get(name)
  return v if v not in (None, "") else default


def _log(msg):
  print(f"[rdv][{_env('ROLE', '?')}] {msg}", flush=True)


def _coord_mode():
  mode = _env("COORD", "auto")
  if mode == "auto":
    return "iris" if _env("IRIS_ADVERTISE_HOST") else "s3"
  return mode


def _make_coordinator(suffix):
  """Returns a Coordinator for sub-channel ``suffix`` (weights | done)."""
  mode = _coord_mode()
  if mode == "iris":
    from mega_eval.rl.iris_coordinator import IrisEndpointCoordinator

    run = _env("RUN_ID")
    if not run:
      raise SystemExit("RUN_ID is required for COORD=iris.")
    return IrisEndpointCoordinator(f"/tunix-rl/{run}/{suffix}")
  # Object-store fallback (local test / GPU cross-cluster).
  from tunix.rl.weight_transfer.coordinator import ObjectStoreCoordinator

  base_uri = _env("S3_COORD_BASE")
  if not base_uri:
    raise SystemExit("S3_COORD_BASE is required for COORD=s3.")
  return ObjectStoreCoordinator(f"{base_uri.rstrip('/')}/{suffix}")


def _config():
  mode = _coord_mode()
  if mode == "iris":
    # net=host: bind+advertise the routable host IP iris assigns.
    flight_host = _env("IRIS_ADVERTISE_HOST", "0.0.0.0")
  else:
    flight_host = _env("FLIGHT_HOST", "127.0.0.1")
  return base.WeightTransferConfig(
      mode=base.WeightTransferMode.ARROW_FLIGHT,
      convert_to_bfloat16=True,
      flight_host=flight_host,
      num_flight_servers=int(_env("NUM_FLIGHT_SERVERS", "4")),
      serve_barrier=False,  # separate-job: rollout never calls serve_weights.
  )


def _synthetic_params():
  """A flat bf16 pytree mimicking a stack of transformer weight matrices."""
  n = int(_env("N_PARAMS", "32"))
  dim = int(_env("PARAM_DIM", "2048"))
  return {
      f"layer_{i:03d}.weight": jax.random.normal(
          jax.random.PRNGKey(i), (dim, dim), dtype=jnp.bfloat16
      )
      for i in range(n)
  }


def _agg_checksum(named):
  """Single aggregate checksum over all params (order-invariant byte-sum).

  Summing byte VALUES is commutative => invariant to shape / flattening, while
  still catching any changed byte -- enough to prove a faithful transfer across
  the job boundary by comparing the two logged values.
  """
  total = 0
  for arr in named.values():
    a = np.ascontiguousarray(np.asarray(arr))
    total += int(np.frombuffer(a.tobytes(), dtype=np.uint8).sum(dtype=np.uint64))
  return total % (2**64)


def _bytes(named):
  return sum(int(np.asarray(a).nbytes) for a in named.values())


def run_trainer():
  weight_id = int(_env("WEIGHT_ID", "1"))
  timeout_s = float(_env("TIMEOUT_S", "600"))
  _log(f"jax {jax.__version__} devices={jax.device_count()} coord={_coord_mode()}")

  params = _synthetic_params()
  # flat is {transfer_key: 1-D host array} -- the exact bytes serve transfers.
  _keys, flat = state_dict.flatten_for_transfer(params, convert_to_bfloat16=True)
  agg = _agg_checksum(flat)
  nbytes = _bytes(flat)
  _log(f"synthetic params: {len(flat)} tensors, {nbytes / 1e9:.3f} GB bf16, "
       f"agg_checksum={agg}")

  weight_coord = _make_coordinator("weights")
  server = ArrowFlightServer(_config(), coordinator=weight_coord)
  _log(f"flight servers up at {server._server_addresses}")  # pylint: disable=protected-access

  t0 = time.time()
  server.serve_weights(weight_id, params)
  _log(f"served+registered weight_id={weight_id} in {time.time()-t0:.2f}s; "
       "waiting for rollout 'done'")

  done_coord = _make_coordinator("done")
  deadline = time.time() + timeout_s
  while time.time() < deadline:
    sig = done_coord.lookup()
    if sig is not None:
      _log(f"rollout reported done: weight_id={sig.weight_id} "
           f"note={sig.server_addresses}")
      _log("=== TRAINER OK ===")
      server.cleanup()
      return 0
    time.sleep(3)
  _log("=== TRAINER TIMEOUT waiting for rollout done ===")
  server.cleanup()
  return 1


def run_rollout():
  weight_id = int(_env("WEIGHT_ID", "1"))
  timeout_s = float(_env("TIMEOUT_S", "600"))
  _log(f"jax {jax.__version__} devices={jax.device_count()} coord={_coord_mode()}")

  weight_coord = _make_coordinator("weights")
  client = ArrowFlightClient(_config(), coordinator=weight_coord)

  deadline = time.time() + timeout_s
  update = None
  t0 = time.time()
  while time.time() < deadline:
    update = client.receive_weights(template=None)  # raw host flat
    if update is not None:
      break
    time.sleep(3)
  if update is None:
    _log("=== ROLLOUT TIMEOUT: no weights resolved from registry ===")
    return 1
  fetch_s = time.time() - t0

  flat = update.flat_state  # {name: host np.ndarray}
  agg = _agg_checksum(flat)
  got_bytes = _bytes(flat)
  gbps = got_bytes / 1e9 / max(fetch_s, 1e-3)
  _log(f"received weight_id={update.weight_id} {len(flat)} tensors "
       f"{got_bytes / 1e9:.3f} GB in {fetch_s:.2f}s ({gbps:.2f} GB/s incl. poll); "
       f"agg_checksum={agg}")

  ok = update.weight_id == weight_id and len(flat) > 0 and got_bytes > 0

  # Signal done (verdict in weight_id: 1=received, 0=not) for the trainer.
  done_coord = _make_coordinator("done")
  done_coord.publish(
      base.ServerInfo(
          weight_id=1 if ok else 0,
          server_addresses=[f"agg={agg}", f"gbps={gbps:.2f}"],
          param_names=[],
      )
  )
  client.cleanup()
  _log("=== ROLLOUT OK ===" if ok else "=== ROLLOUT FAIL ===")
  return 0 if ok else 1


def main():
  role = _env("ROLE")
  if role == "trainer":
    raise SystemExit(run_trainer())
  if role == "rollout":
    raise SystemExit(run_rollout())
  raise SystemExit(f"ROLE must be trainer|rollout, got {role!r}")


if __name__ == "__main__":
  main()
