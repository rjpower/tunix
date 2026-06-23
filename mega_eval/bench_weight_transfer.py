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

"""Weight-transfer performance benchmark: 1 trainer -> N inference workers.

Measures real GB/s (and NIC saturation) for moving a model's worth of weights
from the trainer to N rollout workers, for each pluggable transport:

* ``nccl``         -- in-JAX-world cross-mesh reshard (GPU = NCCL collectives,
                      all devices participate). Trainer mesh -> N rollout meshes
                      in ONE process. Best for GPU (single 8xH100 node).
* ``arrow_flight`` -- host-staged Apache Arrow Flight over gRPC. Two layouts:
                      - single process: server + N client threads (loopback;
                        measures serialize/transfer/decode throughput).
                      - multi host (jax.process_count() > 1): process 0 serves,
                        processes 1.. each pull (measures real cross-NIC GB/s).
                      This is the portable TPU cross-host path (off-Pathways
                      cross-host device_put SIGSEGVs).

Driven by env vars so it can be submitted to iris unchanged:

  BENCH_MODE     nccl | arrow_flight                         (default nccl)
  MODEL_PRESET   tiny | qwen3-8b                             (default tiny)
  N_CLIENTS      number of inference workers                 (default 4)
  N_SYNC         timed sync rounds (median reported)         (default 5)
  CONVERT_BF16   1 to cast floats to bf16 on the wire        (default 1)
  TRAINER_FRAC   fraction of devices for the trainer mesh    (default 0.5)
  NIC_GBPS       advertised NIC bandwidth for % saturation   (default 25.0)
  NUM_FLIGHT_SERVERS  Arrow Flight servers (0=auto cpu//4)   (default 0)

Run locally (CPU smoke):
  XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  BENCH_MODE=nccl MODEL_PRESET=tiny .venv/bin/python mega_eval/bench_weight_transfer.py
"""

import concurrent.futures
import gc
import json
import os
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tunix.rl.weight_transfer import base
from tunix.rl.weight_transfer import coordinator as coordinator_lib
from tunix.rl.weight_transfer import nccl as nccl_lib

# Arrow Flight is imported lazily (needs pyarrow) only when BENCH_MODE selects it.

_GIB = 1024**3


def _env(name, default):
  return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Synthetic parameter pytrees with realistic shape distributions.
# ---------------------------------------------------------------------------

# (hidden, n_layers, intermediate, vocab, n_heads, n_kv_heads, head_dim)
_PRESETS = {
    # ~30 MB bf16 — CPU smoke. Real transformer shape distribution, shrunk.
    "tiny": (256, 4, 688, 4096, 8, 2, 32),
    # Qwen3-8B faithful dims (~8.2B params; ~16 GB bf16 / ~33 GB fp32).
    "qwen3-8b": (4096, 36, 12288, 151936, 32, 8, 128),
}


def _build_params(mesh: Mesh, preset: str, dtype) -> dict:
  """Builds a sharded transformer-shaped parameter pytree on ``mesh``.

  Big matmuls are sharded on the model axis ("tp"); norms/scalars replicate.
  Values are deterministic so a client can verify a round-trip if desired.
  """
  h, layers, inter, vocab, n_heads, n_kv, head_dim = _PRESETS[preset]
  q_out = n_heads * head_dim
  kv_out = n_kv * head_dim
  axes = mesh.axis_names
  tp = "tp" if "tp" in axes else axes[-1]

  def shard(spec):
    return NamedSharding(mesh, spec)

  def randn(key, shape, spec):
    g = jax.random.normal(jax.random.PRNGKey(key), shape, dtype) * 0.02
    return jax.device_put(g, shard(spec))

  params = {}
  params["embed"] = randn(1, (vocab, h), P(tp, None))
  params["final_norm"] = randn(2, (h,), P(None))
  params["lm_head"] = randn(3, (vocab, h), P(tp, None))
  for i in range(layers):
    b = 100 + i * 10
    params[f"layer_{i}"] = {
        "q_proj": randn(b + 1, (h, q_out), P(None, tp)),
        "k_proj": randn(b + 2, (h, kv_out), P(None, tp)),
        "v_proj": randn(b + 3, (h, kv_out), P(None, tp)),
        "o_proj": randn(b + 4, (q_out, h), P(tp, None)),
        "gate_proj": randn(b + 5, (h, inter), P(None, tp)),
        "up_proj": randn(b + 6, (h, inter), P(None, tp)),
        "down_proj": randn(b + 7, (inter, h), P(tp, None)),
        "input_norm": randn(b + 8, (h,), P(None)),
        "post_norm": randn(b + 9, (h,), P(None)),
    }
  return params


def _build_template(mesh: Mesh, preset: str, dtype=jnp.bfloat16) -> dict:
  """Rollout-side template on a (different) mesh, in the inference dtype.

  Same structure/shapes/sharding as the trainer params but on the rollout
  ``mesh`` and in ``dtype`` (bf16 by default -- inference runs bf16, and a
  fully-replicated fp32 copy would OOM a 32GB v6e chip). The values are
  placeholders: the transport overwrites every leaf, so only the leaf
  structure / shape / dtype / sharding are load-bearing. Keeping the tp-sharded
  layout means a 1/Ntp shard lands on each rollout device (no OOM) and the
  transfer is a genuine cross-mesh reshard.
  """
  return _build_params(mesh, preset, dtype)


def _pytree_bytes(tree) -> int:
  return sum(
      int(np.prod(l.shape)) * jnp.dtype(l.dtype).itemsize
      for l in jax.tree_util.tree_leaves(tree)
  )


# ---------------------------------------------------------------------------
# Device partitioning: trainer mesh + N rollout meshes from one device pool.
# ---------------------------------------------------------------------------


def _partition_devices(n_clients: int, trainer_frac: float):
  devs = jax.devices()
  n = len(devs)
  n_trainer = max(1, int(round(n * trainer_frac)))
  n_trainer = min(n_trainer, n - n_clients) if n - n_clients >= 1 else 1
  trainer_devs = devs[:n_trainer]
  rest = devs[n_trainer:] or devs[:1]
  # Split the remaining devices across N rollout meshes (round-robin chunks).
  per = max(1, len(rest) // n_clients)
  rollout_device_sets = []
  for i in range(n_clients):
    chunk = (
        rest[i * per : (i + 1) * per] if (i + 1) * per <= len(rest) else rest
    )
    rollout_device_sets.append(chunk or rest[:1])
  return trainer_devs, rollout_device_sets


def _mesh(devices) -> Mesh:
  return Mesh(np.array(devices).reshape(1, len(devices)), ("dp", "tp"))


# ---------------------------------------------------------------------------
# Benchmark drivers.
# ---------------------------------------------------------------------------


def _bench_nccl(
    preset, n_clients, n_sync, convert_bf16, trainer_frac, nic_gbps
):
  trainer_devs, rollout_sets = _partition_devices(n_clients, trainer_frac)
  trainer_mesh = _mesh(trainer_devs)
  rollout_meshes = [_mesh(s) for s in rollout_sets]
  dtype = jnp.float32

  params = _build_params(trainer_mesh, preset, dtype)
  templates = [_build_template(m, preset, jnp.bfloat16) for m in rollout_meshes]
  model_bytes = _pytree_bytes(params)
  wire_bytes = model_bytes // 2 if convert_bf16 else model_bytes

  cfg = base.WeightTransferConfig(
      mode=base.WeightTransferMode.NCCL, convert_to_bfloat16=convert_bf16
  )
  coord = coordinator_lib.InProcessCoordinator()
  server = nccl_lib.NcclWeightServer(cfg, coordinator=coord)
  clients = [
      nccl_lib.NcclWeightClient(cfg, coordinator=coord)
      for _ in range(n_clients)
  ]

  print(
      "[nccl]"
      f" devices={len(jax.devices())} trainer={len(trainer_devs)} rollouts={[len(s) for s in rollout_sets]} model={model_bytes/_GIB:.3f}GiB"
      f" wire={wire_bytes/_GIB:.3f}GiB"
      f" params={len(jax.tree_util.tree_leaves(params))}"
  )

  per_round = []
  pool = concurrent.futures.ThreadPoolExecutor(max_workers=n_clients)
  for r in range(n_sync + 1):  # +1 warmup (compile)
    t0 = time.time()
    server.serve_weights(r + 1, params)
    # Fan out concurrently: each client's reshard is an independent XLA op on a
    # different target mesh, so dispatching from N threads lets them overlap
    # (a serial list-comprehension would block on each client in turn).
    futs = [
        pool.submit(c.receive_weights, t)
        for c, t in zip(clients, templates)
    ]
    updates = [f.result() for f in futs]
    jax.block_until_ready([u.params for u in updates if u])
    dt = time.time() - t0
    if r == 0:
      print(f"[nccl] warmup round {dt*1000:.0f}ms (compile)")
      continue
    per_round.append(dt)
  pool.shutdown(wait=True)

  med = statistics.median(per_round)
  agg_gbps = (wire_bytes * n_clients) / _GIB / med
  print(
      f"[nccl] RESULT median_sync={med*1000:.1f}ms  per_client_gbps="
      f"{wire_bytes/_GIB/med:.2f}  aggregate_gbps={agg_gbps:.2f}  "
      f"nic%={100*agg_gbps/nic_gbps:.0f}"
  )
  return {
      "mode": "nccl",
      "preset": preset,
      "n_clients": n_clients,
      "devices": len(jax.devices()),
      "model_gib": model_bytes / _GIB,
      "wire_gib": wire_bytes / _GIB,
      "median_sync_s": med,
      "per_client_gbps": wire_bytes / _GIB / med,
      "aggregate_gbps": agg_gbps,
      "nic_pct": 100 * agg_gbps / nic_gbps,
      "server_metrics": vars(server.get_metrics()),
      "client0_metrics": vars(clients[0].get_metrics()),
  }


def _bench_arrow_loopback(
    preset, n_clients, n_sync, convert_bf16, nic_gbps, num_servers
):
  from tunix.rl.weight_transfer import arrow_flight  # pylint: disable=g-import-not-at-top

  mesh = _mesh(jax.devices())
  dtype = jnp.float32
  params = _build_params(mesh, preset, dtype)
  template = _build_template(mesh, preset, jnp.bfloat16)
  model_bytes = _pytree_bytes(params)
  wire_bytes = model_bytes // 2 if convert_bf16 else model_bytes

  cfg = base.WeightTransferConfig(
      mode=base.WeightTransferMode.ARROW_FLIGHT,
      convert_to_bfloat16=convert_bf16,
      num_flight_servers=num_servers,
  )
  server = arrow_flight.ArrowFlightServer(cfg)
  coord = server.coordinator
  clients = [
      arrow_flight.ArrowFlightClient(cfg, coordinator=coord)
      for _ in range(n_clients)
  ]
  print(
      f"[arrow-loopback] flight_servers={len(server._flight_servers)} "  # pylint: disable=protected-access
      f"clients={n_clients} model={model_bytes/_GIB:.3f}GiB "
      f"wire={wire_bytes/_GIB:.3f}GiB"
  )

  per_round = []
  for r in range(n_sync + 1):
    server.serve_weights(r + 1, params)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_clients) as ex:
      futs = [ex.submit(c.receive_weights, template) for c in clients]
      updates = [f.result() for f in futs]
    dt = time.time() - t0
    assert all(u is not None for u in updates), "a client got no weights"
    if r == 0:
      print(f"[arrow-loopback] warmup {dt*1000:.0f}ms")
      continue
    per_round.append(dt)

  med = statistics.median(per_round)
  agg_gbps = (wire_bytes * n_clients) / _GIB / med
  cm = clients[0].get_metrics()
  print(
      f"[arrow-loopback] RESULT median_fanout={med*1000:.1f}ms aggregate_gbps="
      f"{agg_gbps:.2f} nic%={100*agg_gbps/nic_gbps:.0f} "
      f"client0 fetch_mibps={cm.fetch_mib_per_second:.0f} "
      f"decode_mibps={cm.decode_mib_per_second:.0f}"
  )
  server.cleanup()
  return {
      "mode": "arrow_loopback",
      "preset": preset,
      "n_clients": n_clients,
      "flight_servers": len(server._flight_servers),  # pylint: disable=protected-access
      "model_gib": model_bytes / _GIB,
      "wire_gib": wire_bytes / _GIB,
      "median_fanout_s": med,
      "aggregate_gbps": agg_gbps,
      "nic_pct": 100 * agg_gbps / nic_gbps,
      "client0_metrics": vars(cm),
  }


def _bench_arrow_multihost(preset, n_sync, convert_bf16, nic_gbps, num_servers):
  """Process 0 serves continuously; every other process pulls (cross-NIC).

  Runs for a fixed wall-clock window (SERVE_SECONDS) instead of a fixed count,
  then ALL processes meet at one ``sync_global_devices`` barrier before exit so
  jax.distributed shuts down in lockstep (otherwise the trainer and inference
  processes hit the implicit shutdown barrier at different times and it times
  out). ``n_sync`` caps how many timed fetches each worker records.
  """
  from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top
  from tunix.rl.weight_transfer import arrow_flight  # pylint: disable=g-import-not-at-top

  pidx = jax.process_index()
  pcount = jax.process_count()
  is_trainer = pidx == 0
  local_mesh = _mesh(jax.local_devices())
  dtype = jnp.float32
  serve_seconds = float(_env("SERVE_SECONDS", "60"))

  cfg = base.WeightTransferConfig(
      mode=base.WeightTransferMode.ARROW_FLIGHT,
      convert_to_bfloat16=convert_bf16,
      num_flight_servers=num_servers,
  )
  coord = coordinator_lib.JaxKvCoordinator(cfg.coordinator_key)
  result = {"mode": "arrow_multihost", "process": pidx, "pcount": pcount}

  if is_trainer:
    params = _build_params(local_mesh, preset, dtype)
    model_bytes = _pytree_bytes(params)
    wire_bytes = model_bytes // 2 if convert_bf16 else model_bytes
    server = arrow_flight.ArrowFlightServer(cfg, coordinator=coord)
    print(
        f"[arrow-mh p{pidx}/{pcount}] TRAINER serving"
        f" model={model_bytes/_GIB:.3f}GiB wire={wire_bytes/_GIB:.3f}GiB"
        f" flight_servers={len(server._flight_servers)}"  # pylint: disable=protected-access
        f" inference_workers={pcount-1} window={serve_seconds:.0f}s"
    )
    # Continuously publish fresh weight_ids so pulling workers always have
    # something new to fetch; re-serving the same params is cheap on the wire.
    wid = 0
    t_end = time.time() + serve_seconds
    while time.time() < t_end:
      wid += 1
      server.serve_weights(wid, params)
      time.sleep(0.5)
    sm = server.get_metrics()
    print(
        f"[arrow-mh p{pidx}] server serves={wid}"
        f" serialize_mibps={sm.serialize_mib_per_second:.0f}"
        f" materialize_mibps={sm.materialize_mib_per_second:.0f}"
    )
    result.update(
        role="trainer",
        wire_gib=wire_bytes / _GIB,
        serves=wid,
        serialize_mibps=sm.serialize_mib_per_second,
    )
    multihost_utils.sync_global_devices("arrow_bench_done")
    server.cleanup()
    print("BENCH_RESULT_JSON " + json.dumps(result, default=float))
    return result
  else:
    template = _build_template(local_mesh, preset, jnp.bfloat16)
    client = arrow_flight.ArrowFlightClient(cfg, coordinator=coord)
    print(f"[arrow-mh p{pidx}/{pcount}] INFERENCE worker pulling")
    gbps = []
    last = -999
    t_end = time.time() + serve_seconds
    while time.time() < t_end and len(gbps) < n_sync:
      upd = client.receive_weights(template)
      if upd is None or upd.weight_id == last:
        time.sleep(0.05)
        continue
      last = upd.weight_id
      cm = client.get_metrics()
      g = cm.receive_bytes / _GIB / cm.fetch_time if cm.fetch_time else 0
      gbps.append(g)
      print(
          f"[arrow-mh p{pidx}] recv weight_id={upd.weight_id}"
          f" bytes={cm.receive_bytes/_GIB:.3f}GiB fetch={cm.fetch_time:.2f}s"
          f" reshard={cm.reshard_time:.2f}s fetch_gbps={g:.2f}"
          f" nic%={100*g/nic_gbps:.0f}"
      )
    med = statistics.median(gbps) if gbps else 0.0
    best = max(gbps) if gbps else 0.0
    print(
        f"[arrow-mh p{pidx}] RESULT fetches={len(gbps)}"
        f" median_fetch_gbps={med:.2f} best_gbps={best:.2f}"
        f" nic%={100*med/nic_gbps:.0f}"
    )
    result.update(
        role="inference",
        fetches=len(gbps),
        median_fetch_gbps=med,
        best_fetch_gbps=best,
        nic_pct=100 * med / nic_gbps,
    )
    # Barrier BEFORE cleanup so all processes (incl. still-serving trainer) exit
    # together; otherwise jax.distributed's shutdown barrier times out.
    multihost_utils.sync_global_devices("arrow_bench_done")
    client.cleanup()
    print("BENCH_RESULT_JSON " + json.dumps(result, default=float))
    return result


def _maybe_init_distributed():
  """Bring up jax.distributed on multi-host TPU before any backend call.

  Guarded on ``PJRT_DEVICE==TPU`` (the iris TPU signal -- ``JAX_NUM_PROCESSES``
  is empty on iris) so a single-process GPU/CPU run never blocks on a
  coordinator. The Arrow Flight multi-host coordinator (`JaxKvCoordinator`)
  needs this; the single-process GPU NCCL bench does not.
  """
  if os.environ.get("PJRT_DEVICE") == "TPU":
    from mega_eval.training import common  # pylint: disable=g-import-not-at-top

    common.init_distributed()


def _run_one(mode: str):
  """Runs one benchmark `mode` and prints its BENCH_RESULT_JSON line."""
  preset = _env("MODEL_PRESET", "tiny")
  n_clients = int(_env("N_CLIENTS", "4"))
  n_sync = int(_env("N_SYNC", "5"))
  convert_bf16 = _env("CONVERT_BF16", "1") == "1"
  trainer_frac = float(_env("TRAINER_FRAC", "0.5"))
  nic_gbps = float(_env("NIC_GBPS", "25.0"))
  num_servers = int(_env("NUM_FLIGHT_SERVERS", "0"))

  print(
      f"=== bench_weight_transfer mode={mode} preset={preset} "
      f"n_clients={n_clients} n_sync={n_sync} bf16={convert_bf16} "
      f"jax={jax.__version__} backend={jax.default_backend()} ==="
  )

  if mode == "nccl":
    result = _bench_nccl(
        preset, n_clients, n_sync, convert_bf16, trainer_frac, nic_gbps
    )
  elif mode == "arrow_flight":
    if jax.process_count() > 1:
      result = _bench_arrow_multihost(
          preset, n_sync, convert_bf16, nic_gbps, num_servers
      )
    else:
      result = _bench_arrow_loopback(
          preset, n_clients, n_sync, convert_bf16, nic_gbps, num_servers
      )
  else:
    raise ValueError(f"Unknown BENCH_MODE={mode!r}")

  print("BENCH_RESULT_JSON " + json.dumps(result, default=float))


def main():
  _maybe_init_distributed()
  # BENCH_SUITE runs several modes in ONE process (so the iris-wired `python`
  # entrypoint stays a single command -- wrapping in `bash -lc` loses the uv
  # venv on PATH). gc.collect() between modes frees device buffers.
  suite = _env("BENCH_SUITE", "")
  if suite:
    for mode in [m.strip() for m in suite.split(",") if m.strip()]:
      _run_one(mode)
      gc.collect()
    print("BENCH_SUITE_DONE")
    return
  _run_one(_env("BENCH_MODE", "nccl"))


if __name__ == "__main__":
  main()
