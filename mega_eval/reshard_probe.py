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

"""Pin the fast GPU weight-sync path: in-mesh vs disjoint-mesh reshard.

The disjoint-mesh ``device_put`` weight sync clocked ~2.76 GB/s/client on
8xH100 -- ~100x below NVLink, i.e. it is host/PCIe-staged, not NCCL. This probe
isolates the cause by timing, on a single Qwen3-8B-sized bf16 pytree:

  A. in_mesh   -- ONE mesh over all N GPUs, reshard sharding A -> sharding B
                  (XLA should insert NVLink all-to-all collectives -> fast).
  B. disjoint  -- reshard from a mesh on devices[:N/2] to a mesh on
                  devices[N/2:] (cross-mesh transfer -> the slow path).

If A >> B, the NCCL transport should sync within one mesh (trainer + rollout as
sharding *roles*), not across disjoint meshes. Env: MODEL_PRESET, N_SYNC.
"""

import os
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from mega_eval.bench_weight_transfer import _build_params, _pytree_bytes

_GIB = 1024**3


def _reshard_to(tree, sharding_fn):
  out = jax.tree.map(
      lambda x: jax.device_put(x, sharding_fn(x)), tree
  )
  return jax.block_until_ready(out)


def _time(fn, n):
  fn()  # warmup / compile
  ts = []
  for _ in range(n):
    t0 = time.time()
    fn()
    ts.append(time.time() - t0)
  return statistics.median(ts)


def main():
  preset = os.environ.get("MODEL_PRESET", "qwen3-8b")
  n = int(os.environ.get("N_SYNC", "6"))
  devs = jax.devices()
  ndev = len(devs)
  half = ndev // 2
  print(
      f"=== reshard_probe preset={preset} devices={ndev} "
      f"backend={jax.default_backend()} ==="
  )

  # ---- A. in-mesh: one mesh over all GPUs, tp-sharded -> replicated. --------
  m_all = Mesh(np.array(devs).reshape(1, ndev), ("dp", "tp"))
  params_all = _build_params(m_all, preset, jnp.bfloat16)  # tp-sharded source
  wire = _pytree_bytes(params_all)

  def to_replicated():
    return _reshard_to(
        params_all, lambda _x: NamedSharding(m_all, P())
    )

  def to_dp_sharded():
    # tp -> dp shard (a different in-mesh layout; still all-NVLink).
    return _reshard_to(
        params_all, lambda x: NamedSharding(m_all, P("tp", None))
        if x.ndim >= 2 else NamedSharding(m_all, P())
    )

  inmesh_repl = _time(to_replicated, n)
  inmesh_resh = _time(to_dp_sharded, n)
  print(
      f"[A in-mesh] tp->replicated: {inmesh_repl*1000:.0f}ms "
      f"{wire/_GIB/inmesh_repl:.1f} GB/s   |   tp->tp': "
      f"{inmesh_resh*1000:.0f}ms {wire/_GIB/inmesh_resh:.1f} GB/s"
  )

  # ---- B. disjoint: devices[:half] -> devices[half:]. ----------------------
  m_a = Mesh(np.array(devs[:half]).reshape(1, half), ("dp", "tp"))
  m_b = Mesh(np.array(devs[half:]).reshape(1, ndev - half), ("dp", "tp"))
  params_a = _build_params(m_a, preset, jnp.bfloat16)

  def to_mesh_b():
    return _reshard_to(
        params_a,
        lambda x: NamedSharding(m_b, P("tp", None))
        if x.ndim >= 2 else NamedSharding(m_b, P()),
    )

  disjoint = _time(to_mesh_b, n)
  print(
      f"[B disjoint] mesh[:{half}]->mesh[{half}:]: {disjoint*1000:.0f}ms "
      f"{wire/_GIB/disjoint:.1f} GB/s"
  )

  speedup = disjoint / inmesh_resh if inmesh_resh else 0
  print(
      f"PROBE_RESULT wire_gib={wire/_GIB:.2f} inmesh_gbps={wire/_GIB/inmesh_resh:.1f}"
      f" disjoint_gbps={wire/_GIB/disjoint:.1f} inmesh_speedup={speedup:.1f}x"
  )


if __name__ == "__main__":
  main()
