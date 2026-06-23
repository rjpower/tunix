# Weight-transfer performance — real-hardware numbers

Model: **Qwen3-8B**-shaped synthetic params (327 tensors, 30.5 GiB fp32 / **15.3 GiB bf16**
on the wire). 1 trainer → N inference, off-Pathways. Transports: `nccl` (in-JAX-world
cross-mesh reshard) and `arrow_flight` (host-staged gRPC). Bench: `mega_eval/bench_weight_transfer.py`.

## GPU — 8×H100, single node (cw-us-east-02a)

### Cross-mesh reshard throughput (the transport ceiling) — `reshard_probe`
| reshard | layout | GB/s |
|---|---|---|
| **in-mesh** (one 8-GPU mesh, tp→tp′) | sharded→sharded | **78.2** |
| **disjoint-mesh** (devices[:4]→[4:]) | sharded→sharded | **64.0** |

→ JAX/XLA cross-mesh `device_put` lowers to **NCCL collectives using every GPU** (no
rank-0 fold). Disjoint meshes are *not* host-staged — they hit ~64 GB/s. This is the GPU
weight-sync answer; no torch / raw-NCCL needed for single-node colocated sync.

### 4-inference + 1-trainer fan-out (`nccl`, trainer=4 GPU, 4 rollouts=1 GPU each)
- median sync **5.3 s**, per-client **2.87 GB/s**, aggregate **11.5 GB/s**.
- **Why < 64 GB/s:** each 1-GPU rollout target is *replicated* — the full 15 GiB lands on
  one GPU, and XLA's gather-to-a-single-device (~2.9 GB/s) is the bottleneck, NOT the
  interconnect. **Keep inference targets sharded across ≥2 GPUs** (tp) to hit the 64–78 GB/s
  path; an 8B inference shard at tp=2 reshards at NVLink speed.

## TPU — v6e-16, 4 hosts (marin), Arrow Flight cross-NIC

**1 trainer host → 3 inference hosts**, each inference worker pulling the full 15.3 GiB bf16
model over the NIC, sustained 15 rounds:

| worker | fetches | median GB/s | best GB/s |
|---|---|---|---|
| p1 | 15 | 4.46 | 4.89 |
| p2 | 15 | 4.45 | 5.41 |
| p3 | 15 | 4.65 | 5.15 |

- **~4.5 GB/s per worker** (warmup ~3.3, steady-state ~4.5–5.1), **~13.5 GB/s aggregate**
  (≈108 Gbps — near the trainer's NIC line rate).
- Trainer: 18 serves; device→host materialize **5.4 GB/s**; restore/reshard on inference
  0.6–1.3 s.
- **Aggregate is capped by the trainer's single NIC**, not the transport: 3 workers already
  saturate it. To scale past one NIC, serve from multiple trainer hosts (each serving a
  shard) — a future optimization.

## Key findings
1. **GPU disjoint-mesh reshard is fast (64 GB/s), not host-staged.** My first 2.76 GB/s
   reading was a benchmark artifact (serial fan-out + gather-to-1-GPU). Corrected.
2. **Gather-to-single-device is the GPU weight-sync cliff**, not the interconnect — shard
   inference targets.
3. **Arrow Flight saturates the TPU NIC** (~4.5 GB/s/worker) and is the correct cross-host
   path off-Pathways (in-program cross-host `device_put` SIGSEGVs).
4. **Three real transport bugs** the benchmark surfaced and fixed: (a) a global jax barrier
   in `serve_weights` deadlocked disaggregation → gated behind `serve_barrier`; (b)
   `flatten_for_transfer` did an on-device sharded `reshape` → compiled an all-gather that
   crashes multi-host TPU lowering → reshape on host; (c) the jax KV coordinator was
   insert-only → `allow_overwrite=True` + non-blocking `key_value_try_get`.
5. True disaggregation (separate trainer/inference JAX worlds) is the production end state;
   the one-world colocated path already runs off-Pathways at NVLink speed (GPU) / NIC speed
   (TPU).
