# OT-Agent SFT on Qwen3-32B — Tunix/Levanter suitability findings

Charter (#273): evaluate Tunix's fitness for large-scale RL by replicating the
OpenThoughts-Agent SFT (arXiv:2606.24855) — Qwen3-32B on the released 100K
agent-trajectory set at the paper's `cutoff_len=32768` — then an RL pass on top.
Hardware: 32× H100 (4× h100-8x) on CoreWeave `cw-us-east-02a`, via iris.

## TL;DR verdict

The SFT is **feasible but not performant** at 32B / 32k on this 4-node stack.
Every correctness blocker was fixable caller-/config-side (no framework patch),
and the end-to-end pipeline runs: warm-start → train → train-state checkpoint →
bf16 HF export → RL-seed. But measured throughput is **~1,286 tok/s (~0.7% MFU)**,
so the paper's 5-epoch / 100K replication is **~3 months of wall-clock on these 4
nodes** — impractical without either many more nodes or a faster attention kernel.
The 12h budget the user authorized trains **~2% of one epoch**: a genuine
pipeline + scale validation and a partial RL-seed checkpoint, not a replication.

## Framework path: tunix → Levanter

Tunix itself **could not train Qwen3-32B at seq 32768 on GPU**, for two
model/kernel reasons, both architectural to tunix's Qwen3 rather than to our
caller code:

1. **No GPU flash attention.** tunix's Qwen3 flash path is the TPU `splash`
   Pallas kernel only; on GPU `use_flash_attention` fell back to materialized
   `O(seq²)` attention — impossible at 32k. (We prototyped a cuDNN
   `dot_product_attention` path, but it did not make the full step performant.)
2. **Vocab-sized loss tensors.** tunix's `_default_loss_fn` materializes
   `[B,S,V]` fp32 `one_hot`+`log_softmax` over the 152k vocab → 32B train-step
   OOM even with a gather-based CE workaround.

So the SFT was built on **Levanter** (marin-levanter, JAX 0.10), which has a
blockwise flash attention that runs on GPU and a streaming cross-entropy that
never materializes the full-vocab logits. tunix remains the **RL** target; the
SFT→RL bridge is an HF-safetensors checkpoint either framework loads.

## The load-bearing finding: 32B@32k OOM was multi-node FSDP sharding, not activations

The 32B train step OOM'd at 32k. It looked like an activation problem; it was a
**parameter/optimizer-state sharding** problem.

With a mesh of `{replica:1, data:-1, model:8}`, `model=8` consumes all 8
intra-node (ICI/NVLink) GPUs, collapsing the ICI `data` axis to 1. The 4 nodes
then become `replica_dcn=4` — **4 data-parallel replicas**, each holding the
*full* model sharded only 8-way over tensor-parallel. Optimizer state (fp32
master + Adam m,v ≈ 384 GB) / 8 ≈ **48 GB/GPU** (measured base ~46 GiB). No FSDP
across nodes → the step OOM'd.

**Fix (`OTA_DATA_DCN=1`):** shard the FSDP weight axis (`embed`) over *both* the
ICI `data` axis and the DCN `replica_dcn` axis. With `model=8` → `data=1`,
`replica_dcn=4`, so `embed` shards 4-way across nodes (ZeRO-3) × 8-way TP =
32-way. Base dropped from ~46 GiB to **~11.6 GiB**; the step fits with room.
(Mapping `embed→["data","replica_dcn"]` rather than moving `data` to `dcn_axes`,
because Levanter's `axis_shapes()` re-injects a default ICI `data:-1` and would
otherwise raise an axis-overlap error.)

Confirmations that ruled out the activation hypotheses (by reading the kernels,
then measuring):
- Streaming CE saves only x/labels/w/lse in the forward and recomputes logits in
  v-blocks on the backward — O(block), not O(vocab). Measured ~0.5 s.
- Blockwise flash attention is O(seq) fwd+bwd (a `while_loop` over blocks).
  `OTA_FLASH_BLOCK=8192` changed the step only ~7%.
- Nested gradient checkpointing did *not* fix the OOM (correctly — it was never
  the carry stack).

## Throughput characterization

Per-microbatch step time is linear in sequence length with a large
seq-independent floor:

    step(s) ≈ 32.6 (fixed)  +  0.00254 × tokens_per_microbatch

| seq    | s/it (ga=1) | tok/s |
|--------|-------------|-------|
| 32768  | ~107–116    | ~1,200 |
| 16384  | ~74         | ~880  |
| 8192   | ~53 (modeled) | ~610 |

Two consequences:
- The **fixed 32.6 s** is the cross-node FSDP all-gather; it does **not** scale
  with seq, so **tokens/s rises with sequence length** — 32k is both the most
  faithful (no trajectory truncation; the 100K set is median 23.6k / p90 32.7k
  tokens) *and* the highest-throughput choice. We run 32k.
- The all-gather **amortizes over gradient accumulation** (one gather per
  optimizer step, not per microbatch): measured ga=8 step = **815 s** ≈
  `32.6 + 8×97.8`, i.e. ~1,286 tok/s, marginally better than ga=1 despite nested
  checkpointing's ~30% recompute tax (nested is needed for ga>1 HBM headroom).

At ~1,286 tok/s the 100K set (~2.2B tokens packed at 32k) is **~20 days/epoch**;
the paper's 5 epochs ≈ **~3 months** on these 4 nodes.

### Where the time goes / the main future lever

~0.7% MFU is dominated by (a) the un-overlapped cross-node FSDP all-gather and
(b) Levanter's **pure-JAX blockwise flash attention** on GPU (it falls back from
NVIDIA Transformer Engine / cuDNN fused attention). Wiring TE/cuDNN fused
attention and/or overlapping the FSDP gather with compute is the main lever to
move MFU; both are substantial, separately-scoped changes.

## Operational gotchas (CW / iris / Levanter)

- **Host RAM: use `MEM=512GB`; `MEM ≥ ~1024GB` won't schedule.** The schedulable
  ceiling on the shared pool is between 512GB (places in seconds) and 1024GB (hangs
  in `building` indefinitely — observed two 1h+ stalls). 256GB is NOT enough for a
  32B run that both checkpoints and HF-exports: a bf16 export *alone* fits 256GB,
  but when a **train-state checkpoint write coincides with the HF export** (likely
  under a frequent checkpoint cadence) the two host gathers stack and 256GB
  **OOMkills** (`exit 137`, mid-`Saving shard`). 512GB has headroom for the
  concurrent ckpt+export and still schedules — verified live (step-11 checkpoint
  and 14-shard bf16 export completed together at 512GB).
- **bf16 HF export.** Levanter's HF save is shard-by-shard (`~5GB/shard`), but
  fp32 32B is 131GB on disk and the per-shard deshard replicates; `hf_save_dtype=
  bfloat16` halves host/HBM/disk (14 shards, ~65GB). A bf16 checkpoint is a
  faithful RL seed (the fp32 master is a training-time invariant only). Still pair
  it with `MEM=512GB` per above when checkpointing concurrently.
- **Rapid kill→resubmit collides JAX coordinators.** Resubmitting ~30s after
  killing a job reuses a node still running the old JAX coordinator on :8476 →
  `wrong service incarnation` fatal on task 0 → world fails to form. Wait ~3 min
  for node/process cleanup before resubmitting a multi-node job.
- **iris workspace bundle ships committed git only** — uncommitted edits never
  reach the job; commit before every submit.
- **Preemption is real on the shared cluster, and recovery works.** The first
  attempt was preempted (`SIGTERM`, `preemption_notifier.cc`) at step 17 / ~4.3h,
  mid-checkpoint — the cluster reclaimed the nodes. The periodic checkpoint +
  export design held: the step-8 train-state checkpoint and step-11 bf16 export
  survived in S3. Resume needs care: Levanter keys the checkpoint dir by
  `trainer.id` (defaults to the per-job `RUN_ID`), so a fresh job won't auto-find
  it — `OTA_RESUME_ID=<prior job name>` pins `trainer.id` so
  `discover_latest_checkpoint` loads the latest *complete* checkpoint (it correctly
  skipped the SIGTERM-interrupted step-16 and resumed from step-8). **Takeaway for
  the RL phase:** any long run here must checkpoint frequently (we dropped to 30
  min on resume) and be relaunchable by id — preemption is the norm, not the
  exception, on a contended shared pool.

## What the 12h run delivers (job /power/ota-levanter-32b-1782367494)

Qwen3-32B @ 32768, warm-started from the converted base, FSDP via
`OTA_DATA_DCN=1`, nested checkpointing, batch 32 (ga=8), 45 steps (~10.2h at
815 s/step), LR 4e-5 cosine + 0.1 warmup, train-state checkpoint every 2h, bf16
HF export every 11 steps. Deliverables: validated 100K cache build, a partial-SFT
Qwen3-32B checkpoint (~2% of one epoch) as the RL-phase seed, and the throughput
numbers above. Sizing chosen so the cosine schedule completes inside 12h.

**COMPLETED** (initial run `/power/ota-levanter-32b-1782367494`, resumed after one
preemption as `/power/ota-levanter-32b-1782386385`): the full **45/45-step cosine
schedule finished within the 12h budget**. Timeline confirmed live: the 100K cache
built in ~15–20 min via Levanter's distributed "zephyr" tokenizer pool (10 shards);
the **first train step on real 100K data completed in 821.3 s** (steady ~815–825 s/
step — the ~1,286 tok/s model held on the real corpus); **loss 0.40 → 0.249** (min at
step 40) → **0.271 final** (step 45); train-state checkpoints wrote to S3 throughout
(every ~30 min after resume — cadence dropped from 2h to de-risk re-preemption);
and the **bf16 HF export works end-to-end** — 14 safetensors shards of ~4.97 GB
(= ~70 GB bf16; fp32 would be 28 shards / 131 GB), written shard-by-shard to S3.
So the full **train → S3 checkpoint → bf16 HF export → RL seed** path is validated
end-to-end, *including* recovery from a mid-run preemption.

Delivered artifacts (S3, under
`s3://marin-na/users/power/ot-agent-levanter/32b-openthoughts-agent-sft-100k-100k-12h/`,
trainer.id `ota-levanter-32b-1782367494`):
- **RL seeds** (bf16 HF, 23 keys each = 14 shards + index/config/tokenizer):
  `hf/ota-levanter-32b-1782367494/step-{11,22,33,44}`. **step-44 is the final**,
  fully-decayed RL seed.
- **Final train-state checkpoint** (Levanter retains only the latest complete one):
  `checkpoints/ota-levanter-32b-1782367494/step-43` (149 keys). The step-16 partial
  is the SIGTERM-interrupted remnant from the pre-resume preemption — `discover_
  latest_checkpoint` correctly skips it (no completion marker).

This is a **partial SFT** (~2% of one epoch — the 12h budget, not the paper's 5-epoch
replication), serving as the validated RL-phase seed and the source of the throughput
numbers above, not an accuracy reproduction of the paper's 44.8%.

## Suitability bottom line

- **Correctness:** Tunix/Levanter does 32B multi-node GPU SFT; all blockers were
  caller-/config-side, none required a framework fork.
- **Performance:** at 32B/32k on 4 nodes the throughput (~0.7% MFU) makes the
  faithful multi-epoch replication impractical in any reasonable budget. The same
  cross-node-collective and kernel costs will bound the RL phase (rollout +
  training), so RL at this scale needs the attention-kernel/collective-overlap
  work first, or many more nodes.
