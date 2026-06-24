# ot_agent — Qwen3-32B SFT for OpenThoughts-Agent, on 32× H100 (tunix)

Replicate the SFT stage of **OpenThoughts-Agent** (arXiv:2606.24855): fine-tune
`Qwen/Qwen3-32B` on the released 100K agent-trajectory set and (later) an RL pass
on top. The paper reports **44.8%** avg across 7 agentic benchmarks after this
SFT; our charter is to reproduce it with **tunix** on the CoreWeave H100 cluster
(`cw-us-east-02a`) — and, in doing so, to evaluate tunix's fitness for
large-scale, multi-node GPU post-training.

This is the **GPU multi-node** sibling of the in-repo `mega_eval/` project (which
proved the same recipe at Qwen3-8B on TPU). It **reuses `mega_eval` as a
library** — the hard-won assets stay single-sourced — and adds only what
32B-on-4×8-H100 needs.

## What we reuse vs. what's new

| reused from `mega_eval` (unchanged) | new here (`ot_agent/`) |
|---|---|
| ChatML encoder + **assistant-turn loss masking** (`training/agent_sft.py`) | `data.py` — OT-Agent scaling ladder + **process-disjoint** sharding |
| Qwen3 loader w/ key-coverage guards (`models/qwen3_loader.py`) | `distributed.py` — multi-node JAX world (iris registry rendezvous) |
| clipped-AdamW / mesh / metrics glue (`training/common.py`) | `sft.py` — multi-process SFT driver |
| R2 credential + s3 KvStore helpers (`models/checkpoint_staging.py`) | `launch_sft.py` — env-configured entrypoint |
| the `qwen3-32b` registry entry (added to `models/registry.py`) | `export_hf.py` — gather → single HF checkpoint → R2 |
| | `submit_sft.sh` — `iris job run --replicas N` |

## The model fits comfortably on 32× H100

Qwen3-32B, fp32 params (the load-bearing invariant — a 1e-5 AdamW update is below
bf16 ULP, so bf16 storage silently zeroes most updates): params 128 GB + AdamW
m,v 256 GB + grads 128 GB ≈ **512 GB of optimizer state**, FSDP/TP-sharded over
32 GPUs = **16 GB/GPU**, leaving ~64 GB/GPU for activations (decoder remat, bf16
compute). Default mesh is `(fsdp=4, tp=8)`: TP=8 within a node (NVLink, and 8
divides `num_kv_heads=8`), FSDP=4 across nodes (InfiniBand).

## The two things multi-node GPU actually required

Both are framework-level findings about tunix's fitness, not recipe tweaks:

1. **One JAX world across 4 nodes.** tunix has no opinion on rendezvous; iris
   does it (`iris.runtime.jax_init.initialize_jax`: task 0 registers a
   coordinator in the cluster endpoint registry, others poll, all call
   `jax.distributed.initialize`). This is the verified `gpu_gang_smoke` path.
   `mega_eval`'s `init_distributed` *skips* GPU (it only handled TPU + single
   node), so `ot_agent.distributed.init_distributed` is the GPU-multi-node path.

2. **Process-disjoint data.** tunix's training step is multi-host-correct —
   `PeftTrainer` → `sharding_utils.shard_input` builds the global batch from each
   process's *local* slice via `jax.make_array_from_process_local_data`. But the
   `mega_eval` data pipeline has **every process encode the whole corpus**, so on
   4 processes the four local slices are identical → the run sees only ¼ of the
   data per epoch (or silently 4×-replicates each global batch). `data.py` fixes
   this: each process streams a disjoint shard (`idx % N == process_index`) and
   feeds `BATCH_SIZE/N` rows/step. **(`test_data_sharding.py` pins disjoint +
   complete.)** Verdict: tunix's *sharding* is fine; the *example data loader*
   was single-process and had to be made process-aware.

## Checkpointing without a shared filesystem

The 4 CW nodes share no filesystem and orbax can't write `s3://`
(etils.epath/normpath mangles it). So instead of a sharded orbax checkpoint with
nowhere coherent to land, we **gather the actor to host and write one HF-format
safetensors checkpoint** (`export_hf.py`) that the *same* `load_qwen3` reads back
— so eval / RL / serving load it exactly like the base model — then mirror it to
R2. Correctness is anchored on the base model's known torch shapes, making the
loader's `transpose→reshape` transform exactly invertible.
**`test_export_roundtrip.py` proves `loader ∘ exporter == identity`** on a real
tunix Qwen3 (every tensor bit-identical).

## Data: the paper's released ladder

All ungated, all the exact Terminus-2 `conversations` schema (zero encoder
change). Keys map to HF repos in `data.py` (pinned shas in `OT_AGENT_SFT_REVISIONS`):

| key | repo | rows |
|---|---|---|
| `100k` | `open-thoughts/OpenThoughts-Agent-SFT-100K` | 94,334 (headline) |
| `31.6k` / `10k` / `3.16k` / `1k` | `…-SFT-{31.6K,10K,3.16K,1K}` | scaling ladder |
| `coldstart-10k` | `…-SFT-ColdStartForRL-10K` | RL warm-up cold-start |
| `v1` | `…-Agent-v1-SFT` | 15,209 (the 8B-era set) |

(The RL phase set `open-thoughts/OpenThoughts-Agent-v1-RL`, 728 tasks of
`path`+`task_binary`, is for stage 2.)

## Run it — the smoke ladder, then the full run

`submit_sft.sh` has four rungs; **run them in order** (cheap → expensive):

```bash
export HF_TOKEN=...                                  # large HF pulls on the worker
export WANDB_API_KEY=... WANDB_PROJECT=ot-agent      # optional loss curve

STAGE=single   bash ot_agent/submit_sft.sh # 1 node, Qwen3-1.7B: SFT loop + export
STAGE=multi    bash ot_agent/submit_sft.sh # 2 nodes, Qwen3-1.7B: multi-node world + disjoint data
STAGE=bigsmoke bash ot_agent/submit_sft.sh # 4 nodes, Qwen3-32B, ~30 steps: 32B memory FIT
STAGE=full     bash ot_agent/submit_sft.sh # 4 nodes, Qwen3-32B: the 100K replication
```

Watch (iris logs can lag on CW — the job also prints `[ota-*]` tags):
```bash
uv run iris --cluster=cw-us-east-02a job logs /power/<job-name> --follow | grep '\[ota-'
```

The entrypoint is `python -m ot_agent.launch_sft`, configured entirely by env —
see its docstring for the full knob table (`AGENT_MODEL`, `DATASET`, `SFT_STEPS`,
`BATCH_SIZE` (global), `LR`, `MAX_SEQ_LEN`, `TP`, `REMAT`, `EXPORT_DIR`, …).

## Validation status

| layer | status |
|---|---|
| imports / API drift (all modules, CPU) | ✅ green |
| process-disjoint data sharding | ✅ `test_data_sharding.py` |
| HF export = loader inverse (real tunix Qwen3) | ✅ `test_export_roundtrip.py` |
| SFT loop trains + moves weights (PeftTrainer, CPU) | ✅ `test_sft_loop.py` |
| `mega_eval` regression after registry edit | ✅ 46 passed |
| single-node H100: load → SFT → gather → HF export → R2 | ✅ smoke passed (Qwen3-1.7B, loss 0.43, exit 0) |
| multi-node (2-node) JAX world + disjoint data | ✅ both procs one world; `p0`/`p1` read disjoint shards |
| 32B fit + multi-node 32B train on 4×8 H100 | ⏳ `STAGE=bigsmoke` |
| full 100K replication (4 nodes) | ⏳ `STAGE=full` (recipe TBD from paper) |

CPU checks:
```bash
JAX_PLATFORMS=cpu .venv/bin/python -c "import ot_agent.launch_sft"   # import smoke
JAX_PLATFORMS=cpu .venv/bin/python -m pytest ot_agent/tests -q       # 10 passed
```

## Known follow-ups (stage-2 / hardening)

- **Periodic export / resume.** A preemption currently loses the run (export is
  end-of-run only; no shared-FS orbax resume). For the long 32B run, add periodic
  HF export to R2 or a per-node orbax-local + R2-reassembly path.
- **Heartbeat artifact.** iris logs lag on CW; have the job write a status/step
  artifact to R2 to poll progress (per `cw-gpu-cluster-ops-gotchas`).
- **Eval + RL.** Stage 2 wires the 7-benchmark eval and the Dr.GRPO RL pass
  (`OpenThoughts-Agent-v1-RL`) on top of the exported checkpoint — both load it
  via `load_qwen3`, same as the base model.
