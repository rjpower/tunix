# AGENTS.md — working in this repo (`tunix` fork + `mega_eval/`)

Orientation for an agent landing in this repo for the first time. **MARIN.md is the
*cluster + ops* guide** (how to submit iris jobs, the invariants that bite every TPU
run, the slice/model ladder). **This file is the *codebase* guide** — what lives
where, the two subsystems this fork adds, the public seams, and the gotchas specific
to *this code* (not the framework). Read MARIN.md §5 once for the framework
invariants; read this to navigate and extend the code.

If you only read one thing: this repo is the **`tunix` library** (`tunix/`) plus a
**post-training project** (`mega_eval/`) for the *openthoughts-agent* leaderboard
(Qwen3-8B → SFT → Dr.GRPO RL on Terminal-Bench). The fork's two structural additions
are **(A)** a pluggable, de-Pathways `tunix.rl.weight_transfer` subsystem and **(B)**
a **disaggregated** RL system (`mega_eval/rl/`) where trainer and rollout are
*separate iris jobs*.

---

## 1. Repo map

```
tunix/                     the library (models + rl/ + sft/). We CHANGED:
  rl/weight_transfer/      NEW — pluggable weight-sharing (replaces hardcoded Pathways)
  rl/reshard.py            CHANGED — reshard_fns seam (was hardcoded backend list)
  rl/rl_cluster.py         CHANGED — injectable reshard_backend + close() hook
  rl/rollout/*.py          CHANGED — thread reshard_fns through update_params
  utils/{topology,env_utils}.py  CHANGED — single source of truth for capability checks

mega_eval/                 the openthoughts-agent post-training project (NEW)
  training/                SFT: agent_sft.py (ChatML encoder+trainer), common.py
  agent_data/              SFT data: mixtures.py (weighted multi-dataset blend), agent_traces.py
  eval/                    gVisor-sandboxed eval: sandbox.py, agent_loop.py, grade.py, tb_tasks.py
  rl/                      DISAGGREGATED RL: disagg_train.py, disagg_common.py,
                           iris_coordinator.py, trajectory_channel.py, agentic_collect.py,
                           agent.py, chat_parser.py, environment.py
  models/                  registry.py (presets), qwen3_loader.py, checkpoint{,_staging}.py
  docker/                  Dockerfile.agent-task (docker+runsc+buildx baked in)
  launch_sft.py / launch_eval.py / launch_rl.py    in-process stage entrypoints
  rl_disagg_loop.py        the DISAGGREGATED RL entrypoint (ROLE=trainer|rollout)
  *_smoke.py               the validation ladder (run these before trusting a change)
  tests/                   CPU-only unit tests (run under `--group dev pytest`)
  README.md, DATA_PLAN.md, PYPROJECT_NOTES.md   design docs — read for intent

MARIN.md                   cluster/ops guide (submit pattern, invariants, slice ladder)
README.md                  repo intro
```

**Two ways to run RL, don't confuse them:**
- **In-process** (`mega_eval/launch_rl.py`) — one JAX process, `tunix.rl.rl_cluster.RLCluster`,
  actor+rollout on shared or split meshes (`DISAGGREGATE=1` splits *devices*, not jobs).
  This is stock tunix agentic RL. It is forced to `remat=NONE` (the sampler mutates KV
  Params, conflicting with remat) and hits the cross-host reshard SIGSEGV / KV-OOM at scale.
- **Disaggregated** (`mega_eval/rl_disagg_loop.py`) — **two separate iris jobs**, each its
  own JAX world, weights over Arrow Flight + trajectories over GCS. This is the fork's
  answer to the in-process failure modes. **Prefer this for any real TPU RL run** (see §4).

---

## 2. The de-Pathways weight-transfer subsystem (`tunix/rl/weight_transfer/`)

The charter goal "factor out Pathways → pluggable weight sharing." Before: `reshard.py`
had the backend list hardcoded inline (`[pathwaysutils, jax_device_put]`) plus an inline
`'proxy' in JAX_PLATFORMS` check. Now there is a registry + an injectable seam.

**The seam.** `reshard_pytree(..., reshard_fns=None)` (`tunix/rl/reshard.py:183`) takes an
optional ordered factory list. `None` → lazily `select_reshard_fns(AUTO)`, reproducing the
old behavior exactly. `RLCluster` selects it via `ClusterConfig(reshard_backend=...)`
(`rl_cluster.py:191`), builds `LocalWeightTransfer` (`rl_cluster.py:222`), and threads the
resolved list into `_load_model` and `rollout.update_params`. `RLCluster.close()` calls the
transport cleanup hook (`rl_cluster.py:729`).

**Backend selection** (`local.py`): `LocalReshardBackend` enum `AUTO | JAX_DEVICE | PATHWAYS`.
`AUTO` → `[PATHWAYS_FACTORY, JAX_DEVICE_FACTORY]` (try Pathways, fall back). `capabilities()`
introspects the runtime via `env_utils.is_pathways_initialized()` — **the single source of
truth** (there are three *distinct* capability predicates; don't alias them:
`env_utils.is_pathways_initialized` for backend selection, `env_utils.is_pathways_proxy_backend`
for the proxy-env gate, `topology._is_pathways_backend_used` for host-key parsing).

**Three transports** (pick by topology):

| transport | for | mechanism | where |
|---|---|---|---|
| **local** (`local.py`+`reshard.py`) | colocated / same-host disjoint mesh, one JAX program | `jax.device_put` (or pathways) cross-mesh reshard | in-process RL |
| **NCCL** (`nccl.py`) | **GPU**, both meshes in one JAX world | stores live pytree in a process-global dict; client `reshard_pytree` → XLA→NCCL all-to-all over *all* GPUs (no rank-0 fold) | single-process GPU only |
| **Arrow Flight** (`arrow_flight.py`) | **TPU cross-host / cross-job** (no Pathways) | device→bf16→host→flatten→Arrow RecordBatch→gRPC | the disaggregated path |

**Rendezvous** (`coordinator.py`, `Coordinator` ABC: `publish`/`lookup`):
- `InProcessCoordinator` — tests/single-process.
- `JaxKvCoordinator` — within one `jax.distributed` world (KV store). Cannot span jobs.
- `ObjectStoreCoordinator` — cross-job via tensorstore S3 (R2 creds → `AWS_*`). Local/GPU.
- `mega_eval.rl.iris_coordinator.IrisEndpointCoordinator` — **the TPU cross-job answer**:
  the iris cluster-global endpoint registry (see §4).

**The `state_dict` trick** (`state_dict.py`) — *why Arrow can move sharded TPU weights
across hosts without crashing*: a naive `reshape(-1)` of a sharded tensor inside a jit over
a host-local sub-mesh makes XLA compile an all-gather that references device IDs outside the
local mesh → `RET_CHECK device_id < kMaxDeviceCount` SIGSEGV. `flatten_for_transfer` instead
does **only an elementwise `astype(bf16)` on device** (collective-free, lowers per-shard),
then `jax.device_get` to host, then `reshape(-1)` *on host* (free numpy). `restore_from_flat`
reverses it: host reshape/cast, then `device_put` straight to the target sharding — no
single-device staging, no on-device collective. **Any custom transport must reuse these two
functions; do not reshape sharded arrays on device.**

**Public seams** (one import: `from tunix.rl import weight_transfer`):
```python
weight_transfer.LocalReshardBackend            # AUTO | JAX_DEVICE | PATHWAYS
weight_transfer.select_reshard_fns(backend)    # -> ordered factory list
weight_transfer.LocalWeightTransfer(backend)   # injectable holder (.reshard_fns, .close())
weight_transfer.capabilities()                 # {"pathways": bool, "backend": str}
weight_transfer.WeightTransferConfig(mode=WeightTransferMode.ARROW_FLIGHT, ...)
weight_transfer.create_weight_transfer_server(config, coordinator=...)   # lazy import
weight_transfer.create_weight_transfer_client(config, coordinator=...)
# coordinators are NOT re-exported — import from .coordinator
from tunix.rl.weight_transfer.coordinator import JaxKvCoordinator, ObjectStoreCoordinator
from tunix.rl.weight_transfer.state_dict import flatten_for_transfer, restore_from_flat
```

**Gotchas (each cost a run; see MARIN.md §12 for the perf/bug write-up):**
- `WeightTransferConfig.serve_barrier` defaults **False** and must stay False for any
  disaggregated layout — `True` runs a `sync_global_devices` barrier that only the marin
  multi-host *trainer* topology survives; an inference worker that never calls `serve_weights`
  deadlocks. (`base.py:85`)
- You must pass the **same** `Coordinator` object to both server and client. Passing `None`
  to one silently creates a private in-process coordinator → client polls an empty store
  forever and `lookup()` returns `None`.
- `weight_id` is **not** monotonic-enforced — a trainer restored from an earlier checkpoint
  legitimately rolls weights *backwards*; only an exact repeat means "no new weights."
- vLLM / sglang_jax rollouts accept `reshard_fns` but **don't thread it into the sampler yet**
  (P1 TODO; `vllm_rollout.py:130`, `sglang_jax_rollout.py:119`).
- `utils.put_params_on_memory_kind` was fixed from `any`→`all` (`tree_reduce(and_)`): the old
  OR short-circuited on the first on-device leaf and left offloaded embeddings on host → mixed
  memory-kind tree → "memory_space of all inputs ... must be the same" crash at prefill.

---

## 3. The standalone Dr.GRPO step (`mega_eval/rl/disagg_train.py`)

The disaggregated trainer runs a Dr.GRPO optimizer step **without an `RLCluster`**, reusing
tunix's *registered* loss + advantage estimator. The trick that makes a trajectory need only
`(prompt_ids, completion_ids, reward)`:

- `DrGRPOConfig(num_iterations=1, beta=0.0)` ⇒ **no reference model** (`beta=0` →
  `ref_per_token_logps=None`, KL gated off) **and no old-logp pass** (`num_iterations=1` →
  `old_per_token_logps=None` → loss uses `stop_gradient(cur_logps)` as the PPO baseline,
  ratio≡1). On-policy lockstep keeps this exact.
- **`temperature` gotcha:** `DrGRPOConfig` has *no* `temperature` field, but the loss reads
  `algo_config.temperature` to rescale logits when recomputing logps. The in-process learner
  sets it dynamically (`grpo_learner.py:169`); here `build_algo_config` sets `cfg.temperature`
  *on the instance* (the subclass has a `__dict__`). Forget it → loss silently uses the wrong
  temperature relative to what the rollout sampled.
- **`completion_mask`:** `build_train_example(..., completion_mask=None)` derives `!= pad_id`
  for single-turn. For **agentic multi-turn**, the caller passes the **explicit assistant
  mask** (1=model token, 0=env observation) so the loss trains *only* on policy tokens, not
  on injected tool-output turns. Mandatory in agentic mode — without it the loss trains on
  observations and pollutes the gradient.
- `train_step` pulls `function_registry.get_policy_loss_fn("grpo")`, `nnx.value_and_grad`,
  `optimizer.update`. Loss ≈ 0 with mean-centered advantages is **correct** (sum=0); the
  gradient is non-zero → weights move in the PG direction.

---

## 4. The disaggregated RL system (`mega_eval/rl_disagg_loop.py`)

**One entrypoint, two roles, two reward modes.** `main()` dispatches on `ROLE`
(`trainer`|`rollout`) and, for rollout, on `REWARD_MODE` (`placeholder`|`agentic`).

**The lockstep loop** (async / bounded-staleness — trainer trains on whatever batch is
present and logs the policy-version lag):
```
trainer: load policy on (fsdp=1, tp=N) mesh, remat=BLOCK -> serve weight_id=0 (Arrow)
rollout: pull wid=0 -> generate B*G completions -> grade -> put npz batch to GCS (tagged wid)
trainer: drain one batch -> Dr.GRPO step -> serve wid=1
...repeat RL_STEPS times...
trainer: publish "done" endpoint -> both jobs exit
```

**Rendezvous** (`iris_coordinator.py`): both jobs independently build the **same absolute
endpoint name** `"/tunix-rl/<RUN_ID>/weights"`. The leading `/` is load-bearing — it bypasses
iris's per-job namespace so a rollout job resolves an endpoint a *trainer* job registered.
iris tasks run **net=host**, so the trainer's auto-bound Flight port is reachable at
`IRIS_ADVERTISE_HOST:port`. (`COORD=s3` + `ObjectStoreCoordinator` is the local-test / GPU path.)

**Trajectory channel** (`trajectory_channel.py`): GCS-staged rollout→trainer return path.
`<TRAJ_BASE>/pending/<seq>-<uid>.npz`, atomic `put` (write `.tmp` + `mv`), `wait_for_batch`
polls every 2 s, `consume` deletes. Meta rides as a JSON `__meta__` uint8 array inside the npz.

**The agentic collector** (`agentic_collect.py`) — drives tunix's `TrajectoryCollectEngine`
with a **plain `VanillaRollout`**, no RLCluster:
- **Two tokenizers:** the engine needs `adapt_tokenizer(raw) = TokenizerAdapter(raw)` (adds
  `dedup_bos_ids`/`encode`); the **parser and worker take the RAW HF tokenizer**. Mixing them
  up → `AttributeError` at runtime.
- `VanillaModelCall.__call__` renders the running conversation via `TerminusQwenParser.parse(
  ..., add_generation_prompt=True)` → `worker.generate([rendered], rcfg)` with
  `return_logprobs=True`. Mirrors `agentic_rl_learner._model_call`.
- `TerminusQwenParser` (`chat_parser.py`) overrides `_handle_first_message` to emit only
  `bos_token` — suppressing the stock "You are Qwen..." system injection so the rollout prompt
  is **byte-identical** to the SFT/eval encoding. (Same prompt-parity discipline as
  `eval.model_serving.render_chatml`.)
- `collect(mode="Token")` → `prompt_tokens`, `conversation_tokens` (all turns concatenated),
  `conversation_masks` (1=assistant/0=env), `trajectory_reward`, `old_logprobs`.
  `pad_trajectory` left-pads prompt, right-pads completion+mask → `(prompt_ids, completion_ids,
  completion_mask)`.
- **Tasks must be `register_tasks(built)`'d before any `TerminalBenchEnv` is constructed** —
  they live in a module-level `_TASKS` dict; an unregistered `task_id` raises `KeyError` in
  `__init__`. The rollout builds + registers all task images before the weight-polling loop.

**Env knobs** (read via `disagg_common.env`, empty→default):

| var | default | controls |
|---|---|---|
| `ROLE` | (req) | `trainer` \| `rollout` |
| `REWARD_MODE` | `placeholder` | `placeholder` (hash reward, plumbing) \| `agentic` (gVisor+grader) |
| `PRESET` | `tiny` | `tiny` (random smoke) \| `qwen3-1.7b` \| `qwen3-8b` |
| `RL_STEPS` | `4` | trainer optimizer steps then exit |
| `NUM_GENERATIONS` (G) | `4` | completions per prompt (Dr.GRPO group size) |
| `PROMPTS_PER_BATCH` (B) | `4` | prompts/batch → B*G trajectories/step |
| `MAX_NEW_TOKENS` | `32` | single-turn sampler budget |
| `MAX_PROMPT_LEN` | `128` | prompt pad/truncate length |
| `LEARNING_RATE` | `1e-5` | AdamW lr |
| `TEMPERATURE` | `1.0` | rollout temp **and** `algo_config.temperature` for logp recompute |
| `REMAT` | `block` | trainer checkpointing: `none`\|`block`\|`decoder` (safe — trainer never samples) |
| `TRAJ_BASE` | `./_traj` | GCS URI / shared dir for the channel |
| `COORD` | `auto` | `iris`\|`s3`\|`auto` (iris if `IRIS_ADVERTISE_HOST` set) |
| `RUN_ID` | (req for iris) | forms `/tunix-rl/<RUN_ID>/<suffix>` registry names |
| `TIMEOUT_S` | `1200` | wait-for-peer giveup |
| `TASK_LIMIT` / `TASK_IDS` | `4` / `""` | agentic: how many / which Terminal-Bench tasks |
| `MAX_STEPS` | `12` | agentic: max agent turns/episode |
| `MAX_RESPONSE_LEN` | `2048` | agentic: pad completion to this (drives seq len → HBM) |
| `COMMAND_TIMEOUT` | `60` | agentic: per-shell-command timeout |

**Validated milestone ladder (all GREEN on marin, two separate v6e-4 jobs, exit 0):**
- **M0** `rl_rendezvous_smoke.py` — cross-job Arrow via iris registry, byte-exact.
- **M1a** `rl_disagg_smoke.py` — trainer serves real Qwen3-1.7B; rollout pulls cross-job + generates coherent text.
- **M1c** `rl_disagg_loop.py` `REWARD_MODE=placeholder` — full loop, wid 0→3, mean_reward optimizes.
- **M1d** `rl_disagg_loop.py` `REWARD_MODE=agentic` (commit `4304d76`) — real multi-turn
  Terminal-Bench episodes in per-task gVisor sandboxes → graded → trained. Survived a TPU preempt.

**Gotchas / status:** `remat=BLOCK` on the trainer is **only safe because the trainer never
samples** (disaggregation dodges the sampler/KV-vs-remat conflict that pins the in-process
actor to NONE); without it the agentic seq=5120 backward OOMs (96.7G > 31.25G) even at TP=4.
`reward=0.0` on a base 1.7B model is **expected** (it can't solve the tasks) — real signal
needs the SFT checkpoint. The trainer drains **one** batch/step (queue can grow if rollout
outpaces it — intentional bounded-staleness).

---

## 5. The SFT track (`mega_eval/launch_sft.py` + `training/` + `agent_data/`)

**Flow:** `init_distributed()` (skips on non-TPU to avoid single-GPU hang) → resolve preset
(`models/registry.py`: `qwen3-8b` prod, `qwen3-8b-base`, `qwen3-1.7b-base` smoke) → `build_mesh(tp=TP)`
(BATCH_SIZE must divide the `fsdp` axis = `device_count // TP`) → `spec.load_model(..., dtype=bf16,
param_dtype=fp32, remat=DECODER)` → `run_agent_sft`.

**Per-turn loss masking** (`agent_sft.py`, hand-rolled ChatML — NOT `apply_chat_template`, to
dodge Qwen3 `<think>` wrapping): header `<|im_start|>role\n` is always mask=0; body
`{content}<|im_end|>\n` is mask=1 **only for `role=="assistant"`**. The assistant's closing
`<|im_end|>` is *inside* the mask — that's what teaches the rollout stop condition. Rows where
no assistant token survives truncation are dropped (watch the `[agent-sft] scanned=... usable=...
dropped=...` log).

**Weight loading** (`qwen3_loader.py`) has two hard guards: `_assert_key_coverage` (raises on any
unmapped safetensors key — prevents silent random-init) and `_assert_all_params_concrete` (raises
on a leftover `ShapeDtypeStruct`). It also raises on `rope_scaling` (stock tunix Qwen3 can't express
it; Qwen3-8B has none — a guard, not a limit).

**The SFT mixture** (`agent_data/mixtures.py`, issue #265 — "the leaderboard lever"). Intent
(`DATA_PLAN.md`): the precedent 3-epoch run on 15.2k OTA traces scored 0/70; the leaderboard leader
uses a **SWE-heavy** blend. `MIXTURES["swe_heavy"]` = 8 sources (~55% SWE weighting, ~150k traces/epoch),
each with a per-source cap. Three row adapters normalize the schemas to `{"messages": [...]}`:
`terminus2_adapter` (`conversations`), `json_messages_adapter` (a `messages` JSON-string column,
resolved-only filter), `nebius_trajectory_adapter` (`trajectory`/`ai`→`assistant`). `interleave_sources`
does weighted random choice over **raw pyarrow streams** (renormalizing as sources exhaust) — same
contract as HF `interleave_datasets(stopping_strategy="all_exhausted")` but without the `datasets`
builder. Select via `MIXTURE=swe_heavy|ota_only` or an inline `SFT_MIXTURE` JSON spec (wins over `MIXTURE`).

---

## 6. The eval / gVisor-sandbox track (`mega_eval/launch_eval.py` + `eval/`)

**Flow:** restore SFT model from `CKPT_DIR` (`remat=NONE`, flash off — KV-cache sampler needs it) →
wrap as `model_fn(messages)->str` via `render_chatml` (+`top_p=1.0`, seed++ per call for pass@k
diversity) → `load_tb_tasks()` (HF `open-thoughts/OpenThoughts-TB-dev`) → per task: `build_image` →
k× (`GvisorContainerSandbox` → `run_episode` → `grade_task` → close) → `remove_image` (vfs has no
layer sharing) → `prune_ota_images` backstop.

- **RL-gate metric** (`launch_eval.py:149`): `score_spread = max(scores) - min(scores) > 1e-9`. This
  (not binary pass@1) is the go/no-go for Dr.GRPO — it gets advantage from *score variance* even with
  zero full solves. Shard the 70 tasks over jobs with `TASK_OFFSET`+`TASK_LIMIT`.
- **gVisor sandbox** (`eval/sandbox.py`) — the hardest-won asset, CONFIRMED working inside an iris TPU
  task (kernel `4.19.0-gvisor`). `ensure_sandbox_runtime` (idempotent install of docker+runsc+buildx on
  the stock image; no-op on `Dockerfile.agent-task`) → `ensure_dockerd` (`--storage-driver=vfs
  --iptables=false --bridge=none`) → `GvisorContainerSandbox` (`--runtime=runsc --network=none`). runsc
  args `--platform=ptrace --network=sandbox --ignore-cgroups` are each load-bearing (MARIN.md §6).
  **Requires a `--tpu` slice** (iris adds `--privileged` for accelerators; a CPU-only task is *not*
  privileged and the sandbox won't boot). `build` uses `--network=host` (egress for apt); `run` uses
  `--network=none`. `copy_in` single-file must `docker cp` into the *parent* dir (runsc breaks on a
  not-yet-existing exact target). Container names are uuid4 (RL boots G containers concurrently).
- **Grading** (`grade.py`): copy `tests/` in, run `bash /tests/test.sh`, read score from `reward.txt`
  (priority order) or fall back to exit code; ≥1.0 = solved.

---

## 7. Packaging (`pyproject.toml`, see `PYPROJECT_NOTES.md`)

| extra | pulls | when |
|---|---|---|
| `prod` | `jax[tpu]` | TPU jobs (marin). **Mutually exclusive with `gpu`.** |
| `gpu` | `jax[cuda13]==0.10.0` + cublas + nccl-cu13 | CoreWeave H100 (cw-us-east-02a) |
| `mega` | gcsfs + wandb + pyarrow + marin-iris + marin-fray | **all mega_eval iris jobs** — combine with `prod` or `gpu` |

`[tool.uv] conflicts = [{extra=prod},{extra=gpu}]` forks the lock. Submit TPU SFT/RL with
`--extra prod --extra mega`; GPU with `--extra gpu --extra mega`. The `datasets<4` override is
**load-bearing** for the whole graph to co-resolve — don't loosen it (agent_data reads parquet via
pyarrow directly so it's immune anyway). `wandb` is import-gated on `WANDB_PROJECT`.

**Checkpoint staging gotcha:** orbax reads `gs://` natively (gcsfs) but **cannot read `s3://`**
(`os.path.normpath` mangles `s3://`→`s3:/`). `checkpoint_staging.stage_checkpoint_if_remote` mirrors
s3→local NVMe via tensorstore for the R2 cluster (`R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY` →
mapped to `AWS_*`). See MARIN.md §11.2.

---

## 8. Developing & testing (the loop)

1. **CPU smoke first, always** (free; catches import/API/sharding drift before any TPU spend):
   `JAX_PLATFORMS=cpu uv run python marin_smoke.py` → expect `SUCCESS`. (MARIN.md §8.)
2. **Unit tests** (CPU-only, fast): `uv run --group dev pytest mega_eval/tests/ -q` — covers the
   ChatML encoder/masking, mixture weights+adapters, agent loop, RL env contract, sandbox exec/mirror.
3. **The smoke ladder** before trusting an RL change: `sandbox_smoke` (gVisor boots) →
   `agentic_collect_smoke` (one episode collects) → `rl_rendezvous_smoke` (M0) → `rl_disagg_smoke`
   (M1a) → `rl_disagg_loop` placeholder (M1c) → agentic (M1d). Each needs a `--tpu` slice (gVisor +
   real reshard need real hardware); see `scratchpad/submit_*.sh` for the submit commands.
4. **Editable install** — submitting from this repo ships the *working tree* and installs tunix
   **editable**, so edits to `tunix/` reach the worker with no commit/version-bump (MARIN.md §9a).
   Only re-`uv lock` if you changed *dependencies*.

**Code conventions:** match the surrounding style (pyink formatting, two-space indent, Google-style
imports `g-import-not-at-top` for lazy heavy imports). New tunix-library tests go under `tests/`
(not next to the module — repo policy). Keep `top_p` on every sampler, `clip_by_global_norm` on every
optimizer, fp32 actor params — these are *crashes* if dropped, not drift (MARIN.md §5).

---

## 9. Where to go deeper

- **Cluster ops, invariants, slice ladder, submit pattern** → `MARIN.md` (read §5 invariants once).
- **Weight-transfer perf numbers + the three transport bugs** → `MARIN.md` §12, `mega_eval/docs/WEIGHT_TRANSFER_PERF.md`.
- **SFT data design / the leaderboard lever** → `mega_eval/DATA_PLAN.md`.
- **The post-training recipe + the cross-experiment "when is RL essential" law** →
  `~/code/marin-experiments/{tunix-delphi-rl,openthoughts-agent}/{AGENTS,REPORT}.md`.
- **Persistent cross-session findings** → the agent memory index (`MEMORY.md`); the disaggregated
  architecture decision lives in `tunix-rl-disaggregated-separate-jobs`.
