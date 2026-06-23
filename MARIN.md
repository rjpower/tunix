# MARIN.md — running `tunix` on the marin/iris TPU cluster

A bootstrap for future dev work in this repo (`rjpower/tunix`, the fork the marin
experiments will switch to for async-RL). It distills the hard-won, cross-project
knowledge for getting a `google-tunix` workload onto **marin's [iris](https://github.com/marin-community/marin/tree/main/lib/iris)
TPU cluster** and not re-discovering the same five gotchas. Read this once, then
keep the **Invariants** section open while you work.

This is a *cluster + tunix-on-TPU* operations guide. For the **post-training
recipes** that produced the findings, the canonical sources are two sibling
projects under `~/code/marin-experiments/`:

| project | what it proved | read |
|---|---|---|
| `tunix-delphi-rl/` | bootstrapping a **raw 447M base LM** (Delphi) into a tool-user / coder: SFT warm-up → GRPO/Dr.GRPO; the *when-is-RL-essential* law | `AGENTS.md`, `REPORT.md` |
| `openthoughts-agent/` | **Qwen3-8B** terminal agent on Terminal-Bench: ChatML SFT → gVisor-sandboxed agentic Dr.GRPO at 8B/v6e-16 | `AGENTS.md`, `REPORT.md` |

Everything below is what both projects learned in common. When a claim has a
deeper home, the section points to it.

---

## 0. Orientation

- **This repo** is the tunix library itself (`tunix/` = models + `rl/` + `sft/`;
  `tunix.rl.agentic` is the multi-turn/tool-use learner) **plus `mega_eval/`** — the
  openthoughts-agent post-training project (Qwen3-8B SFT → gVisor-sandboxed Dr.GRPO RL)
  layered on top in-repo. The marin experiments still consume tunix as a dependency
  (`google-tunix`, pinned `0.1.7` on PyPI). When you change the fork and want it on a
  worker, see **§9 Developing in this fork**.
- **For the *codebase* (not the cluster), read `AGENTS.md`** — the repo map, the
  de-Pathways `tunix.rl.weight_transfer` subsystem + its public seams, the
  disaggregated separate-jobs RL system (`mega_eval/rl/`), the SFT/eval/data tracks,
  and the code-specific gotchas. This file (MARIN.md) is the cluster/ops half; AGENTS.md
  is the code half. The disaggregated RL *run pattern* is §13 below.
- **iris** is marin's job orchestrator. You submit a *command* + a *resource spec*;
  iris zips your repo (working tree), `uv sync`s the lock on the worker, requests a
  TPU slice, and runs your command. You do **not** ssh to TPUs.
- **The unit of work** is `uv run iris --cluster=marin job run … -- python <entry>.py`
  — see §1 for a verified end-to-end example you can run right now.
- **This repo is wired for iris:** `pyproject.toml` has an `iris` dependency-group
  (`marin-iris` + `marin-fray`) and a `[tool.uv]` prerelease/override block, and
  `marin_smoke.py` is a ready-to-submit hello-world. `uv sync --group iris` and go.
- TPUs are **free on iris** — use them aggressively (parallel sweeps). CPU is for
  compile/import/unit checks *only*; never validate a learning claim on CPU.

---

## 1. Quickstart — run your first job (verified end-to-end)

This is the **whole point of this doc**: how to get a tunix workload onto a marin
TPU and read the result. The flow below was run for real on a **v6e-4** and is the
template for everything else. The artifacts it uses live in this repo:
`marin_smoke.py` (the entrypoint) and the `iris` dependency-group + `[tool.uv]`
block added to `pyproject.toml`.

### 1.1 One-time setup (in this repo)

```bash
# install editable tunix + the iris CLI/client (marin-iris, marin-fray) into .venv.
# the `iris` group + prerelease/override config are already in pyproject.toml.
uv sync --group iris

# the iris CLI is now on the venv; it tunnels to the controller automatically.
uv run iris --cluster=marin cluster status
```

`uv run <cmd>` everywhere — bare `python`/`iris` are not on PATH. If `uv run` is
blocked, call `.venv/bin/python` / `.venv/bin/iris` directly (both work).
Authentication to the controller is via your existing `gcloud` login — the CLI opens
an SSH tunnel to `iris-controller-marin` on its own (you'll see "Tunnel ready" in the
log). Jobs that pull gated HF models or log to wandb need those secrets exported so
`-e VAR "$VAR"` can forward them:

```bash
export HF_TOKEN=...          # gated HF model/dataset pulls happen on the worker
export WANDB_API_KEY=...     # only if you set WANDB_PROJECT on a job
```

### 1.2 Check it on CPU first (free, ~30 s)

`marin_smoke.py` runs on CPU too — always shake out import/API errors here before
spending a TPU (CPU caught two real bugs while writing this: an eager sharded-gather
error and a deprecated mesh context manager):

```bash
JAX_PLATFORMS=cpu uv run python marin_smoke.py
# => [marin-smoke] ... 1 devices, kind='cpu'
#    [marin-smoke] tunix Qwen3-0.6B forward OK -- logits (1, 8, 151936) ...
#    [marin-smoke] SUCCESS
```

### 1.3 Submit to a real v6e-4 (the command that matters)

```bash
uv run iris --cluster=marin job run --no-wait \
  --tpu v6e-4 --enable-extra-resources --extra prod \
  --region europe-west4 --region us-east1 --region us-east5 \
  --cpu 8 --memory 64GB --disk 60GB --max-retries 1 --job-name marin-smoke \
  -- python marin_smoke.py
```

What each part does is in §2. The one repo-specific bit: **`--extra prod`** — that is
*this* repo's optional-dependency group that pulls `jax[tpu]`/libtpu (the marin
experiment repos name the same thing `tpu`; match the name to the repo's
`pyproject.toml` you're submitting from). On submit you'll see the bundle size and
the job id:

```
Workspace bundle size: 13.0 MB
Job submitted: /power/marin-smoke
```

### 1.4 Watch it, then read the result

```bash
# poll state (building -> running -> succeeded); ~20 s to build, ~1 min to run
uv run iris --cluster=marin job summary /power/marin-smoke

# pull the logs and grep your tag out of the libtpu noise
uv run iris --cluster=marin job logs /power/marin-smoke --max-lines 4000 | grep marin-smoke
```

The verified output — note it landed a real **4-chip v6 lite** slice and ran a tunix
model on it:

```
[marin-smoke] google-tunix 0.1.7, jax 0.10.2
[marin-smoke] process 0/1, 4 devices, kind='TPU v6 lite'
[marin-smoke] pmap sum-of-squares per device: [3.57e+08 2.50e+09 6.80e+09 1.32e+10]
[marin-smoke] tunix Qwen3-0.6B forward OK -- logits (1, 8, 151936) dtype float32
[marin-smoke] SUCCESS
```

That confirms the full path: iris zipped the repo → the worker `uv sync --extra prod`
installed tunix + jax[tpu] → JAX saw the 4 attached v6e chips → a real (random-init)
tunix Qwen3-0.6B compiled and ran a forward pass on the TPU. **From here, swap
`marin_smoke.py` for a launcher that does SFT/RL** (see §4 and the experiment repos).

> Anatomy of `marin_smoke.py` (the minimal "tunix run"): guard
> `jax.distributed.initialize()` to multi-host (Invariant 1) → print `jax.devices()`
> → a `pmap` across every chip → build `Qwen3-0.6B` via
> `tunix.models.dummy_model_creator.create_dummy_model` over an `(fsdp, tp)` mesh
> (`axis_types=AxisType.Auto` so GSPMD resolves the sharded embedding gather) → one
> forward under `nnx.jit`. No weights, no env vars.

---

## 2. The iris submit pattern (the one command)

Every TPU job is this shape. Memorize the flags; they are all load-bearing.

```bash
uv run iris --cluster=marin job run --no-wait \
  --tpu v6e-8 --enable-extra-resources --extra prod \
  --region europe-west4 --region us-east1 --region us-east5 \
  --cpu 16 --memory 200GB --disk 100GB --max-retries 3 --job-name my-run \
  -e HF_TOKEN "$HF_TOKEN" -e WANDB_PROJECT my-proj -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e SOME_KNOB value \
  -- python launch_something.py
```

| flag | why it matters |
|---|---|
| `--tpu v6e-N` | requests the accelerator **and** attaches devices. Pick N per the model ladder (§3). |
| `--enable-extra-resources` | opt-in gate required whenever you ask for a TPU/GPU or ≥4 GB mem / ≥10 GB disk. It installs nothing; it just unlocks the heavy request. |
| `--extra <name>` | a `uv` optional-dependency group `uv sync`'d on the worker. **Use `--extra prod`** from *this* repo (it pulls `jax[tpu]`/libtpu); the experiment repos call the same thing `--extra tpu`. Required for any TPU job — without the TPU jax plugin the worker runs on CPU. |
| `--region … (×3)` | **mandatory.** v6e exists in only 3 zones; pass **all three** or the job sits `PENDING`. (zones: `europe-west4-a`, `us-east1-d`, `us-east5-b`.) |
| `--disk NGB` | request **≤ a single node's free space**. `300GB` gets rejected (`insufficient_resources: disk`); 100GB runs an 8B RL probe. gVisor's `vfs` driver doesn't share image layers, so multi-task eval disk ≈ Σ task-image sizes (200–250GB). |
| `--max-retries N` | preemptible v6e gets reclaimed. Use `3` for long runs that checkpoint+resume; `0` when you want a clean single-shot (a stale retry can mask a real failure). |
| `--no-wait` | return immediately (the job runs in the background); inspect with the commands below. |
| `--job-name` | becomes the job id `/power/<name>`. Keep it unique per submit. |
| `-e KEY VAL` | env vars. The entrypoints are configured **entirely by env** (no argparse). `-e HF_TOKEN "$HF_TOKEN"` forwards your shell secret. |

**Pin a region when colocating with data.** A GCS bucket read is cheapest
in-region: `gs://marin-us-east5/…` ⇒ run in `--region us-east5` (v6e is in
`us-east5-b`). Ledger/cache availability is **not** uniform across the three v6e
regions — a cache that loads in one may EOF-fail in another; when contended, be
patient rather than region-hopping. (See memory `marin-v6e-region-data-topology`.)

### Watch / inspect / stop

```bash
uv run iris --cluster=marin job summary /power/my-run                       # state + per-task
uv run iris --cluster=marin job logs    /power/my-run --max-lines 5000      # full logs
uv run iris --cluster=marin job logs    /power/my-run --follow              # stream
uv run iris --cluster=marin job stop    /power/my-run
```

Grep your launcher's log tag (e.g. `grep '\[ota-'`) to cut through libtpu noise.

### Fanning out a sweep (zsh gotcha)

v6e-4/-8 slices are plentiful — shard sequential work (e.g. per-task eval) across
many small jobs instead of one long one. **The Bash tool runs zsh, which does NOT
word-split unquoted `$var`** — `for i in $LIST` runs once with the whole string.
Use a literal list:

```bash
for i in 11 12 13 14 15; do
  uv run iris --cluster=marin job run --no-wait --tpu v6e-4 … \
    -e TASK_OFFSET $i -e TASK_LIMIT 1 --job-name sweep-$i -- python launch_eval.py
done
```

### What ships in the bundle (and the 25 MB cap)

On submit, iris zips the workspace (cwd) and uploads it. The selection
(`iris.cluster.client.bundle`) is **`git ls-files --cached --others
--exclude-standard`**, then the zip reads each file's **current working-tree
contents**. Consequences, all verified:

- **Uncommitted edits to tracked files DO ship** (the zip reads disk, not the git
  blob), and **untracked-but-not-gitignored files ship too** (e.g. a freshly
  written `marin_smoke.py` and `uv.lock` shipped with no `git add`). Only
  **gitignored** files are excluded — so the "commit-before-submit" habit is about
  *reproducibility*, not about getting your change onto the worker.
- A built-in `DEFAULT_EXCLUDE` drops `.git`, `__pycache__`, `.venv`, `*.egg-info`,
  `node_modules`, and **`docs/images`/`docs/figures`/`docs/static`** before zipping —
  which is why this 58 MB repo bundles to **13.0 MB** (the heavy example notebooks
  compress; the doc images are excluded).
- **Hard cap: 25 MB** (`MAX_BUNDLE_SIZE_BYTES`); over it the submit raises
  `Bundle size … exceeds maximum 25MB`. Keep large data/checkpoints out of the
  workspace (use `gs://`), not in the bundle.
- **Make sure `.venv/` is gitignored** before you `uv sync` in a new repo, or it
  becomes an untracked-non-ignored tree and blows the cap. (This repo's `.gitignore`
  already lists `/.venv/`.)
- **Re-`uv lock` before submit if stale** — `marin-*` are nightly `0.2.x.dev`
  prereleases; the worker `uv sync`s the lock that ships in the bundle.

---

## 3. TPU slice & model-size ladder (what fits where)

Memory is the binding constraint; the lever is **tensor parallelism** (`TP`) + flash
attention + remat. Validated fits:

| model | task | slice | config | notes |
|---|---|---|---|---|
| Delphi 447M | SFT+GRPO (tool/coding) | **v6e-4** | TP=1 | the whole `tunix-delphi-rl` recipe runs here in minutes/stage |
| Qwen3-1.7B | agentic RL (machinery smoke) | **v6e-4** | **TP=4** | TP=1 OOMs at `remat=NONE` (42G temporaries); TP shards per-seq activations → fits |
| Qwen3-1.7B | coding curriculum RL | **v6e-8** | — | the heavier rollout shape OOMs v6e-4 (see `tunix-1.7b-rl-win` memory) |
| Qwen3-8B | SFT | **v6e-16** | **TP=4**, flash, batch 4, seq 8192 | memory ladder: 70.6G → 33.4G → fits |
| Qwen3-8B | agentic RL | **v6e-16** | **TP=8** | restore→rollout(G=2)→grade→train_step fits at prompt 4096 / resp 768 / 3 turns; larger G or longer episodes ⇒ TP=16 or shorter seqs |

Rules of thumb:
- **`BATCH_SIZE` must be divisible by the fsdp axis** (`device_count // TP`).
- **`--disk` ≤ one node's free space** (see §2).
- **8B RL timing** (v6e-16 probe): restore ~11 min, image build ~2 min, ~4 min per
  step at G=2 / 3 turns. Per-step cost scales with **G × turns × tokens**.
- Multi-host (v6e-8 spans 2 hosts, v6e-16 spans 4) **works** for agentic RL — both
  1.7B/v6e-8 and 8B/v6e-16 ran the full loop with no collective desync — **but only
  if** you call `jax.distributed.initialize()` first (Invariant 1).

---

## 4. The 3-stage recipe (distilled)

Both projects implement the same loop pointed at harder behaviors: **put a behavior
in-distribution with an SFT warm-up, then amplify it with group-relative RL on the
same in-memory actor** (no checkpoint round-trip between SFT and RL).

1. **SFT for format** — teach the transcript token format with **per-turn loss
   masking**: train (mask 1) on the model's own turns, mask out (mask 0) the env's
   lines (the model must *copy/read* those at RL time, not produce them).
2. **SFT for the call surface** — teach the tool/action surface + the result-copy.
   For raw base LMs, bare-text surfaces (`CALC(a*b)`, an `END`-sentinel program)
   beat Qwen JSON, which is OOD (`tool_call_rate ≈ 0`). For instruct/agent models
   use their native ChatML. *(Stages 1 & 2 are usually one warm-up.)*
3. **RL** — GRPO (tool use) or **Dr.GRPO** (coding/agentic; more robust on small
   actors: group-mean-centered advantage, **no std division**, constant-normalized
   loss). In `tunix.rl.agentic`, Dr.GRPO is `GRPOConfig(advantage_estimator="drgrpo",
   loss_agg_mode="sequence-mean-token-scale")`.

**The cross-experiment law (the single most useful finding).** RL earns its keep
*only* to amplify a **narrow** behavior the base model samples too rarely for SFT to
cover (e.g. copying a tool result forward). When the target is **fully demonstrable
by SFT** (writing a program), **SFT does the work and RL is marginal**. Both hit the
same wall — *RL only sharpens what the base policy already puts mass on* — so:

> **Before an RL run, prove the SFT policy solves the chosen tasks *sometimes*:
> `pass@k > pass@1 > 0`, measured WITH sampling.** Otherwise every generation
> scores the same, the group advantage is 0, and nothing updates ("the bimodal
> wall"). Use a pass@k eval (temperature > 0) to pick RL-trainable tasks. For
> continuous graders, the gate is **score *spread* across samples**, not binary
> solve — Dr.GRPO gets advantage from score variance even with 0 full solves.

See `tunix-delphi-rl/AGENTS.md` (§A–§D invariants, the cross-experiment table) and
`openthoughts-agent/REPORT.md` (the deep-ckpt pass@k gate).

---

## 5. Hard-won invariants — the gotchas that bite EVERY run

These are framework-level (tunix + TPU + iris), not recipe-specific. Each one cost a
failed run to find. Carry them or re-break them.

1. **`jax.distributed.initialize()` before any other jax call on multi-host.**
   v6e-8/-16 span 2/4 hosts; orbax barriers + collectives crash without it. Make it
   line 1 of `main()`. A single-host smoke "works" small and dies at scale — so
   validate the *multi-host* path explicitly.

2. **fp32 actor params for training.** A `1e-5` AdamW update is below bf16 ULP for
   unit-scale weights → bf16 *storage* silently zeroes the update. Store params
   fp32; do compute in bf16 (`config.dtype`); the KL reference (if `beta > 0`) can
   be bf16.

3. **Gradient clipping is load-bearing — its absence is a *crash*, not drift.**
   Unclipped GRPO hits `inf`/`NaN` grads that surface as a libtpu **SIGSEGV** (dies
   mid-run, loses everything). Always `optax.chain(clip_by_global_norm(1.0),
   adamw(...))` — for the SFT phase *and* every RL phase, from one shared factory.

4. **tunix `Sampler` decodes GREEDILY unless you pass `top_p`.** It silently ignores
   `temperature` **and** `seed` otherwise → every pass@k draw is byte-identical →
   fake `pass@1 == pass@k`. **Pass `top_p=1.0` (or <1) to every eval/rollout
   sampler.** This single bug invalidated a whole "no RL headroom" diagnosis once.
   (See memory `tunix-sampler-greedy-bug`.)

5. **GRPO group members must vary.** tunix 0.1.7 generates each group member with a
   fixed `RolloutConfig.seed`, so all `num_generations` samples come out identical →
   zero advantage → no gradient. Install a per-generation rollout seed (the Delphi
   project's `install_per_call_rollout_seed`). This is the same root cause as #4 on
   the rollout side.

6. **Generation conflicts with remat.** The Sampler mutates KV-cache Params, tripping
   remat's trace level. **Train with remat; load with `remat=NONE` to sample.** So an
   SFT job that also evals must gate generation off (`EVAL_GEN=0`), and eval/RL load
   the actor with `remat=NONE`.

7. **RL backprop OOMs at `remat=NONE`** (forced by the sampler). **TP shards the
   per-sequence activations** — this is *why* 1.7B needs TP=4 on v6e-4 and 8B needs
   TP=8 on v6e-16 (§3).

8. **`kv_cache_size ≥ max_prompt_length + max_tokens_to_generate`** or the sampler
   hard-errors. Drivers add `+8` headroom. The cache size is **fixed at Sampler
   construction** — over-budget requests must be clamped/rejected, not grown.

9. **Agent context grows every turn** (terminal output / tool results accumulate).
   Trim the prompt to `max_prompt_length` each turn or the KV cache overflows.

10. **Raw base LMs never emit EOS.** Single-line turns stop on a terminal newline;
    **multi-line turns (programs) can't** — they stop on a sentinel token (`END`).
    Use a custom stop-token set, not just `"\n"`. (Delphi-specific, but the trap
    generalizes to any base-without-chat-template model.)

11. **`PeftTrainer.maybe_restore` resumes params + step from `CKPT_DIR`.** A preempted
    long run continues on retry against the *same* dir; point a fresh run at a **new**
    dir to train from base. (How the deep multi-epoch SFT survived preemption.)

---

## 6. gVisor sandbox (agentic RL that execs untrusted shell)

When the agent runs model-generated shell (Terminal-Bench, code execution), isolate
it under **gVisor (`runsc`)**. CONFIRMED working inside an iris TPU task: a
`--runtime=runsc` container reports kernel `4.19.0-gvisor` vs the host's `6.8.0-gcp`.

- **Requires a `--tpu` slice.** TPU tasks run `--privileged` (iris adds it for
  accelerators); a **CPU-only iris task is NOT privileged**, so the sandbox can't
  come up there. (Smallest proof slice: a v6e-4 running `eval.gvisor_smoke`.)
- The custom task image bakes `runsc`+`docker` in; the stock iris image
  downloads the static binaries at runtime — handle both.
- Three flags were each required (each found by a failed smoke — keep them):
  - **dockerd:** `--storage-driver=vfs --iptables=false --bridge=none` (nested
    overlayfs + bridge/iptables fail in a container).
  - **runsc runtimeArgs:** `--ignore-cgroups` (restricted task cgroup),
    `--platform=ptrace` (no `/dev/kvm`), `--network=sandbox`.
  - **image builds:** `--network=host` (dockerd is bridgeless) so apt has egress.
  - sandbox images need **bash** (`bash -lc` exec) — use `debian:stable-slim`, not
    alpine (busybox-only).

Full write-up: `openthoughts-agent/AGENTS.md` → "gVisor sandbox", and memory
`gvisor-on-iris-tpu`.

---

## 7. Serving a trained model as a live iris endpoint

A trained tunix actor can be served as a long-lived HTTP endpoint reachable through
the iris proxy. The stack (untracked `serving/` in `tunix-delphi-rl`):
**export HF safetensors → serve job (tunix `Sampler` + FastAPI) → register endpoint
→ query via `ProxyResolver`**.

- Submit the serve job via the **Python `IrisClient.submit(ports=["http"], …)`** path
  — the `iris job run` CLI has no named-port flag.
- The endpoint name registers namespace-prefixed (`/<ns>/<name>`); callers resolve
  the slash-prefixed full name.
- Endpoint visibility is tied to job liveness — a crashed serve drops its endpoint;
  submit with generous retries + a warmup.
- GCS reads/writes from the worker use **gcsfs** (in the locked venv) — `gsutil`/
  `gcloud` are not on the worker image.
- **Serving re-hits the greedy-sampler bug** (#4): pass `top_p` for sampling; omit
  it deliberately for greedy.

Details + the live base-vs-RL demo: `serving/SERVING_PLAN.md`, memory
`serving-tunix-iris-endpoints`.

---

## 8. Local / CPU checks before you submit

Catch breakage before paying for a TPU. **Never** validate a learning claim here.
The model code runs on CPU (slow, but it compiles + executes), so a CPU run shakes
out import errors, API drift, and sharding/dtype mistakes for free.

```bash
# the hello-world entrypoint runs on CPU end-to-end (~30 s) — do this before every submit
JAX_PLATFORMS=cpu uv run python marin_smoke.py        # expect "[marin-smoke] SUCCESS"

# imports only (fastest API-drift check for a launcher you're editing)
JAX_PLATFORMS=cpu uv run python -c "import marin_smoke; print('ok')"

# tunix's own unit suite (from this repo); add --group dev for pytest/chex
uv run --group dev pytest tunix/  -q -x        # scope to the area you touched
```

A CPU run breaks in two distinguishable ways: a **stale/again-needed lock** (re-`uv
lock`) vs **API drift** in the code you call (fix the call site). The §1 smoke caught
both an eager-sharding error and a deprecated mesh API on CPU before any TPU spend —
that is exactly the point. (Memory `cpu-only-for-compile`, `dry-run-smoke-test`.)

---

## 9. Developing in this fork (getting fork changes onto a worker)

There are two ways to run a tunix change on iris. Pick by where you submit from.

**(a) Submit from this repo — the fast loop (what §1 does).** Because the bundle
ships the **working tree** and `uv sync` installs this repo **editable**
(`__editable__.google_tunix-…` appears on the worker's `sys.path`), your local
edits to `tunix/` go straight to the worker with **no commit, no version bump, no
re-lock** (unless you changed dependencies). Edit → `JAX_PLATFORMS=cpu uv run python
marin_smoke.py` → submit. This is the loop to use for iterating on tunix itself.

**(b) Pull the fork into an experiment repo — the pinned loop.** The marin
experiments depend on `google-tunix` from **PyPI** (0.1.7) and do *not* bundle this
dir, so a local path install won't reach their workers. To run a fork change there:

1. Commit + push in this repo (`rjpower/tunix`).
2. In the experiment's `pyproject.toml`, pin tunix to the fork by git URL:
   `google-tunix @ git+https://github.com/rjpower/tunix.git@<sha>` (or a branch).
3. `uv lock` in the experiment, then submit (the lock ships; the worker builds the
   fork from git).

Packaging note: `google-tunix` + `marin-iris`/`marin-fray` only co-resolve with the
`[tool.uv]` `prerelease = "allow"` + `override-dependencies` block (already in this
repo's `pyproject.toml`, mirrored from `tunix-delphi-rl`). `marin-levanter` has a
historical conflict with tunix; don't add it here. (Memory `tunix-on-iris`.)

If you only changed dependencies (not just code), **`uv lock` again** so the shipped
lock matches — otherwise the worker resolves the old graph.

---

## 10. Reference docs (where the deeper findings live)

| topic | source |
|---|---|
| iris CLI, job/cluster lifecycle, config | `~/code/marin/lib/iris/README.md`, `OPS.md` |
| Delphi recipe + invariants A–D + cross-experiment law | `~/code/marin-experiments/tunix-delphi-rl/AGENTS.md`, `REPORT.md` |
| 8B agent SFT/eval/RL + 8B memory ladder + gVisor | `~/code/marin-experiments/openthoughts-agent/AGENTS.md`, `REPORT.md` |
| serving a trained actor as an endpoint | `~/code/marin-experiments/tunix-delphi-rl/serving/SERVING_PLAN.md` |
| shared coding/style/testing conventions | `~/code/marin-experiments/AGENTS.md` |
| tunix algorithms (GRPO/Dr.GRPO/agentic) | https://tunix.readthedocs.io |

**Golden rules, one more time:** `--extra prod` + `--enable-extra-resources` + all
three `--region`s on every TPU submit; CPU-check (`JAX_PLATFORMS=cpu`) before you
spend a TPU; pass `top_p`; clip grads; fp32 actor params;
`jax.distributed.initialize()` first on multi-host; `remat=NONE` to sample; prove
`pass@k > pass@1 > 0` before RL; re-`uv lock` if stale.

---

## 11. Running tunix on GPU (CoreWeave H100 via iris `cw-us-east-02a`)

tunix runs off-GCP on NVIDIA H100s through iris's CoreWeave cluster. **Verified
end-to-end:** `marin_smoke.py` ran a real tunix Qwen3-0.6B forward on an H100
(`kind='NVIDIA H100 80GB HBM3'`, jax 0.10.0/cuda13, `logits (1, 8, 151936)`, SUCCESS).

### 11.1 One-time setup
- **`gpu` optional-dependency group** (in `pyproject.toml`): `jax[cuda13]==0.10.0`,
  `nvidia-cublas`, `nvidia-nccl-cu13`. It is **mutually exclusive with `prod`**
  (`jax[tpu]`) via a `[tool.uv] conflicts` block — `uv` forks the lock. Submit TPU jobs
  with `--extra prod`, GPU jobs with `--extra gpu`. Re-`uv lock` after editing it.
- **Local CLI must reach the k8s controller:** the iris dep-group pulls
  `marin-iris[controller]` (the `kubernetes` client). Without it, `cluster status` raises
  `Install iris[controller] to use CloudK8sService`. Also needs `~/.kube/coreweave-iris-gpu`
  + `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY` in env (already present on this box).
- Prove reachability first: `uv run iris --cluster=cw-us-east-02a cluster status`
  (controller `Healthy: True`). NB: this cluster is **Pod-per-task k8s** (no Iris worker
  daemons/autoscaler), so `cluster status` legitimately shows `Workers: 0/0` while jobs
  still dispatch as pods onto the static 32×H100 NodePool.

### 11.2 Submit pattern
```bash
uv run iris --cluster=cw-us-east-02a job run --no-wait \
  --cpu 8 --memory 64GB --gpu H100x1 --enable-extra-resources --extra gpu \
  --max-retries 1 --job-name my-gpu-job -- python my_entry.py
```
- `--gpu H100xN` requests N GPUs (e.g. `H100x8` for a full node); `--enable-extra-resources`
  is required (same gate as TPU). **No `--region` flags** (single-region CoreWeave).
- Full node: `--gpu H100x8 --cpu 32 --memory 512GB`. Multi-node uses the SDK
  (`replicas=N`, `ports=["jax"]`, `coscheduling(group_by="leafgroup")`).
- Data: CoreWeave **cannot read `gs://` for JAX IO** — it's wired to **R2 `s3://marin-na/…`**
  (pods auto-get `AWS_*`/`FSSPEC_S3`). Stream HF datasets/weights on the worker; mirror
  checkpoints to R2. NCCL uses `NCCL_SOCKET_IFNAME=enp157s0np0` (set in cluster config).

### 11.3 Weight transfer / reshard on GPU
The de-Pathways `tunix.rl.weight_transfer` backend registry selects the reshard fn by
capability (AUTO → pathways-if-proxy else `jax_device`). On GPU there is no proxy, so the
**`jax_device`** backend (plain `jax.device_put` cross-mesh reshard, XLA collectives = NCCL)
is used. Colocated RL (ACTOR==ROLLOUT, or same-host disjoint meshes) uses all GPUs by
construction (XLA collectives) — no rank-0 fold. **Verified on 8×H100
(single node):** `reshard_pytree` moved a pytree across disjoint chip meshes
[0:4]→[4:8] with values + physical placement correct (`OVERALL=PASS`), same as the
TPU spike. Cross-node GPU reshard rides NCCL (untested here; the multi-node colocated
path uses one `jax.distributed` world over `replicas=N`).

### 11.4 Multi-host init guard (TPU and GPU)
iris keys TPU multi-host off `PJRT_DEVICE=TPU`, **not** `JAX_NUM_PROCESSES` (which iris
leaves empty) — so `marin_smoke.py`'s `JAX_NUM_PROCESSES>1` guard is a no-op on iris
multi-host. Guard `jax.distributed.initialize()` on `PJRT_DEVICE`/`JAX_PLATFORMS startswith
tpu` (TPU) or the GPU multi-host rendezvous env. A single v6e-8 marin slice can come back as
1 host × 8 chips; use v6e-16 for a true 4-host test.

### 11.5 Off-Pathways cross-host reshard caveat (TPU)
Off-Pathways, **same-host** disjoint-mesh reshard is production-safe, but **cross-host**
disjoint-mesh `device_put` returns correct values yet SIGSEGVs on teardown in XLA's
cross-host receive notifier (v6e/jax 0.10.2). Keep trainer/rollout meshes on the same
host(s) for colocated TPU RL; a different-host topology needs a real transport (Arrow/NCCL/
remote control plane). On GPU this rides NCCL (not XLA TPU cross-host) — verify separately.

## 12. Weight-transfer performance (measured, off-Pathways)
Bench: `mega_eval/bench_weight_transfer.py` (env-driven: `BENCH_MODE`, `MODEL_PRESET`,
`N_SYNC`, `SERVE_SECONDS`, `NIC_GBPS`). Model = Qwen3-8B-shaped, **15.3 GiB bf16** on the
wire (30.5 GiB fp32 in HBM). Two transports under one registry:
`nccl` (in-JAX-world cross-mesh reshard, XLA→NCCL) and `arrow_flight` (host-staged gRPC, the
cross-host path that sidesteps the §11.5 SIGSEGV).

**GPU — 8×H100, single node (`cw-us-east-02a`):**
- Cross-mesh reshard ceiling (`mega_eval/reshard_probe.py`): **in-mesh 78 GB/s**,
  **disjoint-mesh 64 GB/s**, sharded→sharded. XLA lowers `device_put` to NCCL over **all**
  GPUs (no rank-0 fold). This is the GPU weight-sync answer — no torch / raw-NCCL needed for
  single-node colocated sync.
- 1 trainer (4 GPU) + 4 rollouts (1 GPU each, **replicated**): aggregate **11.5 GB/s**,
  per-client 2.87 GB/s. The cap is **gather-to-a-single-device** (~2.9 GB/s), *not* the
  interconnect — **shard inference targets across ≥2 GPUs (tp)** to ride the 64–78 GB/s path.

**TPU — v6e-16, 4 hosts (marin), `arrow_flight` cross-NIC:**
- 1 trainer host → 3 inference hosts, each pulling the full 15.3 GiB bf16 model, 15 rounds:
  per-worker median **~4.5 GB/s** (best ~5.4), **aggregate ~13.5 GB/s** (≈108 Gbps — near the
  trainer's NIC line rate). Device→host materialize 5.4 GB/s; restore/reshard 0.6–1.3 s.
- Aggregate is capped by the **trainer's single NIC** (3 workers already saturate it); to
  scale past one NIC, serve a shard from each trainer host.

**Three transport bugs the bench surfaced (all fixed):**
1. `serve_weights` ran a global `sync_global_devices` barrier (marin's multi-host-trainer
   sync) that deadlocks a disaggregated trainer↔inference layout → gated behind
   `WeightTransferConfig.serve_barrier` (default **off**; Arrow is a network transport, the
   only collective belongs to the caller).
2. `flatten_for_transfer` did an **on-device `reshape(-1)` of a sharded tensor** → compiles an
   all-gather; on multi-host TPU a jit over one host's local sub-mesh makes XLA reference
   another host's `device_id` and aborts (`RET_CHECK device_id < kMaxDeviceCount`). Fixed:
   cast bf16 elementwise (collective-free) → `device_get` local shards → reshape on host.
   `restore_from_flat` likewise `device_put`s the host array straight to the target sharding
   (no single-device staging, no on-device collective).
3. The `jax.distributed` KV coordinator used insert-only `key_value_set` → 2nd serve hit
   `ALREADY_EXISTS` on `/latest`. Fixed: `allow_overwrite=True` + non-blocking
   `key_value_try_get` (no per-poll 1 s timeout).

---

## 13. Disaggregated RL — the two-job run pattern (TPU)

**For any real TPU RL run, do NOT use the in-process `RLCluster`** (`launch_rl.py`): on
TPU it is forced to `remat=NONE` (the sampler mutates KV-cache Params, which conflicts
with remat — Invariant 6/7) and at scale it hits the cross-host reshard SIGSEGV (§11.5)
and the KV-cache OOM. Instead run **trainer and rollout as two SEPARATE iris jobs**, each
its own JAX world / TPU slice, exchanging weights over **Arrow Flight** (the §11.5-safe
cross-host transport) and trajectories over a **GCS-staged channel**. One entrypoint
(`mega_eval/rl_disagg_loop.py`), `ROLE` selects the half. Code-level detail (the standalone
Dr.GRPO step, the agentic collector, the env-knob table) is in **AGENTS.md §3–§4**; this is
the *run* recipe.

**Why two jobs win** (each kills a whole failure class): no shared/cross-host mesh ⇒ no
reshard SIGSEGV; rollout chips hold only inference weights + replicated KV (no fp32
master/AdamW) ⇒ no KV-OOM, no offload; and the **trainer can now use `remat=BLOCK`** because
it *never samples* — the single biggest fit win (without it the agentic seq=5120 backward
OOMs 96.7G > 31.25G even at TP=4).

**Cross-job rendezvous = the iris endpoint registry**, not an object store. Both jobs build
the SAME absolute (`/`-prefixed) name `/tunix-rl/<RUN_ID>/weights`; the leading `/` bypasses
iris's per-job namespace so the rollout job resolves the endpoint the trainer registered.
iris tasks run **net=host**, so the trainer's auto-bound Flight port is reachable at
`IRIS_ADVERTISE_HOST:port` with **no iris port allocation**. (Local/GPU fallback: `COORD=s3`
+ an object-store coordinator; never S3 for TPU coordination.) Needs `marin-iris` importable
in-job → it's in the `mega` extra.

**Submit both jobs** (same `RUN_ID`, same `TRAJ_BASE`; agentic mode shown):

```bash
RUN_ID=ota-rl-$(date +%s); TRAJ=gs://marin-us-central2/users/power/tunix-rl-traj/$RUN_ID
COMMON="--cluster=marin job run --no-wait --tpu v6e-4 --enable-extra-resources \
  --extra prod --extra mega --region europe-west4 --region us-east1 --region us-east5 \
  --cpu 8 --memory 64GB --disk 90GB --max-retries 1 \
  -e RUN_ID $RUN_ID -e TRAJ_BASE $TRAJ -e PRESET qwen3-1.7b \
  -e RL_STEPS 4 -e NUM_GENERATIONS 2 -e PROMPTS_PER_BATCH 1 -e REWARD_MODE agentic \
  -e TASK_LIMIT 1 -e MAX_STEPS 6 -e MAX_RESPONSE_LEN 1024 -e MAX_PROMPT_LEN 4096"

uv run iris $COMMON -e ROLE trainer --job-name ota-rl-trainer -- python -m mega_eval.rl_disagg_loop
uv run iris $COMMON -e ROLE rollout --job-name ota-rl-rollout -- python -m mega_eval.rl_disagg_loop
```

- **`RUN_ID` and `TRAJ_BASE` MUST match** across the two jobs (the registry name and the GCS
  channel are derived from them). `RUN_ID` is required for `COORD=iris`.
- **Order doesn't matter** — the rollout polls `receive_weights` until the trainer serves
  `weight_id=0`, and the trainer waits on `wait_for_batch`. Submit both, then watch:
  `iris job summary /power/ota-rl-trainer`, `iris job logs … | grep '\[m1c\]'`.
- **Disk ≤ 90GB** per VM (§2). **`REMAT=block`** is the default and safe here (trainer never
  samples). `--disk` only needs to hold the per-task gVisor images (§2 vfs note).
- **`reward=0.0` on a base model is expected** — the loop is *mechanically* correct but a 1.7B
  base model can't solve Terminal-Bench. Point `PRESET`/`CKPT_DIR` at the **SFT checkpoint** for
  real signal, and gate tasks by `score_spread > 0` first (§4 / AGENTS.md §6).

**Validated ladder (all GREEN on marin, two separate v6e-4 jobs, exit 0):** M0 byte-exact
cross-job Arrow (`rl_rendezvous_smoke.py`) → M1a real-weights pull+generate (`rl_disagg_smoke.py`)
→ M1c full placeholder loop (`rl_disagg_loop.py`, wid 0→3) → **M1d real agentic Terminal-Bench
reward** (`REWARD_MODE=agentic`, commit `4304d76`; survived a TPU preemption). The infra
build-out (M0–M1d) is complete; the open work is *quality* (run from the SFT checkpoint).
