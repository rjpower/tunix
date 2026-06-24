# PYPROJECT_NOTES — dependency integration for the `mega_eval/` SFT+eval track

Status: **the repo's existing `pyproject.toml` already satisfies the hard
requirement.** This file proposes *additions only*; do **not** run `uv lock` /
`uv sync` as part of this prep (per the prep-only scope). Re-lock before the first
job submit if you adopt any change here (MARIN.md §9a).

## TL;DR

The single load-bearing pin the SFT track needs — `datasets<4` — is **already
present** in this repo's `[tool.uv].override-dependencies`. Everything else the
ported code imports (`grain`, `huggingface_hub`, `pyarrow`, `orbax-checkpoint`,
`transformers`, `optax`, `jax`/`flax`) is already a direct or transitive
`google-tunix` dependency and is installed in the current `.venv`. The CPU import
smoke for all `mega_eval` entrypoints passes against the venv **as-is** — so no
change is strictly required to import/run the SFT+eval code. The only genuinely
optional add is **`wandb`** (loss/reward curves), and it is import-gated.

## What the ported code imports, and where it already comes from

| import in `mega_eval/` | provider in this repo | already present? |
|---|---|---|
| `datasets` (pinned `<4`) | tunix core dep + `[tool.uv]` override `datasets>=3.1.0,<4.0.0` | **yes** (`3.6.0` in venv) |
| `grain.python` (SFT data pipeline) | tunix core dep `grain` | **yes** (`0.2.18`) |
| `huggingface_hub.snapshot_download` | tunix core dep `huggingface_hub` | **yes** (`1.20.1`) |
| `pyarrow.parquet` (the `List`-feature workaround in `agent_data/agent_traces.py`) | transitive (datasets → pyarrow) | **yes** (`24.0.0`) |
| `orbax.checkpoint` (PeftTrainer ckpt) | tunix core dep `orbax-checkpoint>=0.12.0` | **yes** (`0.12.0`) |
| `transformers.AutoTokenizer` | tunix core dep `transformers!=4.57.2` | **yes** (`5.12.1`) |
| `optax`, `jax`, `jax.numpy`, `flax.nnx`, `numpy` | tunix core | **yes** |
| `tomllib` (TB task.toml parse) | Python 3.11+ stdlib (the repo requires `>=3.11`) | **yes** (no `tomli` needed) |

## Why `datasets<4` matters (keep the existing pin)

The OpenThoughts agent traces (and the new `*-terminus-2` corpora — see
`DATA_PLAN.md`) record the `conversations` column as a `List` feature type written
by a newer `datasets`. A `datasets>=4` builder parses that fine, but tunix's other
pins don't co-resolve with `datasets>=4` in the marin-iris graph, hence the
existing `<4` override. **The ported `agent_data/agent_traces.py` sidesteps the
issue entirely** by reading the parquet shards directly with `pyarrow` (which
ignores the HF `List` metadata), so it works under either `datasets` major — but
the pin must stay for the rest of the resolution to converge. **No change needed:
the override is already there.** Verified: `datasets 3.6.0` resolved in the venv.

## Proposed ADDITIONS (optional; pick what you want)

These are the only things not already guaranteed by `google-tunix`. None is
required to import or run a plain (no-wandb) SFT/eval job.

### 1. `wandb` — optional, for the deep-SFT / RL loss curves (RECOMMENDED)

`tunix.sft.metrics_logger` constructs a `WandbBackend` only when `wandb` is
importable, and logs `WandbBackend skipped: 'wandb' library not installed` and
continues otherwise. The deep multi-epoch SFT and RL stages have **no other loss
signal** (`PeftTrainer` prints nothing to stdout), so wandb is the de-facto way to
watch a long run. It is gated on the `WANDB_PROJECT` env var in
`training/common.py:metrics_logging_options` (returns `None` when unset → stdout
only), so adding the package never *forces* a wandb dependency at runtime.

Add to a `[project.optional-dependencies]` group so it ships only when asked
(mirrors the precedent's own `wandb` dep). Suggested — fold it into `prod` (the
extra the iris submit already passes via `--extra prod`) or a new `mega` extra:

```toml
[project.optional-dependencies]
# existing prod extra pulls jax[tpu]; add wandb so --extra prod also gets metrics.
prod = [
  "jax[tpu]>=0.6.0,!=0.7.2",
  "wandb",                      # ADD: gated metrics (no-op unless WANDB_PROJECT set)
]
```

If you prefer to keep `prod` strictly TPU-runtime, add a dedicated extra instead
and pass `--extra prod --extra mega` on submit:

```toml
[project.optional-dependencies]
mega = ["wandb"]               # mega_eval metrics; pair with --extra mega on iris
```

### 2. (NOT needed) `pyarrow`, `grain`, `tomli`, `datasets` as explicit deps

Do **not** add these — they are already pulled by `google-tunix`. Adding an
explicit `pyarrow`/`grain` line risks a redundant/looser pin fighting the existing
resolution. `tomli` is unnecessary on Python ≥3.11 (`tomllib` is stdlib); the
ported `eval/tb_tasks.py` already falls back to `tomli` only on <3.11, which this
repo excludes.

## What does NOT change

- `[tool.uv]` `prerelease = "allow"` + the `override-dependencies` block — keep
  verbatim. They make `google-tunix` + `marin-iris`/`marin-fray` co-resolve, and
  they already include the `datasets<4` pin the SFT track depends on. **Do not add
  `marin-levanter`** (historical tunix conflict — MARIN.md §9).
- The `iris` / `dev` dependency-groups — unchanged; they drive `uv run iris …`.
- The `prod` extra remains the TPU-runtime extra the submit passes as
  `--extra prod` (the precedent repo calls the same thing `tpu`; match this repo's
  name on submit).

## Re-lock note

If you adopt the `wandb` add: edit `pyproject.toml`, then `uv lock` (the shipped
lock is what the worker `uv sync`s — MARIN.md §2 "What ships in the bundle"). If
you adopt nothing, the current `uv.lock` already covers the import/run path.
```
```
