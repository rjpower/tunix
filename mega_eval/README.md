# mega_eval — terminal-agent SFT + gVisor eval, staged in this tunix fork

Model-training track for the OpenThoughts-Agent leaderboard effort, ported into
this `google-tunix` fork from the proven precedent at
`~/code/marin-experiments/openthoughts-agent/` (Qwen3-8B, ChatML SFT with
assistant-turn loss masking on Terminus-2 traces, gVisor-sandboxed Terminal-Bench
eval, Dr.GRPO RL). Launchers sit at the package root, mirroring `marin_smoke.py`'s
"entrypoint ships in the working-tree bundle" pattern (MARIN.md §9a). Submit jobs
from this repo so the editable `tunix` fork installs on the worker.

This is **prep-only staging**: imports verified on CPU, no training/eval run, no
cluster submit.

## Layout (one entrypoint per stage; everything else is a library)

```
mega_eval/
  launch_sft.py      # stage 1: SFT Qwen3-8B on agent traces -> orbax ckpt (gs://)
  launch_eval.py     # stage 2: per TB task -> build -> gVisor -> agent loop -> grade (pass@k)
  launch_rl.py       # stage 3: Dr.GRPO via tunix.rl.agentic (gVisor = reward env)
  training/          # common.py (mesh/optimizer/input fn), agent_sft.py (ChatML encoder + grain + PeftTrainer)
  models/            # qwen3_loader.py, registry.py, checkpoint.py (HF Qwen3 -> tunix nnx; orbax restore)
  agent_data/        # agent_traces.py (parquet/pyarrow streamer; datasets<4 List-feature workaround)
  eval/              # sandbox.py (gVisor/runsc), agent_loop.py, grade.py, model_serving.py, tb_tasks.py, *_smoke.py
  rl/                # agent.py (TerminusAgent), environment.py (TerminalBenchEnv)
  docker/            # Dockerfile.agent-task (iris task base + runsc + docker + buildx)
  tests/             # CPU tests: encoder/masking, agent loop, RL env contract
  PYPROJECT_NOTES.md # dependency-integration notes (no uv lock run)
  DATA_PLAN.md       # Terminus+SWE SFT mixture proposal + new SWE/terminal datasets
```

## What changed vs. the precedent

Only the **first-party import roots** were re-rooted under the package
(`from models.X` → `from mega_eval.models.X`, likewise `training`, `eval`,
`agent_data`, `rl`) so `import mega_eval.launch_sft` works. **Every `tunix.*` /
third-party import is unchanged.** The gVisor sandbox flags in `eval/sandbox.py`
and `docker/Dockerfile.agent-task` are copied **verbatim** (the hardest-won asset):
dockerd `--storage-driver=vfs --iptables=false --bridge=none`; runsc
`--platform=ptrace --network=sandbox --ignore-cgroups`; builds `--network=host`.

## tunix API drift: NONE found

Every `tunix.*` symbol the ported code calls was grepped in this fork and matches
the precedent's `google-tunix 0.1.7` usage exactly. The CPU import smoke for all
three entrypoints + every submodule passes against the editable fork, and the
16 ported CPU tests pass:

| tunix symbol used by mega_eval | status in this fork |
|---|---|
| `tunix.models.qwen3.model`: `ModelConfig` (+ `qwen3_8b`/`qwen3_1p7b` presets), `RematConfig.{NONE,BLOCK,DECODER}`, `Qwen3` | present; `ModelConfig` fields the loader sets all exist (fork even adds optional `rope_scaling`, which the loader guards against and Qwen3-8B doesn't use) |
| `tunix.models.qwen3.params`: `create_model_from_safe_tensors`, `_get_key_and_transform_mapping` | present; signature matches (`file_dir, config, mesh, dtype, mode`) |
| `tunix.utils.torch_utils.torch_key_to_jax_key` | present |
| `tunix.generate.sampler`: `Sampler`, `CacheConfig` | present; `__call__` accepts every kwarg passed (`input_strings, max_generation_steps, max_prompt_length, echo, eos_tokens, temperature, top_p, seed`); `top_p`-greedy behavior unchanged |
| `tunix.sft.peft_trainer`: `PeftTrainer`, `TrainingConfig` (`checkpoint_root_directory`, `checkpointing_options`, `metrics_logging_options`, `eval_every_n_steps`, `max_steps`), `.with_gen_model_input_fn`, `.train` | present |
| `tunix.sft.utils`: `build_positions_from_mask`, `make_causal_attn_mask` | present |
| `tunix.sft.checkpoint_manager.CheckpointManager.maybe_restore` | present |
| `tunix.sft.metrics_logger.MetricsLoggerOptions` | present (fields `log_dir, project_name, run_name, flush_every_n_steps, backend_kwargs`) |
| `tunix.rl.rl_cluster`: `ClusterConfig`, `Role`, `RLTrainingConfig`, `RLCluster` | present |
| `tunix.rl.rollout.base_rollout.RolloutConfig` | present |
| `tunix.rl.agentic.*`: `agentic_grpo_learner.{GRPOConfig,GRPOLearner}`, `parser…QwenChatTemplateParser`, `agents.{agent_types,base_agent}`, `environments.base_environment` | present |

(`orbax.checkpoint.ContinuousCheckpointingPolicy` + `CheckpointManagerOptions`
also verified present.)

## CPU checks (what was run here — no TPU, no training)

```bash
JAX_PLATFORMS=cpu .venv/bin/python -c "import mega_eval.launch_sft"   # OK
JAX_PLATFORMS=cpu .venv/bin/python -c "import mega_eval.launch_eval"  # OK
JAX_PLATFORMS=cpu .venv/bin/python -c "import mega_eval.launch_rl"    # OK
OTA_SANDBOX=local JAX_PLATFORMS=cpu .venv/bin/python -m pytest mega_eval/tests -q  # 16 passed
```

## Dataset accessibility (read-only checks, 2026-06-23, HF_TOKEN set)

All resolve via `huggingface_hub` except gated Salesforce/xlam (license not
accepted on this token — but its content is reachable un-gated via smoltalk2's
`xlam_traces_no_think` split).

| dataset | resolves | rows | size | format / split notes |
|---|---|---|---|---|
| `open-thoughts/OpenThoughts-Agent-v1-SFT` (pinned rev `c5dc8969…`) | yes (ungated) | **15,209** | 110 MB | `conversations` list (role/content), `agent=terminus-2` + 8 metadata cols; the SFT core |
| `open-thoughts/OpenThoughts-TB-dev` (pinned rev `0d54f719…`) | yes (ungated) | **70 tasks** (792 files) | 30 MB | per-task dirs: `task.toml`, `instruction.md`, `environment/`, `tests/`, `solution/`; the eval set |
| `HuggingFaceTB/smoltalk2` (SFT config) | yes (ungated) | tool splits: `smolagents_toolcalling_traces_think` **9,079**; `hermes_function_calling_v1_no_think` **8,961**; `xlam_traces_no_think` **59,962** | 88 GB total | `messages` + `chat_template_kwargs` + `source`; 25 SFT splits |
| `nvidia/Nemotron-Post-Training-Dataset-v1` | yes (ungated) | `tool_calling` split **310,051** (of ~5.7M total) | 203 GB total | `messages` + `metadata.tools` (function schema) + `reasoning` flag |
| `Salesforce/xlam-function-calling-60k` | **gated (auto), 403** on this token | ~60,000 (single `xlam_function_calling_60k.json`) | 100 MB | `query`/`tools`/`answers`; **un-gated equivalent = smoltalk2 `xlam_traces_no_think`** |
| `TIGER-Lab/AceCode-89K` (pinned rev `13216309…`) | yes (ungated) | **87,149** | 1.0 GB | `question`/`test_cases`/`inferences[].completion`/`context_messages`; code SFT, filter by `pass_rate` |

## New SWE/terminal datasets found (full detail + mixture in `DATA_PLAN.md`)

The precedent used only `OpenThoughts-Agent-v1-SFT`; marin has tool/code slices
but **no SWE/terminal trajectories**. Four high-value, ungated adds:

1. `DCAgent/neulab-nebius-swe-agent-trajectories-sandboxes-traces-terminus-2`
   (12,015) — **identical Terminus-2 `conversations` schema → zero-code drop-in.**
2. `mlfoundations-dev/stackexchange-unix-sandboxes-traces-terminus-2` (9,987) +
   siblings (`superuser` 9,983, `staqc`, `swesmith_with_plain_docker`, `codereview`,
   `tor`) — a whole Terminus-2 corpus from the OpenThoughts pipeline.
3. `SWE-bench/SWE-smith-trajectories` (76,002) — large `messages` SWE-agent SFT
   with `resolved`/`patch` (success-filterable).
4. `nebius/SWE-agent-trajectories` (80,036) — large `messages` SWE-agent SFT.

## Submitting (when you move past prep — same shape as the precedent's AGENTS.md)

`--extra prod` (this repo's TPU extra) + `--enable-extra-resources` + all three
`--region`s; CPU-check first; `-e CKPT_DIR gs://…`. The `mega_eval.*` module path
means entrypoints are `python -m mega_eval.launch_sft` (or adjust the submit `--`
command accordingly). See MARIN.md §1–2 for the verified submit pattern.
```
```
