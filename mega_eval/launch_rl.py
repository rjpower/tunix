# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris entrypoint: Dr.GRPO RL for the Qwen3-8B terminal agent (tunix agentic).

Stage 3. Loads the SFT'd actor, then runs tunix's *agentic* GRPO learner: per
rollout it drives a :class:`rl.agent.TerminusAgent` against a
:class:`rl.environment.TerminalBenchEnv` (gVisor sandbox + grader), collects G
generations per task, and updates the policy with Dr.GRPO advantages
(``advantage_estimator="drgrpo"``: group-mean-centered, no std normalization).

Reward is the sparse grader score (``reward_fns=None`` -> the agentic reward
manager uses the env trajectory reward). For Dr.GRPO to learn, a task's G
generations must show reward SPREAD (pass@k > pass@1 > 0); pick RL tasks the SFT
policy already solves *sometimes* (see REPORT.md / the bimodal-wall note).

Rollouts are synchronous and run the sandbox containers in-process, so validate
on a single-host slice first (v6e-4, a small model). Multi-host agentic rollout
(v6e-8/-16 for 8B) is the next rung.

Config via env:
  * ``AGENT_MODEL`` (qwen3-8b), ``CKPT_DIR`` (SFT checkpoint to start from;
    unset = base model, for a machinery smoke only), ``RL_CKPT_DIR`` (where to
    save RL checkpoints).
  * ``RL_STEPS`` (200), ``NUM_GENERATIONS`` (G, 8), ``PROMPTS_PER_BATCH`` (4),
    ``TASK_LIMIT`` (tasks to train on), ``MAX_TURNS`` (15), ``MAX_RESPONSE_TOKENS``
    (episode response budget; default MAX_TURNS*512), ``MAX_PROMPT_LEN`` (8192),
    ``TEMPERATURE`` (1.0), ``LR`` (1e-6),
    ``BETA`` (0.0 KL; 0 => no reference model), ``TP`` (1),
    ``MAX_CONCURRENCY`` (8), ``COMMAND_TIMEOUT`` (60), ``EPISODE_TIMEOUT`` (1200).

Smoke (single host, machinery only):

    uv run iris --cluster=marin job run --no-wait \
      --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 8 --memory 120GB --disk 200GB --max-retries 1 --job-name ota-rl-smoke \
      -e HF_TOKEN "$HF_TOKEN" \
      -e AGENT_MODEL qwen3-1.7b-base -e TASK_LIMIT 2 -e NUM_GENERATIONS 4 \
      -e PROMPTS_PER_BATCH 1 -e RL_STEPS 3 -e MAX_TURNS 6 -e MAX_CONCURRENCY 4 \
      -- python launch_rl.py
"""

import itertools
import os

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download

from tunix.models.qwen3 import model as qm
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from mega_eval.rl.chat_parser import TerminusQwenParser
from tunix.rl.rollout import base_rollout

from mega_eval.eval.sandbox import build_image, prune_ota_images
from mega_eval.eval.tb_tasks import load_tb_tasks
from mega_eval.models.checkpoint import restore_sft_model
from mega_eval.models.registry import get_model_spec
from mega_eval.rl.agent import TerminusAgent
from mega_eval.rl.environment import TerminalBenchEnv, register_tasks
from mega_eval.training.common import build_mesh, build_mesh_over, clipped_adamw, init_distributed, metrics_logging_options


def _ensure_model(repo: str, model_dir: str) -> str:
  if not os.path.exists(os.path.join(model_dir, "config.json")):
    snapshot_download(repo_id=repo, local_dir=model_dir)
  return model_dir


def _solve_rate_metric(prompts, completions, rewards, advantages, **kwargs):
  """Per-step RL diagnostics: solve rate + reward spread (what Dr.GRPO needs)."""
  del prompts, completions, advantages, kwargs
  r = np.asarray(rewards, dtype=np.float32)
  return {
      "rl/solve_rate": (float((r >= 1.0).mean()), np.mean),
      "rl/mean_reward": (float(r.mean()), np.mean),
      "rl/reward_nonzero_frac": (float((r > 0).mean()), np.mean),
  }


def build_prompt_dataset(tasks, prompts_per_batch: int, num_steps: int):
  """Yields ``num_steps`` batches of ``prompts_per_batch`` tasks (cycling)."""
  pool = itertools.cycle(tasks)
  for _ in range(num_steps):
    batch = list(itertools.islice(pool, prompts_per_batch))
    yield {
        "prompts": [t.instruction for t in batch],
        "task_id": [t.task_id for t in batch],
    }


def main() -> None:
  init_distributed()  # must precede any jax call (multi-host)
  model_name = os.environ.get("AGENT_MODEL", "qwen3-8b")
  ckpt_dir = os.environ.get("CKPT_DIR")  # SFT checkpoint to start from (optional)
  rl_ckpt_dir = os.environ.get("RL_CKPT_DIR")
  steps = int(os.environ.get("RL_STEPS", "200"))
  num_generations = int(os.environ.get("NUM_GENERATIONS", "8"))
  prompts_per_batch = int(os.environ.get("PROMPTS_PER_BATCH", "4"))
  task_limit = os.environ.get("TASK_LIMIT")
  task_limit = int(task_limit) if task_limit else None
  # TASK_IDS (comma-separated) trains on a SPECIFIC set -- e.g. the score_spread
  # tasks surfaced by eval -- instead of TASK_LIMIT's "first N" selection.
  task_ids = os.environ.get("TASK_IDS")
  max_turns = int(os.environ.get("MAX_TURNS", "15"))
  # Total response-token budget per EPISODE (shared across turns; the collect
  # engine decrements it each turn). tunix requires the rollout's
  # max_tokens_to_generate to EQUAL the learner's max_response_length.
  max_response_tokens = int(os.environ.get("MAX_RESPONSE_TOKENS", str(max_turns * 512)))
  max_prompt_len = int(os.environ.get("MAX_PROMPT_LEN", "8192"))
  temperature = float(os.environ.get("TEMPERATURE", "1.0"))
  learning_rate = float(os.environ.get("LR", "1e-6"))
  beta = float(os.environ.get("BETA", "0.0"))
  tp = int(os.environ.get("TP", "1"))
  # The vanilla rollout allocates the KV cache REPLICATED on every chip (it is
  # created eagerly via jnp.zeros in the sampler and fed to jax.jit with no
  # in_shardings, so neither tp nor fsdp shards it). With the colocated actor's
  # fp32 params+AdamW resident, an 8B leaves only ~8 GB free -> the prefill OOMs
  # for any real batch x kv. OFFLOAD_TO_CPU=1 parks the actor optimizer/master on
  # host RAM during the rollout, freeing ~all HBM for the cache. Costs a host
  # round-trip of the actor state per step; required to scale batch>~2 at full kv.
  offload_to_cpu = os.environ.get("OFFLOAD_TO_CPU", "0") == "1"
  max_concurrency = int(os.environ.get("MAX_CONCURRENCY", "8"))
  command_timeout = float(os.environ.get("COMMAND_TIMEOUT", "60"))
  episode_timeout = float(os.environ.get("EPISODE_TIMEOUT", "1200"))

  # On CW (no gs://) the SFT checkpoint lives on R2 (s3://); orbax can't read
  # s3:// directly, so stage it to local NVMe first (no-op for local/gs:// paths).
  if ckpt_dir and ckpt_dir.startswith("s3://"):
    from mega_eval.models.checkpoint_staging import stage_checkpoint_if_remote
    print(f"[ota-rl] staging checkpoint {ckpt_dir} -> local NVMe ...", flush=True)
    ckpt_dir = stage_checkpoint_if_remote(ckpt_dir, local_root="./_staged_ckpt")

  spec = get_model_spec(model_name)
  base_dir = _ensure_model(spec.repo, os.environ.get("AGENT_MODEL_DIR") or f"./{spec.name}")
  print(f"[ota-rl] jax {jax.__version__} devices={jax.device_count()} model={spec.name} "
        f"ckpt={ckpt_dir} G={num_generations} ppb={prompts_per_batch} steps={steps} "
        f"tp={tp} offload_cpu={offload_to_cpu} kv={max_prompt_len + max_response_tokens + 16}", flush=True)

  # Topology: colocated (default; ACTOR+ROLLOUT share one mesh, used on TPU) OR
  # disaggregated (DISAGGREGATE=1; ROLLOUT runs on its OWN GPUs so it doesn't
  # share HBM with the actor optimizer/master -- the GPU path). The actor/reference
  # load on the ACTOR mesh; tunix reshards actor params -> rollout mesh each step.
  disaggregate = os.environ.get("DISAGGREGATE", "0") == "1"
  if disaggregate:
    devs = list(jax.devices())
    rollout_gpus = int(os.environ.get("ROLLOUT_GPUS", "2"))
    rollout_tp = int(os.environ.get("ROLLOUT_TP", str(rollout_gpus)))
    if not 0 < rollout_gpus < len(devs):
      raise ValueError(f"ROLLOUT_GPUS={rollout_gpus} must be in (0, {len(devs)}).")
    actor_devs, rollout_devs = devs[:len(devs) - rollout_gpus], devs[len(devs) - rollout_gpus:]
    # The Qwen3 dims (hidden 4096=2^12, vocab 151936=2^7*1187) require power-of-2
    # mesh axes, so each side's GPU count must be a power of 2 (else a deep jax
    # IndivisibleError mid-load). On 8 GPUs the clean split is ROLLOUT_GPUS=4.
    _pow2 = lambda n: n > 0 and (n & (n - 1)) == 0
    if not (_pow2(len(actor_devs)) and _pow2(rollout_gpus)):
      raise ValueError(
          f"DISAGGREGATE needs power-of-2 actor ({len(actor_devs)}) and rollout "
          f"({rollout_gpus}) GPU counts; on {len(devs)} GPUs use ROLLOUT_GPUS=4 "
          f"(actor 4 + rollout 4).")
    mesh = build_mesh_over(actor_devs, tp=tp)
    rollout_mesh = build_mesh_over(rollout_devs, tp=rollout_tp)
    print(f"[ota-rl] DISAGGREGATED actor={[d.id for d in actor_devs]}(tp={tp}) "
          f"rollout={[d.id for d in rollout_devs]}(tp={rollout_tp})", flush=True)
  else:
    mesh = build_mesh(tp=tp)
    rollout_mesh = mesh
  tokenizer = spec.load_tokenizer(base_dir)
  # remat=NONE: the rollout sampler mutates the KV-cache Params, which conflicts
  # with remat's trace level. restore_sft_model already loads remat=NONE.
  if ckpt_dir:
    actor = restore_sft_model(base_dir, ckpt_dir, mesh=mesh)
  else:
    actor = spec.load_model(base_dir, mesh=mesh, dtype=jnp.bfloat16,
                            param_dtype=jnp.float32, remat=qm.RematConfig.NONE)
  reference = None
  if beta > 0.0:  # KL needs a frozen reference copy
    reference = (restore_sft_model(base_dir, ckpt_dir, mesh=mesh) if ckpt_dir
                 else spec.load_model(base_dir, mesh=mesh, dtype=jnp.bfloat16,
                                      param_dtype=jnp.float32, remat=qm.RematConfig.NONE))
  print("[ota-rl] LOAD OK", flush=True)

  # ---- Prebuild task images once, then register for env lookup. ----
  if task_ids:
    wanted = [t.strip() for t in task_ids.split(",") if t.strip()]
    by_id = {t.task_id: t for t in load_tb_tasks()}
    missing = [w for w in wanted if w not in by_id]
    if missing:
      raise SystemExit(f"[ota-rl] TASK_IDS not found in TB-dev: {missing}")
    tasks = [by_id[w] for w in wanted]
    print(f"[ota-rl] training on {len(tasks)} explicit TASK_IDS", flush=True)
  else:
    tasks = load_tb_tasks(limit=task_limit)
  print(f"[ota-rl] prebuilding {len(tasks)} task images", flush=True)
  built = []
  for t in tasks:
    res = build_image(t.environment_dir, t.image_tag)
    if res.exit_code == 0:
      built.append(t)
    else:
      print(f"[ota-rl]   build FAILED {t.task_id}: {res.stderr[-300:]}", flush=True)
  if not built:
    raise RuntimeError("no task images built; cannot run RL")
  register_tasks(built)
  print(f"[ota-rl] {len(built)}/{len(tasks)} images built", flush=True)

  im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
  eos_id = int(tokenizer.eos_token_id)
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=max_response_tokens,
      max_prompt_length=max_prompt_len,
      kv_cache_size=max_prompt_len + max_response_tokens + 16,
      temperature=temperature,
      top_p=1.0,  # tunix sampler decodes greedily without top_p
      eos_tokens=[im_end_id, eos_id],
      return_logprobs=True,  # agentic learner uses rollout logprobs (use_rollout_logps)
  )
  metrics = metrics_logging_options(
      os.environ.get("RUN_NAME", f"{spec.name}-agent-rl"),
      config={"stage": "rl", "model": spec.name, "G": num_generations,
              "ppb": prompts_per_batch, "lr": learning_rate, "beta": beta,
              "max_turns": max_turns},
  )
  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: mesh,
          rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
          **({rl_cluster_lib.Role.REFERENCE: mesh} if reference is not None else {}),
      },
      rollout_engine="vanilla",  # native JAX rollout (no vLLM on these TPUs)
      offload_to_cpu=offload_to_cpu,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=clipped_adamw(learning_rate),
          eval_every_n_steps=10**9,
          max_steps=steps,
          # mini_batch_size is in PROMPTS and must divide the per-step prompt
          # count (PROMPTS_PER_BATCH); one mini-batch = the whole prompt batch.
          mini_batch_size=prompts_per_batch,
          train_micro_batch_size=1,
          metrics_logging_options=metrics,
          checkpoint_root_directory=rl_ckpt_dir,
      ),
      rollout_config=rollout_config,
  )
  rl_cluster = rl_cluster_lib.RLCluster(
      actor=actor, reference=reference, tokenizer=tokenizer, cluster_config=cluster_config,
  )

  grpo_config = GRPOConfig(
      num_generations=num_generations,
      num_iterations=1,
      beta=beta,
      epsilon=0.2,
      advantage_estimator="drgrpo",  # Dr.GRPO: mean-centered, no std norm
      loss_agg_mode="sequence-mean-token-scale",  # Dr.GRPO aggregation
      system_prompt="",  # the env folds the Terminus-2 preamble into turn 0
      max_response_length=max_response_tokens,  # must match rollout max_tokens_to_generate
      max_concurrency=max_concurrency,
      episode_timeout=episode_timeout,
      overlong_filter=False,  # still grade max-steps episodes (keep their reward)
  )
  learner = GRPOLearner(
      rl_cluster=rl_cluster,
      algo_config=grpo_config,
      reward_fns=None,  # reward = env trajectory (grader) reward
      chat_parser=TerminusQwenParser(tokenizer, enable_thinking=True),
      metric_fns=[_solve_rate_metric],
      agent_class=TerminusAgent,
      env_class=TerminalBenchEnv,
      env_kwargs={"max_steps": max_turns, "command_timeout": command_timeout},
  )

  print(f"[ota-rl] training: {steps} steps x {prompts_per_batch} tasks x {num_generations} gens", flush=True)
  try:
    learner.train(build_prompt_dataset(built, prompts_per_batch, steps))
    print(f"[ota-rl] RL COMPLETE (ckpt={rl_ckpt_dir})", flush=True)
  finally:
    # RL builds every task image up front and reuses them across steps; free them
    # at the end (or on failure) so a long run / retry doesn't accrete vfs disk.
    pruned = prune_ota_images()
    if pruned:
      print(f"[ota-rl] cleanup: pruned {pruned} task image(s)", flush=True)


if __name__ == "__main__":
  main()
