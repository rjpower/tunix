# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris entrypoint: evaluate an SFT'd Qwen3-8B agent on OpenThoughts Terminal-Bench.

For each TB-dev task we build its environment image, run it under gVisor (runsc),
let the Terminus-2 agent loop drive the SFT'd policy to issue shell commands in
the sandbox, then run the task's grader (`tests/test.sh`) for pass/fail. Requires
the custom task image (`docker/Dockerfile.agent-task`, which ships runsc+docker)
selected via `iris job run --task-image ...`, and the SFT checkpoint in CKPT_DIR.

Config via env:
  * ``AGENT_MODEL`` (qwen3-8b), ``CKPT_DIR`` (required: the SFT checkpoint root).
  * ``TASK_LIMIT`` (unset = all TB-dev tasks; set small to start).
  * ``TASK_OFFSET`` (0): skip the first N tasks. Pair with ``TASK_LIMIT`` to shard a
    sweep across parallel jobs (e.g. offset 0/14/28/... limit 14 -> 5 disjoint jobs).
  * ``MAX_TURNS`` (20), ``COMMAND_TIMEOUT`` (60), ``MAX_NEW_TOKENS`` (1024).
  * ``TP`` (1), ``MAX_PROMPT_LEN`` (8192), ``TEMPERATURE`` (0.2).
  * ``K_SAMPLES`` (1): episodes per task. >1 runs each task K times (fresh sandbox
    each) and reports pass@1 (mean solve), pass@k (any solve), and per-task score
    stats. For meaningful diversity raise ``TEMPERATURE`` (~0.7-1.0) — k identical
    greedy draws give no spread. The **RL gate** is ``score_spread`` (the continuous
    grader score VARIES across the k samples), NOT binary ``0<pass1<1``: the RL env
    reward is the continuous score (rl/environment.py), so Dr.GRPO gets a usable
    advantage whenever scores differ, even with 0 full solves (see REPORT.md).
  * ``OTA_SANDBOX`` (gvisor) -- set ``local`` only for harness debugging.

Submit (custom image + checkpoint):

    uv run iris --cluster=marin job run --no-wait \
      --task-image ghcr.io/<org>/openthoughts-agent-task:latest \
      --tpu v6e-8 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 16 --memory 200GB --disk 200GB --max-retries 1 --job-name ota-eval \
      -e HF_TOKEN "$HF_TOKEN" \
      -e CKPT_DIR gs://marin-us-central2/openthoughts-agent/qwen3-8b-agent-sft \
      -e TASK_LIMIT 10 -- python launch_eval.py
"""

import json
import os
import traceback

import jax
from huggingface_hub import snapshot_download

from mega_eval.eval.agent_loop import run_episode
from mega_eval.eval.grade import grade_task
from mega_eval.eval.model_serving import make_tunix_model_fn
from mega_eval.eval.sandbox import GvisorContainerSandbox, build_image, prune_ota_images, remove_image
from mega_eval.eval.tb_tasks import load_tb_tasks
from mega_eval.models.checkpoint import restore_sft_model
from mega_eval.models.registry import get_model_spec
from mega_eval.training.common import build_mesh, init_distributed


def _ensure_model(repo: str, model_dir: str) -> str:
  if not os.path.exists(os.path.join(model_dir, "config.json")):
    snapshot_download(repo_id=repo, local_dir=model_dir)
  return model_dir


def main() -> None:
  init_distributed()  # must precede any jax call (orbax multi-host barriers)
  model_name = os.environ.get("AGENT_MODEL", "qwen3-8b")
  checkpoint_dir = os.environ["CKPT_DIR"]
  task_limit = os.environ.get("TASK_LIMIT")
  task_limit = int(task_limit) if task_limit else None
  task_offset = int(os.environ.get("TASK_OFFSET", "0"))  # shard: skip first N tasks
  max_turns = int(os.environ.get("MAX_TURNS", "20"))
  command_timeout = float(os.environ.get("COMMAND_TIMEOUT", "60"))
  max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
  max_prompt_len = int(os.environ.get("MAX_PROMPT_LEN", "8192"))
  temperature = float(os.environ.get("TEMPERATURE", "0.2"))
  k_samples = int(os.environ.get("K_SAMPLES", "1"))
  tp = int(os.environ.get("TP", "1"))

  spec = get_model_spec(model_name)
  base_dir = _ensure_model(spec.repo, os.environ.get("AGENT_MODEL_DIR") or f"./{spec.name}")
  print(f"[ota-eval] jax {jax.__version__} devices={jax.device_count()} ckpt={checkpoint_dir}", flush=True)

  mesh = build_mesh(tp=tp)
  tokenizer = spec.load_tokenizer(base_dir)
  model = restore_sft_model(base_dir, checkpoint_dir, mesh=mesh)
  model_fn = make_tunix_model_fn(
      model, tokenizer, mesh,
      max_prompt_length=max_prompt_len,
      max_new_tokens=max_new_tokens,
      temperature=temperature,
  )

  # Shard for parallel sweeps: load all, then take tasks[offset : offset+limit].
  # Fan N jobs out with disjoint TASK_OFFSET windows to cover all 70 tasks fast.
  all_tasks = load_tb_tasks()
  end = task_offset + task_limit if task_limit else len(all_tasks)
  tasks = all_tasks[task_offset:end]
  print(f"[ota-eval] {len(tasks)} TB tasks to evaluate (offset={task_offset} "
        f"of {len(all_tasks)}), k={k_samples} @ temp={temperature}", flush=True)

  def _one_episode(task) -> dict:
    """Boot a fresh sandbox, run the agent, grade. Isolated per sample."""
    out = {"solved": False, "score": 0.0, "turns": None, "parse_failures": None, "error": None}
    sandbox = None
    try:
      sandbox = GvisorContainerSandbox(task.image_tag)
      episode = run_episode(
          model_fn, sandbox, task.instruction,
          max_turns=max_turns, command_timeout=command_timeout,
      )
      grade = grade_task(sandbox, task)
      out.update(solved=grade.solved, score=grade.score,
                 turns=episode.turns, parse_failures=episode.parse_failures)
    except Exception as e:  # one sample's failure must not kill the rest
      out["error"] = f"{type(e).__name__}: {e}"
      print(f"[ota-eval]   TRACE {task.task_id}:\n{traceback.format_exc()}", flush=True)
    finally:
      if sandbox is not None:
        sandbox.close()
    return out

  records = []
  for i, task in enumerate(tasks):
    print(f"[ota-eval] ({i+1}/{len(tasks)}) task={task.task_id}", flush=True)
    rec = {"task_id": task.task_id, "k": k_samples, "samples": [],
           "pass1": 0.0, "passk": False, "best_score": 0.0, "error": None}
    records.append(rec)  # record up front so a mid-task crash can't drop the task
    build = build_image(task.environment_dir, task.image_tag)
    if build.exit_code != 0:
      rec["error"] = f"image build failed: {build.stderr[-500:]}"
      print(f"[ota-eval]   -> {json.dumps(rec)}", flush=True)
      continue
    try:
      for s in range(k_samples):
        sample = _one_episode(task)
        rec["samples"].append(sample)
        print(f"[ota-eval]   sample {s+1}/{k_samples} -> "
              f"solved={sample['solved']} score={sample['score']:.3f} "
              f"turns={sample['turns']} parse_fail={sample['parse_failures']}", flush=True)
      solves = [bool(x["solved"]) for x in rec["samples"]]
      scores = [float(x["score"]) for x in rec["samples"]]
      rec["pass1"] = sum(solves) / max(len(solves), 1)  # mean solve over k
      rec["passk"] = any(solves)
      rec["best_score"] = max(scores) if scores else 0.0
      rec["score_mean"] = sum(scores) / max(len(scores), 1)
      rec["spread"] = 0.0 < rec["pass1"] < 1.0  # binary-solve spread (strict)
      # The RL gate that actually matches the env reward: the env returns the
      # CONTINUOUS grader score (rl/environment.py), so Dr.GRPO gets a usable
      # advantage whenever the k scores VARY -- even with 0 full solves. A task
      # where every sample scores the same (incl. all-zero) gives 0 advantage.
      rec["score_spread"] = (max(scores) - min(scores) > 1e-9) if scores else False
      print(f"[ota-eval]   = task pass1={rec['pass1']:.3f} passk={rec['passk']} "
            f"score[min/mean/max]={min(scores):.3f}/{rec['score_mean']:.3f}/{rec['best_score']:.3f} "
            f"score_spread={rec['score_spread']}", flush=True)
    finally:
      # Free the image now that this task's samples are graded (or on error):
      # vfs doesn't share layers, so keeping every image blows a small disk.
      # Bounds usage to ~1 image at a time.
      remove_image(task.image_tag)

  n = len(records)
  # micro pass@1 = mean solve over ALL (task,sample); macro pass@k = tasks any-solved.
  all_solves = [bool(x["solved"]) for r in records for x in r["samples"]]
  pass1 = sum(all_solves) / max(len(all_solves), 1)
  passk = sum(1 for r in records if r["passk"]) / max(n, 1)
  spread_tasks = [r["task_id"] for r in records if r.get("spread")]
  # The RL go/no-go: tasks whose continuous scores VARY across the k samples
  # (what Dr.GRPO turns into advantage), regardless of whether any fully solved.
  score_spread_tasks = [r["task_id"] for r in records if r.get("score_spread")]
  print(f"[ota-eval] ===== RESULTS ({n} tasks, k={k_samples}) =====", flush=True)
  print(f"[ota-eval] pass@1={pass1:.3f} (over {len(all_solves)} samples) | "
        f"pass@{k_samples}={passk:.3f} (tasks any-solved)", flush=True)
  print(f"[ota-eval] binary-solve spread (0<pass1<1): {len(spread_tasks)} tasks "
        f"{json.dumps(spread_tasks)}", flush=True)
  print(f"[ota-eval] RL-TRAINABLE (continuous score varies across k): "
        f"{len(score_spread_tasks)} tasks {json.dumps(score_spread_tasks)}", flush=True)
  print(f"[ota-eval] PER_TASK_JSON {json.dumps(records)}", flush=True)
  # Backstop the per-task removals: sweep any task image left resident by a crash
  # between build and grade (best-effort; the iris task's vfs store dies with it).
  pruned = prune_ota_images()
  if pruned:
    print(f"[ota-eval] cleanup: pruned {pruned} leftover task image(s)", flush=True)


if __name__ == "__main__":
  main()
