"""M1d step 2: validate the standalone agentic collector end-to-end on marin.

Loads a real Qwen3 VanillaRollout, builds ONE Terminal-Bench task image, and runs
the full agentic episode (TerminusAgent driving shell commands in a gVisor sandbox,
graded at the end) via mega_eval.rl.agentic_collect -- WITHOUT an RLCluster. Proves
the model_call adapter + collect engine + agent + env + sandbox + grader all work
together and emit the multi-turn token/mask/reward arrays the trainer consumes.

Env: PRESET (default qwen3-1.7b), TASK_LIMIT (#tasks to build, default 1),
MAX_STEPS, MAX_RESPONSE_LEN, MAX_PROMPT_LEN, TEMPERATURE.
"""

import os
import sys
import traceback

import jax
import numpy as np

from mega_eval.eval.sandbox import ensure_dockerd
from mega_eval.eval.sandbox import build_image
from mega_eval.eval.tb_tasks import load_tb_tasks
from mega_eval.rl.environment import register_tasks
from mega_eval.rl import agentic_collect as ac
from mega_eval.rollout_workers_smoke import _mesh


def _env(name, default):
  return os.environ.get(name, default)


def _log(m):
  print(f"[acsmoke] {m}", flush=True)


def main():
  preset = _env("PRESET", "qwen3-1.7b")
  task_limit = int(_env("TASK_LIMIT", "1"))
  max_steps = int(_env("MAX_STEPS", "8"))
  max_response_len = int(_env("MAX_RESPONSE_LEN", "1024"))
  max_prompt_len = int(_env("MAX_PROMPT_LEN", "4096"))
  temperature = float(_env("TEMPERATURE", "0.7"))
  kv_cache = max_prompt_len + max_response_len + 64

  try:
    _log(f"jax {jax.__version__} devices={jax.device_count()} preset={preset}")

    _log("prebuild task image(s)")
    ensure_dockerd()
    tasks = load_tb_tasks(limit=task_limit)
    built = []
    for t in tasks:
      res = build_image(t.environment_dir, t.image_tag)
      if res.exit_code == 0:
        built.append(t)
      else:
        _log(f"  build FAILED {t.task_id}: {res.stderr[-300:]}")
    if not built:
      raise RuntimeError("no task images built")
    register_tasks(built)
    task_id = built[0].task_id
    _log(f"built {len(built)} image(s); collecting for task_id={task_id}")

    _log("load worker (VanillaRollout + real tokenizer)")
    mesh = _mesh(jax.devices())
    worker, raw_tok, _config = ac.build_worker(mesh, preset, kv_cache)
    parser = ac.make_chat_parser(raw_tok)          # parser takes RAW tokenizer
    engine_tok = ac.adapt_tokenizer(raw_tok)       # engine needs the adapter
    model_call = ac.VanillaModelCall(
        worker, parser,
        max_prompt_length=max_prompt_len, kv_cache_size=kv_cache,
        temperature=temperature, top_p=1.0, eos_tokens=ac.eos_token_ids(raw_tok),
    )

    _log(f"collect episode (max_steps={max_steps}, max_response_len={max_response_len})")
    traj = ac.collect_trajectory(
        worker=worker, tokenizer=engine_tok, chat_parser=parser,
        model_call=model_call, task_id=task_id, max_steps=max_steps,
        max_response_length=max_response_len,
    )

    ct = np.asarray(traj["conversation_tokens"])
    cm = np.asarray(traj["conversation_masks"])
    pt = np.asarray(traj["prompt_tokens"])
    n_assistant = int(cm.sum())
    _log("=== TRAJECTORY ===")
    _log(f"  status            : {traj.get('status')}")
    _log(f"  trajectory_reward : {traj.get('trajectory_reward')}")
    _log(f"  prompt_tokens     : {pt.shape}")
    _log(f"  conversation_tok  : {ct.shape}  (assistant tokens trained on: {n_assistant})")
    _log(f"  conversation_mask : {cm.shape} sum={n_assistant} (1=assistant,0=env)")
    n_turns = len(traj.get("conversation_text") or [])
    _log(f"  conversation turns: {n_turns}")
    last = ""
    for m in (traj.get("conversation_text") or []):
      if m.get("role") == "assistant":
        last = m.get("content", "")
    _log(f"  last assistant msg (160c): {last[:160]!r}")

    assert ct.shape == cm.shape, "tokens/masks shape mismatch"
    assert n_assistant > 0, "no assistant tokens to train on"
    assert traj.get("trajectory_reward") is not None, "no reward"
    _log("=== AGENTIC COLLECT SMOKE OK ===")
    return 0
  except Exception as e:  # pylint: disable=broad-except
    _log(f"!!! FAILED: {e!r}")
    traceback.print_exc()
    return 1


if __name__ == "__main__":
  sys.exit(main())
