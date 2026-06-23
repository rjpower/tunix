"""M1c: the disaggregated Dr.GRPO loop across two iris jobs.

Closes the loop proven piece-by-piece in M0/M1a/M1b:
  * trainer  serves the policy weights over Arrow Flight (iris-registry discovery),
             then loops: drain a trajectory batch from the channel -> Dr.GRPO
             optimizer step -> serve the new weights (weight_id++).
  * rollout  loops: pull the latest weights into its sampler -> generate G
             completions per prompt -> score (a PLACEHOLDER reward for now; the
             agentic Terminal-Bench env replaces it in M1d) -> put the trajectory
             batch [B*G] to the channel, tagged with the weight_id it used.

Weights go trainer->rollout over Arrow (mega_eval.rl.disagg_common); trajectories
go rollout->trainer over the GCS-staged channel (mega_eval.rl.trajectory_channel);
the Dr.GRPO step is mega_eval.rl.disagg_train. Async / bounded-staleness: the
trainer trains on whatever batch is available and logs the policy-version lag.

Env: ROLE, PRESET, COORD/RUN_ID/S3_COORD_BASE, TRAJ_BASE (gs://... or a shared
dir), RL_STEPS, NUM_GENERATIONS (G), PROMPTS_PER_BATCH (B), MAX_NEW_TOKENS,
MAX_PROMPT_LEN, LEARNING_RATE, TEMPERATURE, TIMEOUT_S.
"""

import itertools
import time

import jax
import numpy as np
from flax import nnx

from tunix.rl.rollout import base_rollout
from tunix.rl.weight_transfer import base
from tunix.rl.weight_transfer.arrow_flight import ArrowFlightClient
from tunix.rl.weight_transfer.arrow_flight import ArrowFlightServer

from mega_eval.rl import disagg_train as dt
from mega_eval.rl.disagg_common import coord_mode
from mega_eval.rl.disagg_common import env
from mega_eval.rl.disagg_common import flight_config
from mega_eval.rl.disagg_common import make_coordinator
from mega_eval.rl.trajectory_channel import TrajectoryChannel
from mega_eval.rollout_workers_smoke import _load_on_mesh
from mega_eval.rollout_workers_smoke import _make_worker
from mega_eval.rollout_workers_smoke import _mesh

_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "The opposite of hot is",
    "Water is made of hydrogen and",
    "The sun rises in the",
    "A group of wolves is called a",
    "The first president of the United States was",
    "The chemical symbol for gold is",
]


def _log(msg):
  print(f"[m1c][{env('ROLE', '?')}] {msg}", flush=True)


def _cfg():
  return dict(
      preset=env("PRESET", "tiny"),
      steps=int(env("RL_STEPS", "4")),
      g=int(env("NUM_GENERATIONS", "4")),
      b=int(env("PROMPTS_PER_BATCH", "4")),
      max_new=int(env("MAX_NEW_TOKENS", "32")),
      max_prompt=int(env("MAX_PROMPT_LEN", "128")),
      lr=float(env("LEARNING_RATE", "1e-5")),
      temperature=float(env("TEMPERATURE", "1.0")),
      timeout_s=float(env("TIMEOUT_S", "1200")),
      traj_base=env("TRAJ_BASE", "./_traj"),
      # agentic (REWARD_MODE=agentic) extras:
      reward_mode=env("REWARD_MODE", "placeholder"),
      task_limit=int(env("TASK_LIMIT", "4")),
      task_ids=env("TASK_IDS", ""),
      max_steps=int(env("MAX_STEPS", "12")),
      max_response_len=int(env("MAX_RESPONSE_LEN", "2048")),
      command_timeout=float(env("COMMAND_TIMEOUT", "60")),
  )


def _rollout_config(c):
  return base_rollout.RolloutConfig(
      max_tokens_to_generate=c["max_new"],
      max_prompt_length=c["max_prompt"],
      temperature=c["temperature"],
      top_p=1.0,
      kv_cache_size=c["max_prompt"] + c["max_new"],
  )


def _placeholder_reward(completion_row, pad_id):
  """A scalar reward with within-group variance, so advantages are non-degenerate.

  PLACEHOLDER for the M1c plumbing test only -- the agentic Terminal-Bench grader
  replaces this in M1d. Deterministic per completion, varies across samples.
  """
  toks = [int(t) for t in np.asarray(completion_row).ravel() if int(t) != pad_id]
  if not toks:
    return 0.0
  return (sum(toks) % 97) / 97.0


def _right_pad(rows, length, pad_id):
  out = np.full((len(rows), length), pad_id, dtype=np.int32)
  for i, r in enumerate(rows):
    r = np.asarray(r, dtype=np.int32).ravel()[:length]
    out[i, : len(r)] = r
  return out


# ---------------------------------------------------------------------------
def _trainer_remat():
  """RematConfig for the trainer (it NEVER samples, so gradient checkpointing is
  safe -- unlike the in-process actor forced to NONE by the sampler/KV conflict).
  Defaults to BLOCK (remat the attn block -> kills the O(seq^2) backward peak)."""
  mode = env("REMAT", "block").lower()
  if mode == "none" or c_preset_is_tiny():
    return None
  from tunix.models.qwen3 import model as qm  # pylint: disable=g-import-not-at-top
  return {"block": qm.RematConfig.BLOCK, "decoder": qm.RematConfig.DECODER}.get(
      mode, qm.RematConfig.BLOCK)


def c_preset_is_tiny():
  return env("PRESET", "tiny") == "tiny"


def run_trainer():
  c = _cfg()
  mesh = _mesh(jax.devices())
  remat = _trainer_remat()
  _log(f"jax {jax.__version__} devices={jax.device_count()} preset={c['preset']} "
       f"coord={coord_mode()} steps={c['steps']} G={c['g']} B={c['b']} "
       f"mesh=(fsdp=1,tp={jax.device_count()}) remat={env('REMAT', 'block')}")
  model, _tok, _config = _load_on_mesh(mesh, c["preset"], remat=remat)
  optimizer = dt.build_optimizer(model, c["lr"])
  algo = dt.build_algo_config(num_generations=c["g"], temperature=c["temperature"])

  server = ArrowFlightServer(flight_config(), coordinator=make_coordinator("weights"))
  channel = TrajectoryChannel(c["traj_base"])
  done = make_coordinator("done")
  _log(f"flight servers up at {server._server_addresses}")  # pylint: disable=protected-access

  weight_id = 0
  server.serve_weights(weight_id, nnx.state(model, nnx.Param))
  _log(f"served initial weight_id={weight_id}; waiting for trajectories")

  for step in range(c["steps"]):
    keys = channel.wait_for_batch(c["timeout_s"])
    if not keys:
      _log(f"step {step}: TIMEOUT waiting for trajectories")
      break
    key = keys[0]
    arrays, meta = channel.get(key)
    pad_id, eos_id = int(meta["pad_id"]), int(meta["eos_id"])
    rewards = arrays["rewards"]
    # Agentic trajectories ship an explicit assistant mask (1=model token,
    # 0=env observation); single-turn ships none (derived from pad). Shape-agnostic.
    te = dt.build_train_example(
        arrays["prompt_ids"], arrays["completion_ids"], rewards,
        c["g"], algo.advantage_estimator, pad_id,
        completion_mask=arrays.get("completion_mask"),
    )
    loss, _aux = dt.train_step(model, optimizer, te, algo, pad_id, eos_id)
    channel.consume(key)
    weight_id += 1
    server.serve_weights(weight_id, nnx.state(model, nnx.Param))
    lag = weight_id - 1 - int(meta.get("weight_id", weight_id - 1))
    _log(f"step {step}: loss={loss:.5f} mean_reward={float(np.mean(rewards)):.4f} "
         f"-> served weight_id={weight_id} (traj policy-lag={lag})")

  done.publish(base.ServerInfo(weight_id=weight_id, server_addresses=["done"],
                               param_names=[]))
  _log(f"=== TRAINER DONE after weight_id={weight_id} ===")
  server.cleanup()
  return 0


# ---------------------------------------------------------------------------
def run_rollout():
  c = _cfg()
  rcfg = _rollout_config(c)
  mesh = _mesh(jax.devices())
  _log(f"jax {jax.__version__} devices={jax.device_count()} preset={c['preset']} "
       f"coord={coord_mode()} G={c['g']} B={c['b']}")
  worker = _make_worker(mesh, c["preset"], rcfg.kv_cache_size, "vanilla", rcfg)
  pad_id, eos_id = worker.pad_id(), worker.eos_id()
  template = nnx.state(worker.model(), nnx.Param)

  client = ArrowFlightClient(flight_config(), make_coordinator("weights"))
  channel = TrajectoryChannel(c["traj_base"])
  done = make_coordinator("done")

  prompts = (_PROMPTS * ((c["b"] // len(_PROMPTS)) + 1))[: c["b"]]
  expanded = [p for p in prompts for _ in range(c["g"])]  # G per prompt, grouped

  cur_wid = -1
  last_generated = -1
  deadline = time.time() + c["timeout_s"]
  rounds = 0
  while time.time() < deadline:
    if done.lookup() is not None:
      _log("trainer signalled done")
      break
    update = client.receive_weights(template=template)
    if update is not None and update.weight_id != cur_wid:
      worker.update_params(update.params, filter_types=(nnx.Param,), reshard_fns=None)
      cur_wid = update.weight_id
      _log(f"pulled weights weight_id={cur_wid}")
    # Generate one batch per weight version (near-on-policy; no channel backlog).
    if cur_wid < 0 or cur_wid == last_generated:
      time.sleep(1)
      continue
    last_generated = cur_wid

    out = worker.generate(expanded, rcfg)
    comp_rows = out.tokens
    prompt_ids = np.asarray(out.left_padded_prompt_tokens, dtype=np.int32)
    completion_ids = _right_pad(comp_rows, c["max_new"], pad_id)
    rewards = np.array(
        [_placeholder_reward(completion_ids[i], pad_id)
         for i in range(len(completion_ids))],
        dtype=np.float32,
    )
    channel.put(
        {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "rewards": rewards},
        meta={"weight_id": int(cur_wid), "num_generations": c["g"],
              "pad_id": int(pad_id), "eos_id": int(eos_id)},
    )
    rounds += 1
    _log(f"round {rounds}: generated {len(completion_ids)} completions @wid={cur_wid} "
         f"mean_reward={float(np.mean(rewards)):.4f} -> channel")

  _log(f"=== ROLLOUT DONE after {rounds} rounds ===")
  client.cleanup()
  return 0


# ---------------------------------------------------------------------------
def run_rollout_agentic():
  """Rollout that collects REAL agentic Terminal-Bench trajectories (gVisor).

  For each weight version: collect G episodes per task (TerminusAgent driving
  shell commands in a per-task gVisor sandbox, graded at the end), pad each to
  fixed shape, and ship [B*G] prompt_ids/completion_ids/completion_mask(assistant)
  /rewards to the channel -- the trainer's Dr.GRPO step is unchanged (it just gets
  an explicit assistant mask + the grader's [0,1] reward).
  """
  from mega_eval.eval.sandbox import build_image, ensure_dockerd  # pylint: disable=g-import-not-at-top
  from mega_eval.eval.tb_tasks import load_tb_tasks  # pylint: disable=g-import-not-at-top
  from mega_eval.rl.environment import register_tasks  # pylint: disable=g-import-not-at-top
  from mega_eval.rl import agentic_collect as ac  # pylint: disable=g-import-not-at-top

  c = _cfg()
  mesh = _mesh(jax.devices())
  kv = c["max_prompt"] + c["max_response_len"] + 64
  _log(f"jax {jax.__version__} devices={jax.device_count()} preset={c['preset']} "
       f"coord={coord_mode()} AGENTIC G={c['g']} B={c['b']} max_steps={c['max_steps']}")
  worker, raw_tok, _config = ac.build_worker(mesh, c["preset"], kv)
  parser = ac.make_chat_parser(raw_tok)
  engine_tok = ac.adapt_tokenizer(raw_tok)
  pad_id, eos_id = worker.pad_id(), worker.eos_id()
  model_call = ac.VanillaModelCall(
      worker, parser, max_prompt_length=c["max_prompt"], kv_cache_size=kv,
      temperature=c["temperature"], top_p=1.0, eos_tokens=ac.eos_token_ids(raw_tok),
  )
  template = nnx.state(worker.model(), nnx.Param)

  # Build the per-task gVisor images once (reused across all rounds).
  ensure_dockerd()
  if c["task_ids"]:
    wanted = [t.strip() for t in c["task_ids"].split(",") if t.strip()]
    by_id = {t.task_id: t for t in load_tb_tasks()}
    tasks = [by_id[w] for w in wanted if w in by_id]
  else:
    tasks = load_tb_tasks(limit=c["task_limit"])
  built = [t for t in tasks if build_image(t.environment_dir, t.image_tag).exit_code == 0]
  if not built:
    raise RuntimeError("no task images built; cannot run agentic rollout")
  register_tasks(built)
  task_ids = [t.task_id for t in built]
  _log(f"built {len(built)}/{len(tasks)} task images")

  client = ArrowFlightClient(flight_config(), make_coordinator("weights"))
  channel = TrajectoryChannel(c["traj_base"])
  done = make_coordinator("done")
  task_cycle = itertools.cycle(task_ids)

  cur_wid = -1
  last_generated = -1
  deadline = time.time() + c["timeout_s"]
  rounds = 0
  while time.time() < deadline:
    if done.lookup() is not None:
      _log("trainer signalled done")
      break
    update = client.receive_weights(template=template)
    if update is not None and update.weight_id != cur_wid:
      worker.update_params(update.params, filter_types=(nnx.Param,), reshard_fns=None)
      cur_wid = update.weight_id
      model_call.policy_version = cur_wid
      _log(f"pulled weights weight_id={cur_wid}")
    if cur_wid < 0 or cur_wid == last_generated:
      time.sleep(1)
      continue
    last_generated = cur_wid

    batch_tasks = [next(task_cycle) for _ in range(c["b"])]
    prompt_rows, comp_rows, mask_rows, rewards = [], [], [], []
    for tid in batch_tasks:  # G episodes per task, grouped contiguously
      for _g in range(c["g"]):
        traj = ac.collect_trajectory(
            worker=worker, tokenizer=engine_tok, chat_parser=parser,
            model_call=model_call, task_id=tid, max_steps=c["max_steps"],
            max_response_length=c["max_response_len"], command_timeout=c["command_timeout"],
        )
        p, cc, mm = ac.pad_trajectory(
            traj, max_prompt_len=c["max_prompt"],
            max_response_len=c["max_response_len"], pad_id=pad_id,
        )
        prompt_rows.append(p)
        comp_rows.append(cc)
        mask_rows.append(mm)
        rewards.append(float(traj.get("trajectory_reward") or 0.0))
    rewards = np.array(rewards, dtype=np.float32)
    channel.put(
        {"prompt_ids": np.stack(prompt_rows), "completion_ids": np.stack(comp_rows),
         "completion_mask": np.stack(mask_rows), "rewards": rewards},
        meta={"weight_id": int(cur_wid), "num_generations": c["g"],
              "pad_id": int(pad_id), "eos_id": int(eos_id), "agentic": True},
    )
    rounds += 1
    solved = int((rewards >= 1.0).sum())
    _log(f"round {rounds}: {len(rewards)} agentic trajectories @wid={cur_wid} "
         f"mean_reward={float(rewards.mean()):.4f} solved={solved}/{len(rewards)} -> channel")

  _log(f"=== AGENTIC ROLLOUT DONE after {rounds} rounds ===")
  client.cleanup()
  return 0


def main():
  role = env("ROLE")
  if role == "trainer":
    raise SystemExit(run_trainer())
  if role == "rollout":
    if env("REWARD_MODE", "placeholder") == "agentic":
      raise SystemExit(run_rollout_agentic())
    raise SystemExit(run_rollout())
  raise SystemExit(f"ROLE must be trainer|rollout, got {role!r}")


if __name__ == "__main__":
  main()
