"""M1a: disaggregated REAL-model weight transfer trainer->rollout + sampler generate.

Extends the M0 rendezvous smoke from synthetic params to a REAL tunix model. The
trainer loads the model and serves its ``nnx.Param`` weights over Arrow Flight
(discovered via the iris registry). The rollout job -- a SEPARATE JAX world --
builds a random-init ``VanillaRollout`` of the SAME architecture, pulls the
trainer's weights via Arrow (resharded onto its own mesh, the template path M0
skipped), loads them into the sampler, and runs a test ``generate()``. This proves
the real weight-transfer + reshard-onto-template path and that the rollout samples
with the trainer's weights -- the foundation for the M1 learner<->rollout loop.

Env: ROLE=trainer|rollout, PRESET=tiny|1.7b|8b, plus the COORD/RUN_ID/S3 vars from
mega_eval.rl.disagg_common; MAX_NEW_TOKENS, MAX_PROMPT_LEN, PROMPT, WEIGHT_ID,
TIMEOUT_S.
"""

import time

import jax
from flax import nnx

from tunix.rl.rollout import base_rollout
from tunix.rl.weight_transfer import base
from tunix.rl.weight_transfer import state_dict
from tunix.rl.weight_transfer.arrow_flight import ArrowFlightClient
from tunix.rl.weight_transfer.arrow_flight import ArrowFlightServer

from mega_eval.rl.disagg_common import agg_checksum
from mega_eval.rl.disagg_common import coord_mode
from mega_eval.rl.disagg_common import env
from mega_eval.rl.disagg_common import flight_config
from mega_eval.rl.disagg_common import make_coordinator
from mega_eval.rollout_workers_smoke import _load_on_mesh
from mega_eval.rollout_workers_smoke import _make_worker
from mega_eval.rollout_workers_smoke import _mesh
from mega_eval.rollout_workers_smoke import _param_bytes


def _log(msg):
  print(f"[m1a][{env('ROLE', '?')}] {msg}", flush=True)


def _rollout_config():
  max_new = int(env("MAX_NEW_TOKENS", "32"))
  max_prompt = int(env("MAX_PROMPT_LEN", "512"))
  return base_rollout.RolloutConfig(
      max_tokens_to_generate=max_new,
      max_prompt_length=max_prompt,
      temperature=0.7,
      top_p=1.0,
      kv_cache_size=max_prompt + max_new,
  )


def run_trainer():
  preset = env("PRESET", "tiny")
  weight_id = int(env("WEIGHT_ID", "1"))
  timeout_s = float(env("TIMEOUT_S", "900"))
  mesh = _mesh(jax.devices())
  _log(f"jax {jax.__version__} devices={jax.device_count()} preset={preset} "
       f"coord={coord_mode()}")

  model, _tokenizer, _config = _load_on_mesh(mesh, preset)
  params = nnx.state(model, nnx.Param)
  _keys, flat = state_dict.flatten_for_transfer(params, convert_to_bfloat16=True)
  agg = agg_checksum(flat)
  nbytes = _param_bytes(model)
  _log(f"loaded model: {nbytes / 1e9:.3f} GB params, agg_checksum={agg}")

  server = ArrowFlightServer(flight_config(), coordinator=make_coordinator("weights"))
  _log(f"flight servers up at {server._server_addresses}")  # pylint: disable=protected-access
  t0 = time.time()
  server.serve_weights(weight_id, params)
  _log(f"served+registered weight_id={weight_id} in {time.time() - t0:.2f}s; "
       "waiting for rollout done")

  done = make_coordinator("done")
  deadline = time.time() + timeout_s
  while time.time() < deadline:
    sig = done.lookup()
    if sig is not None:
      _log(f"rollout done: weight_id={sig.weight_id} note={sig.server_addresses}")
      _log("=== TRAINER OK ===")
      server.cleanup()
      return 0
    time.sleep(3)
  _log("=== TRAINER TIMEOUT ===")
  server.cleanup()
  return 1


def run_rollout():
  preset = env("PRESET", "tiny")
  weight_id = int(env("WEIGHT_ID", "1"))
  timeout_s = float(env("TIMEOUT_S", "900"))
  prompt = env("PROMPT", "The capital of France is")
  rcfg = _rollout_config()
  mesh = _mesh(jax.devices())
  _log(f"jax {jax.__version__} devices={jax.device_count()} preset={preset} "
       f"coord={coord_mode()}")

  worker = _make_worker(mesh, preset, rcfg.kv_cache_size, "vanilla", rcfg)
  template = nnx.state(worker.model(), nnx.Param)

  client = ArrowFlightClient(flight_config(), make_coordinator("weights"))
  deadline = time.time() + timeout_s
  update = None
  t0 = time.time()
  while time.time() < deadline:
    update = client.receive_weights(template=template)
    if update is not None:
      break
    time.sleep(3)
  if update is None:
    _log("=== ROLLOUT TIMEOUT: no weights resolved ===")
    return 1
  agg = agg_checksum(update.flat_state)
  _log(f"received weight_id={update.weight_id} in {time.time() - t0:.2f}s "
       f"agg_checksum={agg}")

  # Load the transferred weights (already on the rollout mesh) into the sampler.
  worker.update_params(update.params, filter_types=(nnx.Param,), reshard_fns=None)
  _log("loaded weights into sampler; generating...")
  g0 = time.time()
  out = worker.generate([prompt], rcfg)
  text = out.text[0] if getattr(out, "text", None) else str(out)
  _log(f"generate OK in {time.time() - g0:.2f}s: {prompt!r} -> {text!r}")

  ok = update.weight_id == weight_id and isinstance(text, str) and len(text) > 0
  done = make_coordinator("done")
  done.publish(
      base.ServerInfo(
          weight_id=1 if ok else 0,
          server_addresses=[f"agg={agg}", f"gen_len={len(text)}"],
          param_names=[],
      )
  )
  client.cleanup()
  _log("=== ROLLOUT OK ===" if ok else "=== ROLLOUT FAIL ===")
  return 0 if ok else 1


def main():
  role = env("ROLE")
  if role == "trainer":
    raise SystemExit(run_trainer())
  if role == "rollout":
    raise SystemExit(run_rollout())
  raise SystemExit(f"ROLE must be trainer|rollout, got {role!r}")


if __name__ == "__main__":
  main()
