# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Infra smoke test: 1 trainer + N INDEPENDENT rollout workers on one node.

Validates the building blocks for a *future* multi-worker agentic learner:
disaggregated trainer->worker weight sync + N rollout workers that generate
independently and concurrently. Tunix today only drives ONE rollout
(``RLCluster`` -> a single ``VanillaRollout``); ``RolloutEngineGroup`` /
``rollout_traffic_router`` are empty stubs. This harness exercises the
N-worker topology so we can measure it before that abstraction exists.

TOPOLOGY (one JAX process, in-world cross-mesh reshard)
-------------------------------------------------------
One device pool (e.g. 8 GPUs) is partitioned -- reusing
``bench_weight_transfer._partition_devices`` -- into::

    +-------------------+        update_params (cross-mesh reshard)
    |  TRAINER mesh     |  --------------------------------------+
    |  (frac of devs)   |                                        |
    +-------------------+      +------------+   +------------+    |
                               | worker 0   |   | worker 1   | ...|  (N workers)
                               | own mesh   |   | own mesh   |    |
                               | own Sampler|   | own Sampler|<---+
                               | own KV $   |   | own KV $   |
                               +------------+   +------------+
                                  .generate()      .generate()   (concurrent)

For 8 devices, N=4, TRAINER_FRAC=0.5 => trainer 4 devices, 4 workers x 1 device.

"INDEPENDENT workers" here = each worker is a separate ``VanillaRollout`` on its
OWN ``jax.sharding.Mesh``, with its OWN ``Sampler`` and OWN KV cache -- i.e.
independent rollout *state*, all living in ONE JAX process. The trainer->worker
sync is an in-JAX-world cross-mesh reshard (``reshard.reshard_pytree`` via
``VanillaRollout.update_params(..., reshard_fns=select_reshard_fns(...))``), the
same path ``bench_weight_transfer``'s ``nccl`` mode benchmarks (NCCL collectives
on GPU; ``jax.device_put`` on CPU).

This is NOT yet multi-PROCESS isolation. True per-worker process isolation
(separate ``jax.distributed`` processes, separate HBM, a worker dying without
killing the trainer) needs the host-staged Apache Arrow Flight transport
(``tunix/rl/weight_transfer/arrow_flight.py``); see "PROCESS ISOLATION" below.

TUNIX APIs USED
---------------
* ``tunix.rl.rollout.vanilla_rollout.VanillaRollout(model, tokenizer,
  cache_config_or_size)`` -- one rollout worker.
* ``VanillaRollout.update_params(params, filter_types=(nnx.Param,),
  reshard_fns=...)`` -- cross-mesh reshard trainer params -> this worker's mesh.
* ``VanillaRollout.generate(prompts, RolloutConfig)`` -- runs the worker's
  ``tunix.generate.sampler.Sampler``.
* ``tunix.rl.rollout.base_rollout.{CacheConfig,RolloutConfig}``.
* ``tunix.rl.weight_transfer.select_reshard_fns()`` -- reshard-backend factory
  list passed to ``update_params`` / ``reshard.reshard_pytree``.
* ``tunix.models.dummy_model_creator.create_dummy_model`` -- random sharded
  tunix Qwen3 for the ``tiny`` preset (no checkpoint download on CPU).
* ``mega_eval.bench_weight_transfer._partition_devices`` -- device pool split.
* ``mega_eval.models.registry.get_model_spec`` -- real Qwen3 loaders.

ENV KNOBS (all optional)
------------------------
  MODEL_PRESET    tiny | qwen3-1.7b | qwen3-8b               (default tiny)
  N_WORKERS       number of independent rollout workers      (default 4)
  TRAINER_FRAC    fraction of the device pool for trainer     (default 0.5)
  ROUNDS          sync+rollout rounds                         (default 3)
  MAX_NEW_TOKENS  tokens generated per prompt                 (default 32)
  MAX_PROMPT_LEN  prompt pad/truncate length                  (default 512)
  BATCH           prompts per worker per round                (default 2)
  PROMPT          the prompt every worker rolls out           (default below)
  RESHARD_BACKEND auto | jax_device | pathways (select_reshard_fns) (default auto)

CPU SMOKE (no GPU, 8 simulated devices)
---------------------------------------
``--xla_force_host_platform_device_count=8`` is set on this module BEFORE jax is
imported (see top of file), so a plain CPU python sees 8 devices::

  MODEL_PRESET=tiny N_WORKERS=4 ROUNDS=2 MAX_NEW_TOKENS=8 \
      .venv/bin/python mega_eval/rollout_workers_smoke.py
  # or: uv run python mega_eval/rollout_workers_smoke.py

IRIS / CW LAUNCH (8xH100 node, real model)
------------------------------------------
Start with the small real model to validate true GPU resharding, then scale up::

  uv run iris --cluster=cw-us-east-02a job run \
      --gpu H100x8 --enable-extra-resources --extra gpu --extra mega \
      --env MODEL_PRESET=qwen3-1.7b --env N_WORKERS=4 --env TRAINER_FRAC=0.5 \
      --env ROUNDS=3 --env MAX_NEW_TOKENS=32 \
      -- python -m mega_eval.rollout_workers_smoke

On 8xH100 with TRAINER_FRAC=0.5 + N_WORKERS=4 that is trainer=4 GPUs and
4 workers x 1 GPU; on GPU the cross-mesh reshard runs as NCCL collectives. Bump
MODEL_PRESET=qwen3-8b once qwen3-1.7b is green (the 8B trainer shard fits in 4
H100s; each 1-GPU worker holds a bf16 copy).

PROCESS ISOLATION (follow-up, not done here)
--------------------------------------------
To make the N workers TRUE separate processes (one dying must not kill the
trainer; each owns its HBM), swap the in-world reshard for the host-staged
Arrow Flight transport: launch N+1 ``jax.distributed`` processes, process 0 =
trainer running ``arrow_flight.ArrowFlightServer.serve_weights``, processes
1..N = workers each running ``arrow_flight.ArrowFlightClient.receive_weights``
into a local ``VanillaRollout``. ``bench_weight_transfer._bench_arrow_multihost``
already demonstrates that server/client wiring (minus the rollout).
"""

import concurrent.futures
import os
import time

# Simulate an 8-device pool on CPU. MUST be set before jax is imported (XLA
# reads it at backend init). On a real accelerator node we must NOT set this: the
# host-platform flag is technically ignored for device selection (jax defaults to
# the GPU/TPU backend), but skipping it removes any ambiguity so jax.devices()
# returns the physical GPUs/TPUs. Detect accelerators without importing jax.
_HAS_ACCEL = (
    os.path.exists("/dev/nvidia0")
    or os.path.exists("/proc/driver/nvidia")
    or bool(os.environ.get("TPU_WORKER_ID"))
    or os.environ.get("PJRT_DEVICE", "").upper() in ("CUDA", "GPU", "TPU")
    or os.environ.get("JAX_PLATFORMS", "").lower() in ("cuda", "gpu", "tpu")
)
if not _HAS_ACCEL:
  os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402  pylint: disable=g-import-not-at-top
import jax.numpy as jnp  # noqa: E402  pylint: disable=g-import-not-at-top
import numpy as np  # noqa: E402  pylint: disable=g-import-not-at-top
from flax import nnx  # noqa: E402  pylint: disable=g-import-not-at-top
from jax.sharding import Mesh  # noqa: E402  pylint: disable=g-import-not-at-top

from mega_eval.bench_weight_transfer import _partition_devices  # noqa: E402  pylint: disable=g-import-not-at-top
from tunix.rl.rollout import base_rollout  # noqa: E402  pylint: disable=g-import-not-at-top
from tunix.rl.rollout import vanilla_rollout  # noqa: E402  pylint: disable=g-import-not-at-top
from tunix.rl.weight_transfer import select_reshard_fns  # noqa: E402  pylint: disable=g-import-not-at-top


def _env(name, default):
  return os.environ.get(name, default)


# Cloudflare R2 (S3-compatible) endpoint for the marin-na bucket. tensorstore's
# s3 driver reads AWS_* env; map our R2_* creds onto it when AWS_* is unset.
_R2_ENDPOINT = "https://74981a43be0de7712369306c7b19133d.r2.cloudflarestorage.com"


def _map_r2_to_aws_env():
  if "R2_ACCESS_KEY_ID" in os.environ:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ["R2_ACCESS_KEY_ID"])
  if "R2_SECRET_ACCESS_KEY" in os.environ:
    os.environ.setdefault(
        "AWS_SECRET_ACCESS_KEY", os.environ["R2_SECRET_ACCESS_KEY"]
    )
  os.environ.setdefault("AWS_ENDPOINT_URL", _R2_ENDPOINT)
  os.environ.setdefault("AWS_REGION", "auto")


def _persist_result_s3(uri, payload):
  """Write a JSON result blob to an ``s3://bucket/key`` URI via tensorstore's s3
  kvstore. CW pod logs are unreliable (pods GC on success, flaky log server), so
  the run persists a durable artifact we can read back regardless of logs.
  Best-effort: logs and swallows any error so it never fails the smoke."""
  import json  # noqa: g-import-not-at-top
  try:
    import tensorstore as ts  # noqa: g-import-not-at-top

    assert uri.startswith("s3://"), uri
    bucket, _, key = uri[len("s3://") :].partition("/")
    parent, _, leaf = key.rpartition("/")
    _map_r2_to_aws_env()
    kv = ts.KvStore.open({
        "driver": "s3",
        "bucket": bucket,
        "path": (parent + "/") if parent else "",
        "endpoint": os.environ.get("AWS_ENDPOINT_URL", _R2_ENDPOINT),
        "aws_region": os.environ.get("AWS_REGION", "auto"),
    }).result()
    kv.write(leaf, json.dumps(payload, indent=2).encode()).result()
    print(f"[result] persisted -> {uri}", flush=True)
    return True
  except Exception as e:  # pylint: disable=broad-except
    print(f"[result] PERSIST FAILED ({type(e).__name__}): {e}", flush=True)
    return False


_DEFAULT_PROMPT = (
    "You are a terminal agent. The repo fails to build. "
    "Run the test suite, read the first error, and propose a one-line fix. "
    "Begin:\n$ "
)


# ---------------------------------------------------------------------------
# Meshes. Qwen3's default ShardingConfig references the "fsdp" and "tp" axes, so
# every mesh here is named ("fsdp", "tp") (NOT the bench's ("dp","tp") -- the
# bench builds synthetic params it shards by hand, whereas we build a REAL
# tunix Qwen3 whose param sharding comes from ShardingConfig). We reuse the
# bench's _partition_devices for the device split, then name the axes ourselves.
# ---------------------------------------------------------------------------


def _mesh(devices) -> Mesh:
  """A 1 x len(devices) ('fsdp','tp') mesh: pure tensor-parallel over the set."""
  return Mesh(np.array(devices).reshape(1, len(devices)), ("fsdp", "tp"))


# ---------------------------------------------------------------------------
# Tiny preset: a small REAL tunix Qwen3 with random weights + a stub tokenizer.
# No checkpoint download, CPU-friendly, but exercises the real Sampler / KV
# cache / reshard code paths (not synthetic pytrees).
# ---------------------------------------------------------------------------


class _StubTokenizer:
  """Minimal byte-level tokenizer satisfying tunix's TokenizerAdapter contract.

  TokenizerAdapter requires ``encode/decode/bos_id/eos_id/pad_id``. We map each
  prompt to its UTF-8 bytes offset into the model vocab (ids stay < vocab_size),
  which is enough to drive the Sampler end to end for a smoke test (the tiny
  model's outputs are gibberish -- we only assert it RAN and produced text).
  """

  def __init__(self, vocab_size: int):
    self._vocab = vocab_size
    # Reserve 3 special ids at the top of the vocab.
    self._pad = vocab_size - 1
    self._eos = vocab_size - 2
    self._bos = vocab_size - 3

  def encode(self, text: str, **_):
    return [b % (self._vocab - 3) for b in text.encode("utf-8")] or [0]

  def decode(self, ids, **_):
    keep = [i for i in ids if i not in (self._pad, self._eos, self._bos)]
    return bytes(i % 256 for i in keep).decode("utf-8", errors="replace")

  def bos_id(self):
    return self._bos

  def eos_id(self):
    return self._eos

  def pad_id(self):
    return self._pad


# Tiny Qwen3 dims: real architecture (GQA, SwiGLU, RoPE), shrunk to be CPU-fast.
# vocab kept small so the stub tokenizer's byte ids stay in range and the
# embed/lm_head matrices are tiny. Head/feature counts are multiples of 8 so the
# default ShardingConfig (which shards heads/vocab/hidden by the "tp" axis)
# divides evenly for ANY mesh width up to the full 8-device pool -- the trainer
# mesh tp width = round(8*TRAINER_FRAC), so this stays valid as TRAINER_FRAC
# varies (e.g. trainer tp=4 at the default 0.5).
_TINY_QWEN3 = dict(
    num_layers=2,
    vocab_size=512,
    embed_dim=256,
    hidden_dim=512,
    num_heads=8,
    head_dim=32,
    num_kv_heads=8,
    rope_theta=1_000_000,
    norm_eps=1e-6,
)


def _build_tiny(mesh: Mesh):
  """Returns (model, tokenizer, config) for the CPU-smoke 'tiny' preset."""
  from tunix.models.dummy_model_creator import create_dummy_model  # pylint: disable=g-import-not-at-top
  from tunix.models.qwen3 import model as qm  # pylint: disable=g-import-not-at-top

  config = qm.ModelConfig(**_TINY_QWEN3)
  model = create_dummy_model(qm.Qwen3, config, mesh=mesh, dtype=jnp.float32)
  tokenizer = _StubTokenizer(config.vocab_size)
  return model, tokenizer, config


_REAL_PRESETS = {
    "qwen3-1.7b": "qwen3-1.7b-base",
    "qwen3-8b": "qwen3-8b",
}


def _build_real(mesh: Mesh, preset: str, remat=None):
  """Loads a real Qwen3 + tokenizer (downloads from HF) onto ``mesh``.

  Used on GPU (iris). ``mega_eval.models.registry`` maps the friendly preset to
  a ModelSpec whose ``load_model`` is mega_eval.models.qwen3_loader.load_qwen3.

  ``remat`` (a ``qwen3.model.RematConfig``, or None) enables gradient
  checkpointing -- pass ``BLOCK``/``DECODER`` for a TRAINING model to fit long
  sequences in the backward pass. Leave None (=> ``RematConfig.NONE``) for a
  rollout/inference model, since the sampler mutating the KV cache trips remat's
  trace level. Disaggregation makes this safe: the trainer never samples.
  """
  from huggingface_hub import snapshot_download  # pylint: disable=g-import-not-at-top
  from mega_eval.models.registry import get_model_spec  # pylint: disable=g-import-not-at-top

  spec = get_model_spec(_REAL_PRESETS[preset])
  model_dir = snapshot_download(spec.repo)
  # bf16 params for inference (a replicated fp32 8B copy per worker would OOM).
  kw = {"remat": remat} if remat is not None else {}
  model = spec.load_model(model_dir, mesh=mesh, dtype=jnp.bfloat16, **kw)
  tokenizer = spec.load_tokenizer(model_dir)
  return model, tokenizer, model.config


def _load_on_mesh(mesh: Mesh, preset: str, remat=None):
  if preset == "tiny":
    return _build_tiny(mesh)
  if preset in _REAL_PRESETS:
    return _build_real(mesh, preset, remat=remat)
  raise ValueError(
      f"Unknown MODEL_PRESET={preset!r}; "
      f"known: ['tiny', {', '.join(repr(k) for k in _REAL_PRESETS)}]."
  )


def _param_bytes(model) -> int:
  """Total bytes of a model's params (for weight-sync throughput in GB/s).

  Leaves on a sharded mesh report their GLOBAL nbytes, which is exactly the
  amount the cross-mesh reshard moves -- so bytes/sync_time is the transfer rate.
  """
  leaves = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
  return int(sum(getattr(x, "nbytes", 0) for x in leaves))


def _hf_repo(preset: str) -> str:
  """HF repo id for a real preset (sglang_jax needs it as model_version)."""
  if preset not in _REAL_PRESETS:
    return ""
  from mega_eval.models.registry import get_model_spec  # pylint: disable=g-import-not-at-top

  return get_model_spec(_REAL_PRESETS[preset]).repo


def _make_worker(mesh: Mesh, preset: str, kv_cache_size: int, engine: str,
                 rollout_config):
  """Builds one independent rollout worker (own model+sampler+KV) on ``mesh``.

  engine=vanilla   -> tunix VanillaRollout (replicated KV; in-world reshard sync).
  engine=sglang_jax-> tunix SglangJaxRollout (paged/tp-shardable KV; the sampler
                      reshards weights internally). Needs the ``sgl_jax`` package
                      and a real (HF) preset so ``model_version`` resolves a config.
  """
  model, tokenizer, config = _load_on_mesh(mesh, preset)
  if engine == "sglang_jax":
    from tunix.rl.rollout import sglang_jax_rollout  # pylint: disable=g-import-not-at-top

    return sglang_jax_rollout.SglangJaxRollout(
        model, tokenizer, mesh, rollout_config
    )
  cache_config = base_rollout.CacheConfig(
      cache_size=kv_cache_size,
      num_layers=config.num_layers,
      num_kv_heads=config.num_kv_heads,
      head_dim=config.head_dim,
  )
  return vanilla_rollout.VanillaRollout(model, tokenizer, cache_config)


def main():
  preset = _env("MODEL_PRESET", "tiny")
  n_workers = int(_env("N_WORKERS", "4"))
  trainer_frac = float(_env("TRAINER_FRAC", "0.5"))
  rounds = int(_env("ROUNDS", "3"))
  max_new_tokens = int(_env("MAX_NEW_TOKENS", "32"))
  max_prompt_len = int(_env("MAX_PROMPT_LEN", "512"))
  batch = int(_env("BATCH", "2"))
  prompt = _env("PROMPT", _DEFAULT_PROMPT)
  reshard_backend = _env("RESHARD_BACKEND", "auto")
  engine = _env("ENGINE", "vanilla")  # vanilla | sglang_jax

  devices = jax.devices()
  trainer_devs, rollout_sets = _partition_devices(n_workers, trainer_frac)
  trainer_mesh = _mesh(trainer_devs)
  worker_meshes = [_mesh(s) for s in rollout_sets]

  print(
      f"=== rollout_workers_smoke preset={preset} n_workers={n_workers} "
      f"trainer_frac={trainer_frac} rounds={rounds} "
      f"max_new_tokens={max_new_tokens} batch={batch} "
      f"jax={jax.__version__} backend={jax.default_backend()} ===",
      flush=True,
  )
  print(
      f"[devices] total={len(devices)} "
      f"trainer={len(trainer_devs)} "
      f"workers={[len(s) for s in rollout_sets]}",
      flush=True,
  )

  # KV cache must cover prompt + generated tokens.
  kv_cache_size = max_prompt_len + max_new_tokens

  # ----- Trainer: ONE real model on the trainer mesh (the weights we sync). ---
  t0 = time.time()
  trainer_model, _, _ = _load_on_mesh(trainer_mesh, preset)
  model_bytes = _param_bytes(trainer_model)
  print(f"[trainer] model ready on {len(trainer_devs)} device(s) "
        f"in {(time.time()-t0):.1f}s param_bytes={model_bytes/1e9:.2f}GB",
        flush=True)

  # RolloutConfig is built BEFORE workers because SglangJaxRollout.__init__ needs
  # it (the sglang engine reads model_version/page_size/etc. at construction).
  reshard_fns = select_reshard_fns()  # AUTO: [pathways, jax_device] fallback.
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=max_new_tokens,
      max_prompt_length=max_prompt_len,
      temperature=0.7,
      top_p=1.0,
      kv_cache_size=kv_cache_size,
  )
  if engine == "sglang_jax":
    repo = _hf_repo(preset)
    if not repo:
      raise ValueError("ENGINE=sglang_jax needs a real (HF) preset for "
                       "model_version; got MODEL_PRESET=%r." % preset)
    rollout_config.rollout_sglang_jax_model_version = repo
    rollout_config.rollout_sglang_jax_context_length = max_prompt_len + max_new_tokens
    # Each worker gets its own GPU(s); init random then overwrite via the synced
    # nnx weights (load_checkpoint in SglangJaxRollout.__init__).
    rollout_config.rollout_sglang_jax_init_with_random_weights = True
    print(f"[engine] sglang_jax model_version={repo} "
          f"mem_fraction_static={rollout_config.rollout_sglang_jax_mem_fraction_static} "
          f"page_size={rollout_config.rollout_sglang_jax_page_size}", flush=True)

  # ----- N independent rollout workers, each on its own mesh. -----------------
  workers = []
  for wid, wmesh in enumerate(worker_meshes):
    tw = time.time()
    workers.append(
        _make_worker(wmesh, preset, kv_cache_size, engine, rollout_config))
    print(f"[worker {wid}] {engine} rollout ready on "
          f"{[d.id for d in wmesh.devices.flatten()]} "
          f"in {(time.time()-tw):.1f}s", flush=True)

  prompts = [prompt] * batch

  def sync_one(worker):
    t = time.time()
    # Cross-mesh reshard: trainer params (on trainer_mesh) -> this worker's mesh.
    # Vanilla takes reshard_fns; sglang_jax reshards internally (no reshard_fns).
    if engine == "sglang_jax":
      worker.update_params(nnx.state(trainer_model), filter_types=(nnx.Param,))
    else:
      worker.update_params(
          nnx.state(trainer_model),
          filter_types=(nnx.Param,),
          reshard_fns=reshard_fns,
      )
    return (time.time() - t) * 1000.0

  def rollout_one(worker):
    t = time.time()
    out = worker.generate(prompts, rollout_config)
    jax.block_until_ready(out.tokens)
    return out, (time.time() - t) * 1000.0

  pool = concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)
  all_ok = True
  rounds_data = []
  for r in range(rounds):
    print(f"\n--- round {r} (backend={reshard_backend}) ---", flush=True)

    # (1) SYNC: fan out the trainer->worker reshard across threads so the N
    #     independent cross-mesh reshards can overlap (like the bench fan-out).
    sync_futs = {pool.submit(sync_one, w): i for i, w in enumerate(workers)}
    sync_ms = {}
    for f in concurrent.futures.as_completed(sync_futs):
      sync_ms[sync_futs[f]] = f.result()

    # (2) ROLLOUT: each worker generates CONCURRENTLY on its own mesh+sampler.
    roll_futs = {pool.submit(rollout_one, w): i for i, w in enumerate(workers)}
    results = {}
    for f in concurrent.futures.as_completed(roll_futs):
      results[roll_futs[f]] = f.result()

    # (3) Per-worker report.
    produced = 0
    worker_rows = []
    for i in range(n_workers):
      out, gen_ms = results[i]
      text = out.text[0] if out.text else ""
      snippet = text.replace("\n", " ")[:48]
      n_gen = int(np.asarray(out.tokens[0]).shape[-1]) if out.tokens else 0
      ok = bool(out.text) and n_gen > 0
      produced += int(ok)
      all_ok = all_ok and ok
      devs = [d.id for d in worker_meshes[i].devices.flatten()]
      worker_rows.append({
          "worker": i,
          "devices": devs,
          "sync_ms": round(sync_ms[i], 1),
          "generate_ms": round(gen_ms, 1),
          "gen_tokens": n_gen,
          "ok": ok,
          "text_snippet": snippet,
      })
      print(
          f"[round {r}][worker {i}] devices={devs} "
          f"sync={sync_ms[i]:6.1f}ms generate={gen_ms:7.1f}ms "
          f"gen_tokens={n_gen} {'OK' if ok else 'FAIL'} "
          f"text={snippet!r}",
          flush=True,
      )
    rounds_data.append({"round": r, "produced": produced, "workers": worker_rows})
    print(
        f"[round {r}] AGGREGATE workers_produced={produced}/{n_workers} "
        f"{'ALL_OK' if produced == n_workers else 'INCOMPLETE'}",
        flush=True,
    )

  pool.shutdown(wait=True)
  status = "PASS" if all_ok else "FAIL"
  print(
      f"\n=== SMOKE {status}: {n_workers} workers x {rounds} rounds, "
      f"each synced from 1 trainer and generated independently ===",
      flush=True,
  )

  # ----- PERF SUMMARY: weight-sync GB/s + rollout tok/s (WARM = drop round 0). -
  # Round 0 is JIT-dominated (compile of reshard + sampler), so steady-state perf
  # excludes it when >1 round was run.
  warm = rounds_data[1:] if len(rounds_data) > 1 else rounds_data
  perf = {"model_bytes": model_bytes, "warm_rounds": len(warm),
          "per_worker": [], "agg_sync_gbps": 0.0, "agg_gen_tok_s": 0.0}
  print(f"\n--- PERF ({engine}, warm over {len(warm)} round(s), "
        f"model={model_bytes/1e9:.2f}GB, {batch} rollout(s)/sync/worker) ---",
        flush=True)
  for i in range(n_workers):
    syncs = [rd["workers"][i]["sync_ms"] for rd in warm]
    gens = [rd["workers"][i]["generate_ms"] for rd in warm]
    toks = [rd["workers"][i]["gen_tokens"] for rd in warm]
    mean_sync = sum(syncs) / len(syncs)
    mean_gen = sum(gens) / len(gens)
    gbps = (model_bytes / (mean_sync / 1e3)) / 1e9 if mean_sync else 0.0
    tok_s = (batch * (sum(toks) / len(toks))) / (mean_gen / 1e3) if mean_gen else 0.0
    perf["per_worker"].append({
        "worker": i, "mean_sync_ms": round(mean_sync, 1),
        "sync_gbps": round(gbps, 2), "mean_gen_ms": round(mean_gen, 1),
        "gen_tok_s": round(tok_s, 1)})
    perf["agg_sync_gbps"] += gbps
    perf["agg_gen_tok_s"] += tok_s
    print(f"[perf][worker {i}] sync {mean_sync:7.1f}ms = {gbps:6.2f} GB/s | "
          f"generate {mean_gen:7.1f}ms = {tok_s:7.1f} tok/s", flush=True)
  perf["agg_sync_gbps"] = round(perf["agg_sync_gbps"], 2)
  perf["agg_gen_tok_s"] = round(perf["agg_gen_tok_s"], 1)
  print(f"[perf][AGGREGATE] {n_workers} workers: sync {perf['agg_sync_gbps']} GB/s | "
        f"rollout {perf['agg_gen_tok_s']} tok/s", flush=True)

  # Durable artifact (CW logs are unreliable): write the result to s3 if asked.
  result_s3 = _env("RESULT_S3", "")
  if result_s3:
    _persist_result_s3(result_s3, {
        "status": status,
        "all_ok": all_ok,
        "engine": engine,
        "preset": preset,
        "n_workers": n_workers,
        "trainer_frac": trainer_frac,
        "gpus_per_worker": len(rollout_sets[0]) if rollout_sets else 0,
        "rounds": rounds,
        "max_new_tokens": max_new_tokens,
        "max_prompt_len": max_prompt_len,
        "batch": batch,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "device_kind": devices[0].device_kind if devices else "",
        "trainer_devices": [d.id for d in trainer_devs],
        "reshard_backend": reshard_backend,
        "model_bytes": model_bytes,
        "perf": perf,
        "rounds_data": rounds_data,
    })
  return 0 if all_ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
