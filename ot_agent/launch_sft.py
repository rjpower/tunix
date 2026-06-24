# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris entrypoint: SFT Qwen3-32B on the OpenThoughts-Agent 100K set (4x8 H100).

Stage 1 of the OT-Agent replication (arXiv:2606.24855). Loads ``Qwen/Qwen3-32B``
(fp32 params for stable AdamW, bf16 compute, decoder remat to fit 32B at long
sequence), SFTs it on ``open-thoughts/OpenThoughts-Agent-SFT-100K`` in real Qwen3
ChatML with assistant-turn loss masking, and writes a single HF-format
safetensors checkpoint (mirrored to R2) so the actor can be evaluated / RL'd /
served downstream.

Runs as **N iris tasks joined into one JAX world** (N=4 nodes x 8 H100 = 32
devices, one ``(fsdp=N, tp=8)`` mesh). Each task streams a disjoint shard of the
corpus and feeds ``BATCH_SIZE/N`` rows per step; tunix gathers them into the
global batch across the mesh. Submit with ``ot_agent/submit_sft.sh`` (multi-node
``iris job run --replicas N``). See ``ot_agent/README.md``.

Config via env (all optional unless noted):
  * ``AGENT_MODEL`` (qwen3-32b) -- registry key. ``AGENT_MODEL_DIR`` (./<name>).
  * ``DATASET`` (100k) -- OT-Agent ladder key (1k|3.16k|10k|31.6k|100k|
    coldstart-10k|v1) or a full HF dataset id. ``DATASET_REVISION`` (default branch).
  * ``SFT_STEPS`` (2000, OPTIMIZER steps), ``BATCH_SIZE`` (32, the GLOBAL
    per-step batch), ``LR`` (1e-5).
  * ``GRAD_ACCUM`` (1) -- gradient-accumulation steps. Effective (global) batch
    = ``BATCH_SIZE * GRAD_ACCUM``; the per-step ``BATCH_SIZE`` bounds the
    activation peak (use a small per-device microbatch + GRAD_ACCUM to reach a
    large effective batch on 32B without OOM).
  * ``LOW_MEM_LOSS`` (1) -- use the gather-based cross-entropy instead of tunix's
    one-hot loss; required to fit the 32B loss over the 152k vocab. Set 0 only
    to A/B against the stock loss on a small model.
  * ``CE_CHUNKS`` (0) -- if >0, use the chunked CE (this many sequence chunks):
    runs the body with skip_lm_head then projects+scores in chunks (rematerialized),
    so the full ``[B,S,V]`` logits never materialize. Lets the per-device
    microbatch grow at long context (e.g. seq 32768). Supersedes LOW_MEM_LOSS.
  * ``WARMUP_RATIO`` (0.0 = constant LR; the paper uses 0.1 = cosine schedule
    with linear warmup over the first 10% of steps then cosine decay to ~0).
  * ``MAX_SEQ_LEN`` (8192), ``SEED`` (0), ``TP`` (8; tensor-parallel width,
    must divide num_kv_heads=8 and the per-node device count).
  * ``DATA_LIMIT`` (unset = all rows) -- cap global rows scanned (smoke knob).
  * ``REMAT`` (decoder|block|none) -- activation rematerialization. ``FLASH``
    (0|1, default 0) -- flash attention. tunix now dispatches by platform: GPU
    -> cuDNN flash (``jax.nn.dot_product_attention``), TPU -> splash. REQUIRED on
    GPU for long context (>~8k): without it attention is materialized O(seq^2).
  * ``CKPT_DIR`` (unset) -- orbax checkpoint root. Leave UNSET on multi-node CW
    (no shared filesystem); use ``EXPORT_DIR`` instead. Useful for single-node /
    gs:// (TPU) resume.
  * ``EXPORT_DIR`` (unset) -- where to write the HF-format safetensors checkpoint
    after training (local dir or ``s3://marin-na/...`` for R2). The coherent
    downstream artifact; written from process 0 after gathering the actor.
  * ``WANDB_PROJECT`` / ``TB_LOG_DIR`` -- opt-in metrics (loss curve).

NOTE: ``BATCH_SIZE`` (global) must be divisible by both the fsdp axis
(device_count // TP) and the number of JAX processes.
"""

import os

import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from tunix.models.qwen3 import model as qm

# init_distributed() must run before any other jax call (multi-host barriers);
# import it eagerly but call it first thing in main().
from ot_agent.data import (
    build_sharded_sft_dataset,
    default_revision,
    per_process_batch_size,
    resolve_repo,
    rows_per_process,
)
from ot_agent.distributed import init_distributed
from ot_agent.sft import run_sharded_sft
from mega_eval.models.registry import get_model_spec
from mega_eval.training.common import build_mesh, metrics_logging_options

_REMAT = {
    "decoder": qm.RematConfig.DECODER,
    "block": qm.RematConfig.BLOCK,
    "none": qm.RematConfig.NONE,
}


def _ensure_model(repo: str, model_dir: str) -> str:
  if not os.path.exists(os.path.join(model_dir, "config.json")):
    snapshot_download(repo_id=repo, local_dir=model_dir)
  return model_dir


def main() -> None:
  init_distributed()  # FIRST: bring up the multi-node JAX world.

  model_name = os.environ.get("AGENT_MODEL", "qwen3-32b")
  dataset_key = os.environ.get("DATASET", "100k")
  dataset_revision = os.environ.get("DATASET_REVISION") or None
  steps = int(os.environ.get("SFT_STEPS", "2000"))
  global_batch = int(os.environ.get("BATCH_SIZE", "32"))
  grad_accum = int(os.environ.get("GRAD_ACCUM", "1"))
  learning_rate = float(os.environ.get("LR", "1e-5"))
  warmup_ratio = float(os.environ.get("WARMUP_RATIO", "0.0"))
  low_mem_loss = os.environ.get("LOW_MEM_LOSS", "1") == "1"
  ce_chunks = int(os.environ.get("CE_CHUNKS", "0"))
  max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "8192"))
  seed = int(os.environ.get("SEED", "0"))
  tp = int(os.environ.get("TP", "8"))
  data_limit = os.environ.get("DATA_LIMIT")
  data_limit = int(data_limit) if data_limit else None
  remat = _REMAT[os.environ.get("REMAT", "decoder").lower()]
  use_flash = os.environ.get("FLASH", "0") == "1"
  checkpoint_dir = os.environ.get("CKPT_DIR") or None
  export_dir = os.environ.get("EXPORT_DIR") or None

  process_count = jax.process_count()
  process_index = jax.process_index()
  device_count = jax.device_count()

  if device_count % tp != 0:
    raise ValueError(f"TP={tp} does not divide device_count={device_count}.")
  fsdp = device_count // tp
  if global_batch % fsdp != 0:
    raise ValueError(
        f"BATCH_SIZE={global_batch} (global) must be divisible by the fsdp axis "
        f"({fsdp} = device_count {device_count} // TP {tp})."
    )
  per_process_batch = per_process_batch_size(global_batch, process_count)
  # tunix runs steps*grad_accum microbatches; each consumes per_process_batch
  # rows/process, so the process must buffer that many. Effective (global) batch
  # = global_batch * grad_accum.
  n_per_process = rows_per_process(steps * grad_accum, per_process_batch)
  effective_batch = global_batch * grad_accum

  repo_id = resolve_repo(dataset_key)
  if dataset_revision is None:
    dataset_revision = default_revision(repo_id)  # pinned sha if known, else main
  model_spec = get_model_spec(model_name)
  model_dir = os.environ.get("AGENT_MODEL_DIR") or f"./{model_spec.name}"

  if process_index == 0:
    print(f"[ota-sft] jax {jax.__version__} processes={process_count} "
          f"devices={device_count} mesh=(fsdp={fsdp},tp={tp})", flush=True)
    print(
        f"[ota-sft] model={model_spec.name} repo={model_spec.repo} dataset={repo_id} "
        f"steps={steps} global_batch={global_batch} grad_accum={grad_accum} "
        f"effective_batch={effective_batch} per_process_batch={per_process_batch} "
        f"lr={learning_rate} warmup_ratio={warmup_ratio} low_mem_loss={low_mem_loss} "
        f"max_seq_len={max_seq_len} remat={os.environ.get('REMAT','decoder')} "
        f"flash={use_flash} data_limit={data_limit} ckpt={checkpoint_dir} export={export_dir}",
        flush=True,
    )

  _ensure_model(model_spec.repo, model_dir)
  mesh = build_mesh(tp=tp)
  tokenizer = model_spec.load_tokenizer(model_dir)

  # fp32 params (a 1e-5 AdamW update is below bf16 ULP for unit-scale weights, so
  # bf16 storage would silently zero most updates), bf16 compute, decoder remat to
  # fit 32B at long sequence. Sharded on the (fsdp, tp) mesh by the loader.
  model = model_spec.load_model(
      model_dir,
      mesh=mesh,
      dtype=jnp.bfloat16,
      param_dtype=jnp.float32,
      remat=remat,
      use_flash_attention=use_flash,
  )
  print(f"[ota-sft p{process_index}] LOAD OK", flush=True)

  dataset = build_sharded_sft_dataset(
      tokenizer,
      repo_id=repo_id,
      revision=dataset_revision,
      per_process_batch=per_process_batch,
      n_per_process=n_per_process,
      max_seq_len=max_seq_len,
      seed=seed,
      process_index=process_index,
      process_count=process_count,
      limit=data_limit,
  )

  metrics = metrics_logging_options(
      os.environ.get("RUN_NAME", f"{model_spec.name}-ot-agent-sft"),
      config={
          "stage": "sft", "model": model_spec.name, "dataset": repo_id,
          "steps": steps, "global_batch": global_batch, "grad_accum": grad_accum,
          "effective_batch": effective_batch, "lr": learning_rate,
          "warmup_ratio": warmup_ratio, "low_mem_loss": low_mem_loss,
          "max_seq_len": max_seq_len, "tp": tp, "fsdp": fsdp,
          "processes": process_count, "remat": os.environ.get("REMAT", "decoder"),
      },
  )

  model = run_sharded_sft(
      model, tokenizer,
      dataset=dataset,
      steps=steps,
      learning_rate=learning_rate,
      mesh=mesh,
      warmup_ratio=warmup_ratio,
      grad_accum=grad_accum,
      low_mem_loss=low_mem_loss,
      ce_chunks=ce_chunks,
      checkpoint_dir=checkpoint_dir,
      metrics_options=metrics,
  )

  if export_dir:
    # Lazy import: only the export path pulls safetensors/tensorstore.
    from ot_agent.export_hf import export_and_mirror  # noqa: PLC0415
    export_and_mirror(model, model_dir, export_dir, mesh=mesh)

  if process_index == 0:
    print(f"[ota-sft] SFT COMPLETE (model={model_spec.name} dataset={repo_id} "
          f"steps={steps} export={export_dir} ckpt={checkpoint_dir})", flush=True)


if __name__ == "__main__":
  main()
