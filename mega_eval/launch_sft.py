# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris entrypoint: SFT Qwen3-8B on OpenThoughts terminal-agent traces.

Stage 1 of the OpenThoughts-agent-on-Qwen3-8B build. Loads ``Qwen/Qwen3-8B``
(fp32 params for stable AdamW, bf16 compute, decoder remat + flash attention to
fit 8B at long sequence), SFTs it on ``open-thoughts/OpenThoughts-Agent-v1-SFT``
in real Qwen3 ChatML with assistant-turn loss masking, checkpoints the result to
``CKPT_DIR`` (orbax; pass a ``gs://`` path on iris), and prints a before/after
generation on a held-out terminal task so the uptake is visible in the job log.

Config via env:
  * ``AGENT_MODEL`` (qwen3-8b) -- registry key. ``AGENT_MODEL_DIR`` (./<name>).
  * ``SFT_STEPS`` (2000), ``BATCH_SIZE`` (8), ``LR`` (1e-5).
  * ``MAX_SEQ_LEN`` (8192), ``SEED`` (0), ``TP`` (1; set 2 on v6e-16 if OOM).
  * ``DATA_LIMIT`` (unset = all ~15.2k traces; set small for a smoke).
  * ``MIXTURE`` -- name of a registered multi-dataset blend
    (``swe_heavy`` | ``ota_only``); enables weighted multi-corpus SFT.
  * ``SFT_MIXTURE`` -- a JSON list of ``{repo_id, weight, format, ...}`` entries;
    an inline mixture spec that overrides ``MIXTURE``. (See
    ``agent_data/mixtures.sources_from_json``.) When either is set, ``DATA_LIMIT``
    is interpreted as a per-source cap; unset both for the single-corpus path.
  * ``PER_SOURCE_LIMIT`` -- explicit per-source row cap for a mixture (overrides
    the ``DATA_LIMIT`` fallback); small for a smoke.
  * ``CKPT_DIR`` (./checkpoints/<model>-agent-sft).
  * ``EVAL_TOKENS`` (384).
  * ``REMAT`` (decoder | block | none) -- activation rematerialization (decoder
    default; needed for 8B training memory since attention scores are seq^2/layer).
  * ``FLASH`` (0|1) -- TPU splash attention. Default OFF: tunix's splash kernel
    shard-maps the activation batch over ``fsdp``, so it needs BATCH_SIZE
    divisible by the fsdp axis and currently breaks for small batches.
  * ``EVAL_GEN`` (0|1) -- run an in-job before/after generation. Default OFF: the
    tunix Sampler mutates KV-cache Params, which conflicts with remat's trace
    level. Only enable with REMAT=none; otherwise eval in a separate job.

NOTE: ``BATCH_SIZE`` must be divisible by the fsdp axis size
(``device_count // TP``), since FSDP shards the data batch across that axis.

Submit on a v6e-16 (8B fp32 actor + AdamW wants the larger slice):

    uv run iris --cluster=marin job run --no-wait \
      --tpu v6e-16 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 8 --memory 200GB --disk 200GB --max-retries 1 --job-name ota-sft-qwen3-8b \
      -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \
      -e CKPT_DIR gs://<bucket>/openthoughts-agent/qwen3-8b-sft \
      -- python launch_sft.py
"""

import os

import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from tunix.generate import sampler as sampler_lib
from tunix.models.qwen3 import model as qm

from mega_eval.agent_data.mixtures import resolve_mixture
from mega_eval.models.registry import get_model_spec
from mega_eval.training.agent_sft import resolve_chatml_ids, run_agent_sft
from mega_eval.training.common import build_mesh, init_distributed, metrics_logging_options

# A held-out terminal task (not in the SFT stream) used only for a qualitative
# before/after generation -- mirrors the Terminus-2 single-action prompt shape.
HELD_OUT_TASK = (
    "You are an AI assistant solving command-line tasks in a Linux environment. "
    "Respond with a JSON object containing your analysis, plan, and the next "
    "shell command to run.\n\n"
    "Task: Count how many lines in /var/log/app.log contain the word ERROR and "
    "write the count to /workspace/error_count.txt."
)


def _ensure_model(repo: str, model_dir: str) -> str:
  if not os.path.exists(os.path.join(model_dir, "config.json")):
    snapshot_download(repo_id=repo, local_dir=model_dir)
  return model_dir


def _format_agent_prompt(tokenizer, user_text: str) -> str:
  """Renders a single-user-turn ChatML prompt ending at the assistant header."""
  return (
      f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
  )


def _generate(sampler, mesh, prompts, *, max_new, eos_id, im_end_id):
  with mesh:
    out = sampler(
        input_strings=prompts,
        max_generation_steps=max_new,
        max_prompt_length=1024,
        echo=False,
        # Stop on either the chat-turn end or the tokenizer eos.
        eos_tokens=[im_end_id, eos_id],
        temperature=0.7,
        top_p=1.0,  # REQUIRED: without top_p the tunix Sampler decodes greedily.
        seed=0,
    )
  return out.text


def main() -> None:
  init_distributed()  # must precede any jax call (orbax multi-host barriers)
  model_name = os.environ.get("AGENT_MODEL", "qwen3-8b")
  steps = int(os.environ.get("SFT_STEPS", "2000"))
  batch_size = int(os.environ.get("BATCH_SIZE", "8"))
  learning_rate = float(os.environ.get("LR", "1e-5"))
  max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "8192"))
  seed = int(os.environ.get("SEED", "0"))
  tp = int(os.environ.get("TP", "1"))
  data_limit = os.environ.get("DATA_LIMIT")
  data_limit = int(data_limit) if data_limit else None
  eval_tokens = int(os.environ.get("EVAL_TOKENS", "384"))

  # Multi-dataset weighted blend (DATA_PLAN.md). SFT_MIXTURE (inline JSON) wins
  # over MIXTURE (a registry name); neither set => single-corpus path below.
  mixture_json = os.environ.get("SFT_MIXTURE")
  mixture_name = os.environ.get("MIXTURE")
  sources = None
  per_source_limit = None
  if mixture_json or mixture_name:
    sources = resolve_mixture(mixture_json=mixture_json, mixture_name=mixture_name)
    psl = os.environ.get("PER_SOURCE_LIMIT") or os.environ.get("DATA_LIMIT")
    per_source_limit = int(psl) if psl else None
    data_limit = None  # DATA_LIMIT becomes the per-source cap for a blend
  remat = {
      "decoder": qm.RematConfig.DECODER,
      "block": qm.RematConfig.BLOCK,
      "none": qm.RematConfig.NONE,
  }[os.environ.get("REMAT", "decoder").lower()]
  use_flash = os.environ.get("FLASH", "0") == "1"
  eval_gen = os.environ.get("EVAL_GEN", "0") == "1"

  fsdp = jax.device_count() // tp
  if batch_size % fsdp != 0:
    raise ValueError(
        f"BATCH_SIZE={batch_size} must be divisible by the fsdp axis "
        f"({fsdp} = device_count {jax.device_count()} // TP {tp})."
    )

  model_spec = get_model_spec(model_name)
  model_dir = os.environ.get("AGENT_MODEL_DIR") or f"./{model_spec.name}"
  checkpoint_dir = os.environ.get("CKPT_DIR") or f"./checkpoints/{model_spec.name}-agent-sft"

  mixture_desc = (
      f"json[{len(sources)} sources]" if mixture_json
      else (mixture_name if mixture_name else "single:ota-v1")
  )
  print(f"[ota-sft] jax {jax.__version__} devices={jax.devices()}", flush=True)
  print(
      f"[ota-sft] model={model_spec.name} repo={model_spec.repo} steps={steps} "
      f"bs={batch_size} lr={learning_rate} max_seq_len={max_seq_len} tp={tp} "
      f"data_limit={data_limit} mixture={mixture_desc} "
      f"per_source_limit={per_source_limit} ckpt={checkpoint_dir}",
      flush=True,
  )

  _ensure_model(model_spec.repo, model_dir)
  mesh = build_mesh(tp=tp)
  tokenizer = model_spec.load_tokenizer(model_dir)
  # fp32 params (8e-5/1e-5 AdamW updates are below bf16 ULP for unit-scale weights),
  # bf16 compute, decoder remat + flash attention to fit 8B at long sequence.
  model = model_spec.load_model(
      model_dir,
      mesh=mesh,
      dtype=jnp.bfloat16,
      param_dtype=jnp.float32,
      remat=remat,
      use_flash_attention=use_flash,
  )
  print("[ota-sft] LOAD OK", flush=True)

  # In-job before/after generation is a qualitative nicety, but the tunix Sampler
  # mutates KV-cache Params during decode, which conflicts with remat's trace level
  # ("Cannot mutate Param from a different trace level"). Since the SFT job uses
  # remat for 8B memory, generation is OFF by default here; the dedicated eval job
  # loads the checkpoint with remat=NONE for sampling. Set EVAL_GEN=1 to force it
  # (only valid with REMAT=none).
  if eval_gen:
    im_start_id, im_end_id, _ = resolve_chatml_ids(tokenizer)
    eos_id = int(tokenizer.eos_token_id)
    cache_config = sampler_lib.CacheConfig(
        cache_size=1024 + eval_tokens + 16,
        num_layers=model.config.num_layers,
        num_kv_heads=model.config.num_kv_heads,
        head_dim=model.config.head_dim,
    )
    prompts = [_format_agent_prompt(tokenizer, HELD_OUT_TASK)]
    sampler = sampler_lib.Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)
    before = _generate(sampler, mesh, prompts, max_new=eval_tokens, eos_id=eos_id, im_end_id=im_end_id)
    print(f"[ota-sft] BEFORE-SFT action:\n{before[0][:600]!r}", flush=True)

  # ---- SFT ----
  metrics = metrics_logging_options(
      os.environ.get("RUN_NAME", f"{model_spec.name}-agent-sft"),
      config={
          "stage": "sft", "model": model_spec.name, "steps": steps,
          "batch_size": batch_size, "lr": learning_rate, "max_seq_len": max_seq_len,
          "tp": tp, "remat": os.environ.get("REMAT", "decoder"), "flash": use_flash,
          "mixture": mixture_desc,
      },
  )
  model = run_agent_sft(
      model, tokenizer,
      steps=steps,
      batch_size=batch_size,
      learning_rate=learning_rate,
      mesh=mesh,
      max_seq_len=max_seq_len,
      seed=seed,
      limit=data_limit,
      sources=sources,
      per_source_limit=per_source_limit,
      checkpoint_dir=checkpoint_dir,
      metrics_options=metrics,
  )

  if eval_gen:
    sampler = sampler_lib.Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)
    after = _generate(sampler, mesh, prompts, max_new=eval_tokens, eos_id=eos_id, im_end_id=im_end_id)
    print(f"[ota-sft] AFTER-SFT action:\n{after[0][:600]!r}", flush=True)
  print(f"[ota-sft] SFT COMPLETE (model={model_spec.name} steps={steps} ckpt={checkpoint_dir})", flush=True)


if __name__ == "__main__":
  main()
