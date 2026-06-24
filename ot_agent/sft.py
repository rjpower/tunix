# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-process SFT driver: PeftTrainer over a process-sharded dataset.

Thin wrapper around tunix's ``PeftTrainer`` that takes a **pre-built**
process-local grain dataset (see :func:`ot_agent.data.build_sharded_sft_dataset`)
rather than building one internally. Everything else mirrors
``mega_eval.training.agent_sft.run_agent_sft`` -- the global-norm-clipped AdamW
(load-bearing, not a nicety), the assistant-turn loss-mask input fn, and orbax
checkpointing -- and reuses those exact helpers so there is a single source of
truth for the optimizer and the loss-mask plumbing.

The actor must already be FSDP/TP-sharded on ``mesh``. Training runs under
``with mesh:`` so the optimizer state shards on construction and each step's
host batch is gathered into a global array across all JAX processes by
``PeftTrainer``'s ``shard_input`` (multi-host safe).
"""

from __future__ import annotations

from typing import Any

import grain.python as grain
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from tunix.sft.peft_trainer import PeftTrainer, TrainingConfig

from mega_eval.training.common import clipped_adamw, sft_model_input_fn


def low_mem_cross_entropy_loss(
    model,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
    images: jax.Array | None = None,
):
  """Memory-lean drop-in for tunix's ``_default_loss_fn`` (identical math).

  The default loss builds ``jax.nn.one_hot(targets, V)`` *and* a full
  ``jax.nn.log_softmax(logits)`` -- two ``[B, S, V]`` fp32 tensors. At Qwen3's
  ``V=151936`` and a long sequence those dominate HBM (the 32B train step's
  ~60 GB activation peak / 63 GiB single allocation that OOM'd 4x8 H100).

  ``log p(target) = logit[target] - logsumexp(logits)`` gives the same per-token
  NLL using only ``[B, S]`` reductions: ``logsumexp`` reduces over ``V`` (XLA
  fuses the transient) and ``take_along_axis`` gathers the target logit -- no
  ``[B, S, V]`` one-hot, no materialized ``log_softmax``. Equivalent to
  ``optax.softmax_cross_entropy(logits, one_hot).mean()`` over masked tokens;
  ``test_low_mem_loss.py`` pins value+gradient parity against the default.
  """
  kwargs = {} if images is None else {"images": images}
  logits, _ = model(input_tokens, positions, None, attention_mask, **kwargs)

  logits = logits[:, :-1, :]
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]

  log_z = jax.nn.logsumexp(logits, axis=-1)  # [B, S-1]
  target_logit = jnp.take_along_axis(
      logits, target_tokens[..., None], axis=-1
  )[..., 0]  # [B, S-1]
  token_logp = target_logit - log_z  # = log_softmax(logits)[target]

  norm_factor = 1.0 / (jnp.sum(target_mask) + 1e-8)
  return -jnp.sum(token_logp * target_mask.astype(token_logp.dtype)) * norm_factor


def build_optimizer(peak_lr: float, total_steps: int, warmup_ratio: float = 0.0):
  """Global-norm-clipped AdamW, optionally with a cosine+warmup LR schedule.

  ``warmup_ratio <= 0`` reuses ``mega_eval``'s constant-LR ``clipped_adamw`` (the
  proven default). ``warmup_ratio > 0`` matches the OpenThoughts-Agent paper
  recipe (lr_scheduler_type=cosine, warmup_ratio 0.1): linear warmup over
  ``warmup_ratio * total_steps`` then cosine decay to ~0. The global-norm clip is
  load-bearing (unclipped AdamW can NaN and crash), so it wraps both paths.
  """
  if warmup_ratio and warmup_ratio > 0.0:
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, b1=0.9, b2=0.99, weight_decay=0.0),
    )
  return clipped_adamw(peak_lr)


def run_sharded_sft(
    model,
    tokenizer,  # accepted for symmetry / future use; encoding happens upstream
    *,
    dataset: grain.MapDataset,
    steps: int,
    learning_rate: float,
    mesh: jax.sharding.Mesh,
    warmup_ratio: float = 0.0,
    grad_accum: int = 1,
    low_mem_loss: bool = True,
    checkpoint_dir: str | None = None,
    save_interval_secs: int = 600,
    max_to_keep: int = 2,
    metrics_options=None,
) -> Any:
  """SFTs ``model`` in place on a pre-sharded dataset; optionally checkpoints.

  Args:
    model: the Qwen3 ``nnx`` actor (fp32 params), already sharded on ``mesh``.
    tokenizer: the Qwen3 HF tokenizer (unused here; kept for call symmetry).
    dataset: this process's batched grain dataset (per-process batch rows/step).
    steps: number of SFT optimizer steps.
    learning_rate: AdamW lr (clipped at global-norm 1.0).
    mesh: the device mesh the model is sharded on.
    warmup_ratio: >0 enables cosine LR + linear warmup (paper recipe); 0 = const.
    grad_accum: gradient-accumulation steps. ``steps`` counts OPTIMIZER updates;
      tunix runs ``steps * grad_accum`` microbatches, so the effective (global)
      batch = per-step global batch * grad_accum. Lets a small per-device
      microbatch (which bounds the activation peak) reach a large effective batch.
    low_mem_loss: use :func:`low_mem_cross_entropy_loss` (default) instead of
      tunix's one-hot loss -- required to fit 32B's loss over the 152k vocab.
    checkpoint_dir: orbax checkpoint root (local or ``gs://``); ``None`` disables.
      NB: on the CW GPU cluster there is no shared filesystem across nodes, so a
      multi-node run should checkpoint to ``None`` here and rely on the post-run
      HF-safetensors export (``ot_agent.export_hf``) for the coherent artifact.
    save_interval_secs: minimum seconds between periodic checkpoints.
    max_to_keep: number of checkpoints to retain.
    metrics_options: optional tunix metrics logging (wandb/tensorboard).

  Returns:
    The same ``model`` object, now SFT'd.
  """
  optimizer = build_optimizer(learning_rate, steps, warmup_ratio)

  checkpointing_options = None
  if checkpoint_dir:
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_decision_policy=ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
            minimum_interval_secs=save_interval_secs,
        ),
        max_to_keep=max_to_keep,
    )

  trainer = PeftTrainer(
      model=model,
      optimizer=optimizer,
      training_config=TrainingConfig(
          eval_every_n_steps=10**9,
          max_steps=steps,
          gradient_accumulation_steps=grad_accum if grad_accum > 1 else None,
          metrics_logging_options=metrics_options,
          checkpoint_root_directory=checkpoint_dir,
          checkpointing_options=checkpointing_options,
      ),
  )
  trainer.with_gen_model_input_fn(sft_model_input_fn)
  if low_mem_loss:
    trainer.with_loss_fn(low_mem_cross_entropy_loss, has_aux=False)
  print(
      f"[ota-sft] steps={steps} lr={learning_rate} grad_accum={grad_accum} "
      f"low_mem_loss={low_mem_loss} ckpt={checkpoint_dir} "
      f"process={jax.process_index()}/{jax.process_count()}",
      flush=True,
  )
  with mesh:
    trainer.train(dataset)
  print("[ota-sft] training complete", flush=True)
  return model
