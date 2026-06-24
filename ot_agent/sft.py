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
import optax
import orbax.checkpoint as ocp

from tunix.sft.peft_trainer import PeftTrainer, TrainingConfig

from mega_eval.training.common import clipped_adamw, sft_model_input_fn


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
          metrics_logging_options=metrics_options,
          checkpoint_root_directory=checkpoint_dir,
          checkpointing_options=checkpointing_options,
      ),
  )
  trainer.with_gen_model_input_fn(sft_model_input_fn)
  print(
      f"[ota-sft] steps={steps} lr={learning_rate} ckpt={checkpoint_dir} "
      f"process={jax.process_index()}/{jax.process_count()}",
      flush=True,
  )
  with mesh:
    trainer.train(dataset)
  print("[ota-sft] training complete", flush=True)
  return model
