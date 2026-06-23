"""Shared training glue: optimizer, device mesh, and the PeftTrainer input fn.

Lifted from the proven ``tunix-delphi-rl`` plumbing. The only generalisation is
:func:`build_mesh`, which now factorizes the local devices into a 2-D
``(fsdp, tp)`` mesh so the 8B actor can use tensor parallelism on a v6e-16 if
pure FSDP runs out of HBM.
"""

import os

import jax
import numpy as np
import optax

from tunix.sft import utils as sft_utils
from tunix.sft.metrics_logger import MetricsLoggerOptions


def init_distributed() -> None:
  """Bring up the JAX distributed client (multi-host TPU), before any other jax call.

  Required on multi-host slices (v6e-8/-16 span 2/4 hosts): orbax checkpoint
  barriers call into the distributed client and otherwise raise "Distributed
  system is not available". On TPU this auto-detects the coordinator from the
  slice metadata; on a single host it's a harmless no-op. Idempotent: a second
  call (or a harness that already initialized) is ignored.
  """
  try:
    jax.distributed.initialize()
  except RuntimeError as e:  # already initialized
    print(f"[dist] jax.distributed already initialized: {e}", flush=True)


def metrics_logging_options(
    run_name: str,
    *,
    config: dict | None = None,
    log_dir: str | None = None,
    flush_every_n_steps: int = 20,
) -> MetricsLoggerOptions | None:
  """Opt-in tunix metrics logging (wandb + tensorboard), or ``None`` if unconfigured.

  Both the SFT ``PeftTrainer`` and the RL learner take a
  ``metrics_logging_options`` on their training config; threading the same builder
  through both gives loss/reward curves with one switch. Returns ``None`` (logging
  off, stdout only) unless ``WANDB_PROJECT`` or a log dir is set, so a plain run
  never depends on wandb being reachable.

  Config via env:
    * ``WANDB_PROJECT`` -- wandb project; enables the wandb backend (needs
      ``WANDB_API_KEY``, which the iris job inherits).
    * ``TB_LOG_DIR`` (or the ``log_dir`` arg) -- tensorboard dir (local or
      ``gs://``); also the dir tunix's logger needs even for wandb-only.

  Args:
    run_name: the wandb run name (e.g. ``"ota-sft-qwen3-8b"``).
    config: hyperparameters to record on the run.
    log_dir: tensorboard log dir; falls back to ``TB_LOG_DIR``.
    flush_every_n_steps: metric flush cadence.
  """
  project = os.environ.get("WANDB_PROJECT")
  log_dir = log_dir or os.environ.get("TB_LOG_DIR")
  if not project and not log_dir:
    return None
  backend_kwargs: dict = {}
  if project:
    backend_kwargs["wandb"] = {"config": config or {}}
  # tunix's MetricsLoggerOptions requires a log_dir; default to a job-local path
  # (tensorboard backend writes there; harmless if only wandb is wanted).
  return MetricsLoggerOptions(
      log_dir=log_dir or f"/tmp/tb/{run_name}",
      project_name=project or "openthoughts-agent",
      run_name=run_name,
      flush_every_n_steps=flush_every_n_steps,
      backend_kwargs=backend_kwargs,
  )


def clipped_adamw(learning_rate: float) -> optax.GradientTransformation:
  """Global-norm-clipped AdamW (b1=0.9, b2=0.99, wd=0.0).

  The clip is LOAD-BEARING, not a nicety: an occasional exploding update produces
  ``inf``/``NaN`` grads that crash the TPU run with a libtpu ``SIGSEGV``
  mid-training (a hard lesson from the tunix-delphi-rl runs). Clipping the global
  norm to 1.0 bounds the update and keeps the run alive.
  """
  return optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(learning_rate=learning_rate, b1=0.9, b2=0.99, weight_decay=0.0),
  )


def build_mesh(tp: int = 1) -> jax.sharding.Mesh:
  """Builds a 2-D ``(fsdp, tp)`` mesh over all local devices.

  Args:
    tp: tensor-parallel width. ``tp=1`` gives pure FSDP across every device
      (the default). For an 8B actor on a v6e-16, ``tp=2`` keeps each tensor
      shard larger while still sharding the optimizer state over ``fsdp``.

  Returns:
    A ``jax.sharding.Mesh`` with axis names ``("fsdp", "tp")``. The tunix Qwen3
    ``ShardingConfig`` references both axes, so ``tp>1`` activates tensor
    parallelism with no model-code change.

  Raises:
    ValueError: if ``tp`` does not divide the device count.
  """
  ndev = jax.device_count()
  if ndev % tp != 0:
    raise ValueError(f"tp={tp} does not divide device_count={ndev}.")
  fsdp = ndev // tp
  devices = np.asarray(jax.devices()).reshape(fsdp, tp)
  return jax.sharding.Mesh(devices, axis_names=("fsdp", "tp"))


def sft_model_input_fn(batch: dict) -> dict:
  """Expands a batched SFT row into PeftTrainer ``_default_loss_fn`` kwargs.

  ``input_mask`` is the LOSS mask (which tokens to train on). ``positions`` and
  the ``[B, L, L]`` causal ``attention_mask`` are derived from the separate
  PADDING mask (real tokens vs right-padding), matching the rollout loss path.
  """
  pad_mask = batch["pad_mask"]
  return {
      "input_tokens": batch["input_tokens"],
      "input_mask": batch["loss_mask"],
      "positions": sft_utils.build_positions_from_mask(pad_mask),
      "attention_mask": sft_utils.make_causal_attn_mask(pad_mask),
  }
