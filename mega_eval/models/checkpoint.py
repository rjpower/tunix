"""Restore an SFT'd Qwen3 from a tunix/orbax checkpoint for eval or RL.

``training.agent_sft.run_agent_sft`` checkpoints via tunix's ``PeftTrainer``,
which writes an orbax ``Composite{model_params, optimizer_state}`` under the
checkpoint root (one subdir per step). :func:`restore_sft_model` rebuilds the
Qwen3 structure from the base HF config (giving the correct param tree + target
sharding) and lets tunix's ``CheckpointManager.maybe_restore`` overwrite the
weights with the SFT'd params (it reshards to the current mesh as needed).

Eval/RL load with ``remat=NONE`` and flash off so the KV-cache sampler works
(remat conflicts with the sampler's in-place cache mutation).
"""

import jax
import jax.numpy as jnp
from tunix.models.qwen3 import model as qm
from tunix.sft import checkpoint_manager

from mega_eval.models.qwen3_loader import load_qwen3


def restore_sft_model(
    base_model_dir: str,
    checkpoint_dir: str,
    *,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype = jnp.float32,
    step: int | None = None,
) -> qm.Qwen3:
  """Loads the base Qwen3 structure then restores the SFT params over it.

  Args:
    base_model_dir: snapshot dir of the base HF model (for config + structure).
    checkpoint_dir: orbax checkpoint root passed to the SFT trainer.
    mesh: device mesh for sharding (eval uses the same builder as training).
    dtype: compute dtype.
    param_dtype: storage dtype (fp32 to match the SFT'd params).
    step: checkpoint step to restore (None = latest).

  Returns:
    The Qwen3 model with SFT'd params.

  Raises:
    RuntimeError: if no checkpoint is found to restore.
  """
  model = load_qwen3(
      base_model_dir,
      mesh=mesh,
      dtype=dtype,
      param_dtype=param_dtype,
      remat=qm.RematConfig.NONE,
      use_flash_attention=False,
  )
  cm = checkpoint_manager.CheckpointManager(root_directory=checkpoint_dir)
  restored_step, _ = cm.maybe_restore(model, step=step)
  if restored_step == 0:
    raise RuntimeError(f"No checkpoint found under {checkpoint_dir!r}.")
  print(f"[checkpoint] restored SFT params from step {restored_step}", flush=True)
  return model
