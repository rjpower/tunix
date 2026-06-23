"""Trainer-side Dr.GRPO optimizer step for the disaggregated loop.

Reuses tunix's registered GRPO loss + advantage estimator to run ONE optimizer
step on trajectories received from the rollout workers -- no in-process RLCluster
generation. The key simplification (from the seam analysis):

  Dr.GRPO + beta=0 + num_iterations=1  =>  NO reference model AND NO old-logp pass.
    * ref_per_token_logps=None  -> the KL term is gated off in the loss.
    * old_per_token_logps=None  -> the loss uses stop_gradient(cur_logps) as the
      PPO baseline (ratio == 1 on the first/only iteration).

So a trajectory needs only prompt_ids, completion_ids and a scalar reward; the
trainer computes group advantages, builds the `TrainExample`, and steps. Matches
the in-process call in grpo_learner.py:179-186.
"""

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from tunix.rl import algo_core  # registers grpo loss + advantage estimators
from tunix.rl import common as rl_common
from tunix.rl import function_registry
from tunix.rl.grpo import drgrpo_learner

del algo_core  # imported only for its registration side effects.


def build_algo_config(num_generations, epsilon=0.2, beta=0.0, temperature=1.0):
  """Dr.GRPO config: drgrpo advantages, grpo loss, beta=0, single iteration.

  ``temperature`` must match the rollout sampling temperature: the loss reads
  ``algo_config.temperature`` to scale logits when recomputing per-token logps
  (the in-process learner sets it dynamically, grpo_learner.py:169). It is not a
  declared field, so we set it on the instance (the subclass has a ``__dict__``).
  """
  cfg = drgrpo_learner.DrGRPOConfig(
      num_generations=num_generations,
      num_iterations=1,
      beta=beta,
      epsilon=epsilon,
  )
  cfg.temperature = temperature
  return cfg


def build_optimizer(model, learning_rate, max_grad_norm=1.0):
  tx = optax.chain(
      optax.clip_by_global_norm(max_grad_norm),
      optax.adamw(learning_rate),
  )
  return nnx.Optimizer(model, tx, wrt=nnx.Param)


def build_train_example(
    prompt_ids, completion_ids, rewards, num_generations, advantage_estimator,
    pad_id, completion_mask=None,
):
  """Assemble a `TrainExample` (ref/old logps None for Dr.GRPO beta=0, iter=1).

  prompt_ids: [B*G, P] left-padded; completion_ids: [B*G, C] right-padded;
  rewards: [B*G] flat, grouped contiguously by prompt (G per prompt).

  ``completion_mask``: pass None for the single-turn case (derived as
  ``completion_ids != pad_id`` -- every generated token is trained on). For the
  AGENTIC multi-turn case pass the explicit assistant mask (1=model-generated,
  0=env observation), so the loss trains ONLY on the policy's own tokens and not
  on the injected tool outputs (which are non-pad but must not get a gradient).
  """
  prompt_ids = jnp.asarray(prompt_ids)
  completion_ids = jnp.asarray(completion_ids)
  prompt_mask = (prompt_ids != pad_id)
  if completion_mask is None:
    completion_mask = (completion_ids != pad_id).astype(jnp.int32)
  else:
    completion_mask = jnp.asarray(completion_mask).astype(jnp.int32)
  adv_fn = function_registry.get_advantage_estimator(advantage_estimator)
  advantages = jnp.asarray(
      np.asarray(adv_fn(np.asarray(rewards), num_generations), dtype=np.float32)
  )
  return rl_common.TrainExample(
      prompt_ids=prompt_ids,
      prompt_mask=prompt_mask,
      completion_ids=completion_ids,
      completion_mask=completion_mask,
      advantages=advantages,
      ref_per_token_logps=None,
      old_per_token_logps=None,
  )


def train_step(model, optimizer, train_example, algo_config, pad_id, eos_id):
  """One Dr.GRPO optimizer step; returns (loss, aux_metrics)."""
  policy_loss_fn = function_registry.get_policy_loss_fn(
      algo_config.policy_loss_fn
  )

  def _loss(m):
    return policy_loss_fn(
        m,
        train_example,
        algo_config=algo_config,
        pad_id=pad_id,
        eos_id=eos_id,
        compute_logps_chunk_size=0,
    )

  grad_fn = nnx.value_and_grad(_loss, has_aux=True)
  (loss, aux), grads = grad_fn(model)
  optimizer.update(model, grads)
  return float(loss), aux
