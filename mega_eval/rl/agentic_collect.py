"""Standalone agentic trajectory collection for the disaggregated rollout.

Runs tunix's `TrajectoryCollectEngine` (TerminusAgent + TerminalBenchEnv +
multi-turn generation) driven by a plain `VanillaRollout`, WITHOUT an in-process
RLCluster. The engine is decoupled from RLCluster -- it needs only an agent, an
env, a `model_call` callable, a tokenizer and a chat parser -- so the only glue is
the model-call adapter below, which mirrors `agentic_rl_learner._model_call`:
render the running conversation via `TerminusQwenParser`, then `VanillaRollout.
generate` the next assistant turn (the parser already applied the chat template,
so generation is raw-string).

`collect(mode="Token")` returns a dict with the flattened multi-turn arrays the
trainer needs: `prompt_tokens`, `conversation_tokens`, `conversation_masks`
(1=assistant / 0=env observation -- the loss only trains on assistant tokens),
`trajectory_reward` (the env grader's [0,1] score), `old_logprobs`, etc.
"""

import asyncio

import numpy as np

from tunix.generate import tokenizer_adapter
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vanilla_rollout
from tunix.rl.agentic.trajectory import trajectory_collect_engine

from mega_eval.rl.agent import TerminusAgent
from mega_eval.rl.chat_parser import TerminusQwenParser
from mega_eval.rl.environment import TerminalBenchEnv
from mega_eval.rollout_workers_smoke import _load_on_mesh


def build_worker(mesh, preset, kv_cache_size):
  """Loads model+tokenizer once and returns (VanillaRollout, tokenizer, config).

  Mirrors `rollout_workers_smoke._make_worker` (vanilla branch) but also returns
  the tokenizer (the collect engine + parser need it), without double-loading.
  """
  model, tokenizer, config = _load_on_mesh(mesh, preset)
  cache_config = base_rollout.CacheConfig(
      cache_size=kv_cache_size,
      num_layers=config.num_layers,
      num_kv_heads=config.num_kv_heads,
      head_dim=config.head_dim,
  )
  worker = vanilla_rollout.VanillaRollout(model, tokenizer, cache_config)
  return worker, tokenizer, config


def make_chat_parser(tokenizer):
  return TerminusQwenParser(tokenizer, enable_thinking=True)


def adapt_tokenizer(tokenizer):
  """Wrap a raw HF tokenizer in tunix's TokenizerAdapter.

  The collect engine calls adapter-only methods (`dedup_bos_ids`, `encode`) that
  the raw HF tokenizer lacks; RLCluster does the same wrap (rl_cluster.py:279).
  The VanillaRollout and the chat parser take the RAW tokenizer (as launch_rl).
  """
  return tokenizer_adapter.TokenizerAdapter(tokenizer)


def eos_token_ids(tokenizer):
  """The stop tokens for a turn: <|im_end|> and the model eos."""
  ids = []
  im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
  if im_end is not None and im_end >= 0:
    ids.append(int(im_end))
  ids.append(int(tokenizer.eos_token_id))
  return ids


class VanillaModelCall:
  """Adapter: (chat_completions, env, max_generation_steps) -> RolloutOutput.

  Mirrors `agentic_rl_learner._model_call`: stamp policy_version onto env.task,
  render the conversation via the chat parser (chat template applied here), then
  raw-generate the next turn. `policy_version` is mutable so the loop can bump it
  when it pulls fresh weights.
  """

  def __init__(self, worker, chat_parser, *, max_prompt_length, kv_cache_size,
               temperature, top_p, eos_tokens):
    self._worker = worker
    self._parser = chat_parser
    self._max_prompt_length = max_prompt_length
    self._kv_cache_size = kv_cache_size
    self._temperature = temperature
    self._top_p = top_p
    self._eos_tokens = eos_tokens
    self.policy_version = 0

  def __call__(self, chat_completions, env=None, max_generation_steps=None, **_):
    if env is not None:
      env.task["policy_version"] = self.policy_version
    rendered = self._parser.parse(
        messages=chat_completions, add_generation_prompt=True, is_first_msg=True
    )
    rcfg = base_rollout.RolloutConfig(
        max_tokens_to_generate=int(max_generation_steps) if max_generation_steps else 512,
        max_prompt_length=self._max_prompt_length,
        temperature=self._temperature,
        top_p=self._top_p,
        kv_cache_size=self._kv_cache_size,
        eos_tokens=self._eos_tokens,
        return_logprobs=True,
    )
    return self._worker.generate([rendered], rcfg)


def pad_trajectory(traj, *, max_prompt_len, max_response_len, pad_id):
  """One "Token"-mode trajectory -> fixed-shape (prompt_ids, completion_ids, completion_mask).

  Left-pads the turn-0 prompt to ``max_prompt_len``; right-pads the flattened
  multi-turn completion + its assistant mask to ``max_response_len`` -- the exact
  layout `agentic_grpo_learner._process_results` feeds the loss. The mask is the
  ASSISTANT mask (1=model token, 0=env observation), shipped explicitly so the
  trainer trains only on the policy's own tokens.
  """
  prompt_tokens = np.asarray(traj["prompt_tokens"]).ravel()
  completion_tokens = np.asarray(traj["conversation_tokens"]).ravel()
  completion_mask = np.asarray(traj["conversation_masks"]).ravel()
  left_prompt, right_completion, _ = agentic_utils.pad_prompt_and_completion(
      prompt_tokens, completion_tokens, max_prompt_len, max_response_len, pad_id
  )
  padded_mask = agentic_utils.right_pad(completion_mask, max_response_len, 0)
  return (
      np.asarray(left_prompt, dtype=np.int32),
      np.asarray(right_completion[:max_response_len], dtype=np.int32),
      np.asarray(padded_mask[:max_response_len], dtype=np.int32),
  )


def collect_trajectory(
    *, worker, tokenizer, chat_parser, model_call, task_id, max_steps,
    max_response_length, command_timeout=60.0, episode_timeout=900.0,
    group_id=0, pair_index=0,
):
  """Run one agentic episode for `task_id`; returns the "Token"-mode dict."""
  agent = TerminusAgent(system_prompt="")
  env = TerminalBenchEnv(
      {"task_id": task_id},
      group_id=group_id,
      pair_index=pair_index,
      max_steps=max_steps,
      command_timeout=command_timeout,
  )
  engine = trajectory_collect_engine.TrajectoryCollectEngine(
      agent=agent,
      env=env,
      model_call=model_call,
      model_call_kwargs={},
      gamma=1.0,
      max_response_length=max_response_length,
      timeout=episode_timeout,
      tokenizer=tokenizer,
      chat_parser=chat_parser,
      filter_statuses=None,
      overlong_filter=False,
  )
  return asyncio.run(engine.collect(mode="Token"))
