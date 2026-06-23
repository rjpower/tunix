# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Terminal-Bench environment for tunix agentic RL.

One :class:`~tunix.rl.agentic.environments.base_environment.BaseTaskEnv` per
rollout: it boots the task's prebuilt Docker image under gVisor, executes the
agent's shell each turn, and feeds the terminal output back as the next
observation -- the same loop as ``eval.agent_loop.run_episode``, but expressed as
``reset``/``step`` so the tunix collect engine can drive it.

Reward is **sparse and terminal**: every step returns reward 0.0; at episode end
the collect engine calls :attr:`final_reward_fn`, which grades the container's
final state (``tests/test.sh`` -> reward in [0, 1]). With ``reward_fns=None`` the
agentic reward manager uses this trajectory reward directly as the GRPO reward.

Heavyweight :class:`~eval.tb_tasks.TBTask` objects (paths, timeouts) are kept in a
module-level registry keyed by ``task_id`` rather than threaded through the
dataset arrays; :func:`register_tasks` populates it before training.
"""

import numpy as np

from tunix.rl.agentic.environments.base_environment import BaseTaskEnv, EnvStepResult

from mega_eval.eval.agent_loop import SYSTEM_PROMPT, format_observation
from mega_eval.eval.grade import grade_task
from mega_eval.eval.sandbox import make_sandbox

# Mirrors eval.agent_loop's parse-failure reminder so the policy sees the same
# corrective message it did during eval when it emits unparseable output.
_PARSE_FAIL_MSG = (
    "Your response could not be parsed as a JSON action. Respond with a single "
    "JSON object containing analysis, plan, and commands."
)

# task_id -> TBTask, populated by register_tasks() at launch (after image prebuild).
_TASKS: dict = {}


def register_tasks(tasks) -> None:
  """Register TBTasks so envs can look them up by id (call once at launch)."""
  for t in tasks:
    _TASKS[t.task_id] = t


def _as_scalar(value):
  """Unwrap a possibly batched/encoded dataset cell to a python scalar."""
  if isinstance(value, np.ndarray):
    value = value.reshape(-1)[0]
  elif isinstance(value, (list, tuple)):
    value = value[0]
  if isinstance(value, bytes):
    return value.decode("utf-8")
  if isinstance(value, np.generic):
    return value.item()
  return value


class TerminalBenchEnv(BaseTaskEnv):
  """A single Terminal-Bench episode as a tunix agentic environment."""

  def __init__(
      self,
      single_example,
      *,
      group_id=None,
      pair_index=None,
      max_steps: int = 20,
      command_timeout: float = 60.0,
      **kwargs,
  ):
    task_id = _as_scalar(single_example["task_id"])
    self.tbtask = _TASKS[task_id]
    # env.task must be a mutable dict with "prompts" (the learner stamps
    # policy_version onto it and merges it for logging).
    super().__init__(
        task={"task_id": task_id, "prompts": self.tbtask.instruction},
        max_steps=max_steps,
        group_id=group_id,
        pair_index=pair_index,
        **kwargs,
    )
    self.command_timeout = command_timeout
    self.sandbox = None
    # The collect engine calls this (no args) at episode end and adds it to the
    # last step's reward -> the trajectory reward Dr.GRPO consumes.
    self.final_reward_fn = self._grade

  def _initial_observation(self):
    """Boot the task container and return the folded Terminus-2 first turn."""
    self.sandbox = make_sandbox(image=self.tbtask.image_tag)
    folded = f"{SYSTEM_PROMPT}\n\n# Task\n{self.tbtask.instruction}"
    return {"prompts": folded}

  def _step_impl(self, action) -> EnvStepResult:
    """Execute one agent action; reward stays 0 until terminal grading."""
    if not action or not action.get("parse_ok"):
      return EnvStepResult({"prompts": _PARSE_FAIL_MSG}, 0.0, False, {"parse_ok": False})
    if action.get("task_complete") or not action.get("has_commands"):
      # Agent declared done (or emitted no commands). Terminal; grading happens
      # in final_reward_fn against the current container state. Observation is
      # None so no trailing empty user turn is appended (the base agent skips
      # None observations; the engine excludes terminal env turns anyway).
      return EnvStepResult(None, 0.0, True, {"done_reason": "complete"})
    res = self.sandbox.exec(action["shell"], timeout=self.command_timeout)
    obs = format_observation(res.stdout, res.stderr)
    return EnvStepResult({"prompts": obs}, 0.0, False, {})

  def _grade(self) -> float:
    """Grade the container's final state -> reward in [0, 1] (0.0 on any error)."""
    if self.sandbox is None:
      return 0.0
    try:
      return float(grade_task(self.sandbox, self.tbtask).score)
    except Exception as e:  # never let grading crash a rollout
      print(f"[rl-env] grade failed for {self.tbtask.task_id}: {e}", flush=True)
      return 0.0

  def close(self) -> None:
    if self.sandbox is not None:
      self.sandbox.close()
      self.sandbox = None
