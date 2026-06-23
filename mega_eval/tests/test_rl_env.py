# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the RL env + agent (no TPU/docker).

Drives :class:`rl.environment.TerminalBenchEnv` + :class:`rl.agent.TerminusAgent`
with a scripted policy and the local (unsafe) sandbox, replaying the exact call
order the tunix collect engine uses: reset -> update_from_env -> [model ->
update_from_model -> env.step -> update_from_env]* -> final_reward_fn. This pins
the contract our classes must satisfy independent of the tunix engine internals.
"""

import json
import os

import pytest

os.environ.setdefault("OTA_SANDBOX", "local")  # before importing the env's sandbox

from mega_eval.eval.tb_tasks import TBTask  # noqa: E402
from mega_eval.rl import environment as rlenv  # noqa: E402
from mega_eval.rl.agent import TerminusAgent  # noqa: E402
from mega_eval.rl.environment import TerminalBenchEnv  # noqa: E402


def _fake_task(tmp_path) -> TBTask:
  return TBTask(
      task_id="fake-001",
      root=str(tmp_path),
      instruction="Write the word hi to /tmp/ota_hi.txt",
      agent_timeout_sec=60.0,
      verifier_timeout_sec=60.0,
      image_tag="unused:local",  # LocalUnsafeSandbox ignores the image
  )


def _action(commands, *, task_complete=False, prose=""):
  body = {"analysis": "a", "plan": "p", "commands": commands, "task_complete": task_complete}
  return prose + json.dumps(body)


def _drive(agent, env, scripted_responses):
  """Replay the collect engine's loop order against a scripted policy."""
  obs, _ = env.reset()
  agent.reset()
  agent.update_from_env(observation=obs, reward=0.0, done=False, info={})
  done = False
  for response in scripted_responses:
    if done:
      break
    action = agent.update_from_model(response).action
    obs, rew, done, info = env.step(action)
    agent.update_from_env(obs, rew, done, info)
  # Engine grades at episode end via the env's final_reward_fn.
  final_reward = env.final_reward_fn() if hasattr(env, "final_reward_fn") else 0.0
  env.close()
  return final_reward


def test_episode_runs_commands_and_completes(tmp_path):
  task = _fake_task(tmp_path)
  rlenv.register_tasks([task])
  env = TerminalBenchEnv({"task_id": "fake-001"}, group_id=0, pair_index=0, max_steps=5)
  agent = TerminusAgent(system_prompt="")

  responses = [
      _action([{"keystrokes": "echo hi\n", "duration": 0.1}]),
      _action([], task_complete=True),  # declare done
  ]
  _drive(agent, env, responses)

  roles = [m["role"] for m in agent.chat_completions]
  # folded-first-user, assistant, observation-user, assistant(complete)
  assert roles == ["user", "assistant", "user", "assistant"]
  assert agent.chat_completions[0]["content"].startswith("You are an AI assistant")
  assert "# Task" in agent.chat_completions[0]["content"]
  # the command's stdout came back as the next observation
  assert "hi" in agent.chat_completions[2]["content"]
  assert len(agent.trajectory.steps) == 2


def test_parse_failure_yields_reminder(tmp_path):
  task = _fake_task(tmp_path)
  rlenv.register_tasks([task])
  env = TerminalBenchEnv({"task_id": "fake-001"}, group_id=0, pair_index=0, max_steps=3)
  agent = TerminusAgent(system_prompt="")

  obs, _ = env.reset()
  agent.reset()
  agent.update_from_env(observation=obs, reward=0.0, done=False, info={})

  action = agent.update_from_model("totally not json").action
  assert action["parse_ok"] is False
  obs, rew, done, info = env.step(action)
  assert done is False and rew == 0.0
  assert "could not be parsed" in obs["prompts"]
  env.close()


def test_max_steps_terminates(tmp_path):
  task = _fake_task(tmp_path)
  rlenv.register_tasks([task])
  env = TerminalBenchEnv({"task_id": "fake-001"}, group_id=0, pair_index=0, max_steps=2)
  agent = TerminusAgent(system_prompt="")
  obs, _ = env.reset()
  agent.reset()
  agent.update_from_env(observation=obs, reward=0.0, done=False, info={})

  # Never declare complete; the env must truncate at max_steps.
  dones = []
  for _ in range(4):
    action = agent.update_from_model(_action([{"keystrokes": "echo x\n", "duration": 0.1}])).action
    obs, rew, done, info = env.step(action)
    agent.update_from_env(obs, rew, done, info)
    dones.append(done)
    if done:
      break
  assert dones[-1] is True
  assert env.step_count == 2
