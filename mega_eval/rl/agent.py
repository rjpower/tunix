# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Terminus-2 agent for tunix agentic RL.

A thin :class:`~tunix.rl.agentic.agents.base_agent.ConversationAgentBase` that
reuses the eval loop's parsing so RL rollouts are byte-for-byte the same shape the
policy was SFT'd / evaluated on:

  * **No system turn.** The SFT traces fold the Terminus-2 preamble into the first
    user message (their roles are ``[user, assistant, user, ...]``), so the agent
    starts with an empty transcript and the environment supplies that first
    message. Using a separate ChatML ``system`` turn here would be off-distribution.
  * **Assistant turns are the raw JSON action** (the policy's output -- trained),
    user turns are terminal observations (context -- masked). The tunix collect
    engine handles that masking via the ``QwenChatTemplateParser``.
  * :meth:`update_from_model` parses the action with the shared
    :func:`eval.agent_loop.parse_action` and hands the env a small command struct
    (``shell`` + ``task_complete`` + ``parse_ok``); the env executes the shell.
"""

import copy

from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents.base_agent import ConversationAgentBase

from mega_eval.eval.agent_loop import _commands_to_shell, parse_action


class TerminusAgent(ConversationAgentBase):
  """Terminus-2 policy wrapper for the tunix agentic rollout loop."""

  def _init_messages(self, system_prompt: str) -> None:
    # SFT traces have NO system role -- the preamble is folded into the first
    # user message, which the environment provides as the initial observation.
    # Start empty so the transcript matches the training distribution exactly.
    del system_prompt
    self._messages = []

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    """Parse the model's JSON action and record the step.

    Appends the raw response as the assistant turn (so the collect engine trains
    on it), parses it into a shell command struct, and returns that struct as the
    action passed to ``env.step``.
    """
    self.chat_completions.append({"role": "assistant", "content": response})
    parsed = parse_action(response)
    command = {
        "shell": _commands_to_shell(parsed.commands),
        "task_complete": parsed.task_complete,
        "parse_ok": parsed.parse_ok,
        "has_commands": bool(parsed.commands),
    }
    step = agent_types.Step(
        chat_completions=copy.deepcopy(self.chat_completions),
        action=agent_types.Action(action=command),
        model_response=response,
    )
    self.trajectory.steps.append(step)
    return agent_types.Action(action=command)
