# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chat parser for RL rollouts that matches the SFT/eval prompt rendering.

tunix's stock :class:`QwenChatTemplateParser` injects a default
``"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`` system
turn whenever the first message isn't a ``system`` role. But the OpenThoughts
agent SFT data and the eval loop (``eval.model_serving.render_chatml`` /
``training.agent_sft.encode_agent_conversation``) render **only the roles
present** — the Terminus-2 preamble is folded into the turn-0 *user* content, with
no system turn. Using the stock parser at RL rollout time would feed the policy a
system preamble it was never SFT'd on (off-distribution from both SFT and eval),
which can suppress the reward spread Dr.GRPO needs and means RL gains may not
transfer to the (system-less) eval.

:class:`TerminusQwenParser` overrides only the first-message handling to suppress
that injection; everything else (token defitions, assistant masking) is the stock
Qwen behavior. With Qwen3's ``bos_token is None`` the base ``_handle_first_message``
contributes nothing, so the rendered prompt is **byte-identical** to
``render_chatml`` (verified in tests/test_chat_parser.py).
"""

from typing import Dict, List

from tunix.rl.agentic.parser.chat_template_parser.parser import QwenChatTemplateParser


class TerminusQwenParser(QwenChatTemplateParser):
  """Qwen ChatML parser with NO default-system injection (matches SFT/eval)."""

  def _handle_first_message(self, messages: List[Dict[str, str]]) -> str:
    # Base behavior: just the bos token (None/empty for Qwen3 -> contributes
    # nothing), instead of QwenChatTemplateParser's "You are Qwen..." system turn.
    del messages
    return self.tokens.bos_token
