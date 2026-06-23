# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The RL rollout parser must render byte-identically to the SFT/eval format.

Needs only the Qwen3 tokenizer (no weights/accelerator); skips if HF is offline.
Guards the train/eval prompt-consistency invariant: the stock QwenChatTemplateParser
injects a default "You are Qwen..." system turn, which the SFT data and eval
render_chatml never emit; TerminusQwenParser must suppress it and match exactly.
"""

import pytest
from transformers import AutoTokenizer

from mega_eval.eval.model_serving import render_chatml
from mega_eval.rl.chat_parser import TerminusQwenParser
from tunix.rl.agentic.parser.chat_template_parser.parser import QwenChatTemplateParser

TOKENIZER_REPO = "Qwen/Qwen3-8B"

_MSGS = [
    {"role": "user", "content": "PREAMBLE\n\n# Task\ndo the thing"},
    {"role": "assistant", "content": '{"commands":[{"cmd":"ls"}]}'},
    {"role": "user", "content": "file1\nfile2"},
]


@pytest.fixture(scope="module")
def tokenizer():
  try:
    return AutoTokenizer.from_pretrained(TOKENIZER_REPO)
  except Exception as e:  # offline / no network
    pytest.skip(f"cannot load {TOKENIZER_REPO} tokenizer: {e}")


def test_stock_parser_injects_system_turn(tokenizer):
  # Documents the bug we're working around: the stock parser adds a system turn.
  stock = QwenChatTemplateParser(tokenizer, enable_thinking=True).parse(
      _MSGS, add_generation_prompt=True, is_first_msg=True
  )
  assert "You are Qwen" in stock


def test_terminus_parser_matches_render_chatml(tokenizer):
  ref = render_chatml(_MSGS, add_generation_prompt=True)
  fixed = TerminusQwenParser(tokenizer, enable_thinking=True).parse(
      _MSGS, add_generation_prompt=True, is_first_msg=True
  )
  assert "You are Qwen" not in fixed
  assert fixed == ref  # byte-identical to the SFT/eval rendering
