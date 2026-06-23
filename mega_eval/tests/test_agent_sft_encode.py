"""CPU tests for the agent-SFT ChatML encoder + assistant-turn loss masking.

These need only the Qwen3 tokenizer (no model weights, no accelerator), so they
run in the default fast suite. If HF is unreachable the tokenizer fixture skips.
"""

import numpy as np
import pytest
from transformers import AutoTokenizer

from mega_eval.training.agent_sft import encode_agent_conversation, resolve_chatml_ids

TOKENIZER_REPO = "Qwen/Qwen3-8B"


@pytest.fixture(scope="module")
def tokenizer():
  try:
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
  except Exception as e:  # offline / no network
    pytest.skip(f"cannot load {TOKENIZER_REPO} tokenizer: {e}")
  if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
  return tok


@pytest.fixture
def messages():
  return [
      {"role": "user", "content": "You are an agent. Task: list files."},
      {"role": "assistant", "content": '{"plan": "ls", "command": "ls -la"}'},
      {"role": "user", "content": "Current terminal state: file1 file2"},
      {"role": "assistant", "content": '{"plan": "done", "command": "exit"}'},
  ]


def _decode_masked(tokenizer, ids, mask):
  """Decode only the tokens where mask==1."""
  toks = [int(t) for t, m in zip(ids, mask) if m > 0.5]
  return tokenizer.decode(toks)


def test_chatml_ids_resolve(tokenizer):
  im_start, im_end, newline = resolve_chatml_ids(tokenizer)
  # Qwen3 ChatML control tokens are stable ids.
  assert im_start == tokenizer.convert_tokens_to_ids("<|im_start|>")
  assert im_end == tokenizer.convert_tokens_to_ids("<|im_end|>")
  assert im_start != im_end
  assert len(newline) >= 1


def test_loss_mask_only_assistant_turns(tokenizer, messages):
  im_start, im_end, newline = resolve_chatml_ids(tokenizer)
  enc = encode_agent_conversation(
      tokenizer, messages, max_seq_len=256,
      im_start_id=im_start, im_end_id=im_end, newline_ids=newline,
  )
  assert enc is not None
  input_tokens, loss_mask, pad_mask = enc

  # The trained tokens must decode to exactly the two assistant bodies (content +
  # the closing <|im_end|>), and contain NO user content.
  trained = _decode_masked(tokenizer, input_tokens, loss_mask)
  assert '{"plan": "ls", "command": "ls -la"}' in trained
  assert '{"plan": "done", "command": "exit"}' in trained
  assert "list files" not in trained          # user turn 0 not trained
  assert "Current terminal state" not in trained  # observation not trained

  # Every trained span ends on an <|im_end|> (teaches the model to stop). There
  # are two assistant turns => exactly two trained <|im_end|> tokens.
  trained_ids = [int(t) for t, m in zip(input_tokens, loss_mask) if m > 0.5]
  assert trained_ids.count(im_end) == 2


def test_role_headers_are_masked(tokenizer, messages):
  im_start, im_end, newline = resolve_chatml_ids(tokenizer)
  enc = encode_agent_conversation(
      tokenizer, messages, max_seq_len=256,
      im_start_id=im_start, im_end_id=im_end, newline_ids=newline,
  )
  input_tokens, loss_mask, pad_mask = enc
  # The word "assistant" appears in the (masked) role headers; it must never be a
  # trained token, else the model would learn to emit its own header.
  trained = _decode_masked(tokenizer, input_tokens, loss_mask)
  assert "<|im_start|>" not in trained
  # Header markers exist in the full render (4 turns => 4 <|im_start|>).
  assert list(input_tokens).count(im_start) == 4


def test_drop_when_no_assistant_survives_truncation(tokenizer, messages):
  im_start, im_end, newline = resolve_chatml_ids(tokenizer)
  # max_seq_len so small only the (user) turn-0 header fits => no assistant token.
  enc = encode_agent_conversation(
      tokenizer, [messages[0]], max_seq_len=3,
      im_start_id=im_start, im_end_id=im_end, newline_ids=newline,
  )
  assert enc is None


def test_pad_mask_marks_real_tokens(tokenizer, messages):
  im_start, im_end, newline = resolve_chatml_ids(tokenizer)
  enc = encode_agent_conversation(
      tokenizer, messages, max_seq_len=256,
      im_start_id=im_start, im_end_id=im_end, newline_ids=newline,
  )
  input_tokens, loss_mask, pad_mask = enc
  real_len = int(pad_mask.sum())
  assert real_len > 0 and real_len < 256
  # Loss is only ever set inside the real region.
  assert loss_mask[real_len:].sum() == 0
