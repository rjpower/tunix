"""CPU, no-network tests for the multi-dataset weighted SFT mixture.

These assert the three contracts the SWE-heavy blend (DATA_PLAN.md) must satisfy:
  (a) the blend yields ChatML examples encoded with assistant-only loss masks,
  (b) the empirical per-source sampling ratio over many draws matches the
      configured weights,
  (c) every encoded row is truncated to ``MAX_SEQ_LEN``.

All HF/parquet access is mocked: sources stream from in-memory fixture rows via
an injected ``shard_loader``/fake-parquet, so the suite needs no network. The one
test that needs a real Qwen3 tokenizer skips offline. Real dataset downloads are
gated behind ``RUN_MIXTURE_NETWORK_TEST=1`` (off by default).
"""

import os

import numpy as np
import pytest

from mega_eval.agent_data import mixtures as M
from mega_eval.training.agent_sft import (
    _collect_encoded_rows,
    build_agent_sft_dataset,
    resolve_chatml_ids,
)

TOKENIZER_REPO = "Qwen/Qwen3-8B"


# --------------------------------------------------------------------------
# In-memory fixtures: tag each source's assistant content with its source name so
# we can recover which source a drawn row came from (for the ratio test).
# --------------------------------------------------------------------------
def _terminus2_row(tag: str, i: int) -> dict:
  return {
      "conversations": [
          {"role": "user", "content": f"{tag} task {i}: do the thing"},
          {"role": "assistant", "content": f"SRC={tag} action {i}"},
          {"role": "user", "content": "Current terminal state: ok"},
          {"role": "assistant", "content": f"SRC={tag} done {i}"},
      ]
  }


def _fake_stream_factory(rows_by_source: dict[str, list[dict]]):
  """Returns a ``stream_source`` replacement that yields fixture rows in-memory."""

  def _fake_stream_source(src, *, limit=None, row_group_batch=256, shard_loader=None):
    rows = rows_by_source[src.name]
    caps = [c for c in (limit, src.cap) if c is not None]
    hard = min(caps) if caps else None
    emitted = 0
    # Cycle the small fixture pool so a high-weight source never starves the draw.
    i = 0
    while True:
      raw = rows[i % len(rows)]
      i += 1
      ex = src.adapter(raw)
      if ex is None:
        continue
      yield ex
      emitted += 1
      if hard is not None and emitted >= hard:
        return

  return _fake_stream_source


@pytest.fixture
def tokenizer():
  from transformers import AutoTokenizer
  try:
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
  except Exception as e:  # offline / no network
    pytest.skip(f"cannot load {TOKENIZER_REPO} tokenizer: {e}")
  if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
  return tok


# --------------------------------------------------------------------------
# (b) weighted sampling ratio  -- pure, no tokenizer/network needed
# --------------------------------------------------------------------------
def test_interleave_respects_weights(monkeypatch):
  sources = [
      M.DatasetSource(name="A", repo_id="x/a", weight=0.6, adapter=M.terminus2_adapter),
      M.DatasetSource(name="B", repo_id="x/b", weight=0.3, adapter=M.terminus2_adapter),
      M.DatasetSource(name="C", repo_id="x/c", weight=0.1, adapter=M.terminus2_adapter),
  ]
  rows = {s.name: [_terminus2_row(s.name, i) for i in range(50)] for s in sources}
  monkeypatch.setattr(M, "stream_source", _fake_stream_factory(rows))

  N = 6000
  counts = {"A": 0, "B": 0, "C": 0}
  for ex in M.interleave_sources(sources, seed=0):
    tag = ex["messages"][1]["content"].split()[0]  # "SRC=A"
    counts[tag.split("=")[1]] += 1
    if sum(counts.values()) >= N:
      break

  total = sum(counts.values())
  ratios = {k: v / total for k, v in counts.items()}
  # Empirical proportions track the configured (normalized) weights within 4%.
  assert abs(ratios["A"] - 0.6) < 0.04, ratios
  assert abs(ratios["B"] - 0.3) < 0.04, ratios
  assert abs(ratios["C"] - 0.1) < 0.04, ratios


def test_zero_weight_source_excluded(monkeypatch):
  sources = [
      M.DatasetSource(name="A", repo_id="x/a", weight=1.0, adapter=M.terminus2_adapter),
      M.DatasetSource(name="Z", repo_id="x/z", weight=0.0, adapter=M.terminus2_adapter),
  ]
  rows = {s.name: [_terminus2_row(s.name, i) for i in range(10)] for s in sources}
  monkeypatch.setattr(M, "stream_source", _fake_stream_factory(rows))
  seen = set()
  for k, ex in enumerate(M.interleave_sources(sources, seed=1)):
    seen.add(ex["messages"][1]["content"].split("=")[1].split()[0])
    if k >= 200:
      break
  assert seen == {"A"}


def test_per_source_cap_and_exhaustion(monkeypatch):
  # Capped sources exhaust; the blend stops when all are drained.
  sources = [
      M.DatasetSource(name="A", repo_id="x/a", weight=1.0, adapter=M.terminus2_adapter, cap=5),
      M.DatasetSource(name="B", repo_id="x/b", weight=1.0, adapter=M.terminus2_adapter, cap=3),
  ]
  rows = {s.name: [_terminus2_row(s.name, i) for i in range(100)] for s in sources}
  monkeypatch.setattr(M, "stream_source", _fake_stream_factory(rows))
  drawn = list(M.interleave_sources(sources, seed=2))
  tags = [ex["messages"][1]["content"].split("=")[1].split()[0] for ex in drawn]
  assert tags.count("A") == 5 and tags.count("B") == 3
  assert len(drawn) == 8


# --------------------------------------------------------------------------
# Adapter unit tests -- the three schemas in the SWE-heavy plan
# --------------------------------------------------------------------------
def test_adapters_normalize_all_three_schemas():
  # Terminus-2 conversations (in-domain + sandboxes traces).
  t = M.terminus2_adapter(
      {"conversations": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]})
  assert t["messages"] == [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

  # SWE-smith: messages as a JSON string + resolved filter.
  j = M.json_messages_adapter(
      {"messages": '[{"role":"system","content":"s"},{"role":"assistant","content":"a"}]',
       "resolved": True},
      resolved_key="resolved", require_resolved=True)
  assert [m["role"] for m in j["messages"]] == ["system", "assistant"]
  assert M.json_messages_adapter({"messages": "[]", "resolved": False},
                                 resolved_key="resolved", require_resolved=True) is None

  # nebius: trajectory list, ai->assistant, content in `text`, system in system_prompt.
  n = M.nebius_trajectory_adapter(
      {"trajectory": [{"role": "system", "text": "", "system_prompt": "SYS"},
                      {"role": "ai", "text": "patch"},
                      {"role": "user", "text": "obs"}],
       "target": True}, require_resolved=True)
  assert n["messages"][0] == {"role": "system", "content": "SYS"}
  assert n["messages"][1] == {"role": "assistant", "content": "patch"}
  # require_resolved drops un-resolved rows.
  assert M.nebius_trajectory_adapter({"trajectory": [{"role": "ai", "text": "x"}], "target": False},
                                     require_resolved=True) is None


def test_swe_heavy_registry_is_swe_weighted():
  srcs = M.get_mixture("swe_heavy")
  by = {s.name: s for s in srcs}
  # Has the 4 named NEW SWE/terminal datasets from DATA_PLAN.md.
  assert "dcagent-swe-terminus2" in by
  assert "unix-sandboxes-terminus2" in by
  assert "swe-smith" in by
  assert "nebius-swe-agent" in by
  # SWE/terminal weight dominates the in-domain core (SWE-heavy tilt).
  in_domain = by["ota-v1"].weight
  swe = sum(s.weight for n, s in by.items() if n != "ota-v1")
  assert swe > 3 * in_domain
  # Every long-tail SWE corpus is capped so it can't swamp the in-domain buckets.
  assert all(s.cap is not None for s in srcs)


# --------------------------------------------------------------------------
# (a) ChatML + assistant-only masks and (c) MAX_SEQ_LEN truncation
# --------------------------------------------------------------------------
def _decode_masked(tokenizer, ids, mask):
  return tokenizer.decode([int(t) for t, m in zip(ids, mask) if m > 0.5])


def test_blend_yields_chatml_assistant_masked_rows(tokenizer, monkeypatch):
  sources = [
      M.DatasetSource(name="A", repo_id="x/a", weight=0.7, adapter=M.terminus2_adapter),
      M.DatasetSource(name="B", repo_id="x/b", weight=0.3, adapter=M.terminus2_adapter),
  ]
  rows = {s.name: [_terminus2_row(s.name, i) for i in range(40)] for s in sources}
  monkeypatch.setattr(M, "stream_source", _fake_stream_factory(rows))

  MAX = 64
  stream = M.interleave_sources(sources, seed=0)
  encoded = _collect_encoded_rows(tokenizer, stream, n=24, seed=0, max_seq_len=MAX)
  assert len(encoded) == 24

  im_start, im_end, _ = resolve_chatml_ids(tokenizer)
  for input_tokens, loss_mask, pad_mask in encoded:
    # (c) every row truncated to MAX_SEQ_LEN.
    assert input_tokens.shape == (MAX,)
    assert loss_mask.shape == (MAX,)
    # (a) assistant-only mask: trained text contains the SRC= assistant content
    # and never a turn header / user observation.
    trained = _decode_masked(tokenizer, input_tokens, loss_mask)
    assert "SRC=" in trained
    assert "<|im_start|>" not in trained
    assert "Current terminal state" not in trained
    assert "do the thing" not in trained
    # loss only inside the real (non-pad) region.
    real_len = int(pad_mask.sum())
    assert loss_mask[real_len:].sum() == 0
    # at least one trained <|im_end|> (teaches the stop token).
    trained_ids = [int(t) for t, m in zip(input_tokens, loss_mask) if m > 0.5]
    assert im_end in trained_ids


def test_build_dataset_from_sources_batches(tokenizer, monkeypatch):
  sources = [M.DatasetSource(name="A", repo_id="x/a", weight=1.0, adapter=M.terminus2_adapter)]
  rows = {"A": [_terminus2_row("A", i) for i in range(40)]}
  monkeypatch.setattr(M, "stream_source", _fake_stream_factory(rows))

  ds = build_agent_sft_dataset(
      tokenizer, n=16, seed=0, batch_size=4, max_seq_len=48, sources=sources)
  batch = ds[0]
  assert set(batch) == {"input_tokens", "loss_mask", "pad_mask"}
  assert np.asarray(batch["input_tokens"]).shape == (4, 48)
  assert np.asarray(batch["loss_mask"]).shape == (4, 48)


# --------------------------------------------------------------------------
# Real-download integration test (opt-in; never runs in the default CPU suite).
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("RUN_MIXTURE_NETWORK_TEST") != "1",
    reason="set RUN_MIXTURE_NETWORK_TEST=1 to exercise real HF downloads",
)
def test_real_swe_heavy_stream_smoke(tokenizer):
  srcs = M.get_mixture("swe_heavy")
  stream = M.interleave_sources(srcs, seed=0, per_source_limit=2)
  got = []
  for ex in stream:
    assert "messages" in ex and ex["messages"]
    got.append(ex)
    if len(got) >= 8:
      break
  assert got
