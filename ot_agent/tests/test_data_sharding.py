# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Process-disjoint sharding is the correctness crux of the multi-node data path.

If the four data-parallel processes did not read disjoint slices, the run would
see only 1/N of the corpus per epoch (or silently 4x-replicate each global
batch). These tests pin: (1) the shards partition the corpus exactly (disjoint +
complete), (2) the per-process batch arithmetic is sound.
"""

import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ot_agent import data as otd


def _write_fake_dataset(tmpdir: str, n_rows: int) -> None:
  """Writes a tiny parquet shard in the OT-Agent ``conversations`` schema."""
  os.makedirs(os.path.join(tmpdir, "data"), exist_ok=True)
  convs = [
      [{"role": "user", "content": f"task {i}"},
       {"role": "assistant", "content": f"action {i}"}]
      for i in range(n_rows)
  ]
  table = pa.table({
      "conversations": convs,
      "task": [f"t{i}" for i in range(n_rows)],
      "trace_source": ["fake"] * n_rows,
  })
  # Two physical shards to exercise the multi-file path.
  half = n_rows // 2
  pq.write_table(table.slice(0, half), os.path.join(tmpdir, "data", "train-0.parquet"))
  pq.write_table(table.slice(half), os.path.join(tmpdir, "data", "train-1.parquet"))


def _collect_shard(tmpdir, shard_index, shard_count, limit=None, monkeypatch=None):
  monkeypatch.setattr(otd, "_local_parquet_shards", lambda repo_id, revision: sorted([
      os.path.join(tmpdir, "data", f) for f in os.listdir(os.path.join(tmpdir, "data"))
  ]))
  return [ex["messages"][0]["content"]
          for ex in otd.stream_traces("fake/repo", shard_index=shard_index,
                                       shard_count=shard_count, limit=limit)]


def test_shards_are_disjoint_and_complete(monkeypatch):
  with tempfile.TemporaryDirectory() as d:
    _write_fake_dataset(d, 100)
    n = 4
    shards = [set(_collect_shard(d, i, n, monkeypatch=monkeypatch)) for i in range(n)]
    # Disjoint: no row appears in two shards.
    for i in range(n):
      for j in range(i + 1, n):
        assert shards[i].isdisjoint(shards[j]), f"shards {i},{j} overlap"
    # Complete: the union is the whole corpus.
    union = set().union(*shards)
    assert union == {f"task {i}" for i in range(100)}
    # Balanced: 100 / 4 = 25 each.
    assert all(len(s) == 25 for s in shards)


def test_single_process_reads_everything(monkeypatch):
  with tempfile.TemporaryDirectory() as d:
    _write_fake_dataset(d, 40)
    rows = _collect_shard(d, 0, 1, monkeypatch=monkeypatch)
    assert len(rows) == 40
    assert rows == [f"task {i}" for i in range(40)]


def test_limit_caps_global_rows(monkeypatch):
  with tempfile.TemporaryDirectory() as d:
    _write_fake_dataset(d, 100)
    # limit=20 scans only the first 20 GLOBAL rows; with 2 shards each sees 10.
    shards = [set(_collect_shard(d, i, 2, limit=20, monkeypatch=monkeypatch)) for i in range(2)]
    union = set().union(*shards)
    assert union == {f"task {i}" for i in range(20)}
    assert shards[0].isdisjoint(shards[1])


def test_per_process_batch_arithmetic():
  assert otd.per_process_batch_size(32, 4) == 8
  assert otd.per_process_batch_size(16, 2) == 8
  with pytest.raises(ValueError):
    otd.per_process_batch_size(30, 4)  # not divisible
  assert otd.rows_per_process(steps=20, per_process_batch=8) == 22 * 8


def test_resolve_repo():
  assert otd.resolve_repo("100k") == "open-thoughts/OpenThoughts-Agent-SFT-100K"
  assert otd.resolve_repo("100K") == "open-thoughts/OpenThoughts-Agent-SFT-100K"
  assert otd.resolve_repo("org/custom-dataset") == "org/custom-dataset"


def test_shard_index_out_of_range(monkeypatch):
  with tempfile.TemporaryDirectory() as d:
    _write_fake_dataset(d, 10)
    with pytest.raises(ValueError):
      _collect_shard(d, 5, 4, monkeypatch=monkeypatch)
