# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenThoughts-Agent SFT data: the scaling ladder + process-disjoint sharding.

The paper (arXiv:2606.24855) releases its assembled SFT set at six sizes -- a
compute-controlled scaling ladder -- plus the 100K headline set, all under
``open-thoughts/OpenThoughts-Agent-SFT-<size>``. Every one carries the **exact
Terminus-2 ``conversations`` schema** of ``mega_eval``'s in-domain corpus
(``conversations: List[{role, content}]`` + ``agent``/``task``/``trace_source``
metadata), so the proven assistant-turn-masked ChatML encoder
(``mega_eval.training.agent_sft.encode_agent_conversation``) reads it unchanged.

Two responsibilities live here:

1. **Stream** the parquet shards with pyarrow (the ``datasets<4`` builder cannot
   parse the ``List`` feature type -- same workaround as
   ``mega_eval.agent_data.agent_traces``).
2. **Shard across processes.** A 4-node data-parallel run is 4 JAX processes
   feeding one ``("fsdp","tp")`` mesh; tunix's ``shard_input`` assembles the
   global batch from each process's *local* contribution via
   ``jax.make_array_from_process_local_data``. So each process must stream a
   **disjoint** slice of the data -- otherwise the four processes feed identical
   rows and the run sees only ``1/4`` of the corpus per epoch (or, worse,
   silently 4x-replicates each global batch). We shard by global row index
   ``idx % process_count == process_index`` (a strided, disjoint partition;
   batch-position assignment is irrelevant for SFT).
"""

from __future__ import annotations

import glob
import math
import os
from typing import Any, Iterator

import grain.python as grain
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from mega_eval.training.agent_sft import encode_agent_conversation, resolve_chatml_ids

# The paper's SFT scaling ladder + headline set (all ungated, same schema).
# Sizes are the *released* row counts (≈, the repos are named by target size).
OT_AGENT_SFT_DATASETS: dict[str, str] = {
    "1k": "open-thoughts/OpenThoughts-Agent-SFT-1K",
    "3.16k": "open-thoughts/OpenThoughts-Agent-SFT-3.16K",
    "10k": "open-thoughts/OpenThoughts-Agent-SFT-10K",
    "31.6k": "open-thoughts/OpenThoughts-Agent-SFT-31.6K",
    "100k": "open-thoughts/OpenThoughts-Agent-SFT-100K",  # ~94.3k rows; the headline set
    # Cold-start subset distilled for the RL phase warm-up.
    "coldstart-10k": "open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K",
    # The original (smaller) v1 set mega_eval used; kept for parity / ablation.
    "v1": "open-thoughts/OpenThoughts-Agent-v1-SFT",
}

# Pinned commit shas (resolved 2026-06-24) so a rerun reads byte-identical data.
# A caller may override via ``DATASET_REVISION`` / the ``revision`` arg.
OT_AGENT_SFT_REVISIONS: dict[str, str] = {
    "open-thoughts/OpenThoughts-Agent-SFT-1K": "704dc6620eb61c0e4bef5a4943416866081beea6",
    "open-thoughts/OpenThoughts-Agent-SFT-3.16K": "f7350d7ade54407504f388e3faf303150a3cb395",
    "open-thoughts/OpenThoughts-Agent-SFT-10K": "d0f898f7b65290c9312f59089a7e9265fdaed00c",
    "open-thoughts/OpenThoughts-Agent-SFT-31.6K": "430c23f84bc0616ca8795e02bd9447f9f49c413d",
    "open-thoughts/OpenThoughts-Agent-SFT-100K": "45fb28fcc38d352133cb28a1c8a43a2f14fea97b",
    "open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K": "261a70c35c0b16327c01a745ab65cc5651fb00d9",
    "open-thoughts/OpenThoughts-Agent-v1-SFT": "c5dc896981f4e3b7c5382669b1d1be0bc4b6a1a6",
}


def default_revision(repo_id: str) -> str | None:
  """Returns the pinned revision for a known OT-Agent repo, else ``None`` (main)."""
  return OT_AGENT_SFT_REVISIONS.get(repo_id)

_METADATA_FIELDS = (
    "agent", "model", "model_provider", "date", "task", "episode",
    "run_id", "trial_name", "trace_source", "result",
)


def resolve_repo(name_or_repo: str) -> str:
  """Maps a ladder key (e.g. ``"100k"``) to its HF repo id; passes a full id through."""
  key = name_or_repo.strip().lower()
  if key in OT_AGENT_SFT_DATASETS:
    return OT_AGENT_SFT_DATASETS[key]
  return name_or_repo  # already a full ``org/dataset`` id


def _local_parquet_shards(repo_id: str, revision: str | None) -> list[str]:
  """Downloads the dataset's parquet shards and returns their local paths."""
  local_dir = snapshot_download(
      repo_id=repo_id,
      repo_type="dataset",
      revision=revision,
      allow_patterns=["data/*.parquet"],
  )
  shards = sorted(glob.glob(os.path.join(local_dir, "data", "*.parquet")))
  if not shards:
    raise ValueError(f"No parquet shards under {local_dir}/data for {repo_id}.")
  return shards


def stream_traces(
    repo_id: str,
    *,
    revision: str | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    limit: int | None = None,
    row_group_batch: int = 256,
) -> Iterator[dict[str, Any]]:
  """Streams OT-Agent episodes as ``{"messages": [...], "metadata": {...}}``.

  Yields only the rows belonging to this process's shard
  (``global_idx % shard_count == shard_index``), so a multi-process run reads the
  corpus disjointly. ``limit`` caps the number of **global** rows scanned (so
  every shard sees the same dataset prefix), not the number emitted by a shard.

  Args:
    repo_id: HF dataset id (or pass a ladder key through :func:`resolve_repo`).
    revision: pinned git revision (``None`` = the repo default branch).
    shard_index: this process's index in ``[0, shard_count)``.
    shard_count: number of data-parallel processes.
    limit: stop after scanning this many global rows (``None`` = all).
    row_group_batch: pyarrow ``iter_batches`` size.
  """
  if not 0 <= shard_index < shard_count:
    raise ValueError(f"shard_index={shard_index} out of range [0,{shard_count}).")
  shards = _local_parquet_shards(repo_id, revision)
  global_idx = 0
  for shard in shards:
    pf = pq.ParquetFile(shard)
    for batch in pf.iter_batches(batch_size=row_group_batch):
      for ex in batch.to_pylist():
        if limit is not None and global_idx >= limit:
          return
        idx = global_idx
        global_idx += 1
        if idx % shard_count != shard_index:
          continue
        conv = ex.get("conversations") or []
        messages = [
            {"role": str(t.get("role", "user")),
             "content": str(t.get("content") or "")}
            for t in conv
        ]
        metadata = {k: ex.get(k) for k in _METADATA_FIELDS if k in ex}
        yield {"messages": messages, "metadata": metadata}


def collect_encoded_shard(
    tokenizer,
    examples: Iterator[dict[str, Any]],
    n: int,
    seed: int,
    max_seq_len: int,
    *,
    process_index: int = 0,
    scan_cap: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """Encodes a (already process-sharded) stream into ``n`` usable SFT rows.

  Mirrors ``mega_eval.training.agent_sft._collect_encoded_rows`` but reuses the
  public encoder and logs the process index. Rows with no surviving assistant
  token are dropped; the shard is shuffled with ``seed`` and cycled to fill if it
  runs dry before ``n``.
  """
  im_start, im_end, newline_ids = resolve_chatml_ids(tokenizer)
  cap = scan_cap if scan_cap is not None else max(n * 6, 6000)
  rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
  scanned = dropped = truncated = 0
  for ex in examples:
    scanned += 1
    enc = encode_agent_conversation(
        tokenizer, ex.get("messages", []), max_seq_len,
        im_start_id=im_start, im_end_id=im_end, newline_ids=newline_ids,
    )
    if enc is None:
      dropped += 1
    else:
      if int(enc[2].sum()) >= max_seq_len:
        truncated += 1
      rows.append(enc)
      if len(rows) >= n:
        break
    if scanned >= cap:
      break

  if not rows:
    raise ValueError(
        f"[p{process_index}] no usable episodes after scanning {scanned} rows "
        f"(all empty or longer than max_seq_len={max_seq_len})."
    )
  print(
      f"[ota-data p{process_index}] scanned={scanned} usable={len(rows)} "
      f"dropped={dropped} window_full={truncated} (max_seq_len={max_seq_len})",
      flush=True,
  )
  import random  # noqa: PLC0415
  random.Random(seed + process_index).shuffle(rows)
  if len(rows) < n:
    print(f"[ota-data p{process_index}] only {len(rows)}/{n} usable; cycling to fill.",
          flush=True)
    base = list(rows)
    while len(rows) < n:
      rows.append(base[len(rows) % len(base)])
  return rows[:n]


class _RowSource(grain.RandomAccessDataSource):
  """A grain random-access source over pre-encoded ``(tokens, loss, pad)`` rows."""

  def __init__(self, rows):
    self._rows = rows

  def __len__(self):
    return len(self._rows)

  def __getitem__(self, idx):
    return self._rows[idx]


def _to_columns(batch):
  input_tokens, loss_mask, pad_mask = batch
  return {"input_tokens": input_tokens, "loss_mask": loss_mask, "pad_mask": pad_mask}


def build_sharded_sft_dataset(
    tokenizer,
    *,
    repo_id: str,
    per_process_batch: int,
    n_per_process: int,
    max_seq_len: int,
    seed: int,
    process_index: int,
    process_count: int,
    revision: str | None = None,
    limit: int | None = None,
    scan_cap: int | None = None,
) -> grain.MapDataset:
  """Builds this process's batched grain dataset over its disjoint data shard.

  The returned dataset yields ``per_process_batch`` rows per step; tunix's
  ``PeftTrainer`` (``data_sharding_axis=("fsdp",)``) assembles the global batch
  of ``per_process_batch * process_count`` rows across the mesh. Uses grain (not
  HF ``.batch()``) because tunix's collation corrupts HF-batched rows.
  """
  stream = stream_traces(
      repo_id, revision=revision, shard_index=process_index,
      shard_count=process_count, limit=limit,
  )
  rows = collect_encoded_shard(
      tokenizer, stream, n_per_process, seed, max_seq_len,
      process_index=process_index, scan_cap=scan_cap,
  )
  return grain.MapDataset.source(_RowSource(rows)).batch(per_process_batch).map(_to_columns)


def rows_per_process(steps: int, per_process_batch: int) -> int:
  """Number of encoded rows a process must buffer for ``steps`` SFT steps."""
  return (steps + 2) * per_process_batch


def per_process_batch_size(global_batch: int, process_count: int) -> int:
  """Splits the global batch across processes; requires clean divisibility."""
  if global_batch % process_count != 0:
    raise ValueError(
        f"global BATCH_SIZE={global_batch} must be divisible by process_count="
        f"{process_count} (each process feeds an equal share of the global batch)."
    )
  return global_batch // process_count


def ceil_div(a: int, b: int) -> int:
  return math.ceil(a / b)
