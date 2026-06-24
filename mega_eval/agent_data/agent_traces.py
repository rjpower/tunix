"""Stream the OpenThoughts terminal-agent SFT traces from HuggingFace.

``open-thoughts/OpenThoughts-Agent-v1-SFT`` (https://www.openthoughts.ai/blog/agent)
is ~15.2k teacher-agent trajectories generated with the Terminus-2 harness and a
GLM-4.6 teacher. Each row is one episode of a terminal agent: the model reads a
task + terminal observations and emits a structured action per turn.

Row schema (HF datasets-server)::

    conversations : List[{"role": str, "content": str}]   # the trainable field
    agent, model, model_provider, date, task, episode, run_id, trial_name : str

The ``conversations`` list interleaves:
  * a ``user`` turn 0 carrying the system + task instructions,
  * ``assistant`` turns: the agent's action (a JSON object ``{analysis, plan,
    commands, ...}``),
  * ``user`` turns: the terminal observation fed back ("Current terminal state:
    New Terminal Output: ...").

We yield each row as ``{"messages": [...], "metadata": {...}}`` so the SFT
encoder can mask everything except the assistant turns.

NOTE: we read the parquet shards directly with pyarrow rather than via
``datasets.load_dataset``. The dataset was written with a newer ``datasets``
version that records the ``conversations`` column as a ``List`` feature type,
which the tunix-pinned ``datasets<4.0`` cannot parse (``Feature type 'List' not
found``). pyarrow ignores that custom HF metadata, so reading the parquet shards
directly is both robust and lighter (no datasets builder).
"""

import glob
import os
from typing import Any, Iterator

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

DATASET_ID = "open-thoughts/OpenThoughts-Agent-v1-SFT"
# Pinned for reproducibility (resolved 2026-06-21). Override via load arg if needed.
DATASET_REVISION = "c5dc896981f4e3b7c5382669b1d1be0bc4b6a1a6"

_METADATA_FIELDS = (
    "agent",
    "model",
    "model_provider",
    "date",
    "task",
    "episode",
    "run_id",
    "trial_name",
)


def _local_parquet_shards(revision: str) -> list[str]:
  """Downloads the dataset's parquet shards and returns their local paths."""
  local_dir = snapshot_download(
      repo_id=DATASET_ID,
      repo_type="dataset",
      revision=revision,
      allow_patterns=["data/*.parquet"],
  )
  shards = sorted(glob.glob(os.path.join(local_dir, "data", "*.parquet")))
  if not shards:
    raise ValueError(f"No parquet shards under {local_dir}/data for {DATASET_ID}.")
  return shards


def load_agent_traces(
    *,
    revision: str = DATASET_REVISION,
    limit: int | None = None,
    row_group_batch: int = 256,
) -> Iterator[dict[str, Any]]:
  """Streams agent episodes as ``{"messages": [...], "metadata": {...}}``.

  Reads the parquet shards row-group by row-group (effectively streaming) via
  pyarrow, so the whole file is on disk but never fully materialized in memory.

  Args:
    revision: dataset git revision (pinned by default).
    limit: stop after this many rows (``None`` = all ~15.2k).
    row_group_batch: pyarrow ``iter_batches`` size.

  Yields:
    Dicts with ``messages`` (list of ``{"role", "content"}``) and ``metadata``.
  """
  shards = _local_parquet_shards(revision)
  emitted = 0
  for shard in shards:
    pf = pq.ParquetFile(shard)
    for batch in pf.iter_batches(batch_size=row_group_batch):
      for ex in batch.to_pylist():
        conv = ex.get("conversations") or []
        messages = [
            {"role": str(t.get("role", "user")), "content": str(t.get("content") or "")}
            for t in conv
        ]
        metadata = {k: ex.get(k) for k in _METADATA_FIELDS}
        yield {"messages": messages, "metadata": metadata}
        emitted += 1
        if limit is not None and emitted >= limit:
          return
