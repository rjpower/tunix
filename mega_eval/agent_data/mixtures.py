"""Multi-dataset, weight-mixed SFT data loading for the agentic-SFT track.

The single-dataset loader (:mod:`mega_eval.agent_data.agent_traces`) streams one
HF parquet corpus (``open-thoughts/OpenThoughts-Agent-v1-SFT``) and yields
``{"messages": [{"role","content"}, ...]}`` for the assistant-masked encoder. This
module generalizes that to a **weighted blend of several corpora** -- the
SWE-heavy mixture proposed in ``mega_eval/DATA_PLAN.md`` -- without adding any new
heavy dependency: every source is still read shard-by-shard with ``pyarrow`` (the
existing stack; the tunix-pinned ``datasets`` cannot parse the ``List`` feature
type these datasets ship with), and the blend is a probability-weighted round
robin over the per-source streams (same idea as HF ``interleave_datasets`` with
``probabilities``, implemented locally so it works on the raw pyarrow streams).

Three trace schemas appear across the plan; each gets a tiny adapter that
normalizes a parquet row to the canonical ``{"messages": [...], "metadata": {}}``
the encoder (``training/agent_sft.encode_agent_conversation``) consumes:

* ``conversations`` (**Terminus-2**, the in-domain core + the new SWE/unix
  sandboxes traces): a ``list[{"role","content"}]`` -- the *exact* schema of the
  precedent's SFT set, so it is a zero-rename drop-in (:func:`terminus2_adapter`).
* ``messages`` as a **JSON string** (``SWE-bench/SWE-smith-trajectories``): one
  ``json.loads`` then it is already ``[{"role","content"}]`` over
  system/user/assistant/tool turns (:func:`json_messages_adapter`). Carries a
  ``resolved`` flag for an optional high-signal filter.
* ``trajectory`` as a **list with non-standard keys** (``nebius/SWE-agent-
  trajectories``): role ``ai`` -> ``assistant``, content lives in ``text`` (with
  ``system_prompt`` for the system turn) (:func:`nebius_trajectory_adapter``).
  Carries a ``target`` flag (resolved) for the same optional filter.

The encoder masks loss to ``role == "assistant"`` and masks every other role
(system / user / tool / observation) as context, so all three schemas train the
same way once normalized.
"""

import dataclasses
import glob
import json
import os
import random
from typing import Any, Callable, Iterator

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

# A normalized example is ``{"messages": [{"role","content"}, ...], "metadata": {}}``.
Example = dict[str, Any]
# An adapter maps one raw parquet row (a dict) to a normalized Example, or to
# ``None`` to drop the row (e.g. a quality filter rejected it, or it was empty).
RowAdapter = Callable[[dict[str, Any]], Example | None]


# ---------------------------------------------------------------------------
# Row adapters: raw parquet row -> {"messages": [...], "metadata": {...}} or None
# ---------------------------------------------------------------------------
def _norm_messages(raw_msgs: Any, *, role_key: str, content_key: str,
                   role_map: dict[str, str] | None = None,
                   fallback_content_key: str | None = None) -> list[dict[str, str]]:
  """Coerces a list of turn dicts to ``[{"role": str, "content": str}]``."""
  out: list[dict[str, str]] = []
  if not isinstance(raw_msgs, (list, tuple)):
    return out
  for t in raw_msgs:
    if not isinstance(t, dict):
      continue
    role = str(t.get(role_key, "user") or "user")
    if role_map:
      role = role_map.get(role, role)
    content = t.get(content_key)
    if (content is None or content == "") and fallback_content_key:
      content = t.get(fallback_content_key)
    out.append({"role": role, "content": str(content or "")})
  return out


def terminus2_adapter(row: dict[str, Any]) -> Example | None:
  """Terminus-2 ``conversations`` schema (the in-domain / sandboxes-traces core).

  Identical shape to ``open-thoughts/OpenThoughts-Agent-v1-SFT``: a
  ``conversations`` list of ``{"role","content"}``. Zero rename.
  """
  msgs = _norm_messages(row.get("conversations"), role_key="role", content_key="content")
  if not msgs:
    return None
  return {"messages": msgs, "metadata": {}}


def json_messages_adapter(
    row: dict[str, Any],
    *,
    column: str = "messages",
    resolved_key: str | None = None,
    require_resolved: bool = False,
) -> Example | None:
  """``messages`` stored as a JSON string (e.g. SWE-smith).

  Args:
    column: the parquet column holding the JSON-encoded message list.
    resolved_key: optional bool column (e.g. ``"resolved"``); used by the
      ``require_resolved`` high-signal filter.
    require_resolved: if True, drop rows whose ``resolved_key`` is falsy.
  """
  if require_resolved and resolved_key is not None and not row.get(resolved_key):
    return None
  raw = row.get(column)
  if isinstance(raw, str):
    try:
      raw = json.loads(raw)
    except (ValueError, TypeError):
      return None
  msgs = _norm_messages(raw, role_key="role", content_key="content")
  if not msgs:
    return None
  return {"messages": msgs, "metadata": {}}


def list_messages_adapter(
    row: dict[str, Any],
    *,
    column: str = "messages",
    role_key: str = "role",
    content_key: str = "content",
    role_map: dict[str, str] | None = None,
    fallback_content_key: str | None = None,
    resolved_key: str | None = None,
    require_resolved: bool = False,
) -> Example | None:
  """A list-of-dicts message column with configurable role/content keys.

  Covers ``nebius/SWE-agent-trajectories`` (``trajectory`` list, role ``ai`` ->
  ``assistant``, content in ``text``, system content in ``system_prompt``).
  """
  if require_resolved and resolved_key is not None and not row.get(resolved_key):
    return None
  msgs = _norm_messages(
      row.get(column), role_key=role_key, content_key=content_key,
      role_map=role_map, fallback_content_key=fallback_content_key,
  )
  if not msgs:
    return None
  return {"messages": msgs, "metadata": {}}


def nebius_trajectory_adapter(row: dict[str, Any], *, require_resolved: bool = False) -> Example | None:
  """``nebius/SWE-agent-trajectories``: ``trajectory`` list, ``ai`` role, ``text`` content."""
  return list_messages_adapter(
      row,
      column="trajectory",
      role_key="role",
      content_key="text",
      role_map={"ai": "assistant"},
      fallback_content_key="system_prompt",
      resolved_key="target",
      require_resolved=require_resolved,
  )


# ---------------------------------------------------------------------------
# Per-source spec + stream
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class DatasetSource:
  """One corpus in a mixture: where it lives + how to normalize + its weight.

  Attributes:
    name: short label for logging.
    repo_id: HF dataset repo id.
    weight: relative sampling proportion in the blend (normalized across the
      mixture; need not sum to 1).
    adapter: maps a raw parquet row to ``{"messages": [...], ...}`` or ``None``.
    revision: pinned dataset git revision (``None`` = repo default; pin for prod).
    allow_patterns: parquet glob(s) to download (default ``data/*.parquet``).
    parquet_subdir: subdir under the snapshot to glob for shards.
    cap: optional hard cap on usable rows drawn from this source per build, so a
      huge corpus (Nemotron/xlam/AceCode/SWE-smith) cannot swamp the in-domain
      Terminus-2 buckets even if its weight rounds up.
  """

  name: str
  repo_id: str
  weight: float
  adapter: RowAdapter
  revision: str | None = None
  allow_patterns: tuple[str, ...] = ("data/*.parquet",)
  parquet_subdir: str = "data"
  cap: int | None = None


def _source_shards(src: DatasetSource) -> list[str]:
  local_dir = snapshot_download(
      repo_id=src.repo_id,
      repo_type="dataset",
      revision=src.revision,
      allow_patterns=list(src.allow_patterns),
  )
  shards = sorted(glob.glob(os.path.join(local_dir, src.parquet_subdir, "*.parquet")))
  if not shards:
    shards = sorted(glob.glob(os.path.join(local_dir, "**", "*.parquet"), recursive=True))
  if not shards:
    raise ValueError(f"No parquet shards for {src.repo_id} (subdir={src.parquet_subdir}).")
  return shards


def stream_source(
    src: DatasetSource,
    *,
    limit: int | None = None,
    row_group_batch: int = 256,
    shard_loader: Callable[[DatasetSource], list[str]] | None = None,
) -> Iterator[Example]:
  """Streams normalized examples from one :class:`DatasetSource`.

  Reads parquet shards row-group by row-group (pyarrow), applies the source's
  adapter, and skips rows the adapter drops (returns ``None``). Honors both the
  caller ``limit`` and the source's own ``cap`` (whichever is smaller).

  ``shard_loader`` is injectable so tests can supply local fixture shards without
  touching the network.
  """
  caps = [c for c in (limit, src.cap) if c is not None]
  hard = min(caps) if caps else None
  loader = shard_loader or _source_shards
  shards = loader(src)
  emitted = 0
  for shard in shards:
    pf = pq.ParquetFile(shard)
    for batch in pf.iter_batches(batch_size=row_group_batch):
      for raw in batch.to_pylist():
        ex = src.adapter(raw)
        if ex is None:
          continue
        yield ex
        emitted += 1
        if hard is not None and emitted >= hard:
          return


def interleave_sources(
    sources: list[DatasetSource],
    *,
    seed: int = 0,
    per_source_limit: int | None = None,
    shard_loader: Callable[[DatasetSource], list[str]] | None = None,
) -> Iterator[Example]:
  """Weighted round-robin over several source streams (HF-``interleave``-style).

  At each step a source is chosen with probability proportional to its ``weight``
  (among sources not yet exhausted); the next example from that source's stream is
  yielded. This realizes the configured per-dataset sampling proportions in
  expectation -- the same contract as ``datasets.interleave_datasets(...,
  probabilities=..., stopping_strategy="all_exhausted")`` -- but over the raw
  pyarrow streams the rest of the stack already uses. When a source runs dry it is
  dropped and the remaining weights are renormalized; iteration ends when all
  sources are exhausted.

  Args:
    sources: the mixture (weights need not sum to 1).
    seed: PRNG seed for the weighted choice (reproducible blends).
    per_source_limit: optional cap applied to every source (in addition to each
      source's own ``cap``); handy for fast smoke builds.
    shard_loader: injected shard resolver (tests pass fixtures; prod uses HF).
  """
  rng = random.Random(seed)
  live: list[tuple[DatasetSource, Iterator[Example]]] = []
  for src in sources:
    if src.weight <= 0:
      continue
    it = stream_source(src, limit=per_source_limit, shard_loader=shard_loader)
    live.append((src, it))
  weights = [src.weight for src, _ in live]

  while live:
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    pick = 0
    for i, w in enumerate(weights):
      upto += w
      if r <= upto:
        pick = i
        break
    src, it = live[pick]
    try:
      yield next(it)
    except StopIteration:
      live.pop(pick)
      weights.pop(pick)


# ---------------------------------------------------------------------------
# The mixture registry
# ---------------------------------------------------------------------------
# Pinned revisions resolved 2026-06-23 (ungated; verified via huggingface_hub).
_REV_OTA = "c5dc896981f4e3b7c5382669b1d1be0bc4b6a1a6"  # OpenThoughts-Agent-v1-SFT


def _swe_heavy_sources() -> list[DatasetSource]:
  """The DATA_PLAN.md SWE-weighted mixture (~55% SWE), as concrete sources.

  Weights track the DATA_PLAN buckets; caps keep the long-tail SWE corpora from
  swamping the in-domain Terminus-2 buckets. Buckets 6-7 (general tool/code SFT)
  are intentionally omitted from the *default* blend: those are HF-``datasets``
  config/split corpora (smoltalk2 SFT splits, Nemotron tool_calling, AceCode)
  that need the datasets builder this pyarrow path deliberately avoids; they are
  documented in DATA_PLAN.md as the format-breadth tail and can be layered in
  later behind the same adapter contract.
  """
  return [
      # Bucket 1: in-domain core (the proven base; exact eval distribution).
      DatasetSource(
          name="ota-v1",
          repo_id="open-thoughts/OpenThoughts-Agent-v1-SFT",
          revision=_REV_OTA,
          weight=0.12,
          adapter=terminus2_adapter,
          cap=20_000,
      ),
      # Bucket 2: Terminus-2 SWE-agent traces (the single highest-value add).
      DatasetSource(
          name="dcagent-swe-terminus2",
          repo_id="DCAgent/neulab-nebius-swe-agent-trajectories-sandboxes-traces-terminus-2",
          weight=0.18,
          adapter=terminus2_adapter,
          cap=20_000,
      ),
      # Bucket 3: Terminus-2 unix/shell sandboxes traces (broad terminal competence).
      DatasetSource(
          name="unix-sandboxes-terminus2",
          repo_id="mlfoundations-dev/stackexchange-unix-sandboxes-traces-terminus-2",
          weight=0.11,
          adapter=terminus2_adapter,
          cap=15_000,
      ),
      DatasetSource(
          name="superuser-sandboxes-terminus2",
          repo_id="mlfoundations-dev/stackexchange-superuser-sandboxes-traces-terminus-2",
          weight=0.11,
          adapter=terminus2_adapter,
          cap=15_000,
      ),
      # Bucket 4: SWE-bench-style repo-fix trajectories (the leader's SWE lever).
      #   resolved-only -> the high-signal subset.
      DatasetSource(
          name="swe-smith",
          repo_id="SWE-bench/SWE-smith-trajectories",
          weight=0.12,
          adapter=lambda r: json_messages_adapter(
              r, column="messages", resolved_key="resolved", require_resolved=True),
          cap=25_000,
      ),
      DatasetSource(
          name="nebius-swe-agent",
          repo_id="nebius/SWE-agent-trajectories",
          weight=0.10,
          adapter=lambda r: nebius_trajectory_adapter(r, require_resolved=True),
          cap=25_000,
      ),
      # Bucket 5: OpenHands / R2E SWE SFT (small, successful-only, high signal/row).
      DatasetSource(
          name="r2e-gym-sft",
          repo_id="R2E-Gym/R2EGym-SFT-Trajectories",
          weight=0.04,
          adapter=lambda r: json_messages_adapter(r, column="messages"),
          cap=5_000,
      ),
      DatasetSource(
          name="openhands-sft",
          repo_id="SWE-Gym/OpenHands-SFT-Trajectories",
          weight=0.02,
          adapter=lambda r: json_messages_adapter(r, column="messages"),
          cap=2_000,
      ),
  ]


def _ota_only_sources() -> list[DatasetSource]:
  """Single-source control: the precedent's exact SFT set (back-compat blend)."""
  return [
      DatasetSource(
          name="ota-v1",
          repo_id="open-thoughts/OpenThoughts-Agent-v1-SFT",
          revision=_REV_OTA,
          weight=1.0,
          adapter=terminus2_adapter,
      ),
  ]


# name -> () -> list[DatasetSource]. A factory (not a frozen list) so revision
# pins / caps stay together and the (closure-holding) adapters build lazily.
MIXTURES: dict[str, Callable[[], list[DatasetSource]]] = {
    "swe_heavy": _swe_heavy_sources,
    "ota_only": _ota_only_sources,
}


def get_mixture(name: str) -> list[DatasetSource]:
  """Returns the named mixture's sources, or raises with the known names."""
  if name not in MIXTURES:
    raise KeyError(f"Unknown mixture {name!r}; known: {sorted(MIXTURES)}")
  return MIXTURES[name]()


def sources_from_json(spec: str) -> list[DatasetSource]:
  """Builds a mixture from a JSON spec (the ``SFT_MIXTURE`` env var).

  Each entry is ``{"repo_id", "weight", "format"?, "revision"?, "cap"?,
  "column"?, "split"?, "require_resolved"?, "name"?}``. ``format`` is one of
  ``terminus2`` (default), ``json_messages``, ``list_messages`` (with optional
  ``role_key``/``content_key``/``role_map``/``fallback_content_key``), or
  ``nebius_trajectory``. ``split`` is accepted for parity with the spec contract
  and recorded but, because these corpora ship a single ``train`` parquet group,
  it does not change shard selection here.
  """
  entries = json.loads(spec)
  out: list[DatasetSource] = []
  for i, e in enumerate(entries):
    fmt = e.get("format", "terminus2")
    require_resolved = bool(e.get("require_resolved", False))
    column = e.get("column", "messages")
    if fmt == "terminus2":
      adapter: RowAdapter = terminus2_adapter
    elif fmt == "json_messages":
      adapter = lambda r, c=column, rr=require_resolved, rk=e.get("resolved_key", "resolved"): \
          json_messages_adapter(r, column=c, resolved_key=rk, require_resolved=rr)
    elif fmt == "nebius_trajectory":
      adapter = lambda r, rr=require_resolved: nebius_trajectory_adapter(r, require_resolved=rr)
    elif fmt == "list_messages":
      adapter = lambda r, c=column, rk=e.get("role_key", "role"), ck=e.get("content_key", "content"), \
          rm=e.get("role_map"), fck=e.get("fallback_content_key"): \
          list_messages_adapter(r, column=c, role_key=rk, content_key=ck,
                                role_map=rm, fallback_content_key=fck)
    else:
      raise ValueError(f"Unknown mixture entry format {fmt!r} (entry {i}).")
    out.append(DatasetSource(
        name=e.get("name", e["repo_id"]),
        repo_id=e["repo_id"],
        weight=float(e.get("weight", 1.0)),
        adapter=adapter,
        revision=e.get("revision"),
        cap=e.get("cap"),
    ))
  return out


def resolve_mixture(*, mixture_json: str | None = None, mixture_name: str | None = None) -> list[DatasetSource]:
  """Resolves the active mixture from the two env-var entry points.

  ``SFT_MIXTURE`` (JSON, if set) wins over ``MIXTURE`` (a registry name); if
  neither is given, falls back to ``swe_heavy``.
  """
  if mixture_json:
    return sources_from_json(mixture_json)
  if mixture_name:
    return get_mixture(mixture_name)
  return get_mixture("swe_heavy")
