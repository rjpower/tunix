"""Agent-trace SFT for Qwen3-8B: ChatML encoding + assistant-turn loss masking.

Unlike the base-LM chat-SFT in tunix-delphi-rl (which imposes a plain-text
``<|user|>``/``<|assistant|>`` format because Delphi has no chat template), the
OpenThoughts agent is post-trained from ``Qwen/Qwen3-8B`` -- a chat model whose
native format IS ChatML (``<|im_start|>role\\n...<|im_end|>\\n``). We therefore
encode each episode in real Qwen3 ChatML, using the tokenizer's own
``<|im_start|>`` / ``<|im_end|>`` ids, and train ONLY on the assistant turns
(the agent's actions) -- the system/task prompt and the terminal observations are
context (mask 0). Training on the closing ``<|im_end|>`` of each assistant turn
teaches the model to STOP, which is what the rollout harness keys on.

The encoder is intentionally hand-rolled rather than ``apply_chat_template``: the
agent traces are structured-JSON actions (no Qwen3 ``<think>`` blocks), so we want
plain ChatML with no thinking-mode wrapping, and we need per-segment control of
the loss mask. ``tests/test_agent_sft_encode.py`` asserts the rendered string
matches Qwen3's own ``apply_chat_template`` for a non-thinking conversation.

Checkpointing: tunix's ``PeftTrainer`` saves the nnx params via orbax when
``checkpoint_root_directory`` is set (orbax handles ``gs://`` paths). The
tunix-delphi-rl SFT never checkpointed -- this module does, so the SFT'd actor
can be restored in a separate eval/RL job.
"""

import random
from typing import Any, Iterable

import grain.python as grain
import jax
import numpy as np
import orbax.checkpoint as ocp

from tunix.sft.peft_trainer import PeftTrainer, TrainingConfig

from mega_eval.agent_data.agent_traces import load_agent_traces
from mega_eval.agent_data.mixtures import DatasetSource, interleave_sources
from mega_eval.training.common import clipped_adamw, sft_model_input_fn


def resolve_chatml_ids(tokenizer) -> tuple[int, int, list[int]]:
  """Returns ``(im_start_id, im_end_id, newline_ids)`` for the Qwen3 tokenizer.

  Raises:
    ValueError: if the ChatML control tokens are not in the vocabulary (i.e. this
      is not a Qwen/ChatML tokenizer).
  """
  im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
  im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
  unk = tokenizer.unk_token_id
  if im_start is None or im_end is None or im_start == unk or im_end == unk:
    raise ValueError(
        "Tokenizer has no <|im_start|>/<|im_end|> ChatML tokens; "
        "this encoder requires a Qwen3/ChatML tokenizer."
    )
  newline_ids = tokenizer.encode("\n", add_special_tokens=False)
  return int(im_start), int(im_end), [int(t) for t in newline_ids]


def render_chatml(messages: list[dict[str, Any]], *, add_generation_prompt: bool = True) -> str:
  """Renders messages as Qwen3 ChatML text (matches :func:`encode_agent_conversation`).

  Used at eval/rollout time to build the prompt string fed to the sampler, so the
  policy sees the exact format it was SFT'd on. With ``add_generation_prompt`` the
  string ends at ``<|im_start|>assistant\\n`` for the model to continue.
  """
  parts = []
  for msg in messages:
    role = msg.get("role", "user")
    content = msg.get("content") or ""
    parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
  if add_generation_prompt:
    parts.append("<|im_start|>assistant\n")
  return "".join(parts)


def encode_agent_conversation(
    tokenizer,
    messages: list[dict[str, Any]],
    max_seq_len: int,
    *,
    im_start_id: int,
    im_end_id: int,
    newline_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
  """Encodes one episode into ``(input_tokens, loss_mask, pad_mask)`` in ChatML.

  Each turn renders as ``<|im_start|>{role}\\n{content}<|im_end|>\\n``. The loss
  mask is 1 over assistant ``{content}<|im_end|>\\n`` (the model's own output +
  its stop token) and 0 everywhere else (turn headers, system/user/tool content).
  Right-padded to ``max_seq_len``; rows longer are truncated.

  Args:
    tokenizer: the Qwen3 HF tokenizer.
    messages: ``[{"role", "content"}, ...]`` (roles: system/user/assistant/tool).
    max_seq_len: padded length; longer conversations are truncated.
    im_start_id, im_end_id, newline_ids: from :func:`resolve_chatml_ids`.

  Returns:
    The ``(input_tokens, loss_mask, pad_mask)`` triple, or ``None`` if no
    assistant token survived (no training signal).
  """
  ids: list[int] = []
  loss: list[int] = []
  for msg in messages:
    role = msg.get("role", "user")
    content = msg.get("content") or ""
    train = 1 if role == "assistant" else 0
    # Header: <|im_start|>{role}\n  -- always context (mask 0).
    header = [im_start_id] + tokenizer.encode(role + "\n", add_special_tokens=False)
    ids.extend(header)
    loss.extend([0] * len(header))
    # Body: {content}<|im_end|>\n  -- trained only on assistant turns.
    body = tokenizer.encode(content, add_special_tokens=False)
    body.append(im_end_id)
    body.extend(newline_ids)
    ids.extend(body)
    loss.extend([train] * len(body))

  ids = ids[:max_seq_len]
  loss = loss[:max_seq_len]
  if 1 not in loss:  # no assistant token survived truncation -> drop
    return None

  real_len = len(ids)
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else im_end_id
  input_tokens = np.full(max_seq_len, int(pad_id), dtype=np.int32)
  loss_mask = np.zeros(max_seq_len, dtype=np.float32)
  pad_mask = np.zeros(max_seq_len, dtype=np.bool_)
  input_tokens[:real_len] = np.asarray(ids, dtype=np.int32)
  loss_mask[:real_len] = np.asarray(loss, dtype=np.float32)
  pad_mask[:real_len] = True
  return input_tokens, loss_mask, pad_mask


def _collect_encoded_rows(
    tokenizer,
    examples: Iterable[dict[str, Any]],
    n: int,
    seed: int,
    max_seq_len: int,
    *,
    scan_cap: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """Encodes a stream of ChatML examples into the first ``n`` usable SFT rows.

  Consumes any iterator of ``{"messages": [{"role","content"}, ...]}`` (a single
  corpus or a weighted blend), masks loss to assistant turns via
  :func:`encode_agent_conversation`, drops rows with no surviving assistant token,
  caps each row at ``max_seq_len``, shuffles with ``seed``, and cycles to fill if
  the stream runs dry before ``n`` rows. Returns exactly ``n`` rows.
  """
  im_start, im_end, newline_ids = resolve_chatml_ids(tokenizer)
  cap = scan_cap if scan_cap is not None else max(n * 4, 4000)
  rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
  scanned = 0
  dropped = 0
  truncated = 0
  for ex in examples:
    scanned += 1
    msgs = ex.get("messages", [])
    enc = encode_agent_conversation(
        tokenizer, msgs, max_seq_len,
        im_start_id=im_start, im_end_id=im_end, newline_ids=newline_ids,
    )
    if enc is None:
      dropped += 1
    else:
      # Flag rows that filled the whole window (likely truncated mid-episode).
      if int(enc[2].sum()) >= max_seq_len:
        truncated += 1
      rows.append(enc)
      if len(rows) >= n:
        break
    if scanned >= cap:
      break

  if not rows:
    raise ValueError(
        f"No usable agent episodes after scanning {scanned} rows "
        f"(all empty or longer than max_seq_len={max_seq_len})."
    )
  print(
      f"[agent-sft] scanned={scanned} usable={len(rows)} dropped={dropped} "
      f"window_full={truncated} (max_seq_len={max_seq_len})",
      flush=True,
  )
  random.Random(seed).shuffle(rows)
  if len(rows) < n:
    print(
        f"[agent-sft] only {len(rows)}/{n} usable rows; cycling to fill.",
        flush=True,
    )
    base = list(rows)
    while len(rows) < n:
      rows.append(base[len(rows) % len(base)])
  return rows[:n]


class _EncodedRowSource(grain.RandomAccessDataSource):
  """A grain random-access source over pre-encoded ``(tokens, loss, pad)`` rows."""

  def __init__(self, rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    self._rows = rows

  def __len__(self) -> int:
    return len(self._rows)

  def __getitem__(self, idx: int):
    return self._rows[idx]


class _AgentSFTSource(_EncodedRowSource):
  """A grain source of pre-tokenized agent episodes streamed from HuggingFace.

  Streams from :func:`agent_data.agent_traces.load_agent_traces`, encodes each
  episode with :func:`encode_agent_conversation`, and keeps the first ``n`` usable
  rows (shuffled with ``seed``; cycled to fill if the stream runs dry).
  """

  def __init__(
      self,
      tokenizer,
      n: int,
      seed: int,
      max_seq_len: int,
      *,
      limit: int | None = None,
      scan_cap: int | None = None,
  ):
    rows = _collect_encoded_rows(
        tokenizer, load_agent_traces(limit=limit), n, seed, max_seq_len,
        scan_cap=scan_cap,
    )
    super().__init__(rows)


class _MixtureSFTSource(_EncodedRowSource):
  """A grain source over a *weighted blend* of several trace corpora.

  Weight-interleaves the ``sources`` (per :func:`mixtures.interleave_sources`),
  encodes each example with assistant-turn masking, and keeps the first ``n``
  usable rows. The empirical per-source sampling ratio over the drawn rows matches
  the configured weights in expectation (modulo per-source caps / exhaustion).
  """

  def __init__(
      self,
      tokenizer,
      sources: list[DatasetSource],
      n: int,
      seed: int,
      max_seq_len: int,
      *,
      per_source_limit: int | None = None,
      scan_cap: int | None = None,
      shard_loader=None,
  ):
    names = ", ".join(f"{s.name}:{s.weight:g}" for s in sources)
    print(f"[agent-sft] mixture sources -> {names}", flush=True)
    stream = interleave_sources(
        sources, seed=seed, per_source_limit=per_source_limit,
        shard_loader=shard_loader,
    )
    # The blend can be very large; bound the scan so a build is finite even when
    # every source has many shards. Default: enough to fill n with headroom.
    cap = scan_cap if scan_cap is not None else max(n * 6, 6000)
    rows = _collect_encoded_rows(
        tokenizer, stream, n, seed, max_seq_len, scan_cap=cap,
    )
    super().__init__(rows)


def _to_columns(batch):
  input_tokens, loss_mask, pad_mask = batch
  return {
      "input_tokens": input_tokens,
      "loss_mask": loss_mask,
      "pad_mask": pad_mask,
  }


def build_agent_sft_dataset(
    tokenizer,
    n: int,
    seed: int,
    batch_size: int,
    max_seq_len: int,
    *,
    limit: int | None = None,
    sources: list[DatasetSource] | None = None,
    per_source_limit: int | None = None,
    shard_loader=None,
) -> grain.MapDataset:
  """Batched grain dataset of agent-SFT rows.

  Uses grain (not HF ``.batch()``) because tunix's ``jax.tree.map(np.repeat, ...)``
  collation corrupts HF-batched rows.

  If ``sources`` is given it builds a **weighted blend** of those corpora
  (multi-dataset mixing); otherwise it streams the single in-domain corpus
  (``load_agent_traces``), preserving the original behavior.
  """
  if sources:
    source: grain.RandomAccessDataSource = _MixtureSFTSource(
        tokenizer, sources, n, seed, max_seq_len,
        per_source_limit=per_source_limit, shard_loader=shard_loader,
    )
  else:
    source = _AgentSFTSource(tokenizer, n, seed, max_seq_len, limit=limit)
  return grain.MapDataset.source(source).batch(batch_size).map(_to_columns)


def run_agent_sft(
    model,
    tokenizer,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    mesh: jax.sharding.Mesh,
    max_seq_len: int,
    seed: int = 0,
    limit: int | None = None,
    sources: list[DatasetSource] | None = None,
    per_source_limit: int | None = None,
    checkpoint_dir: str | None = None,
    save_interval_secs: int = 600,
    max_to_keep: int = 2,
    metrics_options=None,
) -> Any:
  """SFTs ``model`` in place on agent traces and (optionally) checkpoints it.

  The actor must already be FSDP-sharded on ``mesh`` (the loader arranges this via
  ``mesh=``). When ``checkpoint_dir`` is set, ``PeftTrainer`` saves the nnx params
  via orbax periodically and forces a final save on close.

  Args:
    model: the Qwen3 ``nnx`` actor (fp32 params), already sharded on ``mesh``.
    tokenizer: the Qwen3 HF tokenizer.
    steps: number of SFT optimizer steps.
    batch_size: episodes per step.
    learning_rate: AdamW lr (clipped at global-norm 1.0).
    mesh: the device mesh the model is sharded on.
    max_seq_len: padded episode length; longer episodes are truncated.
    seed: PRNG seed for row shuffling.
    limit: cap on how many HF rows to stream from the single in-domain corpus
      (``None`` = all ~15.2k). Ignored when ``sources`` is given.
    sources: optional list of :class:`mixtures.DatasetSource` for a weighted
      multi-dataset blend; when set, the single-corpus path is bypassed.
    per_source_limit: optional cap on usable rows drawn per source in the blend
      (in addition to each source's own ``cap``); small for smoke runs.
    checkpoint_dir: orbax checkpoint root (local or ``gs://``); ``None`` disables.
    save_interval_secs: minimum seconds between periodic checkpoints.
    max_to_keep: number of checkpoints to retain.

  Returns:
    The same ``model`` object, now SFT'd.
  """
  n = (steps + 2) * batch_size
  dataset = build_agent_sft_dataset(
      tokenizer, n, seed, batch_size, max_seq_len, limit=limit,
      sources=sources, per_source_limit=per_source_limit,
  )
  optimizer = clipped_adamw(learning_rate)

  checkpointing_options = None
  if checkpoint_dir:
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_decision_policy=ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
            minimum_interval_secs=save_interval_secs,
        ),
        max_to_keep=max_to_keep,
    )

  trainer = PeftTrainer(
      model=model,
      optimizer=optimizer,
      training_config=TrainingConfig(
          eval_every_n_steps=10**9,
          max_steps=steps,
          metrics_logging_options=metrics_options,
          checkpoint_root_directory=checkpoint_dir,
          checkpointing_options=checkpointing_options,
      ),
  )
  trainer.with_gen_model_input_fn(sft_model_input_fn)
  print(
      f"[agent-sft] steps={steps} bs={batch_size} lr={learning_rate} "
      f"max_seq_len={max_seq_len} ckpt={checkpoint_dir}",
      flush=True,
  )
  with mesh:
    trainer.train(dataset)
  print("[agent-sft] complete", flush=True)
  return model
