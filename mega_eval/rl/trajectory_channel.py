"""GCS-staged trajectory channel for disaggregated RL: rollout workers PUT
trajectory batches, the trainer POLLs + CONSUMEs them.

This is the rollout->trainer return path (the counterpart to the Arrow weight
path trainer->rollout). A durable object-store buffer is the natural fit for the
async N-rollouts -> 1-trainer topology: workers append at their own pace, the
trainer drains the backlog, and nothing is lost if either side restarts. The user
endorsed GCS for TPU; we use fsspec so the same code drives ``gs://`` on the
cluster and a local dir in tests.

Layout under ``base_uri``:
  pending/<seq>-<uid>.npz   a serialized batch waiting to be trained on
The trainer lists ``pending/``, reads each, trains, then ``consume()``s (deletes)
it. Writes go to a ``.tmp`` key then are renamed so a lister never sees a partial
object (GCS object writes are atomic; the rename keeps local-fs tests honest too).

A "batch" is a flat ``{name: np.ndarray}`` dict (+ JSON-able scalars under
``__meta__``) -- e.g. the GRPO rollout fields prompt_ids / completion_ids / masks
/ old_logprobs / rewards / policy_version, stacked over the [B*G] group. Serialized
with ``np.savez_compressed`` (portable, no pickle).
"""

import io
import json
import time
import uuid

import numpy as np


def serialize_batch(arrays: dict, meta: dict | None = None) -> bytes:
  """{name: np.ndarray} (+ JSON-able meta) -> npz bytes."""
  buf = io.BytesIO()
  payload = {k: np.asarray(v) for k, v in arrays.items()}
  if meta is not None:
    payload["__meta__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )
  np.savez_compressed(buf, **payload)
  return buf.getvalue()


def deserialize_batch(data: bytes) -> tuple[dict, dict]:
  """npz bytes -> ({name: np.ndarray}, meta dict)."""
  with np.load(io.BytesIO(data), allow_pickle=False) as npz:
    arrays = {k: npz[k] for k in npz.files if k != "__meta__"}
    meta = {}
    if "__meta__" in npz.files:
      meta = json.loads(bytes(npz["__meta__"]).decode("utf-8"))
  return arrays, meta


class TrajectoryChannel:
  """Object-store queue of trajectory batches (one RL run = one ``base_uri``)."""

  def __init__(self, base_uri: str):
    self._base = base_uri.rstrip("/")
    self._pending = f"{self._base}/pending"
    self._fs = self._make_fs(base_uri)
    self._seq = 0

  @staticmethod
  def _make_fs(base_uri: str):
    import fsspec  # pylint: disable=g-import-not-at-top

    if base_uri.startswith("gs://"):
      return fsspec.filesystem("gcs")
    if base_uri.startswith("s3://"):
      return fsspec.filesystem("s3")
    fs = fsspec.filesystem("file")
    return fs

  # ---- rollout side -------------------------------------------------------
  def put(self, arrays: dict, meta: dict | None = None) -> str:
    """Serialize + append a batch; returns its key. Atomic via tmp+rename."""
    self._seq += 1
    name = f"{self._seq:08d}-{uuid.uuid4().hex[:8]}.npz"
    final = f"{self._pending}/{name}"
    tmp = f"{final}.tmp"
    data = serialize_batch(arrays, meta)
    try:
      # Local fs needs the parent dir; GCS/S3 have no real dirs (no-op there).
      self._fs.makedirs(self._pending, exist_ok=True)
    except Exception:  # pylint: disable=broad-except
      pass
    with self._fs.open(tmp, "wb") as f:
      f.write(data)
    self._fs.mv(tmp, final)
    return name

  # ---- trainer side -------------------------------------------------------
  def list_pending(self) -> list[str]:
    """Sorted keys of complete pending batches (ignores .tmp)."""
    try:
      paths = self._fs.ls(self._pending, detail=False)
    except FileNotFoundError:
      return []
    names = [p.rsplit("/", 1)[-1] for p in paths]
    return sorted(n for n in names if n.endswith(".npz"))

  def get(self, key: str) -> tuple[dict, dict]:
    with self._fs.open(f"{self._pending}/{key}", "rb") as f:
      return deserialize_batch(f.read())

  def consume(self, key: str) -> None:
    try:
      self._fs.rm(f"{self._pending}/{key}")
    except FileNotFoundError:
      pass

  def drain(self, max_items: int | None = None):
    """Yields (key, arrays, meta) for pending batches; caller consume()s them."""
    keys = self.list_pending()
    if max_items is not None:
      keys = keys[:max_items]
    for key in keys:
      arrays, meta = self.get(key)
      yield key, arrays, meta

  def wait_for_batch(self, timeout_s: float, poll_s: float = 2.0):
    """Blocks until >=1 pending batch exists; returns its keys (or [] on timeout)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
      keys = self.list_pending()
      if keys:
        return keys
      time.sleep(poll_s)
    return []
