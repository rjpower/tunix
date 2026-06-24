# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export an SFT'd tunix Qwen3 actor to a HuggingFace-format safetensors model.

Why not orbax? On the CW GPU cluster the 4 nodes share **no filesystem** and
orbax cannot write ``s3://`` (etils.epath has no s3 backend; ``os.path.normpath``
mangles ``s3://`` -> ``s3:/``). A multi-host orbax checkpoint therefore has
nowhere coherent to land. Instead we **gather** the sharded actor to host and
write a single HF-format safetensors checkpoint that the *same* ``load_qwen3``
loader reads back (so eval / RL / serving all load it exactly like the base
model), then mirror it to R2.

Correctness is anchored on the **base model's** safetensors headers: every torch
key's exact shape and dtype are known, so inverting the loader's
``(transpose, reshape)`` transform is exact (no shape guessing). The forward
transform (``tunix.models.safetensors_loader``) is ``transpose(permute)`` then
``reshape(reshape)``; we invert it as ``reshape(post_transpose_shape)`` then
``transpose(argsort(permute))``. A bidirectional coverage assert (every model
param maps to a torch key and vice-versa) fails loudly on any drift.

Multi-host gather: each param is collected to a replicated host array via
``jax.experimental.multihost_utils.process_allgather`` (a no-op device_get when
single-process); only process 0 writes + mirrors.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import struct
from typing import Any

import jax
import numpy as np

from tunix.models.qwen3 import params as qp
from tunix.models.qwen3 import model as qm
from tunix.utils.torch_utils import torch_key_to_jax_key

# Reuse the R2 credential mapping + s3 KvStore from the staging helper.
from mega_eval.models.checkpoint_staging import _map_r2_to_aws_env, _s3_kvstore

_SHARD_BYTES = 5 * 1024**3  # ~5 GiB safetensors shards (HF convention)

# safetensors dtype tags for the numpy dtypes we emit.
_ST_DTYPE = {
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
    np.dtype("bfloat16"): "BF16",  # numpy has no bf16; handled via ml_dtypes below
}


def _read_safetensors_headers(model_dir: str) -> dict[str, dict[str, Any]]:
  """Returns ``{torch_key: {"shape": [...], "dtype": "BF16", ...}}`` for the base model."""
  headers: dict[str, dict[str, Any]] = {}
  shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
  if not shards:
    raise ValueError(f"No .safetensors in {model_dir} to anchor export shapes.")
  for path in shards:
    with open(path, "rb") as f:
      n = struct.unpack("<Q", f.read(8))[0]
      hdr = json.loads(f.read(n).decode("utf-8"))
    for k, meta in hdr.items():
      if k != "__metadata__":
        headers[k] = meta
  return headers


def _build_nnx_to_torch(config: qm.ModelConfig, torch_headers: dict[str, dict[str, Any]]):
  """Maps each nnx param key -> (torch_key, transform, torch_shape, torch_dtype)."""
  key_map = qp._get_key_and_transform_mapping(config)  # pylint: disable=protected-access
  out: dict[str, tuple] = {}
  for tk, meta in torch_headers.items():
    jax_key, transform = torch_key_to_jax_key(key_map, tk)
    out[jax_key] = (tk, transform, tuple(meta["shape"]), meta["dtype"])
  return out


def _dotted(path) -> str:
  """Joins an nnx tree path into a dotted key (e.g. ``layers.5.attn.q_proj.w``)."""
  parts = []
  for p in path:
    k = getattr(p, "key", p)
    parts.append(str(k))
  key = ".".join(parts)
  # The nnx Param array sometimes nests under a trailing ``value`` key.
  if key.endswith(".value"):
    key = key[: -len(".value")]
  return key


def _gather_host(x: jax.Array) -> np.ndarray:
  """Gathers a (possibly multi-host sharded) array to a replicated host numpy."""
  if jax.process_count() > 1:
    from jax.experimental import multihost_utils  # noqa: PLC0415
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))
  return np.asarray(jax.device_get(x))


def _invert_transform(arr: np.ndarray, transform, torch_shape: tuple[int, ...]) -> np.ndarray:
  """Inverts the loader's ``transpose -> reshape`` to recover the torch tensor."""
  if transform is None:
    return arr.reshape(torch_shape)
  permute, _reshape = transform
  if permute:
    # post-transpose shape = torch_shape permuted; undo reshape into it, then
    # undo the transpose with the inverse permutation.
    post_transpose_shape = tuple(torch_shape[i] for i in permute)
    arr = arr.reshape(post_transpose_shape)
    inv = tuple(int(i) for i in np.argsort(permute))
    arr = arr.transpose(inv)
  return arr.reshape(torch_shape)


def _to_torch_state(model, model_dir: str) -> dict[str, np.ndarray]:
  """Gathers + converts the nnx actor into a ``{torch_key: np.ndarray}`` state dict."""
  from flax import nnx  # noqa: PLC0415

  config = model.config
  torch_headers = _read_safetensors_headers(model_dir)
  nnx_to_torch = _build_nnx_to_torch(config, torch_headers)

  _, state = nnx.split(model)
  pure = state.to_pure_dict()

  collected: dict[str, np.ndarray] = {}
  seen_nnx: set[str] = set()

  leaves_with_paths = jax.tree_util.tree_leaves_with_path(pure)
  for path, leaf in leaves_with_paths:
    if not isinstance(leaf, jax.Array) and not isinstance(leaf, np.ndarray):
      continue
    nnx_key = _dotted(path)
    if nnx_key not in nnx_to_torch:
      raise ValueError(
          f"nnx param {nnx_key!r} has no torch mapping; export would drop it "
          f"(known: {sorted(nnx_to_torch)[:4]}...)."
      )
    tk, transform, torch_shape, torch_dtype = nnx_to_torch[nnx_key]
    host = _gather_host(leaf)
    torch_arr = _invert_transform(host, transform, torch_shape)
    collected[tk] = (torch_arr, torch_dtype)
    seen_nnx.add(nnx_key)

  missing = set(nnx_to_torch) - seen_nnx
  if missing:
    raise ValueError(f"{len(missing)} base-model torch keys had no matching nnx "
                     f"param: {sorted(missing)[:6]}...")
  return collected


def _cast(arr: np.ndarray, torch_dtype: str) -> np.ndarray:
  """Casts a gathered fp32 array back to the base model's stored dtype."""
  import ml_dtypes  # noqa: PLC0415

  table = {"F32": np.float32, "F16": np.float16, "BF16": ml_dtypes.bfloat16}
  np_dtype = table.get(torch_dtype, np.float32)
  return arr.astype(np_dtype)


def _write_sharded_safetensors(state: dict[str, tuple], out_dir: str) -> None:
  """Writes the state dict as HF-convention sharded safetensors + an index."""
  import safetensors.numpy as safe_np  # noqa: PLC0415

  os.makedirs(out_dir, exist_ok=True)
  # Greedy bin-pack keys into ~_SHARD_BYTES shards.
  items = list(state.items())  # [(torch_key, (arr, dtype_tag))]
  shards: list[list[str]] = [[]]
  sizes = [0]
  for tk, (arr, dt) in items:
    nbytes = arr.size * _cast(arr[:0] if arr.size else arr, dt).dtype.itemsize
    if sizes[-1] and sizes[-1] + nbytes > _SHARD_BYTES:
      shards.append([])
      sizes.append(0)
    shards[-1].append(tk)
    sizes[-1] += nbytes

  total = len(shards)
  weight_map: dict[str, str] = {}
  for i, keys in enumerate(shards):
    fname = (f"model-{i + 1:05d}-of-{total:05d}.safetensors"
             if total > 1 else "model.safetensors")
    tensors = {tk: _cast(state[tk][0], state[tk][1]) for tk in keys}
    safe_np.save_file(tensors, os.path.join(out_dir, fname),
                      metadata={"format": "pt"})
    for tk in keys:
      weight_map[tk] = fname
    print(f"[ota-export] wrote {fname} ({len(keys)} tensors)", flush=True)

  if total > 1:
    total_bytes = int(sum(sizes))
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
      json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, f)


def _copy_aux_files(model_dir: str, out_dir: str) -> None:
  """Copies config.json, tokenizer, generation config -- everything but weights."""
  for fn in os.listdir(model_dir):
    if fn.endswith(".safetensors") or fn.endswith(".safetensors.index.json"):
      continue
    src = os.path.join(model_dir, fn)
    if os.path.isfile(src):
      shutil.copy(src, os.path.join(out_dir, fn))


def _mirror_to_s3(local_dir: str, s3_uri: str) -> None:
  """Uploads every file under ``local_dir`` to an ``s3://`` (R2) prefix via tensorstore."""
  _map_r2_to_aws_env()
  bucket, _, prefix = s3_uri[len("s3://"):].partition("/")
  kv = _s3_kvstore(bucket, prefix)
  files = [os.path.relpath(os.path.join(r, f), local_dir)
           for r, _, fs in os.walk(local_dir) for f in fs]
  for rel in files:
    with open(os.path.join(local_dir, rel), "rb") as fh:
      kv.write(rel, fh.read()).result()
    print(f"[ota-export] mirrored s3 <- {rel}", flush=True)


def export_and_mirror(model, model_dir: str, export_dir: str, *, mesh=None,
                      local_staging: str = "./_export_hf") -> str:
  """Gathers ``model``, writes a HF-format checkpoint, mirrors it to R2 if remote.

  Args:
    model: the SFT'd tunix Qwen3 nnx actor (sharded on ``mesh``).
    model_dir: the base model dir (anchors export shapes + supplies config/tokenizer).
    export_dir: destination -- a local dir or ``s3://marin-na/...`` (R2).
    mesh: the actor's device mesh (entered for the gather collectives).
    local_staging: local scratch dir used when ``export_dir`` is ``s3://``.

  Returns:
    The local directory the HF checkpoint was written to.
  """
  is_remote = export_dir.startswith("s3://")
  local_dir = local_staging if is_remote else export_dir

  ctx = mesh if mesh is not None else _nullcontext()
  with ctx:
    state = _to_torch_state(model, model_dir)

  if jax.process_index() == 0:
    print(f"[ota-export] gathered {len(state)} tensors; writing -> {local_dir}", flush=True)
    if os.path.exists(local_dir):
      shutil.rmtree(local_dir)
    _write_sharded_safetensors(state, local_dir)
    _copy_aux_files(model_dir, local_dir)
    if is_remote:
      _mirror_to_s3(local_dir, export_dir)
    print(f"[ota-export] DONE -> {export_dir}", flush=True)
  # Barrier so non-zero processes do not exit before process 0 finishes writing.
  if jax.process_count() > 1:
    from jax.experimental import multihost_utils  # noqa: PLC0415
    multihost_utils.sync_global_devices("ota-export-done")
  return local_dir


class _nullcontext:
  def __enter__(self):
    return None

  def __exit__(self, *a):
    return False
