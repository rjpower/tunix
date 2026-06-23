# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flat state-dict (de)serialization for transferable Tunix parameter pytrees.

marin's Arrow Flight transfer leans on Haliax ``state_dict`` to turn a model
into ``{name: flat_array}`` and back. Tunix models are plain JAX/flax pytrees
(``nnx.state(model)`` is itself a pytree), so we do the same thing with
``jax.tree_util`` keypaths -- no Haliax dependency:

* `flatten_for_transfer(params)` casts floats to bf16 (optional), flattens each
  leaf to 1-D *on device* (so the device->host copy is already bf16 + reshaped),
  materializes to host, and returns ``(ordered_keys, {key: np.ndarray})``.
* `restore_from_flat(flat, template)` reshapes each flat array back to the
  matching *template* leaf's shape, casts to the template dtype, and
  ``device_put``s it onto the template leaf's sharding -- so the restored pytree
  has exactly the inference worker's mesh/sharding. The wire format never needs
  to carry shapes or shardings: the receiver's template is the source of truth.

The key for a leaf is ``jax.tree_util.keystr(path)``; it is identical on both
sides as long as both flatten the same pytree structure (the trainer's params
and the rollout worker's template share structure).
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree


def flat_keys(tree: PyTree) -> list[str]:
  """Returns the ordered `keystr` keys of ``tree``'s leaves."""
  items, _ = jax.tree_util.tree_flatten_with_path(tree)
  return [jax.tree_util.keystr(path) for path, _ in items]


def _cast_bf16(params: PyTree, convert_to_bfloat16: bool) -> PyTree:
  """Casts floating leaves to bf16 (optional). Elementwise only.

  IMPORTANT: this is intentionally *not* jit'd and does *not* reshape on device.
  A device-side ``reshape(-1)`` of a sharded tensor compiles an all-gather; on
  multi-host TPU a jit confined to one host's local-device mesh (a subset of the
  global topology) makes XLA's collective lowering reference devices on other
  hosts and crash (RET_CHECK ``device_id < kMaxDeviceCount``). An elementwise
  ``astype`` has no cross-device dependency, so it lowers per-shard and is safe
  on a sub-global mesh; the reshape happens on host (see `flatten_for_transfer`).
  """

  def f(x):
    if convert_to_bfloat16 and jnp.issubdtype(x.dtype, jnp.floating):
      return x.astype(jnp.bfloat16)
    return x

  return jax.tree.map(f, params)


def flatten_for_transfer(
    params: PyTree, *, convert_to_bfloat16: bool = True
) -> tuple[list[str], dict[str, np.ndarray]]:
  """Flattens ``params`` to host-resident 1-D arrays keyed by leaf keypath.

  The bf16 cast happens on device (elementwise, collective-free); the array is
  then ``device_get``-assembled to host from its local shards, and reshaped to
  1-D *on host* (free numpy view). Doing the reshape on host avoids an on-device
  all-gather that breaks multi-host TPU lowering.

  Args:
    params: The parameter pytree to serialize (flax/nnx leaves).
    convert_to_bfloat16: Cast floating leaves to bf16 before the host copy.

  Returns:
    ``(ordered_keys, flat)`` where ``flat[key]`` is a contiguous 1-D
    ``np.ndarray`` (the leaf's bytes; dtype carries the bf16 cast if applied).
    ``ordered_keys`` preserves leaf order for stable iteration.
  """
  cast_tree = _cast_bf16(params, convert_to_bfloat16)
  host_tree = jax.device_get(cast_tree)
  items, _ = jax.tree_util.tree_flatten_with_path(host_tree)
  keys: list[str] = []
  flat: dict[str, np.ndarray] = {}
  for path, value in items:
    key = jax.tree_util.keystr(path)
    keys.append(key)
    flat[key] = np.ascontiguousarray(np.asarray(value).reshape(-1))
  return keys, flat


def restore_from_flat(flat: dict[str, np.ndarray], template: PyTree) -> PyTree:
  """Rebuilds a pytree shaped/sharded like ``template`` from ``flat``.

  Each template leaf at keypath ``k`` is filled with ``flat[k]`` reshaped to the
  template leaf's shape, cast to its dtype, and ``device_put`` onto its sharding
  (so sharded targets are placed correctly). The flat arrays may be bf16 (from
  `flatten_for_transfer`); the cast to the template dtype undoes that for the
  inference worker as desired.

  Args:
    flat: ``{key: 1-D host array}`` as produced by `flatten_for_transfer`.
    template: Pytree carrying the target structure / shape / dtype / sharding.

  Returns:
    A pytree with ``template``'s structure and shardings, filled from ``flat``.

  Raises:
    KeyError: If a template leaf's key is missing from ``flat``.
  """
  items, treedef = jax.tree_util.tree_flatten_with_path(template)
  out_leaves = []
  for path, tmpl_leaf in items:
    key = jax.tree_util.keystr(path)
    if key not in flat:
      raise KeyError(f"Transferred state is missing parameter {key!r}.")
    value = np.asarray(flat[key]).reshape(jnp.shape(tmpl_leaf))
    target_dtype = getattr(tmpl_leaf, "dtype", value.dtype)
    if np.dtype(value.dtype) != jnp.dtype(target_dtype):
      value = value.astype(target_dtype)  # cast on host, not on device
    sharding = getattr(tmpl_leaf, "sharding", None)
    # device_put a host array straight to the target sharding: jax scatters the
    # shards directly (no full-tensor staging on one device, and no on-device
    # collective -- both matter on multi-host TPU).
    if sharding is not None:
      out_leaves.append(jax.device_put(value, sharding))
    else:
      out_leaves.append(jax.device_put(value))
  return jax.tree_util.tree_unflatten(treedef, out_leaves)


def summarize(flat: dict[str, np.ndarray]) -> tuple[int, int, str | None]:
  """Returns ``(total_bytes, largest_param_bytes, largest_param_name)``."""
  total = 0
  largest = 0
  largest_name: str | None = None
  for name, value in flat.items():
    nbytes = int(np.asarray(value).nbytes)
    total += nbytes
    if nbytes > largest:
      largest = nbytes
      largest_name = name
  return total, largest, largest_name
