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

"""Pluggable local weight-transfer (reshard) backend registry.

This module is the single source of truth for which *local* reshard backend
Tunix uses to move a pytree from a source sharding/mesh to a target
sharding/mesh inside one JAX program. It selects a backend by capability and
supports an optional explicit override, instead of hardcoding a Pathways-first
fallback in `reshard.py`.

Two backends exist today:

* ``PATHWAYS``  -- uses ``pathwaysutils.experimental.reshard`` when a Pathways
  proxy backend is attached (GCP single-controller multi-slice path).
* ``JAX_DEVICE`` -- uses plain ``jax.device_put``; correct for standard
  multi-controller JAX (CPU/GPU/TPU without Pathways) and disjoint in-program
  meshes.

`select_reshard_fns()` returns an ordered list of backend *factories* (each of
which is tried in turn by `reshard._get_reshard_fn`, falling through on
ImportError/EnvironmentError). The ``AUTO`` order ``[pathways, jax_device]`` is
byte-identical to the historical graceful-degradation order, so the default
behavior is unchanged.

IMPORTANT: this is LOCAL reshard only. No remote/server-client/NCCL/Arrow
transport lives here; that is a separate issue (#261). The backend objects only
carry a no-op `close()` hook so future remote transports can release handles.
"""

import enum
from typing import Any, Callable

from tunix.rl import reshard
from tunix.utils import env_utils


# A "reshard factory" is one of the `_get_reshard_fn_*` callables in
# `reshard.py`. Each accepts the keyword args (cache_resharding_plans, donate,
# use_experimental_pre_reshard) and returns a reshard function
# `(x, sharding) -> x`. `reshard._get_reshard_fn` tries them in order.
ReshardFactory = Callable[..., Callable[..., Any]]


@enum.unique
class LocalReshardBackend(enum.Enum):
  """Local reshard backend selector threaded through cluster config.

  Attributes:
    AUTO: Try Pathways first, then plain ``jax.device_put`` -- identical to the
      historical graceful-degradation order. Default; behavior unchanged.
    JAX_DEVICE: Only use ``jax.device_put``. Skips the Pathways import attempt,
      which is cleaner/faster off-GCP and avoids a noisy import log line.
    PATHWAYS: Only use the Pathways backend. Errors clearly if Pathways is not
      available (e.g. ``JAX_PLATFORMS`` lacks ``proxy``).
  """

  AUTO = "auto"
  JAX_DEVICE = "jax_device"
  PATHWAYS = "pathways"


# Named, documented references to the local reshard backend factories. These
# live in `reshard.py` (so the model-load path can keep importing them there),
# and are re-exported here so callers have a single, documented registry.
PATHWAYS_RESHARD_FACTORY: ReshardFactory = (
    reshard._get_reshard_fn_pathwaysutils  # pylint: disable=protected-access
)
JAX_DEVICE_RESHARD_FACTORY: ReshardFactory = (
    reshard._get_reshard_fn_jax_device_put  # pylint: disable=protected-access
)

# AUTO order. Kept as a module constant so it is provably the same list the
# historical `reshard_pytree` used inline.
_AUTO_RESHARD_FACTORIES: list[ReshardFactory] = [
    PATHWAYS_RESHARD_FACTORY,
    JAX_DEVICE_RESHARD_FACTORY,
]


def is_pathways_available() -> bool:
  """Returns whether a Pathways backend is currently attached.

  This is the single source of truth for the Pathways capability used by the
  weight-transfer selection. It wraps `env_utils.is_pathways_initialized()`,
  which imports ``pathwaysutils`` and asks the runtime, returning False when
  ``pathwaysutils`` is absent.

  Returns:
    True if a Pathways backend is in use, otherwise False.
  """
  return env_utils.is_pathways_initialized()


def capabilities() -> dict[str, Any]:
  """Returns the resolved local weight-transfer capabilities.

  Returns:
    A dict with:
      * ``pathways`` (bool): whether a Pathways backend is attached.
      * ``backend`` (str): the backend ``AUTO`` would prefer right now, i.e.
        ``"pathways"`` when Pathways is attached, else ``"jax_device"``.
  """
  pathways = is_pathways_available()
  return {
      "pathways": pathways,
      "backend": (
          LocalReshardBackend.PATHWAYS.value
          if pathways
          else LocalReshardBackend.JAX_DEVICE.value
      ),
  }


def select_reshard_fns(
    backend: LocalReshardBackend = LocalReshardBackend.AUTO,
) -> list[ReshardFactory]:
  """Returns the ordered reshard-factory list for the requested backend.

  The returned list is consumed by `reshard.reshard_pytree` /
  `reshard._get_reshard_fn`, which tries each factory in order and falls through
  to the next on ImportError/EnvironmentError.

  Args:
    backend: Which local backend to use. ``AUTO`` (default) reproduces the
      historical ``[pathways, jax_device]`` graceful-degradation order.

  Returns:
    A new list of reshard factories (a fresh list each call so callers may not
    mutate shared state).

  Raises:
    ValueError: If ``backend`` is not a known `LocalReshardBackend`.
  """
  if backend == LocalReshardBackend.AUTO:
    return list(_AUTO_RESHARD_FACTORIES)
  if backend == LocalReshardBackend.JAX_DEVICE:
    return [JAX_DEVICE_RESHARD_FACTORY]
  if backend == LocalReshardBackend.PATHWAYS:
    return [PATHWAYS_RESHARD_FACTORY]
  raise ValueError(f"Unknown local reshard backend: {backend!r}.")


class LocalWeightTransfer:
  """Local, in-program reshard backend with an injectable factory list.

  This is the dependency injected into `RLCluster`/rollout construction instead
  of a global mutable singleton. It resolves the reshard-factory list once from
  the requested `LocalReshardBackend` and exposes:

  * `reshard_fns`: the resolved factory list to pass to
    `reshard.reshard_pytree(..., reshard_fns=...)`.
  * `close()`: a no-op cleanup hook. Local resharding holds no out-of-band
    handles, so there is nothing to release today; the hook exists so a future
    remote transport (separate issue) can release sockets/buffers symmetrically
    from `RLCluster.close()`.
  """

  def __init__(self, backend: LocalReshardBackend = LocalReshardBackend.AUTO):
    """Initializes the local weight transfer.

    Args:
      backend: The local reshard backend to use. Defaults to ``AUTO`` so the
        behavior is byte-identical to the historical fallback order.
    """
    self._backend = backend
    self._reshard_fns = select_reshard_fns(backend)

  @property
  def backend(self) -> LocalReshardBackend:
    """The configured local reshard backend selector."""
    return self._backend

  @property
  def reshard_fns(self) -> list[ReshardFactory]:
    """The resolved, ordered reshard-factory list for `reshard_pytree`."""
    return list(self._reshard_fns)

  def close(self) -> None:
    """Releases any transport handles. No-op for local resharding.

    Local resharding moves data with ``jax.device_put`` / pathwaysutils and
    owns no sockets, files, or pinned buffers, so this does nothing today. It is
    called from `RLCluster.close()` so that a future remote weight-transfer
    backend can release its handles through the same hook without changing the
    cluster shutdown path.
    """
    return None
