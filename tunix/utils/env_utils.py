# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environment utils."""

import os

import flax


def setup_sharding_environment():
  """Sets up the sharding environment."""
  if hasattr(flax.config, 'flax_always_shard_variable'):
    flax.config.update('flax_always_shard_variable', False)


def is_internal_env():
  """Checks if the code is running within the internal environment."""
  try:
    from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile  # noqa: F401

    return True
  except ImportError:
    return False


def is_pathways_initialized():
  """Checks if Pathways is initialized."""
  try:
    import pathwaysutils  # noqa: F401
    return pathwaysutils.is_pathways_backend_used()
  except ImportError:
    return False


def is_pathways_proxy_backend():
  """Checks whether the Pathways proxy is selected as the JAX backend.

  This is the single source of truth for the ``'proxy' in JAX_PLATFORMS``
  predicate: it is the capability that gates the Pathways reshard backend and
  the Pathways persistence-API checkpoint path. It is intentionally distinct
  from `is_pathways_initialized()`, which asks pathwaysutils whether a Pathways
  backend is in use; this one only inspects the requested JAX backend env, which
  is what those call sites historically branched on.

  Returns:
    True if ``JAX_PLATFORMS`` requests the Pathways proxy backend, else False.
  """
  return 'proxy' in os.getenv('JAX_PLATFORMS', '')


SGLANG_JAX_TP_AXIS_NAME = os.getenv('SGLANG_JAX_TP_AXIS_NAME', 'tensor')
