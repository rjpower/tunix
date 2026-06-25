# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-node JAX rendezvous for the OT-Agent GPU run.

A 32x H100 SFT run is 4 iris tasks (one per 8-GPU node) that must join **one**
JAX distributed world so a single ``(fsdp, tp)`` mesh spans all 32 devices.
Unlike a TPU slice (where the runtime auto-discovers the coordinator) and unlike
a single 8x H100 box (where no distributed client is needed), a multi-node GPU
job needs an explicit coordinator address that every process agrees on.

iris already solves this: ``iris.runtime.jax_init.initialize_jax`` has task 0
register its coordinator address in the cluster-global endpoint registry and the
other tasks poll for it, then all call ``jax.distributed.initialize`` with the
shared address. This is the path the verified ``gpu_gang_smoke`` (4-node H100,
cross-host NCCL all-reduce) uses, so we delegate to it rather than re-implement
the rendezvous.

``initialize_jax`` covers every case we hit:
  * **TPU** -- calls ``jax.distributed.initialize()`` (runtime autodiscovery).
  * **single iris task** (1x 8-GPU node) -- explicit single-process init.
  * **multi-task iris job** (our 4-node run) -- registry rendezvous.
  * **not inside an iris job** (local CPU dev) -- no-op (``get_job_info`` None).

It MUST run before any other JAX call (orbax barriers + collectives need the
client up). Make it line 1 of ``main()``. Idempotent: a second call is a no-op.

(This is deliberately separate from ``mega_eval.training.common.init_distributed``,
which guards on ``PJRT_DEVICE==TPU`` and *skips* GPU because bare
``jax.distributed.initialize()`` autodiscovery hangs on GPU. ``initialize_jax``
passes an explicit coordinator, so it does not hang -- it is the correct GPU
path.)
"""

from __future__ import annotations

import os


def init_distributed() -> None:
  """Bring up the JAX distributed client via the iris endpoint registry.

  Safe on TPU, single-node GPU, multi-node GPU, and off-cluster (CPU). Prints a
  diagnostic so the job log shows the resolved world size. Never raises on the
  off-cluster path -- if ``marin-iris`` is not importable (a bare CPU checkout)
  we skip init, which is correct for a single local process.
  """
  try:
    from iris.runtime.jax_init import initialize_jax  # noqa: PLC0415
  except Exception as e:  # pylint: disable=broad-except
    # No marin-iris (pure-CPU dev box). A single local process needs no
    # distributed client; JAX defaults are correct.
    print(f"[ota-dist] iris.runtime.jax_init unavailable ({e!r}); "
          "skipping distributed init (single-process).", flush=True)
    return

  initialize_jax()

  # Import jax only after init so we never touch the backend before the
  # distributed client is up.
  import jax  # noqa: PLC0415

  print(
      f"[ota-dist] jax {jax.__version__} "
      f"process {jax.process_index()}/{jax.process_count()} "
      f"local_devices={jax.local_device_count()} global_devices={jax.device_count()} "
      f"host={os.uname().nodename}",
      flush=True,
  )
