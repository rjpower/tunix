"""Most-basic tunix run on the marin/iris TPU cluster.

This is the "hello world" that proves the whole path works end-to-end:
iris ships this repo's bundle -> the worker `uv sync`s google-tunix + deps ->
JAX sees the attached v6e TPU -> a real tunix model runs on it. No weights are
downloaded and no env vars are required.

Submitted to iris as the literal command `python marin_smoke.py` on a v6e slice
(see MARIN.md "Hello world" for the exact submit command). It:

  1. (multi-host only) initializes JAX distributed before any other jax call;
  2. prints the JAX / tunix versions and the TPU devices iris attached;
  3. runs a trivial sharded computation across every TPU device;
  4. instantiates a real (random-init) tunix Qwen3-0.6B over an (fsdp, tp) TPU
     mesh and runs one forward pass -- exercising the tunix model + sharding +
     XLA-on-TPU compile path without any checkpoint.

Each line is tagged ``[marin-smoke]`` so it is easy to grep out of libtpu noise:
    uv run iris --cluster=marin job logs /power/<job> --max-lines 4000 | grep marin-smoke
"""

import os
from importlib.metadata import version

import jax
import jax.numpy as jnp
from flax import nnx

from tunix.models.dummy_model_creator import create_dummy_model
from tunix.models.qwen3 import model as qwen3


def main() -> None:
  # v6e-8/-16 span >1 host; orbax/collectives need this before any other jax
  # call. A single-host v6e-4 sets JAX_NUM_PROCESSES=1 and skips it. (Invariant
  # in MARIN.md: jax.distributed.initialize() first on multi-host.)
  if int(os.environ.get("JAX_NUM_PROCESSES", "1")) > 1:
    jax.distributed.initialize()

  devices = jax.devices()
  print(f"[marin-smoke] google-tunix {version('google-tunix')}, jax {jax.__version__}")
  print(f"[marin-smoke] process {jax.process_index()}/{jax.process_count()}, "
        f"{len(devices)} devices, kind={devices[0].device_kind!r}")
  print(f"[marin-smoke] devices: {devices}")

  # (3) trivial computation on every device -- proves the TPU actually computes.
  n = len(devices)
  x = jnp.arange(n * 1024, dtype=jnp.float32).reshape(n, 1024)
  sq = jax.pmap(lambda a: jnp.sum(a * a))(x)
  print(f"[marin-smoke] pmap sum-of-squares per device: {sq}")

  # (4) a real tunix Qwen3-0.6B (random init) sharded over the TPU, one forward.
  # The default sharding references the 'fsdp' and 'tp' axes, so the mesh must
  # carry both (tp=1 => no tensor-parallel split, fine for a smoke). axis_types
  # = Auto puts the mesh in GSPMD mode so XLA resolves the sharded embedding
  # gather (the default "explicit" sharding-in-types mode rejects it).
  mesh = jax.make_mesh(
      (n, 1), ("fsdp", "tp"),
      axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
  )
  cfg = qwen3.ModelConfig.qwen3_0p6b()
  model = create_dummy_model(qwen3.Qwen3, cfg, mesh=mesh)

  batch, seqlen = 1, 8
  tokens = jnp.ones((batch, seqlen), dtype=jnp.int32)
  positions = jnp.arange(seqlen, dtype=jnp.int32)[None, :]
  causal = jnp.tril(jnp.ones((seqlen, seqlen), dtype=jnp.bool_))[None, :, :]

  # Run the forward under jit: XLA's GSPMD partitioner resolves the sharded
  # embedding gather (eager execution can't). This is how tunix runs the model
  # everywhere (training + sampling).
  @nnx.jit
  def forward(m, toks, pos, mask):
    return m(toks, pos, None, mask)

  with jax.set_mesh(mesh):
    logits, _ = forward(model, tokens, positions, causal)
  logits.block_until_ready()
  print(f"[marin-smoke] tunix Qwen3-0.6B forward OK -- logits {logits.shape} "
        f"dtype {logits.dtype}")
  print("[marin-smoke] SUCCESS")


if __name__ == "__main__":
  main()
