# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# One-time HF -> Levanter checkpoint conversion for the OpenThoughts-Agent SFT.
#
# WHY THIS EXISTS
# ---------------
# Qwen3-32B at seq 32768 fits the *train step* on 32xH100 only with 2D sharding
# (TP=8 over NVLink + FSDP over the rest; see ot_agent/levanter_sft.py). But the
# one-shot HF->2D-sharded weight load is a different problem: when the trainer has
# already built the ~22GB optimizer state, the monolithic HF->2D conversion jit's
# ~52GB reshard transient stacked on top exceeds 80GB -> OOM mid-load. There is no
# single mesh that clears both that load *and* the train step.
#
# THE FIX: decouple the HF conversion from training. Convert HF -> a Levanter
# Tensorstore checkpoint ONCE, with NO optimizer state and NO activations in play,
# then have the SFT run warm-start from it via a gentle per-array tensorstore load
# into the 2D layout (ot_agent/levanter_sft.py::_patch_load_pretrained_from_checkpoint).
#
# HOW THE CONVERT ITSELF AVOIDS OOM (the important part)
# -----------------------------------------------------
# Levanter's loader is already sharding-aware: `_load_safe_tensors` streams each
# safetensors tensor from R2/HF and `device_put`s it sharded over the active mesh's
# `data` axis (`best_effort_sharding`), and `load_pretrained` builds the model
# inside a `named_jit` whose outputs shard per the axis mapping. So under a real
# multi-GPU mesh, NOTHING is ever materialized whole on one device.
#
# The earlier CPU version (`use_cpu_device()` / JAX_PLATFORMS=cpu) defeated exactly
# this: a 1-device CPU "mesh" can't shard, so the 64GB state_dict + the 64GB model
# build + the GQA q/k/v reshape and resize-vocab intermediates ALL piled into host
# RAM at once (>300GB) -> OOMKilled even at a 256GB cap, and "fixing" it with
# MEM=1800GB just made the job unschedulable. The model is 64GB; converting it
# should never need 1800GB of anything.
#
# So we run the convert on ONE 8xH100 node at TP=1 (pure FSDP, mesh data=8/model=1).
# Every Qwen3 weight shards 8-way over `data` (the embed/feature dim or the stacked
# block axis), inputs stream in already-sharded, there's no optimizer state and no
# activations (a convert runs no train step -- which is why TP=1 is clean here even
# though a TP=1 *train step* would OOM on a replicated activation). Peak ~20GB/GPU.
# Single host => no jax.distributed setup. The saved checkpoint is sharding-agnostic,
# so training warm-starts it at TP=8/2D regardless of the mesh used to write it.
#
# Usage (one 8xH100 node is plenty; the convert takes a few minutes):
#   OTA_MODEL=32b OTA_CONVERT=1 REPLICAS=1 \
#   OTA_CONVERT_OUTPUT=s3://marin-na/users/power/ot-agent-levanter/qwen3-32b-base-levanter \
#   bash ot_agent/submit_levanter_sft.sh
#
# The resulting path is then passed to the SFT launcher as OTA_INIT_FROM.

import logging
import time

import jax
import jax.numpy as jnp
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager

from haliax.partitioning import set_mesh

import fsspec

from levanter.checkpoint import save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef, load_tokenizer
from levanter.main import export_hf_to_lm
from levanter.utils.fsspec_utils import mkdirs
from levanter.utils.mesh import MeshConfig, create_mesh_from_axis_specs

from ot_agent.levanter_sft import _env, _qwen3_8b, _qwen3_32b_real

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_config() -> export_hf_to_lm.ImportHfConfig:
    model_name = _env("OTA_MODEL", "8b").lower()
    # seq_len is irrelevant to the saved weights (no parameter carries the Pos
    # axis; rope is recomputed at train time), so a nominal value is fine here.
    seq_len = int(_env("OTA_SEQ", "32768"))
    if model_name == "8b":
        model, base_ckpt = _qwen3_8b(seq_len), "Qwen/Qwen3-8B"
    elif model_name == "32b":
        model, base_ckpt = _qwen3_32b_real(seq_len), "Qwen/Qwen3-32B"
    else:
        raise ValueError(f"OTA_MODEL={model_name!r} must be '8b' or '32b'")

    output = _env(
        "OTA_CONVERT_OUTPUT",
        f"s3://marin-na/users/power/ot-agent-levanter/qwen3-{model_name}-base-levanter",
    ).rstrip("/")

    return export_hf_to_lm.ImportHfConfig(
        hf_checkpoint=base_ckpt,
        output_path=output,
        model=model,
        # Build with OUR explicit Qwen3Config (not the HF config) so the saved
        # checkpoint's pytree structure is byte-for-byte what the SFT run rebuilds
        # via the same _qwen3_*; that's what makes the warm-start load_checkpoint
        # match. _qwen3_32b_real already mirrors the real Qwen3-32B arch.
        use_hf_model_config=False,
        dtype="bfloat16",
        # Resize the padded 151936-row embedding down to the tokenizer's 151669 so
        # the checkpoint matches the train run's Vocab axis (see the vocab-resize
        # story in ot_agent/levanter_sft.py::_patch_levanter_vocab_resize).
        resize_vocab_to_match_tokenizer=True,
    )


def _build_fsdp_mesh():
    """A single-node, pure-FSDP mesh (data=all local GPUs, model=1) + its param mapping.

    TP=1 keeps the load clean (every weight shards over `data`, inputs stream in
    already sharded; see module docstring). We still set kv_head->model in the
    mapping for parity with the train config -- it's a no-op at model=1.
    """
    num_devices = jax.device_count()
    mesh_cfg = MeshConfig(
        axes={"replica": 1, "data": -1, "model": 1},
        param_mapping={"embed": "data", "kv_head": "model"},
        compute_mapping={"kv_head": "model"},
    )
    ici_axes, dcn_axes = mesh_cfg.axis_shapes(num_devices=num_devices, num_slices=1)
    mesh = create_mesh_from_axis_specs(ici_axes=ici_axes, dcn_axes=dcn_axes)
    return mesh, mesh_cfg.resolved_param_mapping


def _clean_stale_output(path: str) -> None:
    """Remove any existing checkpoint dir so tensorstore writes fresh arrays.

    tensorstore opens each array with create=true,open=true: if a prior (possibly
    partial or differently-sharded) convert left arrays under this prefix, the new
    write's chunk layout won't match the stale array's metadata and tensorstore
    raises FAILED_PRECONDITION (e.g. q_proj chunk [64,8,8,128,640] for a data=8
    convert vs a leftover [16,8,8,128,5120] from a data=4 run). Clearing the prefix
    makes the convert idempotent regardless of what a previous attempt left behind.
    """
    fs, _, (plain,) = fsspec.get_fs_token_paths(path)
    try:
        if fs.exists(plain):
            logger.info("Removing stale checkpoint dir before convert: %s", path)
            fs.rm(plain, recursive=True)
    except FileNotFoundError:
        pass


def convert(cfg: export_hf_to_lm.ImportHfConfig) -> None:
    start = time.time()
    hf_ref = cfg.hf_checkpoint if isinstance(cfg.hf_checkpoint, RepoRef) else RepoRef.from_string(cfg.hf_checkpoint)
    logger.info("HF->Levanter convert: %s -> %s", hf_ref, cfg.output_path)
    logger.info("jax devices: %s", jax.devices())

    mesh, param_mapping = _build_fsdp_mesh()
    logger.info("convert mesh axes=%s param_mapping=%s", dict(mesh.shape), param_mapping)

    tokenizer = load_tokenizer(hf_ref.model_name_or_path)
    converter = cfg.model.hf_checkpoint_converter()
    converter = converter.replaced(reference_checkpoint=hf_ref, tokenizer=tokenizer)

    dtype = getattr(jnp, cfg.dtype) if cfg.dtype else None
    with set_mesh(mesh):
        # load_state_dict (inside load_pretrained) streams each safetensors tensor
        # sharded over `data`; the named_jit builds the model sharded per
        # param_mapping. Nothing is ever whole on one device.
        model = converter.load_pretrained(
            cfg.model.model_type,
            config=cfg.model if not cfg.use_hf_model_config else None,
            axis_mapping=param_mapping,
            resize_vocab_to_match_tokenizer=cfg.resize_vocab_to_match_tokenizer,
            dtype=dtype,
        )

        _clean_stale_output(cfg.output_path)
        mkdirs(cfg.output_path)
        logger.info("Saving Levanter checkpoint (step=0) to %s", cfg.output_path)
        manager = GlobalAsyncCheckpointManager()

        def _committed():
            logger.info("Checkpoint committed to Tensorstore. Total time: %.1fs", time.time() - start)

        save_checkpoint(
            tree=model,
            checkpoint_path=cfg.output_path,
            manager=manager,
            commit_callback=_committed,
            step=0,
            is_temporary=False,
        )
        # save_checkpoint kicks the GlobalAsyncCheckpointManager commit off
        # asynchronously and returns immediately. Levanter only auto-waits when IT
        # created the manager (tree_serialize_leaves_tensorstore: `if manager_was_none`),
        # so when we pass our OWN manager we MUST block here -- otherwise convert()
        # returns and the interpreter tears the in-flight writer thread down
        # mid-commit -> SIGSEGV (exit 139) and a truncated checkpoint with no
        # metadata.json (the commit_callback that writes it never fires).
        manager.wait_until_finished()
        manager.check_for_errors()
    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    convert(build_config())
