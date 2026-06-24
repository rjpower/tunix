# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# One-time HF -> Levanter checkpoint conversion for the OpenThoughts-Agent SFT.
#
# WHY THIS EXISTS
# ---------------
# Qwen3-32B at seq 32768 fits the *train step* on 32xH100 only with 2D sharding
# (TP=8 over NVLink + FSDP over the rest; see ot_agent/levanter_sft.py). But the
# one-shot HF->2D-sharded weight load is a different problem: Levanter converts the
# whole 64GB safetensors state dict to the 2D layout inside a single named_jit
# (all 64 layers + the GQA q/k/v reshapes at once), and that reshard transient
# (~52GB) stacked on the ~22GB optimizer state blows past 80GB -> OOM mid-load.
# At TP=1 (1D) the load is clean but the *train step* then OOMs on a replicated
# 50GB activation. There is no single mesh that clears both phases.
#
# THE FIX: decouple the expensive HF conversion from training. Levanter's
# `export_hf_to_lm` loads the HF model on the CPU (`use_cpu_device()`, host RAM is
# 256GB so it never touches device memory) and writes a Tensorstore checkpoint.
# The SFT run then warm-starts from that checkpoint via `trainer.initialize_from`
# (+ allow_partial_checkpoint), which deserializes each array straight into the 2D
# device layout per-array (gentle, ~24GB peak) -- no monolithic conversion jit.
#
# Usage (one node is plenty -- the load is CPU-bound, the save streams to R2):
#   OTA_MODEL=32b \
#   OTA_CONVERT_OUTPUT=s3://marin-na/users/power/ot-agent-levanter/qwen3-32b-base-levanter \
#   python -m ot_agent.convert_hf_to_levanter
#
# The resulting path is then passed to the SFT launcher as OTA_INIT_FROM.

import logging
import os

import jax

from levanter.main import export_hf_to_lm

from ot_agent.levanter_sft import _env, _qwen3_8b, _qwen3_32b_real

logging.basicConfig(level=logging.INFO)


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


if __name__ == "__main__":
    cfg = build_config()
    logging.info("HF->Levanter convert: %s -> %s", cfg.hf_checkpoint, cfg.output_path)
    logging.info("jax devices: %s", jax.devices())
    export_hf_to_lm.main(cfg)
