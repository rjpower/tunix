# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenThoughts-Agent SFT of Qwen3 on the CoreWeave H100 cluster, via Levanter.

Replicates the OT-Agent recipe (arXiv:2606.24855): full-parameter SFT of
Qwen3-32B on agentic chat trajectories at the paper's ``cutoff_len`` of 32768.
This is the Levanter port of the hand-rolled tunix pipeline -- Levanter has, as
battle-tested kernels, the three things we had to build by hand in tunix and
which *still* didn't fit Qwen3-32B at seq 32768 on 32xH100:

  * fused cross-entropy (no ``[B, S, 151936]`` fp32 logit materialization; a
    dedicated Triton/Pallas GPU kernel + vocab-parallel ``shard_map``),
  * NVTE/cuDNN flash attention auto-selected on H100 (O(seq) memory -- our
    ``jax.nn.dot_product_attention(cudnn)`` path materialized the scores at 32k),
  * sequence packing with segment-id flash masking (``pack=True``).

It drives **Levanter directly** (``levanter.main.train_lm.main``) -- one process
does HF init -> tokenize/cache -> train -> periodic HF-safetensors export. The
only dependency is ``marin-levanter[gpu]`` (a published PyPI package; see the
``levanter`` extra in pyproject.toml). No marin executor / experiments package,
so the whole pipeline lives in this repo.

``submit_levanter_sft.sh`` submits this module as an iris GPU job. Knobs come in
as ``OTA_*`` env vars.

Env knobs (all optional; defaults give the 8B smoke):
  OTA_MODEL     8b (default) | 32b      -- 32b uses the REAL Qwen3-32B arch
  OTA_DATASET   HF id (default open-thoughts/OpenThoughts-Agent-SFT-1K)
  OTA_SEQ       train/model seq len     (default 32768)
  OTA_BATCH     global batch in seqs    (default 8)
  OTA_PDP       per_device_parallelism  (default 1; < batch/devices => grad accum)
  OTA_TP        tensor-parallel axis    (default 1; FSDP over the rest)
  OTA_STEPS     optimizer steps         (default 40)
  OTA_LR        peak LR                 (default 4e-5, the paper's)
  OTA_WARMUP    warmup fraction         (default 0.1, the paper's)
  OTA_HF_EXPORT hf_save_steps           (default = OTA_STEPS, i.e. export once)
  OTA_OUTPUT    fsspec output root      (default s3://marin-na/users/power/ot-agent-levanter)
  OTA_RUN       run id suffix           (default from RUN_ID or "manual")
"""

import os

import jmp
import levanter.main.train_lm as train_lm
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import (
    ChatLmDatasetFormat,
    DatasetComponent,
    HfDatasetSourceConfig,
    LmDataConfig,
)
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.mesh import MeshConfig

from ot_agent._qwen3_chat_template import QWEN_3_CHAT_TEMPLATE

# The released OpenThinkerAgent-32B-SFT-100K model card: lr 4e-5, cosine +
# warmup_ratio 0.1, effective batch 96, 5 epochs, bf16, cutoff_len 32768.
PAPER_LR = 4e-5
PAPER_WARMUP = 0.1


def _qwen3_8b(seq_len: int) -> Qwen3Config:
    # Matches Qwen/Qwen3-8B config.json (head_dim 4096/32 = 128).
    return Qwen3Config(
        max_seq_len=seq_len,
        hidden_dim=4096,
        intermediate_dim=12288,
        num_heads=32,
        num_kv_heads=8,
        num_layers=36,
        activation_function=ActivationFunctionEnum.silu,
        initializer_range=0.02,
        layer_norm_epsilon=1e-6,
        tie_word_embeddings=False,
        reference_checkpoint="Qwen/Qwen3-8B",
        rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    )


def _qwen3_32b_real(seq_len: int) -> Qwen3Config:
    # The REAL Qwen/Qwen3-32B (config.json): 64 q-heads, explicit head_dim 128
    # (not 5120/64=80), intermediate 25600, untied embeddings, Qwen3 RoPE theta
    # 1e6, QK-norm (Qwen3Config default). NOT marin's ``qwen3_32b`` constant,
    # which is an olmo-shaped 5120/40-head/Llama3-rope model.
    return Qwen3Config(
        max_seq_len=seq_len,
        hidden_dim=5120,
        intermediate_dim=25600,
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        num_layers=64,
        activation_function=ActivationFunctionEnum.silu,
        initializer_range=0.02,
        layer_norm_epsilon=1e-6,
        tie_word_embeddings=False,
        reference_checkpoint="Qwen/Qwen3-32B",
        rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    )


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def build_config() -> train_lm.TrainLmConfig:
    model_name = _env("OTA_MODEL", "8b").lower()
    dataset = _env("OTA_DATASET", "open-thoughts/OpenThoughts-Agent-SFT-1K")
    seq_len = int(_env("OTA_SEQ", "32768"))
    batch = int(_env("OTA_BATCH", "8"))
    pdp = int(_env("OTA_PDP", "1"))
    tp = int(_env("OTA_TP", "1"))
    steps = int(_env("OTA_STEPS", "40"))
    lr = float(_env("OTA_LR", str(PAPER_LR)))
    warmup = float(_env("OTA_WARMUP", str(PAPER_WARMUP)))
    hf_export = int(_env("OTA_HF_EXPORT", str(steps)))
    output = _env("OTA_OUTPUT", "s3://marin-na/users/power/ot-agent-levanter").rstrip("/")
    run = _env("OTA_RUN", os.environ.get("RUN_ID", "manual"))

    if model_name == "8b":
        model, base_ckpt = _qwen3_8b(seq_len), "Qwen/Qwen3-8B"
    elif model_name == "32b":
        model, base_ckpt = _qwen3_32b_real(seq_len), "Qwen/Qwen3-32B"
    else:
        raise ValueError(f"OTA_MODEL={model_name!r} must be '8b' or '32b'")

    slug = dataset.split("/")[-1].lower()
    run_dir = f"{output}/{model_name}-{slug}-{run}"
    # Tokenized cache is keyed by dataset+tokenizer, not run -- reuse across runs.
    cache_dir = f"{output}/cache/{slug}-qwen3"

    # Assistant-turn-only loss: the Qwen3 template's {% generation %} blocks tag
    # exactly the assistant/tool-call tokens; mask_user_turns drops the rest.
    # pack=True greedily packs trajectories to seq_len with segment-id attention
    # masking so packed docs don't attend across boundaries.
    chat_format = ChatLmDatasetFormat(
        messages_field="conversations",
        chat_template=QWEN_3_CHAT_TEMPLATE,
        mask_user_turns=True,
        pack=True,
    )
    source = HfDatasetSourceConfig(id=dataset, cache_dir=cache_dir, format=chat_format)
    data = LmDataConfig(
        tokenizer=base_ckpt,  # Qwen3-8B/-32B share one tokenizer/vocab (151936)
        cache_dir=cache_dir,
        components={"ot_agent": DatasetComponent(source=source, cache_dir=cache_dir, format=chat_format)},
        train_weights={"ot_agent": 1.0},
        shuffle=True,
    )

    # WandB if a key is present (loss curve), else a local JSON logger.
    if os.environ.get("WANDB_API_KEY"):
        tracker = WandbConfig(
            project=os.environ.get("WANDB_PROJECT", "ot-agent"),
            name=f"{model_name}-{slug}-{run}",
            tags=["ot-agent", "qwen3", model_name, "sft"],
        )
    else:
        tracker = JsonLoggerConfig()

    trainer = TrainerConfig(
        tracker=tracker,
        mp=jmp.get_policy("p=f32,c=bfloat16"),  # bf16 compute, fp32 master/optimizer
        train_batch_size=batch,
        per_device_parallelism=pdp,
        num_train_steps=steps,
        steps_per_eval=10**9,  # smoke: no periodic validation
        checkpointer=CheckpointerConfig(base_path=f"{run_dir}/checkpoints"),
        # FSDP shards params+optimizer over the cross-device ``data`` axis;
        # ``model`` is intra-node tensor parallel (1 = pure FSDP).
        mesh=MeshConfig(axes={"replica": 1, "data": -1, "model": tp}),
        allow_nondivisible_batch_size=True,
    )

    optimizer = AdamConfig(
        learning_rate=lr,
        weight_decay=0.0,
        warmup=warmup,
        lr_schedule="cosine",
        min_lr_ratio=0.0,
        max_grad_norm=1.0,
    )

    return train_lm.TrainLmConfig(
        data=data,
        trainer=trainer,
        model=model,
        optimizer=optimizer,
        train_seq_len=seq_len,
        initialize_from_hf=base_ckpt,
        pad_tokenizer_to_match_model=True,  # Qwen pads embed vocab past the tokenizer
        hf_save_path=f"{run_dir}/hf",
        hf_save_steps=hf_export,
    )


if __name__ == "__main__":
    train_lm.main(build_config())
