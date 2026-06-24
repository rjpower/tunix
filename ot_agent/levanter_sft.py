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
  OTA_CKPT_MINUTES train-state ckpt cadence in minutes (default 120)
  OTA_OUTPUT    fsspec output root      (default s3://marin-na/users/power/ot-agent-levanter)
  OTA_CACHE     tokenized-cache root    (default {OTA_OUTPUT}/cache; MUST be shared
                                         storage -- the cache build fans out to
                                         zephyr worker pods, so node-local /tmp
                                         is invisible to the consolidating driver)
  OTA_RUN       run id suffix           (default from RUN_ID or "manual")
"""

import datetime
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
    # Checkpoint cadence (minutes). A 32B train state is hundreds of GB; the
    # Levanter default (15 min) would saturate R2 over a 12h run, so default to 2h.
    ckpt_minutes = int(_env("OTA_CKPT_MINUTES", "120"))
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
    # The cache MUST live on shared storage (S3/GCS): Levanter's cache build fans
    # out to separate zephyr worker pods (each with its own /tmp), and the driver
    # consolidates their shard outputs -- node-local paths aren't visible across
    # pods. OTA_CACHE overrides the cache root independently of OTA_OUTPUT so the
    # checkpoints/HF export can stay node-local (driver-only) while the cache is
    # shared. tensorstore's built-in s3 driver reads/writes it (no s3fs needed).
    cache_root = _env("OTA_CACHE", f"{output}/cache").rstrip("/")
    cache_dir = f"{cache_root}/{slug}-qwen3"

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
        checkpointer=CheckpointerConfig(
            base_path=f"{run_dir}/checkpoints",
            save_interval=datetime.timedelta(minutes=ckpt_minutes),
        ),
        # FSDP shards params+optimizer over the cross-device ``data`` axis;
        # ``model`` is intra-node tensor parallel (1 = pure FSDP).
        #
        # Qwen3 attention q/k/v projections carry the GQA ``kv_head`` axis, which
        # Levanter's default shared_mapping ({mlp,heads}->model) does NOT cover --
        # so at TP>1 they'd shard only over ``data``, leaving attention params/opt
        # under-sharded (the 32B init OOM). Map kv_head->model so q/k/v shard over
        # the TP axis (kv_head=8 == model=8 for Qwen3-32B); o_proj/MLP already shard
        # via heads/mlp->model. Also map it in compute so the attention activations
        # shard (else they replicate -> the 50GB activation OOM). No-op at TP=1.
        mesh=MeshConfig(
            axes={"replica": 1, "data": -1, "model": tp},
            param_mapping={"embed": "data", "kv_head": "model"},
            compute_mapping={"kv_head": "model"},
        ),
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
        # NB: pad_tokenizer_to_match_model is intentionally False; see
        # _patch_levanter_vocab_resize for the Qwen padded-vocab story.
        pad_tokenizer_to_match_model=False,
        hf_save_path=f"{run_dir}/hf",
        hf_save_steps=hf_export,
    )


def _patch_levanter_vocab_resize() -> None:
    """Resize the HF model down to the tokenizer vocab on load (Qwen padded-vocab fix).

    Qwen3's HF embedding is padded to 151936, but the Qwen3 tokenizer only has
    151669 real tokens (ids 151669..151935 are never emitted). Levanter's
    train_lm builds the optimizer state from ``len(config.data.the_tokenizer)``
    (= 151669) via ``trainer.initial_state``, but ``initialize_from_hf`` then loads
    the model at the HF size (151936) and swaps only the model into the state --
    the opt state is never rebuilt. The first ``optimizer.update`` then raises a
    vocab-axis pytree mismatch (151936 vs 151669).

    ``pad_tokenizer_to_match_model=True`` does NOT fix this: it pads a *copy* of
    the tokenizer held by the HF converter (``as_hf_tokenizer`` + dataclasses
    .replace), not the shared ``the_tokenizer`` that drives the Vocab axis. The
    robust fix is to resize the loaded model *down* to the tokenizer vocab so the
    model and opt state agree at 151669; the dropped rows are unused padding.
    ``HFCheckpointConverter.load_pretrained`` supports this via
    ``resize_vocab_to_match_tokenizer`` but ``TrainLmConfig`` doesn't expose it,
    so flip its default here. ``vocab`` is absent from our ``param_mapping`` so the
    axis is unsharded and ``round_axis_for_partitioning`` is a no-op (151669 for
    any mesh) -- consistent for the 8B smoke and the 32B/TP run alike.
    """
    import levanter.compat.hf_checkpoints as hfc

    orig = hfc.HFCheckpointConverter.load_pretrained
    if getattr(orig, "_ota_resize_patched", False):
        return

    def patched(self, *args, **kwargs):
        kwargs.setdefault("resize_vocab_to_match_tokenizer", True)
        return orig(self, *args, **kwargs)

    patched._ota_resize_patched = True
    hfc.HFCheckpointConverter.load_pretrained = patched


if __name__ == "__main__":
    _patch_levanter_vocab_resize()
    train_lm.main(build_config())
