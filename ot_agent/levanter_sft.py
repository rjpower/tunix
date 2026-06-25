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
import logging
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

logger = logging.getLogger(__name__)


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
        #
        # vocab->model is in param_mapping ONLY, NOT compute_mapping. The big-vocab
        # CE cost is handled by the fused Pallas GPU streaming kernel (engaged via
        # _patch_fused_ce_block_size -> block_size=64), which tiles over vocab and
        # never materializes the [seq, 151672] logits. That kernel's shard_map
        # (levanter/models/loss.py) shards only the BATCH axis and keeps the FULL
        # vocab + hidden on each device -- it does NO cross-vocab-shard reduction.
        # So putting vocab->model in *compute* gives each shard a partial,
        # incorrect softmax and forces an all-gather of the vocab -> the XLA
        # fallback materializes the full [seq, 151672] fp32 logits (~38GiB) -> OOM.
        # Keeping vocab out of compute makes the CE correct and lets Pallas stream.
        # Keeping vocab->model in *param* still shards the lm_head/embedding param,
        # gradient, and Adam optimizer state over model=8 (rounded 151672 % 8 == 0;
        # train_lm's round_axis_for_partitioning + the converted checkpoint both
        # already agree on 151672). The lm_head is gathered to full vocab for the
        # CE (~1.5GB bf16 transient) and the dW reduce-scattered back to the sharded
        # layout -- standard FSDP, cheap next to what Pallas streaming frees.
        mesh=MeshConfig(
            axes={"replica": 1, "data": -1, "model": tp},
            param_mapping={"embed": "data", "kv_head": "model", "vocab": "model"},
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
        # Always go through Levanter's initialize_from_hf flow. For 8B it loads HF
        # directly; for the 32B warm-start (OTA_INIT_FROM) we keep this set so the
        # same code path runs, but _patch_load_pretrained_from_checkpoint replaces
        # the HF conversion with a gentle tensorstore load of the pre-converted
        # checkpoint (see __main__). use_hf_model_config stays False (its default)
        # so train_lm passes our explicit Qwen3Config to load_pretrained.
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


def _patch_fused_ce_block_size(v_block: int = 64) -> None:
    """Force the fused cross-entropy onto the Pallas GPU *streaming* kernel.

    THE 32B@32k FIT HINGES ON THIS. Without it, the loss silently falls back to
    the XLA path, which all-gathers the vocab and materializes the full
    ``[seq, 151672]`` fp32 logits + gradient (~38GiB) on one device -> OOM. The
    Pallas GPU kernel instead tiles over the vocab and never materializes them.

    Why the fallback happens by default: ``maybe_fused_next_token_loss`` is called
    with ``block_size=None``, so the kernel infers block sizes from a tuned table
    (``tuned_block_sizes.py``). That table has NO shape bucket for Qwen3-32B's
    hidden=5120 (its NVIDIA buckets cap at h=4096 "llama3-ish" / h<=3072), so it
    returns ``BlockSizes.get_default()`` = (b=1024, h=512, v=1024). On H100 the
    Pallas weight tile must fit ~101KB of shared memory: 512*1024*2B = 1MB >> 101KB,
    so ``pallas_gpu`` raises PallasUnsupportedError and the kernel falls back to XLA.

    The fix: pass an explicit ``block_size`` (= the kernel's ``v_block_size``),
    which both *bypasses* the broken inference AND keeps the default h=512:
    512 * 64 * 2B = 64KB < 101KB -> the tile fits and ``pallas_gpu`` (the default
    first-choice impl on GPU) runs. v_block=64 is the largest power of two that
    fits with h=512 (v=128 -> 128KB > budget). ``compute_next_token_loss`` doesn't
    thread a block_size and ``TrainLmConfig`` exposes no knob, so we inject it by
    wrapping ``maybe_fused_next_token_loss`` where the model imported it
    (``lm_model``); Qwen3 inherits that base ``compute_next_token_loss`` unchanged.
    """
    import levanter.models.lm_model as lmm

    orig = lmm.maybe_fused_next_token_loss
    if getattr(orig, "_ota_block_patched", False):
        return

    def patched(*args, **kwargs):
        kwargs.setdefault("block_size", v_block)
        return orig(*args, **kwargs)

    patched._ota_block_patched = True
    lmm.maybe_fused_next_token_loss = patched
    logger.info("OTA: forcing fused CE onto Pallas GPU streaming kernel (block_size=%d)", v_block)


def _patch_load_pretrained_from_checkpoint(ckpt_path: str) -> None:
    """Warm-start 32B from a pre-converted Levanter checkpoint, gently.

    The one-shot HF->2D-sharded weight load OOMs at 32B: Levanter converts the whole
    safetensors state dict to the 2D (data x model) layout inside a single named_jit
    (all layers + the GQA q/k/v reshapes at once), and that ~52GB reshard transient
    stacked on the ~22GB optimizer state (already built by ``trainer.initial_state``)
    exceeds 80GB. ``ot_agent/convert_hf_to_levanter.py`` does the HF conversion once
    on the CPU (host RAM) and writes a Tensorstore checkpoint; here we replace
    ``HFCheckpointConverter.load_pretrained`` so the in-training "load" just builds
    the model template and deserializes that checkpoint **per array** straight into
    the 2D layout (``load_checkpoint`` with ``axis_mapping``) -- no monolithic
    conversion jit, peak ~= opt(22GB) + model(2GB). This mirrors train_lm's own
    ``initialize_from_checkpoint_path`` recipe (build -> load_checkpoint -> shard),
    minus ``subpath="model"`` because export_hf_to_lm saves the bare model tree.

    The converted checkpoint already has the vocab resized to the tokenizer's 151669
    (export_hf_to_lm resize_vocab_to_match_tokenizer=True), so no resize is needed
    here and this patch is used *instead of* _patch_levanter_vocab_resize.
    """
    import haliax
    import jax
    import levanter.compat.hf_checkpoints as hfc
    from haliax.partitioning import round_axis_for_partitioning
    from levanter.checkpoint import discover_latest_checkpoint, load_checkpoint

    orig = hfc.HFCheckpointConverter.load_pretrained
    if getattr(orig, "_ota_ckpt_patched", False):
        return

    def patched(self, lm_model_cls, ref=None, config=None, axis_mapping=None,
                resize_vocab_to_match_tokenizer=False, dtype=None):
        # train_lm passes config=config.model (use_hf_model_config defaults False);
        # fall back to the HF arch if it ever passes None.
        if config is None:
            config = self.config_from_hf_config(self.hf_config_from_hf_checkpoint(ref))
        # Round the vocab axis up to be divisible by its sharding (vocab->model=8 =>
        # 151669 -> 151672), EXACTLY as train_lm does for the opt state it builds
        # (round_axis_for_partitioning). The template, the opt state, and the converted
        # checkpoint must all agree on 151672 or the swap/deserialize shape-mismatches.
        Vocab = round_axis_for_partitioning(haliax.Axis("vocab", self.Vocab.size), axis_mapping)
        # export_hf_to_lm writes the model tree directly to ckpt_path (step=0, no
        # step-N subdir). discover_latest_checkpoint returns None if there's no
        # step-subdir layout, so fall back to the bare path. (latest_checkpoint_path
        # would *raise* instead of returning None -- don't use it here.)
        path = discover_latest_checkpoint(ckpt_path) or ckpt_path
        logger.info("OTA warm-start: loading Levanter checkpoint %s into the 2D layout", path)
        template = config.build(Vocab, key=jax.random.PRNGKey(0))
        model = load_checkpoint(template, path, axis_mapping=axis_mapping)
        if axis_mapping is not None:
            model = haliax.shard(model, axis_mapping)
        return model

    patched._ota_ckpt_patched = True
    hfc.HFCheckpointConverter.load_pretrained = patched


if __name__ == "__main__":
    # Always engage the Pallas GPU streaming CE (see _patch_fused_ce_block_size):
    # the XLA fallback OOMs at 32B@32k regardless of warm-start vs fresh-HF init.
    _patch_fused_ce_block_size(int(_env("OTA_CE_VBLOCK", "64")))
    _init_from = _env("OTA_INIT_FROM", "").rstrip("/")
    if _init_from:
        _patch_load_pretrained_from_checkpoint(_init_from)
    else:
        _patch_levanter_vocab_resize()
    train_lm.main(build_config())
