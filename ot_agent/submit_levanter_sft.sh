#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Submit the OpenThoughts-Agent Qwen3 SFT (Levanter) to the CoreWeave H100
# cluster. Unlike the tunix path (ot_agent/submit_sft.sh), this runs Levanter
# directly: one job does HF init -> tokenize/cache -> train -> HF export, using
# Levanter's fused CE (no [B,S,V] logits) + O(seq) blockwise FlashAttention +
# packing. The job image is built from THIS repo with the `gpu`+`levanter` extras
# (marin-levanter, jax[cuda13] from `gpu`; NOT marin-levanter[gpu] -- its
# torch-pulling kernel deps clash with jax's cuDNN, see pyproject `levanter`); the
# launcher is `python -m ot_agent.levanter_sft`, configured by OTA_* env vars.
#
# Default = the 8B de-risk smoke: 1 node (8xH100), OT-Agent-1K, seq 32768, ~40
# steps, output to node-local disk (no S3 needed). It validates every mechanism
# (HF load, packing + assistant mask, O(seq) flash, fused CE at 32k, HF export)
# before committing the 32B run.
#
# Usage:
#   export HF_TOKEN=... WANDB_API_KEY=...
#   bash ot_agent/submit_levanter_sft.sh                 # 8B smoke (local output)
#   OTA_MODEL=32b OTA_REPLICAS... (later, with s3 output) # the faithful run
#
# Real (persisted) runs: set OTA_OUTPUT=s3://marin-na/users/power/ot-agent-levanter
# and R2 creds (R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY); add s3fs for the HF export.
set -euo pipefail

CLUSTER="${CLUSTER:-cw-us-east-02a}"
GPUS="${GPUS:-H100x8}"
REPLICAS="${REPLICAS:-1}"
R2_ENDPOINT="${R2_ENDPOINT:-https://74981a43be0de7712369306c7b19133d.r2.cloudflarestorage.com}"
# Smoke default: node-local output (ephemeral) so no S3/tensorstore-s3/s3fs deps.
export OTA_OUTPUT="${OTA_OUTPUT:-/tmp/ota-levanter}"
TS="$(date +%s)"
_TAG="${OTA_MODEL:-8b}"
if [[ "${OTA_CONVERT:-0}" == "1" ]]; then _TAG="convert-${_TAG}"; fi
NAME="ota-levanter-${_TAG}-${TS}"

ENVS=(-e HF_TOKEN "${HF_TOKEN:-}" -e RUN_ID "${NAME}")
# 32B@32k: the train step needs ~61GB/GPU (< 80GB physical), but the HF-load +
# opt-state-init + first-step-compile phase transiently stacks load buffers on top
# of the optimizer state. With XLA's default preallocation that transient slams
# into the fixed BFC wall and OOMs mid-load. PREALLOCATE=false lets the load
# buffers be freed and the memory reused for the step, sharing the full card
# dynamically; MEM_FRACTION then just caps growth (0.93*80=74.4GB, ~5.6GB left for
# NCCL/CUDA context). 8B (1 node, ample headroom) is unaffected.
ENVS+=(-e XLA_PYTHON_CLIENT_PREALLOCATE "${XLA_PYTHON_CLIENT_PREALLOCATE:-false}")
ENVS+=(-e XLA_PYTHON_CLIENT_MEM_FRACTION "${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.93}")
# 32B@32k first-step compile OOMs in XLA GPU autotuning, NOT execution: the
# autotuner profiles several candidate configs for the fused-CE backward fusion,
# each allocating ~2.9GB scratch on top of the ~61GB persistent step -> the
# compile-time probing spike exceeds the 74GB BFC cap (one task OOMs, then the
# multi-host shutdown barrier times out and aborts the run). Disabling autotuning
# removes that probing spike; the default kernels still fit and run within the
# established envelope. Override XLA_FLAGS to re-tune (e.g. add other flags).
ENVS+=(-e XLA_FLAGS "${XLA_FLAGS:---xla_gpu_autotune_level=0}")
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  ENVS+=(-e WANDB_API_KEY "${WANDB_API_KEY}" -e WANDB_PROJECT "${WANDB_PROJECT:-ot-agent}")
fi
# R2 creds matter when ANY of the output / cache / convert-target / warm-start
# source is on s3:// (all use tensorstore's built-in s3 driver, no s3fs).
if [[ ( "${OTA_OUTPUT}" == s3://* || "${OTA_CACHE:-}" == s3://* \
        || "${OTA_CONVERT_OUTPUT:-}" == s3://* || "${OTA_INIT_FROM:-}" == s3://* ) \
      && -n "${R2_ACCESS_KEY_ID:-}" ]]; then
  ENVS+=(-e AWS_ACCESS_KEY_ID "${R2_ACCESS_KEY_ID}" -e AWS_SECRET_ACCESS_KEY "${R2_SECRET_ACCESS_KEY}" \
         -e AWS_ENDPOINT_URL "${R2_ENDPOINT}")
fi
# Forward OTA_* knobs that are set in the environment.
for v in OTA_MODEL OTA_DATASET OTA_SEQ OTA_BATCH OTA_PDP OTA_TP OTA_STEPS OTA_LR OTA_WARMUP OTA_HF_EXPORT OTA_CKPT_MINUTES OTA_OUTPUT OTA_CACHE OTA_RUN OTA_INIT_FROM OTA_CONVERT_OUTPUT; do
  if [[ -n "${!v:-}" ]]; then ENVS+=(-e "$v" "${!v}"); fi
done

# Entrypoint: the one-time HF->Levanter checkpoint conversion (OTA_CONVERT=1) or
# the SFT trainer. The convert runs ON the GPUs at TP=1 (pure FSDP): every weight
# shards over `data` and inputs stream in already-sharded, so ~20GB/GPU with no
# optimizer state -- 1 node (8xH100) suffices. (The old CPU path needed >300GB host
# RAM because a 1-device CPU mesh can't shard; see convert_hf_to_levanter.py.)
if [[ "${OTA_CONVERT:-0}" == "1" ]]; then
  ENTRYPOINT=(python -m ot_agent.convert_hf_to_levanter)
else
  ENTRYPOINT=(python -m ot_agent.levanter_sft)
fi

REPLICA_ARGS=()
if [[ "${REPLICAS}" -gt 1 ]]; then REPLICA_ARGS=(--replicas "${REPLICAS}"); fi

# NB: do not `set -x` here -- ENVS carries secrets (HF_TOKEN, WANDB/R2 keys).
echo "submitting ${NAME}: model=${OTA_MODEL:-8b} dataset=${OTA_DATASET:-OT-Agent-1K} seq=${OTA_SEQ:-32768} steps=${OTA_STEPS:-40} output=${OTA_OUTPUT}"
uv run iris --cluster="${CLUSTER}" job run --no-wait \
  --enable-extra-resources --extra gpu --extra levanter \
  --gpu "${GPUS}" "${REPLICA_ARGS[@]}" --cpu 32 --memory "${MEM:-256GB}" --disk 512GB --max-retries 0 \
  --job-name "${NAME}" "${ENVS[@]}" \
  -- "${ENTRYPOINT[@]}"

echo "submitted ${NAME}"
echo "watch: uv run iris --cluster=${CLUSTER} job logs /power/${NAME} --follow"
