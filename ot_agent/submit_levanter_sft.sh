#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Submit the OpenThoughts-Agent Qwen3 SFT (Levanter) to the CoreWeave H100
# cluster. Unlike the tunix path (ot_agent/submit_sft.sh), this runs Levanter
# directly: one job does HF init -> tokenize/cache -> train -> HF export, using
# Levanter's fused-CE GPU kernel + NVTE flash + packing. The job image is built
# from THIS repo with the `gpu`+`levanter` extras (marin-levanter[gpu]); the
# launcher is `python -m ot_agent.levanter_sft`, configured by OTA_* env vars.
#
# Default = the 8B de-risk smoke: 1 node (8xH100), OT-Agent-1K, seq 32768, ~40
# steps, output to node-local disk (no S3 needed). It validates every mechanism
# (HF load, packing + assistant mask, NVTE flash, fused CE at 32k, HF export)
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
NAME="ota-levanter-${OTA_MODEL:-8b}-${TS}"

ENVS=(-e HF_TOKEN "${HF_TOKEN:-}" -e RUN_ID "${NAME}")
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  ENVS+=(-e WANDB_API_KEY "${WANDB_API_KEY}" -e WANDB_PROJECT "${WANDB_PROJECT:-ot-agent}")
fi
# R2 creds only matter when OTA_OUTPUT is s3://; harmless to forward otherwise.
if [[ "${OTA_OUTPUT}" == s3://* && -n "${R2_ACCESS_KEY_ID:-}" ]]; then
  ENVS+=(-e AWS_ACCESS_KEY_ID "${R2_ACCESS_KEY_ID}" -e AWS_SECRET_ACCESS_KEY "${R2_SECRET_ACCESS_KEY}" \
         -e AWS_ENDPOINT_URL "${R2_ENDPOINT}")
fi
# Forward OTA_* knobs that are set in the environment.
for v in OTA_MODEL OTA_DATASET OTA_SEQ OTA_BATCH OTA_PDP OTA_TP OTA_STEPS OTA_LR OTA_WARMUP OTA_HF_EXPORT OTA_OUTPUT OTA_RUN; do
  if [[ -n "${!v:-}" ]]; then ENVS+=(-e "$v" "${!v}"); fi
done

REPLICA_ARGS=()
if [[ "${REPLICAS}" -gt 1 ]]; then REPLICA_ARGS=(--replicas "${REPLICAS}"); fi

# NB: do not `set -x` here -- ENVS carries secrets (HF_TOKEN, WANDB/R2 keys).
echo "submitting ${NAME}: model=${OTA_MODEL:-8b} dataset=${OTA_DATASET:-OT-Agent-1K} seq=${OTA_SEQ:-32768} steps=${OTA_STEPS:-40} output=${OTA_OUTPUT}"
uv run iris --cluster="${CLUSTER}" job run --no-wait \
  --enable-extra-resources --extra gpu --extra levanter \
  --gpu "${GPUS}" "${REPLICA_ARGS[@]}" --cpu 32 --memory 256GB --disk 512GB --max-retries 0 \
  --job-name "${NAME}" "${ENVS[@]}" \
  -- python -m ot_agent.levanter_sft

echo "submitted ${NAME}"
echo "watch: uv run iris --cluster=${CLUSTER} job logs /power/${NAME} --follow"
