#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Submit the OT-Agent Qwen3-32B SFT to the CoreWeave H100 cluster (cw-us-east-02a).
#
# Multi-node is one flag: `iris job run --replicas N` auto-enables `leafgroup`
# coscheduling for GPUs (all N pods land on one InfiniBand fabric and form one
# JAX world via ot_agent.distributed.init_distributed). The entrypoint is
# `python -m ot_agent.launch_sft`, configured entirely by `-e` env vars.
#
# Four rungs of a smoke ladder (run them in order before the full run):
#   STAGE=single   1 node (8 H100), Qwen3-1.7B  -- validates the SFT loop + export
#   STAGE=multi    2 nodes,         Qwen3-1.7B  -- validates the multi-node JAX
#                                                  world + process-disjoint data
#   STAGE=bigsmoke 4 nodes (32 H100), Qwen3-32B, ~30 steps -- de-risks the 32B
#                                                  memory FIT on 4x8 H100 + export
#   STAGE=full     4 nodes (32 H100), Qwen3-32B -- the 100K replication run
#
# Usage:
#   export HF_TOKEN=...                         # gated/large HF pulls on the worker
#   export WANDB_API_KEY=... WANDB_PROJECT=ot-agent   # optional loss curve
#   STAGE=single bash ot_agent/submit_sft.sh
#   STAGE=multi  bash ot_agent/submit_sft.sh
#   STAGE=full   bash ot_agent/submit_sft.sh
#
# Watch:
#   uv run iris --cluster=cw-us-east-02a job summary /power/<job-name>
#   uv run iris --cluster=cw-us-east-02a job logs    /power/<job-name> --follow | grep '\[ota-'
set -euo pipefail

STAGE="${STAGE:-single}"
CLUSTER="cw-us-east-02a"
# R2 prefix for the exported HF checkpoint (the coherent downstream artifact).
EXPORT_BASE="${EXPORT_BASE:-s3://marin-na/users/power/ot-agent}"
TS="$(date +%s)"

IRIS=(uv run iris --cluster="${CLUSTER}" job run --no-wait
  --enable-extra-resources --extra gpu --extra mega)

# Forward secrets the worker needs (HF pulls; optional wandb).
ENVS=(-e HF_TOKEN "${HF_TOKEN:-}")
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  ENVS+=(-e WANDB_API_KEY "${WANDB_API_KEY}" -e WANDB_PROJECT "${WANDB_PROJECT:-ot-agent}")
fi

case "${STAGE}" in
  single)
    # 1 node, tiny model, tiny data: shake out the loop + export end-to-end.
    NAME="ota-sft-smoke-single-${TS}"
    "${IRIS[@]}" --gpu H100x8 --cpu 16 --memory 200GB --disk 100GB --max-retries 0 \
      --job-name "${NAME}" "${ENVS[@]}" \
      -e AGENT_MODEL qwen3-1.7b-base -e DATASET 1k -e DATA_LIMIT 512 \
      -e SFT_STEPS 20 -e BATCH_SIZE 8 -e TP 1 -e MAX_SEQ_LEN 2048 -e REMAT decoder \
      -e EXPORT_DIR "${EXPORT_BASE}/smoke-single-${TS}/hf" \
      -- python -m ot_agent.launch_sft
    ;;
  multi)
    # 2 nodes (16 H100): the multi-node JAX world + disjoint data-parallel path.
    NAME="ota-sft-smoke-multi-${TS}"
    "${IRIS[@]}" --gpu H100x8 --replicas 2 --cpu 16 --memory 200GB --disk 100GB --max-retries 0 \
      --job-name "${NAME}" "${ENVS[@]}" \
      -e AGENT_MODEL qwen3-1.7b-base -e DATASET 10k -e DATA_LIMIT 4096 \
      -e SFT_STEPS 30 -e BATCH_SIZE 16 -e TP 8 -e MAX_SEQ_LEN 2048 -e REMAT decoder \
      -e EXPORT_DIR "${EXPORT_BASE}/smoke-multi-${TS}/hf" \
      -- python -m ot_agent.launch_sft
    ;;
  bigsmoke)
    # 4 nodes (32 H100): Qwen3-32B for ~12 steps at the FAITHFUL seq 32768 with
    # GPU cuDNN flash attention (FLASH=1) -- the de-risk for the long run: does
    # the new GPU flash path work + fit at the paper's cutoff_len 32768? Without
    # flash, seq-32768 attention is O(seq^2) and OOMs; with cuDNN flash it's
    # O(seq). Small per-device microbatch (global 4 -> 1/device on fsdp=4) +
    # grad-accum, low-mem loss on. Watch the XLA peak-HBM line to size the full
    # run's per-step batch + estimate throughput at 32k.
    # All knobs overridable via env so perf variants don't need an edit, e.g.:
    #   CE_CHUNKS=8 BATCH_SIZE=16 GRAD_ACCUM=1 STAGE=bigsmoke bash ot_agent/submit_sft.sh
    NAME="ota-sft-bigsmoke-32b-${TS}"
    "${IRIS[@]}" --gpu H100x8 --replicas 4 --cpu 32 --memory 512GB --disk 300GB --max-retries 0 \
      --job-name "${NAME}" "${ENVS[@]}" \
      -e AGENT_MODEL qwen3-32b -e DATASET 10k -e DATA_LIMIT "${DATA_LIMIT:-512}" \
      -e SFT_STEPS "${SFT_STEPS:-12}" -e BATCH_SIZE "${BATCH_SIZE:-4}" \
      -e GRAD_ACCUM "${GRAD_ACCUM:-2}" -e CE_CHUNKS "${CE_CHUNKS:-0}" \
      -e LR 1e-5 -e TP "${TP:-8}" -e MAX_SEQ_LEN "${MAX_SEQ_LEN:-32768}" \
      -e FLASH "${FLASH:-1}" -e REMAT "${REMAT:-decoder}" -e RUN_NAME "${NAME}" \
      -e EXPORT_DIR "${EXPORT_DIR-${EXPORT_BASE}/bigsmoke-32b-${TS}/hf}" \
      -- python -m ot_agent.launch_sft
    # ^ set EXPORT_DIR= (empty) to skip the export -- useful for perf-only runs.
    ;;
  full)
    # 4 nodes (32 H100): Qwen3-32B on the 100K set -- the FAITHFUL replication.
    # Recipe from the released OpenThinkerAgent-32B-SFT-100K model card:
    #   lr 4e-5, cosine + warmup_ratio 0.1, EFFECTIVE batch 96, 5 epochs, bf16,
    #   cutoff_len 32768. 5 epochs of 94,334 rows @ eff batch 96 = 4914 steps.
    # Effective batch = per-step BATCH_SIZE 4 (1/device on fsdp=4) x GRAD_ACCUM 24.
    # GPU cuDNN flash (FLASH=1) makes the 32k context fit (O(seq) attention).
    # WARNING: the full 5-epoch/100K @ 32k run is MULTI-WEEK on 32 H100 (~118k
    # microbatches at seq 32768). For a scoped demonstration, override e.g.
    #   SFT_STEPS=<fewer>  or  DATASET=10k  or  a shorter MAX_SEQ_LEN.
    NAME="ota-sft-qwen3-32b-100k-${TS}"
    "${IRIS[@]}" --gpu H100x8 --replicas 4 --cpu 32 --memory 512GB --disk 300GB --max-retries 1 \
      --job-name "${NAME}" "${ENVS[@]}" \
      -e AGENT_MODEL qwen3-32b -e DATASET "${DATASET:-100k}" \
      -e SFT_STEPS "${SFT_STEPS:-4914}" -e BATCH_SIZE "${BATCH_SIZE:-4}" \
      -e GRAD_ACCUM "${GRAD_ACCUM:-24}" -e CE_CHUNKS "${CE_CHUNKS:-8}" \
      -e LR "${LR:-4e-5}" -e WARMUP_RATIO "${WARMUP_RATIO:-0.1}" \
      -e TP "${TP:-8}" -e MAX_SEQ_LEN "${MAX_SEQ_LEN:-32768}" -e FLASH "${FLASH:-1}" \
      -e REMAT "${REMAT:-decoder}" -e RUN_NAME "${NAME}" \
      -e EXPORT_DIR "${EXPORT_BASE}/qwen3-32b-100k-${TS}/hf" \
      -- python -m ot_agent.launch_sft
    ;;
  *)
    echo "unknown STAGE='${STAGE}' (want: single|multi|bigsmoke|full)" >&2
    exit 2
    ;;
esac

echo "submitted STAGE=${STAGE} as /power/${NAME}"
echo "watch: uv run iris --cluster=${CLUSTER} job logs /power/${NAME} --follow | grep '\\[ota-'"
