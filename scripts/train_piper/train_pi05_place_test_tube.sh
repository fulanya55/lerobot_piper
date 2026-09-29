#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/train_pi05_place_test_tube.yaml"
DATASET_ROOT="/root/wxwu/dataset/EEG/PLACE_THE_TEST_TUBE_NEW/PLACE_THE_TEST_TUBE_FIX"
MODEL_ROOT="/root/wxwu/model/pi05_base"
TOKENIZER_ROOT="/root/wxwu/model/paligemma-3b-pt-224"

NUM_GPUS=8
BATCH_SIZE_PER_GPU=4
GRADIENT_ACCUMULATION_STEPS=6
EPOCHS=40

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-lerobot-piper}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"


NUM_FRAMES="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["total_frames"])' "${DATASET_ROOT}/meta/info.json")"
SAMPLES_PER_MICRO_STEP=$((BATCH_SIZE_PER_GPU * NUM_GPUS))
MIN_MICRO_STEPS=$(((NUM_FRAMES * EPOCHS + SAMPLES_PER_MICRO_STEP - 1) / SAMPLES_PER_MICRO_STEP))
TRAIN_STEPS=$(((MIN_MICRO_STEPS + GRADIENT_ACCUMULATION_STEPS - 1) / GRADIENT_ACCUMULATION_STEPS * GRADIENT_ACCUMULATION_STEPS))
OPTIMIZER_STEPS=$((TRAIN_STEPS / GRADIENT_ACCUMULATION_STEPS))
EFFECTIVE_BATCH_SIZE=$((SAMPLES_PER_MICRO_STEP * GRADIENT_ACCUMULATION_STEPS))
WARMUP_STEPS=$(((TRAIN_STEPS * 3 + 99) / 100))


RUN_ID="${RUN_ID:-piper_pi05_place_test_tube_fix_bs4_ga6_ep4_$(date -u +%Y%m%d_%H%M%S)}"
MASTER_PORT="${MASTER_PORT:-29500}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"

echo "Run ID: ${RUN_ID}"
echo "Dataset: ${DATASET_ROOT} (${NUM_FRAMES} frames)"
echo "GPUs: ${NUM_GPUS}; batch/GPU: ${BATCH_SIZE_PER_GPU}; GA: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective batch: ${EFFECTIVE_BATCH_SIZE}; epochs: ${EPOCHS}"
echo "Micro-steps: ${TRAIN_STEPS}; optimizer updates: ${OPTIMIZER_STEPS}; warmup: ${WARMUP_STEPS} (3%)"
echo "Output: ${OUTPUT_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; configuration checks passed, training was not started."
  exit 0
fi

cd "${REPO_ROOT}"

exec "${REPO_ROOT}/.venv/bin/torchrun" \
  --nnodes=1 \
  --node-rank=0 \
  --nproc-per-node="${NUM_GPUS}" \
  --master-addr=127.0.0.1 \
  --master-port="${MASTER_PORT}" \
  "${REPO_ROOT}/.venv/bin/lerobot-train" \
  --config_path="${CONFIG_PATH}" \
  --batch_size="${BATCH_SIZE_PER_GPU}" \
  --accelerator.gradient_accumulation.steps="${GRADIENT_ACCUMULATION_STEPS}" \
  --steps="${TRAIN_STEPS}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.mode="${WANDB_MODE}" \
  --wandb.project="${WANDB_PROJECT}" \
  --wandb.run_id="${RUN_ID}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_ID}"
