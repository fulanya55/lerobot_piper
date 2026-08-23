#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/train_pi05_full.yaml"
DATASET_ROOT="/root/wwx/dataset/PLACE_THE_TEST_TUBE_SUCCESS_MERGED"
MODEL_ROOT="/root/wwx/model/pi05_base"
TOKENIZER_ROOT="/root/wwx/model/paligemma-3b-pt-224"

NUM_GPUS=8
BATCH_SIZE_PER_GPU=4
GRADIENT_ACCUMULATION_STEPS=6
EPOCHS=8

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-lerobot-piper}"

for required_path in \
  "${REPO_ROOT}/.venv/bin/torchrun" \
  "${REPO_ROOT}/.venv/bin/lerobot-train" \
  "${CONFIG_PATH}" \
  "${DATASET_ROOT}/meta/info.json" \
  "${MODEL_ROOT}/config.json" \
  "${MODEL_ROOT}/model.safetensors" \
  "${MODEL_ROOT}/policy_preprocessor.json" \
  "${MODEL_ROOT}/policy_postprocessor.json" \
  "${TOKENIZER_ROOT}/tokenizer.json"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

# LeRobot counts cfg.steps in micro-batches. Round up to a complete GA cycle so
# the last accumulated gradients are applied. For the current 75,487-frame
# dataset this evaluates to 18,876 micro-steps and 3,146 optimizer updates.
NUM_FRAMES="$(jq -er '.total_frames' "${DATASET_ROOT}/meta/info.json")"
SAMPLES_PER_MICRO_STEP=$((BATCH_SIZE_PER_GPU * NUM_GPUS))
MIN_MICRO_STEPS=$(((NUM_FRAMES * EPOCHS + SAMPLES_PER_MICRO_STEP - 1) / SAMPLES_PER_MICRO_STEP))
TRAIN_STEPS=$(((MIN_MICRO_STEPS + GRADIENT_ACCUMULATION_STEPS - 1) / GRADIENT_ACCUMULATION_STEPS * GRADIENT_ACCUMULATION_STEPS))
OPTIMIZER_STEPS=$((TRAIN_STEPS / GRADIENT_ACCUMULATION_STEPS))
EFFECTIVE_BATCH_SIZE=$((SAMPLES_PER_MICRO_STEP * GRADIENT_ACCUMULATION_STEPS))

RUN_ID="${RUN_ID:-piper_pi05_full_bs4_ga6_ep8_$(date -u +%Y%m%d_%H%M%S)}"
MASTER_PORT="${MASTER_PORT:-29500}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"

echo "Run ID: ${RUN_ID}"
echo "Dataset: ${DATASET_ROOT} (${NUM_FRAMES} frames)"
echo "GPUs: ${NUM_GPUS}; batch/GPU: ${BATCH_SIZE_PER_GPU}; GA: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective batch: ${EFFECTIVE_BATCH_SIZE}; epochs: ${EPOCHS}"
echo "Micro-steps: ${TRAIN_STEPS}; optimizer updates: ${OPTIMIZER_STEPS}"
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
  --wandb.enable=true \
  --wandb.mode=online \
  --wandb.project="${WANDB_PROJECT}" \
  --wandb.run_id="${RUN_ID}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_ID}"
