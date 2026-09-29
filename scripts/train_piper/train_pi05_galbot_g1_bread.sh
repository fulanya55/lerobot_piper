#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/train_pi05_galbot_g1_bread.yaml"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RESUME_ARGS=()
SOURCE_DATASET_ROOT="${DATASET_SOURCE:-/share/project/zhiwei/syj/datasets/Galbot_G1_take_bread_from_microwave_0909_2720}"
DATASET_ROOT="${PROJECTED_DATASET_ROOT:-${REPO_ROOT}/../dataset/Galbot_G1_take_bread_from_microwave_0909_2720_pi05_23d_3cam}"
MODEL_ROOT="/root/wxwu/model/pi05_base"
TOKENIZER_ROOT="/root/wxwu/model/paligemma-3b-pt-224"

NUM_GPUS=8
BATCH_SIZE_PER_GPU=4
GRADIENT_ACCUMULATION_STEPS=6
EPOCHS=5
EXPECTED_FRAMES=73826
EXPECTED_VECTOR_DIM=38

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-lerobot-piper}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

for required_path in \
  "${REPO_ROOT}/.venv/bin/torchrun" \
  "${REPO_ROOT}/.venv/bin/lerobot-train" \
  "${REPO_ROOT}/.venv/bin/python" \
  "${CONFIG_PATH}" \
  "${SOURCE_DATASET_ROOT}/meta/info.json" \
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

read -r NUM_FRAMES ACTION_DIM STATE_DIM < <("${REPO_ROOT}/.venv/bin/python" - "${SOURCE_DATASET_ROOT}/meta/info.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    info = json.load(f)
print(
    info["total_frames"],
    info["features"]["action"]["shape"][0],
    info["features"]["observation.state"]["shape"][0],
)
PY
)

if (( NUM_FRAMES != EXPECTED_FRAMES )); then
  echo "Unexpected dataset frame count: ${NUM_FRAMES} (expected ${EXPECTED_FRAMES})" >&2
  exit 1
fi
if (( ACTION_DIM != EXPECTED_VECTOR_DIM || STATE_DIM != EXPECTED_VECTOR_DIM )); then
  echo "Expected ${EXPECTED_VECTOR_DIM}D action/state, got action=${ACTION_DIM}, state=${STATE_DIM}" >&2
  exit 1
fi

"${REPO_ROOT}/.venv/bin/python" \
  "${SCRIPT_DIR}/project_galbot_g1_for_pi05.py" \
  "${SOURCE_DATASET_ROOT}" \
  "${DATASET_ROOT}" \
  --target-dim 23 \
  --camera-key observation.images.image_head_left \
  --camera-key observation.images.image_arm_left \
  --camera-key observation.images.image_arm_right

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "Projected dataset metadata was not created: ${DATASET_ROOT}/meta/info.json" >&2
  exit 1
fi

SAMPLES_PER_MICRO_STEP=$((BATCH_SIZE_PER_GPU * NUM_GPUS))
MIN_MICRO_STEPS=$(((NUM_FRAMES * EPOCHS + SAMPLES_PER_MICRO_STEP - 1) / SAMPLES_PER_MICRO_STEP))
TRAIN_STEPS=$(((MIN_MICRO_STEPS + GRADIENT_ACCUMULATION_STEPS - 1) / GRADIENT_ACCUMULATION_STEPS * GRADIENT_ACCUMULATION_STEPS))
OPTIMIZER_STEPS=$((TRAIN_STEPS / GRADIENT_ACCUMULATION_STEPS))
EFFECTIVE_BATCH_SIZE=$((SAMPLES_PER_MICRO_STEP * GRADIENT_ACCUMULATION_STEPS))
WARMUP_STEPS=$(((TRAIN_STEPS * 3 + 99) / 100))

if (( TRAIN_STEPS != 11538 || WARMUP_STEPS != 347 )); then
  echo "Unexpected dataset-derived schedule: steps=${TRAIN_STEPS}, warmup=${WARMUP_STEPS}" >&2
  echo "Update ${CONFIG_PATH} before launching." >&2
  exit 1
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_CHECKPOINT="$(realpath -e -- "${RESUME_CHECKPOINT}")"
  CONFIG_PATH="${RESUME_CHECKPOINT}/pretrained_model/train_config.json"
  for required_path in \
    "${CONFIG_PATH}" \
    "${RESUME_CHECKPOINT}/pretrained_model/model.safetensors" \
    "${RESUME_CHECKPOINT}/training_state/training_step.json" \
    "${RESUME_CHECKPOINT}/training_state/optimizer_0/.metadata" \
    "${RESUME_CHECKPOINT}/training_state/scheduler_state.json" \
    "${RESUME_CHECKPOINT}/training_state/rng_state.safetensors"; do
    if [[ ! -f "${required_path}" ]]; then
      echo "Missing resume artifact: ${required_path}" >&2
      exit 1
    fi
  done
  if ! "${REPO_ROOT}/.venv/bin/python" - "${CONFIG_PATH}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
actual = {
    key
    for key in config.get("policy", {}).get("input_features", {})
    if key.startswith("observation.images.")
}
expected = {
    "observation.images.head_left_rgb",
    "observation.images.right_wrist_rgb",
    "observation.images.left_wrist_rgb",
}
if actual != expected:
    print(
        "The resume checkpoint was trained with a different camera set: "
        f"{sorted(actual)}; expected exactly {sorted(expected)}. "
        "Start a new 3-camera run from the base pi05 checkpoint instead.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
  then
    exit 1
  fi
  OUTPUT_DIR="${OUTPUT_DIR:-$(dirname -- "$(dirname -- "${RESUME_CHECKPOINT}")")}" 
  RUN_ID="${RUN_ID:-$(basename -- "${OUTPUT_DIR}")}"
  RESUME_ARGS=(--resume=true)
fi

RUN_ID="${RUN_ID:-piper_pi05_galbot_g1_bread_bs4_ga6_ep5_$(date -u +%Y%m%d_%H%M%S)}"
MASTER_PORT="${MASTER_PORT:-29500}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_ID}}"

# Fail before torchrun if the output path would be overwritten.
if [[ -e "${OUTPUT_DIR}" && -z "${RESUME_CHECKPOINT}" ]]; then
  echo "Output directory already exists: ${OUTPUT_DIR}" >&2
  echo "Set OUTPUT_DIR to a new path before launching." >&2
  exit 1
fi

echo "Run ID: ${RUN_ID}"
echo "Source dataset: ${SOURCE_DATASET_ROOT} (${NUM_FRAMES} frames, action/state=${ACTION_DIM}D)"
echo "Training dataset: ${DATASET_ROOT} (projected action/state=23D, cameras=head-left+left-wrist+right-wrist)"
echo "GPUs: ${NUM_GPUS}; batch/GPU: ${BATCH_SIZE_PER_GPU}; GA: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective batch: ${EFFECTIVE_BATCH_SIZE}; epochs: ${EPOCHS}"
echo "Micro-steps: ${TRAIN_STEPS}; optimizer updates: ${OPTIMIZER_STEPS}; warmup: ${WARMUP_STEPS}"
echo "Output: ${OUTPUT_DIR}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  echo "Resuming model, optimizer, scheduler and data position from: ${RESUME_CHECKPOINT}"
fi

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
  "${RESUME_ARGS[@]}" \
  --policy.gradient_checkpointing=false \
  --policy.compile_model=true \
  --dataset.repo_id=Galbot_G1_take_bread_from_microwave_0909_2720_pi05_23d_3cam \
  --dataset.root="${DATASET_ROOT}" \
  --batch_size="${BATCH_SIZE_PER_GPU}" \
  --accelerator.gradient_accumulation.steps="${GRADIENT_ACCUMULATION_STEPS}" \
  --steps="${TRAIN_STEPS}" \
  --save_freq="${SAVE_FREQ:-0}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.mode="${WANDB_MODE}" \
  --wandb.project="${WANDB_PROJECT}" \
  --wandb.run_id="${RUN_ID}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_ID}"
