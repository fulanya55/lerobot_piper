#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export WANDB_MODE=online
export WANDB_PROJECT=lerobot

RUN_ID=robocoin_pi0_lora_v4
LEARNING_RATE=2e-5
DECAY_LEARNING_RATE=1e-6
BATCH_SIZE_PER_GPU=48
GRADIENT_ACCUMULATION_STEPS=1

uv run torchrun \
  --nnodes=1 \
  --node-rank=0 \
  --nproc-per-node=8 \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  /root/lerobot/.venv/bin/lerobot-train \
  --config_path=/root/lerobot/scripts/train/train_config_1day.yaml \
  --policy.optimizer_lr="${LEARNING_RATE}" \
  --policy.scheduler_decay_lr="${DECAY_LEARNING_RATE}" \
  --batch_size="${BATCH_SIZE_PER_GPU}" \
  --accelerator.gradient_accumulation.steps="${GRADIENT_ACCUMULATION_STEPS}" \
  --steps=9428 \
  --save_freq=5000 \
  --log_freq=1 \
  --num_workers=8 \
  --seed=42 \
  --wandb.enable=true \
  --wandb.project="${WANDB_PROJECT}" \
  --wandb.run_id="${RUN_ID}" \
  --output_dir="/root/lerobot/outputs/train/${RUN_ID}" \
  --job_name="${RUN_ID}"
