# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.rewards import RewardModelConfig
from lerobot.optim import AdamWConfig, LRSchedulerConfig, OptimizerConfig


@RewardModelConfig.register_subclass(name="eeg_assisted_scorer")
@dataclass
class EEGAssistedScorerConfig(RewardModelConfig):
    """Configuration for the training-only EEG-assisted trajectory scorer.

    The reward path only consumes PI0/PI0.5 action-expert behavior features. Raw
    EEG or precomputed EEG latents are accepted exclusively by the auxiliary
    alignment path during scorer training.
    """

    name: str = "eeg_assisted_scorer"

    # Default PI0/PI0.5 ``gemma_300m`` action-expert hidden width.
    behavior_feature_dim: int = 1024
    behavior_dim: int = 256
    pooling_hidden_dim: int = 256
    reward_hidden_dim: int = 256
    eeg_feature_dim: int = 1024
    eeg_predictor_hidden_dim: int = 512
    dropout: float = 0.1

    preference_temperature: float = 1.0
    alignment_weight: float = 1.0
    score_regularization_weight: float = 1e-4

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # BrainMU-H is kept external: the source tree and checkpoint are supplied
    # by the experiment, and its multi-GB weights are never embedded in scorer
    # checkpoints. They are only needed when batches contain raw EEG instead of
    # precomputed ``z_e`` tensors.
    eeg_tokenizer_src: str | None = None
    eeg_checkpoint_path: str | None = None
    eeg_checkpoint_variant: str = "ema"
    eeg_encoder_device: str = "cpu"
    trust_eeg_checkpoint: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        integer_fields = {
            "behavior_feature_dim": self.behavior_feature_dim,
            "behavior_dim": self.behavior_dim,
            "pooling_hidden_dim": self.pooling_hidden_dim,
            "reward_hidden_dim": self.reward_hidden_dim,
            "eeg_feature_dim": self.eeg_feature_dim,
            "eeg_predictor_hidden_dim": self.eeg_predictor_hidden_dim,
        }
        invalid = [name for name, value in integer_fields.items() if value <= 0]
        if invalid:
            raise ValueError(f"EEG scorer dimensions must be positive: {', '.join(invalid)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.preference_temperature <= 0:
            raise ValueError("preference_temperature must be positive")
        if self.alignment_weight < 0 or self.score_regularization_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")
        if self.eeg_checkpoint_variant not in {"ema", "model"}:
            raise ValueError("eeg_checkpoint_variant must be 'ema' or 'model'")

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        # Pairwise tensors use the ``observation.eeg_rlhf.*`` namespace. The
        # exact set depends on whether raw EEG or precomputed latents are used,
        # so runtime validation in the model is authoritative.
        return None
