# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from lerobot.rewards.pretrained import PreTrainedRewardModel

from .brainmu_human import BrainMuHumanEEGEncoder
from .configuration_eeg_assisted_scorer import EEGAssistedScorerConfig

EEG_RLHF_PREFIX = "observation.eeg_rlhf."
BEHAVIOR_FEATURES = EEG_RLHF_PREFIX + "behavior_features"
BEHAVIOR_MASK = EEG_RLHF_PREFIX + "behavior_mask"
WINNER_BEHAVIOR_FEATURES = EEG_RLHF_PREFIX + "winner_behavior_features"
WINNER_BEHAVIOR_MASK = EEG_RLHF_PREFIX + "winner_behavior_mask"
LOSER_BEHAVIOR_FEATURES = EEG_RLHF_PREFIX + "loser_behavior_features"
LOSER_BEHAVIOR_MASK = EEG_RLHF_PREFIX + "loser_behavior_mask"
WINNER_EEG = EEG_RLHF_PREFIX + "winner_eeg"
LOSER_EEG = EEG_RLHF_PREFIX + "loser_eeg"
WINNER_EEG_LATENT = EEG_RLHF_PREFIX + "winner_eeg_latent"
LOSER_EEG_LATENT = EEG_RLHF_PREFIX + "loser_eeg_latent"
WINNER_EEG_VALID = EEG_RLHF_PREFIX + "winner_eeg_valid"
LOSER_EEG_VALID = EEG_RLHF_PREFIX + "loser_eeg_valid"


class MaskedTemporalPooling(nn.Module):
    """Small trainable attention pooling over already-completed action tokens."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.token_projection = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
        self.output = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: Tensor, mask: Tensor | None = None) -> Tensor:
        if features.ndim != 3:
            raise ValueError(f"behavior features must be [B,T,D], got {tuple(features.shape)}")
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if mask.shape != features.shape[:2]:
            raise ValueError(
                f"behavior mask shape {tuple(mask.shape)} does not match {tuple(features.shape[:2])}"
            )
        mask = mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("each trajectory must contain at least one valid behavior token")

        logits = self.attention(torch.tanh(self.token_projection(features))).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = logits.softmax(dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * features, dim=1)
        return self.output(pooled)


def _make_reward_head(config: EEGAssistedScorerConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(config.behavior_dim, config.reward_hidden_dim),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.reward_hidden_dim, 1),
    )


class EEGRolloutScorer(nn.Module):
    """EEG-free scorer snapshot used for candidate ranking during rollout."""

    def __init__(self, behavior_pooling: nn.Module, reward_head: nn.Module) -> None:
        super().__init__()
        self.behavior_pooling = behavior_pooling
        self.reward_head = reward_head

    def forward(self, behavior_features: Tensor, behavior_mask: Tensor | None = None) -> Tensor:
        return self.reward_head(self.behavior_pooling(behavior_features, behavior_mask)).squeeze(-1)

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> EEGRolloutScorer:
        """Load the compact EEG-free snapshot emitted by the scorer trainer."""

        from safetensors.torch import load_file

        directory = Path(directory).expanduser().resolve(strict=True)
        config_path = directory / "rollout_scorer_config.json"
        weights_path = directory / "rollout_scorer.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"expected rollout_scorer_config.json and rollout_scorer.safetensors in {directory}"
            )
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("contains_eeg_modules") is not False:
            raise ValueError("rollout scorer config must explicitly declare contains_eeg_modules=false")
        config = EEGAssistedScorerConfig(
            device="cpu",
            behavior_feature_dim=int(payload["behavior_feature_dim"]),
            pooling_hidden_dim=int(payload["pooling_hidden_dim"]),
            behavior_dim=int(payload["behavior_dim"]),
            reward_hidden_dim=int(payload["reward_hidden_dim"]),
            dropout=float(payload["dropout"]),
        )
        scorer = cls(
            behavior_pooling=MaskedTemporalPooling(
                input_dim=config.behavior_feature_dim,
                hidden_dim=config.pooling_hidden_dim,
                output_dim=config.behavior_dim,
                dropout=config.dropout,
            ),
            reward_head=_make_reward_head(config),
        )
        scorer.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
        scorer.to(device).eval().requires_grad_(False)
        return scorer


class EEGAssistedScorerRewardModel(PreTrainedRewardModel):
    """Pairwise trajectory scorer with training-only BrainMU-H alignment."""

    name = "eeg_assisted_scorer"
    config_class = EEGAssistedScorerConfig

    def __init__(self, config: EEGAssistedScorerConfig, **_: Any) -> None:
        super().__init__(config)
        self.behavior_pooling = MaskedTemporalPooling(
            input_dim=config.behavior_feature_dim,
            hidden_dim=config.pooling_hidden_dim,
            output_dim=config.behavior_dim,
            dropout=config.dropout,
        )
        self.eeg_predictor = nn.Sequential(
            nn.Linear(config.behavior_dim, config.eeg_predictor_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.eeg_predictor_hidden_dim, config.eeg_feature_dim),
        )
        self.reward_head = _make_reward_head(config)
        # Keep the multi-GB frozen foundation encoder outside Module registration
        # so scorer checkpoints contain only trainable scorer parameters.
        object.__setattr__(self, "_eeg_target_encoder", None)

    def _get_behavior(self, batch: dict[str, Tensor], prefix: str = "") -> tuple[Tensor, Tensor | None]:
        feature_key = {
            "": BEHAVIOR_FEATURES,
            "winner": WINNER_BEHAVIOR_FEATURES,
            "loser": LOSER_BEHAVIOR_FEATURES,
        }[prefix]
        mask_key = {
            "": BEHAVIOR_MASK,
            "winner": WINNER_BEHAVIOR_MASK,
            "loser": LOSER_BEHAVIOR_MASK,
        }[prefix]
        if feature_key not in batch:
            raise KeyError(f"missing required scorer tensor {feature_key!r}")
        features = batch[feature_key]
        if features.shape[-1] != self.config.behavior_feature_dim:
            raise ValueError(
                f"{feature_key} last dim must be {self.config.behavior_feature_dim}, got {features.shape[-1]}"
            )
        return features, batch.get(mask_key)

    def encode_behavior(self, features: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.behavior_pooling(features, mask)

    def score_behavior(self, features: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.reward_head(self.encode_behavior(features, mask)).squeeze(-1)

    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        features, mask = self._get_behavior(batch)
        return self.score_behavior(features, mask)

    def _ensure_eeg_target_encoder(self) -> BrainMuHumanEEGEncoder:
        encoder = self._eeg_target_encoder
        if encoder is not None:
            return encoder
        if not self.config.eeg_tokenizer_src or not self.config.eeg_checkpoint_path:
            raise ValueError(
                "raw EEG was provided but eeg_tokenizer_src/eeg_checkpoint_path are not configured; "
                "provide both paths or precompute winner/loser_eeg_latent"
            )
        encoder = BrainMuHumanEEGEncoder(
            tokenizer_src=self.config.eeg_tokenizer_src,
            checkpoint_path=self.config.eeg_checkpoint_path,
            variant=self.config.eeg_checkpoint_variant,
            device=self.config.eeg_encoder_device,
            trust_checkpoint=self.config.trust_eeg_checkpoint,
        )
        if encoder.feature_dim != self.config.eeg_feature_dim:
            raise ValueError(
                f"BrainMU-H produces {encoder.feature_dim}-D features but eeg_feature_dim="
                f"{self.config.eeg_feature_dim}"
            )
        object.__setattr__(self, "_eeg_target_encoder", encoder)
        return encoder

    def _get_eeg_target(self, batch: dict[str, Tensor], prefix: str, device: torch.device) -> Tensor | None:
        latent_key = WINNER_EEG_LATENT if prefix == "winner" else LOSER_EEG_LATENT
        eeg_key = WINNER_EEG if prefix == "winner" else LOSER_EEG
        if latent_key in batch:
            target = batch[latent_key]
        elif eeg_key in batch:
            target = self._ensure_eeg_target_encoder().encode(batch[eeg_key])
        else:
            return None
        if target.ndim != 2 or target.shape[-1] != self.config.eeg_feature_dim:
            raise ValueError(
                f"{prefix} EEG target must be [B,{self.config.eeg_feature_dim}], got {tuple(target.shape)}"
            )
        if not torch.isfinite(target).all():
            raise ValueError(f"{prefix} EEG target contains NaN or Inf")
        return target.detach().to(device=device, dtype=torch.float32)

    @staticmethod
    def _alignment_loss(prediction: Tensor, target: Tensor, valid: Tensor | None) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"EEG prediction/target shapes must match, got {tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        per_sample = (
            (functional.normalize(prediction.float(), dim=-1) - functional.normalize(target, dim=-1))
            .square()
            .mean(-1)
        )
        if valid is None:
            return per_sample.mean()
        valid = valid.to(device=per_sample.device).reshape(-1)
        if valid.shape[0] != per_sample.shape[0]:
            raise ValueError("EEG validity mask batch size does not match targets")
        valid = valid.bool().to(dtype=per_sample.dtype)
        return (per_sample * valid).sum() / valid.sum().clamp_min(1.0)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        winner_features, winner_mask = self._get_behavior(batch, "winner")
        loser_features, loser_mask = self._get_behavior(batch, "loser")
        winner_behavior = self.encode_behavior(winner_features, winner_mask)
        loser_behavior = self.encode_behavior(loser_features, loser_mask)
        winner_reward = self.reward_head(winner_behavior).squeeze(-1)
        loser_reward = self.reward_head(loser_behavior).squeeze(-1)

        reward_margin = winner_reward - loser_reward
        preference_loss = -functional.logsigmoid(reward_margin / self.config.preference_temperature).mean()

        alignment_terms: list[Tensor] = []
        winner_target = self._get_eeg_target(batch, "winner", winner_behavior.device)
        if winner_target is not None:
            alignment_terms.append(
                self._alignment_loss(
                    self.eeg_predictor(winner_behavior),
                    winner_target,
                    batch.get(WINNER_EEG_VALID),
                )
            )
        loser_target = self._get_eeg_target(batch, "loser", loser_behavior.device)
        if loser_target is not None:
            alignment_terms.append(
                self._alignment_loss(
                    self.eeg_predictor(loser_behavior),
                    loser_target,
                    batch.get(LOSER_EEG_VALID),
                )
            )
        alignment_loss = (
            torch.stack(alignment_terms).mean() if alignment_terms else preference_loss.new_zeros(())
        )
        score_regularization = 0.5 * (winner_reward.square().mean() + loser_reward.square().mean())
        loss = (
            preference_loss
            + self.config.alignment_weight * alignment_loss
            + self.config.score_regularization_weight * score_regularization
        )
        metrics = {
            "loss_preference": float(preference_loss.detach()),
            "loss_alignment": float(alignment_loss.detach()),
            "loss_score_regularization": float(score_regularization.detach()),
            "preference_accuracy": float((reward_margin > 0).float().mean().detach()),
            "reward_margin": float(reward_margin.mean().detach()),
        }
        return loss, metrics

    def export_rollout_scorer(self, *, freeze: bool = True) -> EEGRolloutScorer:
        """Copy only the EEG-free modules needed to rank rollout trajectories."""

        scorer = EEGRolloutScorer(
            behavior_pooling=copy.deepcopy(self.behavior_pooling),
            reward_head=copy.deepcopy(self.reward_head),
        )
        scorer.eval()
        if freeze:
            scorer.requires_grad_(False)
        return scorer
