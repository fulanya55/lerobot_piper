# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FlowPRO/RPRO objectives shared by PI0 and PI0.5.

The implicit reward follows FlowPRO equations (3)--(6):
``beta / 2 * (reference_loss - policy_loss)``. Current and frozen-reference
policies must evaluate each sample with the same flow noise and timestep.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor


def _validate_loss_vector(name: str, value: Tensor, expected_shape: torch.Size | None = None) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be a per-sample vector [B], got {tuple(value.shape)}")
    if value.numel() == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match {tuple(expected_shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _weighted_mean(value: Tensor, weight: Tensor | None) -> Tensor:
    if weight is None:
        return value.mean()
    if weight.shape != value.shape:
        raise ValueError(f"pair_weight shape {tuple(weight.shape)} does not match {tuple(value.shape)}")
    weight = weight.detach().to(device=value.device, dtype=value.dtype)
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("pair_weight must be finite and non-negative")
    if not (weight > 0).any():
        raise ValueError("pair_weight must contain at least one positive value")
    return (value * weight).sum() / weight.sum()


def rpro_loss(
    policy_winner_loss: Tensor,
    policy_loser_loss: Tensor,
    reference_winner_loss: Tensor,
    reference_loser_loss: Tensor,
    *,
    beta: float,
    lambda_pro: float = 1.0,
    lambda_sft: float = 1.0,
    pair_weight: Tensor | None = None,
    sft_loss: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute RPRO from matched current/reference flow-matching losses."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    if lambda_pro < 0 or lambda_sft < 0:
        raise ValueError("lambda_pro and lambda_sft must be non-negative")
    _validate_loss_vector("policy_winner_loss", policy_winner_loss)
    for name, value in (
        ("policy_loser_loss", policy_loser_loss),
        ("reference_winner_loss", reference_winner_loss),
        ("reference_loser_loss", reference_loser_loss),
    ):
        _validate_loss_vector(name, value, policy_winner_loss.shape)

    reference_winner_loss = reference_winner_loss.detach()
    reference_loser_loss = reference_loser_loss.detach()
    reward_winner = 0.5 * beta * (reference_winner_loss - policy_winner_loss)
    reward_loser = 0.5 * beta * (reference_loser_loss - policy_loser_loss)

    contrastive = -functional.logsigmoid(reward_winner - reward_loser)
    proximal = -0.5 * (
        functional.logsigmoid(reward_winner)
        + functional.logsigmoid(-reward_winner)
        + functional.logsigmoid(reward_loser)
        + functional.logsigmoid(-reward_loser)
    )
    pro_loss = _weighted_mean(contrastive + proximal, pair_weight)
    if sft_loss is None:
        sft_loss_mean = _weighted_mean(policy_winner_loss, pair_weight)
    else:
        _validate_loss_vector("sft_loss", sft_loss)
        sft_loss_mean = sft_loss.mean()
    total = lambda_pro * pro_loss + lambda_sft * sft_loss_mean
    metrics = {
        "loss_pro": pro_loss.detach(),
        "loss_rpro_sft": sft_loss_mean.detach(),
        "implicit_reward_winner": _weighted_mean(reward_winner.detach(), pair_weight),
        "implicit_reward_loser": _weighted_mean(reward_loser.detach(), pair_weight),
        "implicit_reward_margin": _weighted_mean((reward_winner - reward_loser).detach(), pair_weight),
    }
    return total, metrics


def compute_flow_matching_rpro(
    policy: Any,
    winner_batch: dict[str, Tensor],
    loser_batch: dict[str, Tensor],
    *,
    reference_policy: Any,
    beta: float,
    lambda_pro: float = 1.0,
    lambda_sft: float = 1.0,
    pair_weight: Tensor | None = None,
    success_sft_batch: dict[str, Tensor] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Evaluate one PI flow-matching RPRO update with matched flow draws."""

    if reference_policy is policy:
        raise ValueError("reference_policy must be a frozen snapshot, not the trainable policy")
    # FSDP2 replaces the trainable policy's class in place with a dynamic
    # sharded subclass, while the frozen reference remains an ordinary PI
    # policy (Accelerate permits only one trainable model per FSDP2 wrapper).
    # Accept that representation as long as either object is an instance of
    # the other's concrete policy class; retain the guard for genuinely
    # different policy families.
    if not (isinstance(reference_policy, type(policy)) or isinstance(policy, type(reference_policy))):
        raise TypeError("policy and reference_policy must have the same PI policy type")
    if any(parameter.requires_grad for parameter in reference_policy.parameters()):
        raise ValueError("reference_policy parameters must all have requires_grad=False")

    winner_loss, _, winner_targets = policy(
        winner_batch,
        reduction="none",
        return_flow_targets=True,
    )
    loser_loss, _, loser_targets = policy(
        loser_batch,
        reduction="none",
        return_flow_targets=True,
    )

    reference_was_training = reference_policy.training
    reference_policy.eval()
    try:
        with torch.no_grad():
            reference_winner_loss, _ = reference_policy(
                winner_batch,
                reduction="none",
                flow_noise=winner_targets["noise"],
                flow_time=winner_targets["time"],
            )
            reference_loser_loss, _ = reference_policy(
                loser_batch,
                reduction="none",
                flow_noise=loser_targets["noise"],
                flow_time=loser_targets["time"],
            )
    finally:
        reference_policy.train(reference_was_training)

    success_sft_loss = None
    if success_sft_batch is not None:
        success_sft_loss, _ = policy(success_sft_batch, reduction="none")
    loss, tensor_metrics = rpro_loss(
        winner_loss,
        loser_loss,
        reference_winner_loss,
        reference_loser_loss,
        beta=beta,
        lambda_pro=lambda_pro,
        lambda_sft=lambda_sft,
        pair_weight=pair_weight,
        sft_loss=success_sft_loss,
    )
    return loss, {key: float(value) for key, value in tensor_metrics.items()}


def compute_flow_matching_grpo(
    policy: Any,
    rollout_batch: dict[str, Tensor],
    *,
    reference_policy: Any,
    advantages: Tensor,
    beta: float = 0.1,
    clip_range: float = 0.2,
    kl_coef: float = 0.02,
    lambda_grpo: float = 1.0,
    lambda_sft: float = 0.0,
    success_sft_batch: dict[str, Tensor] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Compute a grouped PPO/GRPO update for PI flow-matching policies.

    PI's flow-matching loss is not an explicit token log-probability.  For a
    fixed flow noise/time draw, ``-loss`` is a consistent energy/log-density
    proxy, so the policy/reference difference supplies the GRPO ratio.  The
    caller must construct ``advantages`` from K rollouts of the same prompt;
    this function does not fabricate online rollouts from offline success and
    failure demonstrations.
    """

    if reference_policy is policy:
        raise ValueError("reference_policy must be a frozen rollout snapshot")
    if type(reference_policy) is not type(policy):
        raise TypeError("policy and reference_policy must have the same PI policy type")
    if any(parameter.requires_grad for parameter in reference_policy.parameters()):
        raise ValueError("reference_policy parameters must all have requires_grad=False")
    if beta <= 0 or clip_range <= 0 or kl_coef < 0 or lambda_grpo < 0 or lambda_sft < 0:
        raise ValueError("invalid GRPO coefficient")

    policy_loss, _, targets = policy(
        rollout_batch,
        reduction="none",
        return_flow_targets=True,
    )
    if advantages.ndim != 1 or advantages.shape != policy_loss.shape:
        raise ValueError(
            f"advantages must be a finite vector with shape {tuple(policy_loss.shape)}, "
            f"got {tuple(advantages.shape)}"
        )
    advantages = advantages.to(device=policy_loss.device, dtype=policy_loss.dtype).detach()
    if not torch.isfinite(advantages).all():
        raise ValueError("advantages contain NaN or Inf")

    was_training = reference_policy.training
    reference_policy.eval()
    try:
        with torch.no_grad():
            reference_loss, _ = reference_policy(
                rollout_batch,
                reduction="none",
                flow_noise=targets["noise"],
                flow_time=targets["time"],
            )
    finally:
        reference_policy.train(was_training)
    reference_loss = reference_loss.detach()
    log_ratio = beta * (reference_loss - policy_loss)
    ratio = log_ratio.clamp(-20.0, 20.0).exp()
    clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    policy_objective = -surrogate.mean()
    approx_kl = 0.5 * log_ratio.square().mean()
    loss = lambda_grpo * policy_objective + kl_coef * approx_kl

    sft_loss = policy_loss.new_zeros(())
    if success_sft_batch is not None and lambda_sft:
        success_sft_loss, _ = policy(success_sft_batch, reduction="none")
        sft_loss = success_sft_loss.mean()
        loss = loss + lambda_sft * sft_loss
    metrics = {
        "loss_grpo": float(policy_objective.detach()),
        "loss_grpo_kl": float(approx_kl.detach()),
        "loss_grpo_sft": float(sft_loss.detach()),
        "grpo_ratio_mean": float(ratio.detach().mean()),
        "grpo_clip_fraction": float((ratio.detach() != clipped_ratio.detach()).float().mean()),
        "grpo_advantage_mean": float(advantages.mean()),
        "grpo_advantage_std": float(advantages.std(unbiased=False)),
    }
    return loss, metrics
