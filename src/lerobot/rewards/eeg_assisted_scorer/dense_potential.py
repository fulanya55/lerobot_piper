r"""Dense, trajectory-level EEG reward supervision.

This module is intentionally different from the old pairwise scorer.  It
consumes a complete sequence of action-window features and predicts a causal
potential :math:`\Phi(s_t, z_t)`.  The reward used by the policy is the
potential difference

``r_t = gamma * Phi_{t+1} - Phi_t``

while a separate terminal outcome head is used only for weak trajectory-end
supervision.  Deployment/GRPO uses only the potential difference and retains
the environment's terminal outcome reward.  Thus the EEG is trained on every
valid window of every first-viewing trajectory, while the final success/failure
annotation only supervises the trajectory endpoint.  The EEG residual is
initialized at zero and a behaviour-only baseline is retained, which makes the
downstream ablation a real safety gate rather than a post-hoc classification
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class DenseRewardOutput:
    phi: Tensor
    behavior_phi: Tensor
    local_reward: Tensor
    terminal_reward: Tensor
    global_logit: Tensor
    eeg_gate: Tensor
    eeg_scale: Tensor
    behavior_embedding: Tensor
    eeg_embedding: Tensor
    valid_mask: Tensor


class DensePotentialRewardModel(nn.Module):
    """Causal behaviour potential with an optional EEG residual pathway.

    Inputs are complete trajectories. ``behavior`` can be ``[B,T,D]`` or
    ``[B,T,K,D]`` (the latter is pooled over the K action tokens). EEG is one
    latent per trajectory window, ``[B,T,E]``. ``eeg_valid`` is a training-time
    mask; at deployment it can be all zeros and the model becomes exactly the
    exported behaviour-only scorer.
    """

    def __init__(
        self,
        behavior_dim: int = 1024,
        eeg_dim: int = 1024,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(behavior_dim, eeg_dim, hidden_dim) <= 0:
            raise ValueError("feature dimensions must be positive")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.behavior_dim = behavior_dim
        self.eeg_dim = eeg_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.behavior_projection = nn.Sequential(
            nn.LayerNorm(behavior_dim),
            nn.Linear(behavior_dim, hidden_dim),
            nn.GELU(),
        )
        self.eeg_projection = nn.Sequential(
            nn.LayerNorm(eeg_dim),
            nn.Linear(eeg_dim, hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.behavior_phi_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.eeg_delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_head = nn.Linear(hidden_dim * 2, 1)
        self.terminal_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # At step zero the EEG contribution is practically zero.  The
        # parameter can grow during training, but the behavior-only model is
        # always available for the downstream positive-gain gate.
        self.eeg_scale_logit = nn.Parameter(torch.tensor(-8.0))

    @staticmethod
    def _pool_behavior(behavior: Tensor) -> Tensor:
        if behavior.ndim == 4:
            behavior = behavior.mean(dim=2)
        if behavior.ndim != 3:
            raise ValueError(f"behavior must be [B,T,D] or [B,T,K,D], got {tuple(behavior.shape)}")
        return behavior

    @staticmethod
    def _last_valid(hidden: Tensor, mask: Tensor) -> Tensor:
        lengths = mask.long().sum(dim=1).clamp_min(1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]

    def forward(
        self,
        behavior: Tensor,
        eeg: Tensor | None = None,
        *,
        valid_mask: Tensor | None = None,
        eeg_valid: Tensor | None = None,
        use_eeg: bool = True,
    ) -> DenseRewardOutput:
        behavior = self._pool_behavior(behavior).float()
        batch, steps, _ = behavior.shape
        if valid_mask is None:
            valid_mask = torch.ones(batch, steps, dtype=torch.bool, device=behavior.device)
        valid_mask = valid_mask.to(device=behavior.device, dtype=torch.bool)
        if valid_mask.shape != (batch, steps) or not valid_mask.any(dim=1).all():
            raise ValueError("valid_mask must be [B,T] and every trajectory needs one valid step")
        if eeg is None:
            eeg = torch.zeros(batch, steps, self.eeg_dim, device=behavior.device, dtype=behavior.dtype)
            eeg_valid = torch.zeros(batch, steps, device=behavior.device, dtype=torch.bool)
        else:
            if eeg.ndim != 3 or eeg.shape[:2] != (batch, steps) or eeg.shape[-1] != self.eeg_dim:
                raise ValueError(f"eeg must be [B,T,{self.eeg_dim}], got {tuple(eeg.shape)}")
            eeg = eeg.to(device=behavior.device, dtype=behavior.dtype)
            if eeg_valid is None:
                eeg_valid = torch.ones(batch, steps, device=behavior.device, dtype=torch.bool)
        eeg_valid = eeg_valid.to(device=behavior.device, dtype=torch.bool) & valid_mask
        behavior_hidden, _ = self.temporal(self.behavior_projection(behavior))
        eeg_hidden = self.eeg_projection(eeg)
        gate = torch.sigmoid(self.gate_head(torch.cat((behavior_hidden, eeg_hidden), dim=-1))).squeeze(-1)
        scale = F.softplus(self.eeg_scale_logit)
        if not use_eeg:
            gate = torch.zeros_like(gate)
        gate = gate * eeg_valid.to(gate.dtype)
        behavior_phi = self.behavior_phi_head(behavior_hidden).squeeze(-1)
        delta_phi = self.eeg_delta_head(eeg_hidden).squeeze(-1)
        phi = behavior_phi + scale * gate * delta_phi
        last_hidden = self._last_valid(behavior_hidden, valid_mask)
        terminal_reward = self.terminal_head(last_hidden).squeeze(-1)
        # Use an absorbing terminal state with zero potential.  The last valid
        # window therefore receives ``-Phi(s_T)``; the sum telescopes and is a
        # policy-invariant potential-based shaping term.
        local_reward = torch.zeros_like(phi)
        if steps > 1:
            local_reward[:, :-1] = self.gamma * phi[:, 1:] - phi[:, :-1]
        last = valid_mask.long().sum(dim=1).clamp_min(1) - 1
        local_reward[torch.arange(batch, device=phi.device), last] = -phi[
            torch.arange(batch, device=phi.device), last
        ]
        local_reward = local_reward.masked_fill(~valid_mask, 0.0)
        # Supervise the discounted sum of the same dense rewards that GRPO will
        # consume.  This avoids training an unrelated classification head.
        discounts = self.gamma ** torch.arange(local_reward.shape[1], device=phi.device, dtype=phi.dtype)
        dense_return = (local_reward * discounts.unsqueeze(0)).sum(dim=1)
        terminal_discount = self.gamma ** (valid_mask.long().sum(dim=1).clamp_min(1) - 1).to(phi.dtype)
        global_logit = dense_return + terminal_discount * terminal_reward
        return DenseRewardOutput(
            phi=phi,
            behavior_phi=behavior_phi,
            local_reward=local_reward,
            terminal_reward=terminal_reward,
            global_logit=global_logit,
            eeg_gate=gate,
            eeg_scale=scale.detach(),
            behavior_embedding=behavior_hidden,
            eeg_embedding=eeg_hidden,
            valid_mask=valid_mask,
        )

    def behavior_only(self, behavior: Tensor, *, valid_mask: Tensor | None = None) -> DenseRewardOutput:
        """Evaluate the same model with EEG disabled for the safety ablation."""

        return self.forward(behavior, valid_mask=valid_mask, use_eeg=False)

    def dense_rewards(self, output: DenseRewardOutput, *, include_terminal: bool = False) -> Tensor:
        """Return safe potential shaping rewards.

        ``include_terminal=True`` is diagnostic-only.  GRPO must leave it
        false and retain the original environment terminal reward.
        """

        rewards = output.local_reward.clone()
        if include_terminal:
            last = output.valid_mask.long().sum(dim=1).clamp_min(1) - 1
            rewards[torch.arange(rewards.shape[0], device=rewards.device), last] += output.terminal_reward
        return rewards.masked_fill(~output.valid_mask, 0.0)


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def dense_potential_loss(
    output: DenseRewardOutput,
    labels: Tensor,
    *,
    smooth_weight: float = 1e-2,
    residual_weight: float = 1e-4,
    pair_weight: float = 0.25,
    alignment_weight: float = 0.1,
    local_label_weight: float = 0.5,
    local_target: Tensor | None = None,
    local_target_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Weak trajectory-end supervision plus structural dense-reward losses.

    ``labels`` supervise only the complete trajectory.  Pairwise terms are
    sampled across arbitrary independent views in a batch; no same-session
    pairing is assumed or required.
    """

    labels = labels.to(device=output.global_logit.device, dtype=output.global_logit.dtype).reshape(-1)
    if labels.shape != output.global_logit.shape:
        raise ValueError("labels must contain one binary outcome per trajectory")
    if ((labels < 0) | (labels > 1)).any():
        raise ValueError("labels must be binary")
    terminal_loss = F.binary_cross_entropy_with_logits(output.global_logit, labels)
    pos = output.global_logit[labels > 0.5]
    neg = output.global_logit[labels <= 0.5]
    if len(pos) and len(neg):
        pair_logits = pos[:, None] - neg[None, :]
        pairwise = -F.logsigmoid(pair_logits).mean()
    else:
        pairwise = terminal_loss.new_zeros(())
    if output.local_reward.shape[1] > 1:
        smooth = _masked_mean(
            (output.local_reward[:, 1:] - output.local_reward[:, :-1]).square(),
            output.valid_mask[:, 1:] & output.valid_mask[:, :-1],
        )
    else:
        smooth = terminal_loss.new_zeros(())
    residual = _masked_mean((output.phi - output.behavior_phi).square(), output.valid_mask)
    aligned = output.valid_mask & (output.eeg_gate > 0)
    alignment = _masked_mean(
        1.0 - F.cosine_similarity(output.behavior_embedding, output.eeg_embedding, dim=-1),
        aligned,
    )
    if local_target is None:
        local_event_loss = terminal_loss.new_zeros(())
        local_event_count = terminal_loss.new_zeros(())
    else:
        if local_target.shape != output.local_reward.shape:
            raise ValueError("local_target must have the same [B,T] shape as local_reward")
        if local_target_mask is None:
            local_target_mask = torch.ones_like(local_target, dtype=torch.bool)
        if local_target_mask.shape != output.local_reward.shape:
            raise ValueError("local_target_mask must have the same [B,T] shape as local_reward")
        event_mask = local_target_mask.to(device=output.local_reward.device, dtype=torch.bool) & output.valid_mask
        # Correct replay nodes should have positive shaping reward and
        # unsmooth/error nodes negative shaping reward.  A softplus margin is
        # scale-stable and does not turn the sparse annotations into another
        # trajectory-level classifier.
        signed_target = local_target.to(output.local_reward).clamp(-1.0, 1.0)
        local_event_loss = _masked_mean(
            F.softplus(-signed_target * output.local_reward),
            event_mask,
        )
        local_event_count = event_mask.to(output.local_reward.dtype).sum().detach()
    loss = (
        terminal_loss
        + pair_weight * pairwise
        + smooth_weight * smooth
        + residual_weight * residual
        + alignment_weight * alignment
        + local_label_weight * local_event_loss
    )
    metrics = {
        "loss": loss.detach(),
        "loss_terminal": terminal_loss.detach(),
        "loss_pairwise": pairwise.detach(),
        "loss_smooth": smooth.detach(),
        "loss_residual": residual.detach(),
        "loss_alignment": alignment.detach(),
        "loss_local_event": local_event_loss.detach(),
        "local_event_count": local_event_count,
        "global_accuracy": ((output.global_logit.sigmoid() > 0.5) == (labels > 0.5)).float().mean().detach(),
        "eeg_scale": output.eeg_scale.detach(),
        "mean_gate": _masked_mean(output.eeg_gate, output.valid_mask).detach(),
    }
    return loss, metrics


def grouped_advantages(rewards: Tensor, group_ids: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Compute GRPO group-relative advantages for a rollout batch."""

    if rewards.ndim != 1 or group_ids.ndim != 1 or rewards.shape != group_ids.shape:
        raise ValueError("rewards and group_ids must be one-dimensional with equal shape")
    advantages = torch.zeros_like(rewards)
    for group in torch.unique(group_ids):
        mask = group_ids == group
        values = rewards[mask]
        advantages[mask] = (values - values.mean()) / values.std(unbiased=False).clamp_min(eps)
    return advantages


__all__ = [
    "DensePotentialRewardModel",
    "DenseRewardOutput",
    "dense_potential_loss",
    "grouped_advantages",
]
