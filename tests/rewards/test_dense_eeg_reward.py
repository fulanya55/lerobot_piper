import copy

import torch
from torch import nn

from lerobot.policies.common.preference import compute_flow_matching_grpo
from lerobot.rewards.eeg_assisted_scorer.dense_potential import (
    DensePotentialRewardModel,
    dense_potential_loss,
    grouped_advantages,
)


def test_dense_potential_uses_complete_trajectory_and_padding() -> None:
    model = DensePotentialRewardModel(behavior_dim=16, eeg_dim=8, hidden_dim=12, gamma=0.99)
    behavior = torch.randn(4, 7, 16)
    eeg = torch.randn(4, 7, 8)
    valid = torch.ones(4, 7, dtype=torch.bool)
    valid[0, -2:] = False
    output = model(behavior, eeg, valid_mask=valid, eeg_valid=valid)
    loss, metrics = dense_potential_loss(output, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    loss.backward()
    assert output.phi.shape == (4, 7)
    assert output.local_reward.shape == (4, 7)
    assert torch.equal(output.local_reward[0, -2:], torch.zeros(2))
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["global_accuracy"] <= 1.0


def test_potential_shaping_telescopes_to_initial_potential() -> None:
    model = DensePotentialRewardModel(behavior_dim=8, eeg_dim=8, hidden_dim=10, gamma=0.99)
    behavior = torch.randn(2, 6, 8)
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[0, -2:] = False
    output = model.behavior_only(behavior, valid_mask=valid)
    rewards = model.dense_rewards(output)
    discounts = model.gamma ** torch.arange(6)
    for index in range(2):
        assert torch.allclose(
            (rewards[index] * discounts).sum(),
            -output.phi[index, 0],
            atol=1e-5,
        )


def test_local_replay_marks_supervise_potential_difference() -> None:
    model = DensePotentialRewardModel(behavior_dim=8, eeg_dim=8, hidden_dim=10, gamma=0.99)
    behavior = torch.randn(2, 5, 8)
    output = model.behavior_only(behavior)
    target = torch.zeros(2, 5)
    target[:, 1] = 1.0
    target[:, 3] = -1.0
    target_mask = torch.zeros(2, 5, dtype=torch.bool)
    target_mask[:, 1] = True
    target_mask[:, 3] = True
    loss, metrics = dense_potential_loss(
        output,
        torch.tensor([1.0, 0.0]),
        local_target=target,
        local_target_mask=target_mask,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["local_event_count"].item() == 4
    assert metrics["loss_local_event"].item() > 0


def test_grouped_advantages_are_centered_per_rollout_group() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0, 5.0, 7.0])
    groups = torch.tensor([0, 0, 1, 1, 1])
    advantage = grouped_advantages(rewards, groups)
    assert torch.allclose(advantage[groups == 0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantage[groups == 1].mean(), torch.tensor(0.0), atol=1e-6)


class _FakeFlowPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        batch,
        reduction="mean",
        *,
        flow_noise=None,
        flow_time=None,
        return_flow_targets=False,
    ):
        x = batch["x"]
        noise = torch.zeros_like(x) if flow_noise is None else flow_noise
        time = torch.full((len(x),), 0.5, device=x.device) if flow_time is None else flow_time
        value = (self.scale * x - noise).square().mean(dim=1)
        output = value if reduction == "none" else value.mean()
        if return_flow_targets:
            return output, {}, {"noise": noise, "time": time}
        return output, {}


def test_flow_matching_grpo_has_trainable_surrogate() -> None:
    policy = _FakeFlowPolicy()
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    batch = {"x": torch.tensor([[1.0, 2.0], [2.0, 3.0]])}
    loss, metrics = compute_flow_matching_grpo(
        policy,
        batch,
        reference_policy=reference,
        advantages=torch.tensor([-1.0, 1.0]),
    )
    loss.backward()
    assert policy.scale.grad is not None
    assert torch.isfinite(policy.scale.grad)
    assert set(metrics) >= {"loss_grpo", "loss_grpo_kl", "grpo_ratio_mean"}
