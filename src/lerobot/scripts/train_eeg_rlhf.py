#!/usr/bin/env python
"""Two-stage EEG-assisted RLHF training for LeRobot PI0/PI0.5.

Stage 1 jointly trains a Bradley--Terry trajectory scorer and its EEG
alignment predictor.  Stage 2 freezes the EEG-free scorer snapshot and runs one
FlowPRO/RPRO policy update while mixing successful SFT samples.  EEG is never
fed to the policy or scorer during rollout.

The script intentionally keeps EEG in the DataLoader path.  Manifest rows are
small metadata records; EEG windows and BrainMU features are computed inside
``__getitem__``/the batch and are not saved as cache files.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from lerobot.datasets import EEGChunkWindowDataset, EEGPreferencePairDataset, EEGWindowConfig, LeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_policy_config, make_pre_post_processors
from lerobot.rewards.eeg_assisted_scorer.modeling_eeg_assisted_scorer import (
    LOSER_EEG_VALID,
    WINNER_EEG_VALID,
    EEGAssistedScorerRewardModel,
)
from lerobot.rewards.eeg_assisted_scorer.configuration_eeg_assisted_scorer import EEGAssistedScorerConfig


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("scorer", "policy", "all"), default="all")
    p.add_argument("--dataset-repo-id", required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--loser-dataset-repo-id")
    p.add_argument("--loser-dataset-root", type=Path)
    p.add_argument("--winner-manifest", type=Path, required=True)
    p.add_argument("--loser-manifest", type=Path, required=True)
    p.add_argument("--policy", choices=("pi0", "pi05"), default="pi05")
    p.add_argument("--pretrained-policy", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--half-window-s", type=float, default=2.5)
    p.add_argument("--scorer-steps", type=int, default=1000)
    p.add_argument("--policy-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--policy-learning-rate", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--alignment-weight", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eeg-tokenizer-src", type=Path)
    p.add_argument("--eeg-checkpoint-path", type=Path)
    p.add_argument("--trust-eeg-checkpoint", action="store_true")
    return p.parse_args()


def _policy_dataset(
    args: argparse.Namespace,
    policy_config: Any,
    manifest: Path,
    encoder: Any | None = None,
    *,
    repo_id: str | None = None,
    root: Path | None = None,
) -> EEGChunkWindowDataset:
    repo_id = repo_id or args.dataset_repo_id
    root = root or args.dataset_root
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    delta = resolve_delta_timestamps(policy_config, metadata)
    dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta, return_uint8=True)
    return EEGChunkWindowDataset(
        dataset,
        manifest,
        config=EEGWindowConfig(chunk_size=args.chunk_size, half_window_s=args.half_window_s),
        encoder=encoder,
    )


def _collate_policy_samples(samples: list[dict[str, Any]]) -> dict[str, Tensor]:
    """Default LeRobot collation with nested EEG metadata kept explicit."""
    keys = set.intersection(*(set(sample) for sample in samples))
    result: dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        if isinstance(values[0], Tensor):
            result[key] = torch.stack(values)
        elif key == "task":
            result[key] = values
    return result


def _pair_collate(items: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Tensor]]:
    return {
        "winner": _collate_policy_samples([item["winner"] for item in items]),
        "loser": _collate_policy_samples([item["loser"] for item in items]),
    }


def _scorer_batch(
    policy: Any,
    pair_batch: dict[str, dict[str, Any]],
    device: torch.device,
    preprocessor: Any | None = None,
) -> dict[str, Tensor]:
    output: dict[str, Tensor] = {}
    for side, prefix in (("winner", "winner"), ("loser", "loser")):
        batch = {key: value for key, value in pair_batch[side].items() if isinstance(value, Tensor)}
        if preprocessor is not None:
            original = pair_batch[side]
            batch = preprocessor({**original, **batch})
        batch = {key: value.to(device) for key, value in batch.items()}
        features, mask = policy.extract_behavior_features(batch, enable_grad=False)
        output[f"observation.eeg_rlhf.{prefix}_behavior_features"] = features
        output[f"observation.eeg_rlhf.{prefix}_behavior_mask"] = mask
        token = batch.get("observation.eeg_rlhf.token")
        if token is not None:
            output[f"observation.eeg_rlhf.{prefix}_eeg_latent"] = token.detach()
        valid = batch.get("observation.eeg_rlhf.window_valid")
        if valid is not None:
            output[WINNER_EEG_VALID if prefix == "winner" else LOSER_EEG_VALID] = valid.bool()
    return output


def _train_scorer(
    args: argparse.Namespace,
    policy: Any,
    winner: EEGChunkWindowDataset,
    loser: EEGChunkWindowDataset,
    device: torch.device,
    preprocessor: Any | None = None,
) -> Path:
    # Pair the same frame windows from the success and failure manifests.  A
    # production run should pass a manifest containing explicit winner/loser
    # pairing; this strict requirement prevents accidental label copying.
    pair = EEGPreferencePairDataset(winner, loser)
    loader = DataLoader(pair, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=_pair_collate)
    config = EEGAssistedScorerConfig(
        device=str(device),
        behavior_feature_dim=int(policy.model.action_out_proj.in_features),
        alignment_weight=args.alignment_weight,
        learning_rate=args.learning_rate,
        eeg_feature_dim=(int(winner[0]["observation.eeg_rlhf.token"].shape[-1]) if args.eeg_tokenizer_src else 1024),
        eeg_tokenizer_src=str(args.eeg_tokenizer_src) if args.eeg_tokenizer_src else None,
        eeg_checkpoint_path=str(args.eeg_checkpoint_path) if args.eeg_checkpoint_path else None,
        trust_eeg_checkpoint=args.trust_eeg_checkpoint,
    )
    scorer = EEGAssistedScorerRewardModel(config).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.learning_rate)
    iterator = iter(loader)
    scorer.train()
    metrics: dict[str, float] = {}
    for step in range(1, args.scorer_steps + 1):
        try:
            pair_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            pair_batch = next(iterator)
        tensors = _scorer_batch(policy, pair_batch, device, preprocessor)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = scorer(tensors)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), config.grad_clip_norm)
        optimizer.step()
        if step == 1 or step == args.scorer_steps or step % 50 == 0:
            print(f"scorer step={step} loss={float(loss):.6f} preference={metrics['preference_accuracy']:.3f}")
    output = args.output_dir / "scorer"
    output.mkdir(parents=True, exist_ok=True)
    scorer.save_pretrained(output)
    (output / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return output


def _train_policy(
    args: argparse.Namespace,
    policy: Any,
    winner: EEGChunkWindowDataset,
    loser: EEGChunkWindowDataset,
    scorer_path: Path,
    device: torch.device,
    preprocessor: Any | None = None,
) -> None:
    from lerobot.rewards.eeg_assisted_scorer.modeling_eeg_assisted_scorer import EEGRolloutScorer

    # The scorer snapshot is deliberately loaded only for candidate weighting;
    # it never receives EEG during this phase.
    rollout_scorer = EEGRolloutScorer.from_pretrained(scorer_path, device=device)
    pair_loader = DataLoader(
        EEGPreferencePairDataset(winner, loser),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_pair_collate,
    )
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.policy_learning_rate)
    iterator = iter(pair_loader)
    policy.train()
    for step in range(1, args.policy_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(pair_loader)
            batch = next(iterator)
        winner_batch = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch["winner"].items()}
        loser_batch = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch["loser"].items()}
        if preprocessor is not None:
            winner_batch = preprocessor(winner_batch)
            loser_batch = preprocessor(loser_batch)
        # The policy update is EEG-free.  The same successful replay is used as
        # the preference winner and SFT anchor; rejected rollouts are supplied
        # by the paired rollout collector in online deployments.
        with torch.no_grad():
            features, mask = policy.extract_behavior_features(winner_batch)
            pair_weight = torch.sigmoid(rollout_scorer(features, mask)).clamp_min(1e-3)
        loss, metrics = policy.compute_rpro_loss(
            winner_batch,
            loser_batch,
            reference_policy=reference,
            beta=args.beta,
            pair_weight=pair_weight,
            lambda_sft=1.0,
            success_sft_batch=winner_batch,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == args.policy_steps or step % 50 == 0:
            print(f"policy step={step} loss={float(loss):.6f} margin={metrics['implicit_reward_margin']:.6f}")
    output = args.output_dir / "policy"
    output.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output)


def main() -> None:
    args = _args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.eeg_tokenizer_src is None or args.eeg_checkpoint_path is None:
        raise ValueError(
            "EEG-RLHF requires --eeg-tokenizer-src and --eeg-checkpoint-path; "
            "raw EEG is encoded in the DataLoader and never cached"
        )
    config = make_policy_config(args.policy, pretrained_path=args.pretrained_policy, device=str(device), push_to_hub=False)
    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    policy = make_policy(config, ds_meta=metadata).to(device)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=metadata.stats,
        dataset_meta=metadata,
    )
    eeg_encoder = None
    if args.eeg_tokenizer_src and args.eeg_checkpoint_path:
        from lerobot.rewards.eeg_assisted_scorer.brainmu_human import BrainMuHumanEEGEncoder

        eeg_encoder = BrainMuHumanEEGEncoder(
            tokenizer_src=args.eeg_tokenizer_src,
            checkpoint_path=args.eeg_checkpoint_path,
            device=args.device,
            trust_checkpoint=args.trust_eeg_checkpoint,
        ).encode
    winner_dataset = _policy_dataset(args, config, args.winner_manifest, eeg_encoder)
    loser_dataset = _policy_dataset(
        args,
        config,
        args.loser_manifest,
        eeg_encoder,
        repo_id=args.loser_dataset_repo_id,
        root=args.loser_dataset_root,
    )
    scorer_path = args.output_dir / "scorer"
    if args.stage in {"scorer", "all"}:
        scorer_path = _train_scorer(args, policy, winner_dataset, loser_dataset, device, preprocessor)
    if args.stage in {"policy", "all"}:
        if not scorer_path.is_dir():
            raise FileNotFoundError(f"stage=policy requires scorer checkpoint at {scorer_path}")
        _train_policy(args, policy, winner_dataset, loser_dataset, scorer_path, device, preprocessor)


if __name__ == "__main__":
    main()
