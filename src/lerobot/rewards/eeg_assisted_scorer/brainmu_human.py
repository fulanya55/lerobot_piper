# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Read-only BrainMU-H integration used by the EEG target branch.

The implementation follows ``Brianmu_tokenizer_human_demo``: the EEG is
encoded with the checkpoint-compatible ``LegacyUniTokAdapter`` and the
pre-quantization 1024-D patch tokens are mean pooled. The external foundation
model remains frozen and is deliberately not registered as a scorer submodule.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor

AUXILIARY_MARKERS = (
    "EOG",
    "HEOG",
    "VEOG",
    "EMG",
    "ECG",
    "EKG",
    "GSR",
    "EDA",
    "RESP",
    "PULSE",
    "PPG",
    "TRIG",
    "STATUS",
    "MARKER",
    "EVENT",
)


def _compact_channel_name(name: str) -> str:
    return "".join(character for character in str(name).upper() if character.isalnum())


def _is_auxiliary_or_padding(name: str) -> bool:
    compact = _compact_channel_name(name)
    return compact.startswith("PADCH") or any(token in compact for token in AUXILIARY_MARKERS)


def select_eeg_channels(channel_names: Sequence[str], limit: int = 19) -> tuple[list[int], list[str]]:
    """Deterministically select channels using the BrainMU-H demo contract."""

    candidates = [
        (str(name).lower(), index, str(name))
        for index, name in enumerate(channel_names)
        if not _is_auxiliary_or_padding(str(name))
    ]
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = candidates[: int(limit)]
    if not selected:
        raise ValueError("No EEG channels remain after excluding auxiliary/padding channels")
    return [item[1] for item in selected], [item[2] for item in selected]


def prepare_brainmu_h_eeg(
    signals: Tensor,
    channel_names: Sequence[str],
    *,
    input_sfreq: float,
    target_sfreq: float = 256.0,
    target_samples: int = 640,
    real_channels: int = 19,
    padded_channels: int = 24,
    crop_mode: str = "left",
) -> tuple[Tensor, list[str]]:
    """Convert ``[C,T]`` or ``[B,C,T]`` EEG to BrainMU-H ``[B,24,640]``."""

    if signals.ndim == 2:
        signals = signals.unsqueeze(0)
    if signals.ndim != 3:
        raise ValueError(f"signals must be [C,T] or [B,C,T], got {tuple(signals.shape)}")
    if signals.shape[1] != len(channel_names):
        raise ValueError(
            f"input has {signals.shape[1]} channels but {len(channel_names)} channel names were provided"
        )
    if input_sfreq <= 0 or target_sfreq <= 0:
        raise ValueError("sampling rates must be positive")
    if padded_channels < real_channels:
        raise ValueError("padded_channels must be >= real_channels")
    if not torch.isfinite(signals).all():
        raise ValueError("input contains NaN or Inf")

    indices, selected_names = select_eeg_channels(channel_names, real_channels)
    eeg = signals.float().index_select(1, torch.tensor(indices, device=signals.device))
    mean = eeg.mean(dim=-1, keepdim=True)
    std = eeg.var(dim=-1, keepdim=True, unbiased=False).sqrt().clamp_min(1e-6)
    eeg = (eeg - mean) / std

    if float(input_sfreq) != float(target_sfreq):
        new_length = max(1, int(round(eeg.shape[-1] * target_sfreq / input_sfreq)))
        eeg = functional.interpolate(eeg, size=new_length, mode="linear", align_corners=False)

    length = int(eeg.shape[-1])
    if length > target_samples:
        starts = {
            "left": 0,
            "center": (length - target_samples) // 2,
            "right": length - target_samples,
        }
        if crop_mode not in starts:
            raise ValueError(f"unsupported crop mode: {crop_mode}")
        start = starts[crop_mode]
        eeg = eeg[..., start : start + target_samples]
    elif length < target_samples:
        eeg = functional.pad(eeg, (0, target_samples - length))

    if eeg.shape[1] < padded_channels:
        eeg = functional.pad(eeg, (0, 0, 0, padded_channels - eeg.shape[1]))
    elif eeg.shape[1] > padded_channels:
        eeg = eeg[:, :padded_channels]
    return eeg.contiguous(), selected_names


@contextmanager
def _torchao_metadata_stub() -> Iterator[None]:
    """Permit trusted legacy optimizer metadata to be deserialized without torchao."""

    try:
        importlib.import_module("torchao.optim.subclass_8bit")
    except ModuleNotFoundError:
        torchao_module = types.ModuleType("torchao")
        torchao_module.__path__ = []  # type: ignore[attr-defined]
        optim_module = types.ModuleType("torchao.optim")
        optim_module.__path__ = []  # type: ignore[attr-defined]
        subclass_module = types.ModuleType("torchao.optim.subclass_8bit")

        class OptimState8bit(torch.Tensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                raise RuntimeError("optimizer metadata tensors must never execute")

        OptimState8bit.__module__ = "torchao.optim.subclass_8bit"
        subclass_module.OptimState8bit = OptimState8bit
        inserted = {
            "torchao": torchao_module,
            "torchao.optim": optim_module,
            "torchao.optim.subclass_8bit": subclass_module,
        }
        sys.modules.update(inserted)
        try:
            yield
        finally:
            for name, module in reversed(list(inserted.items())):
                if sys.modules.get(name) is module:
                    del sys.modules[name]
    else:
        yield


@contextmanager
def _open_clip_architecture_fallback() -> Iterator[None]:
    """Resolve stale ``local-dir:`` architecture paths without changing the checkpoint."""

    try:
        import open_clip
    except ModuleNotFoundError as exc:
        if exc.name == "open_clip":
            message = "BrainMU-H raw EEG encoding requires open-clip-torch. Install lerobot[eeg-rlhf]."
        else:
            message = (
                "The open-clip-torch installation is incomplete; missing dependency "
                f"{exc.name!r}. Reinstall lerobot[eeg-rlhf] with dependencies."
            )
        raise ModuleNotFoundError(message) from exc

    original = open_clip.create_model_from_pretrained
    built_in_models = set(open_clip.list_models())

    def create_read_only(model_name, *args, **kwargs):
        text = str(model_name)
        if not text.startswith("local-dir:"):
            return original(model_name, *args, **kwargs)
        local_path = Path(text.split(":", 1)[1]).expanduser()
        if local_path.is_dir():
            return original(model_name, *args, **kwargs)
        architecture = local_path.name
        if architecture not in built_in_models:
            raise FileNotFoundError(
                f"stale OpenCLIP path {local_path}; unknown architecture {architecture!r}"
            )
        if args:
            raise TypeError("architecture-only fallback rejects positional pretrained arguments")
        if kwargs.pop("return_transform", True):
            raise ValueError("architecture-only fallback requires return_transform=False")
        kwargs.pop("weights_only", None)
        return open_clip.create_model(
            architecture,
            pretrained=None,
            load_weights=False,
            require_pretrained=False,
            **kwargs,
        )

    open_clip.create_model_from_pretrained = create_read_only
    try:
        yield
    finally:
        open_clip.create_model_from_pretrained = original


class BrainMuHumanEEGEncoder:
    """Frozen, external BrainMU-H encoder producing pooled 1024-D targets."""

    feature_dim = 1024

    def __init__(
        self,
        *,
        tokenizer_src: str | Path,
        checkpoint_path: str | Path,
        variant: str = "ema",
        device: str | torch.device = "cpu",
        trust_checkpoint: bool = False,
    ) -> None:
        if not trust_checkpoint:
            raise ValueError(
                "BrainMU-H legacy checkpoints require pickle deserialization. Set "
                "trust_eeg_checkpoint=true only for a checkpoint you trust."
            )
        self.tokenizer_src = Path(tokenizer_src).expanduser().resolve(strict=True)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
        self.device = torch.device(device)
        if not self.tokenizer_src.is_dir() or not self.checkpoint_path.is_dir():
            raise NotADirectoryError("tokenizer_src and checkpoint_path must be directories")

        source_text = str(self.tokenizer_src)
        sys.path.insert(0, source_text)
        try:
            with _open_clip_architecture_fallback(), _torchao_metadata_stub():
                from brainmu_tokenizer.adapter import LegacyUniTokAdapter

                tokenizer = LegacyUniTokAdapter.from_checkpoint(
                    self.checkpoint_path,
                    modality="eeg",
                    variant=variant,
                    device=self.device,
                    strict=False,
                )
        finally:
            if sys.path and sys.path[0] == source_text:
                sys.path.pop(0)

        missing = list(tokenizer.load_info.get("missing_keys", []) or [])
        unexpected = [
            key
            for key in list(tokenizer.load_info.get("unexpected_keys", []) or [])
            if not str(key).startswith("continuous_modality_embeddings.")
        ]
        if missing or unexpected:
            raise RuntimeError(
                "BrainMU-H checkpoint does not fully cover the runtime architecture: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        tokenizer.eval()
        tokenizer.requires_grad_(False)
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def encode(self, signals: Tensor) -> Tensor:
        if signals.ndim != 3 or tuple(signals.shape[-2:]) != (24, 640):
            raise ValueError(
                "BrainMU-H expects preprocessed EEG [B,24,640]; use prepare_brainmu_h_eeg first, "
                f"got {tuple(signals.shape)}"
            )
        if not torch.isfinite(signals).all():
            raise ValueError("BrainMU-H input contains NaN or Inf")
        model = self.tokenizer.model
        encoder = getattr(model, "eeg_encoder", None)
        encode_tokens = getattr(model, "_encode_2d_tokens", None)
        scatter_tokens = getattr(model, "_scatter_2d_quantized_output", None)
        if encoder is None or not callable(encode_tokens) or not callable(scatter_tokens):
            raise AttributeError("BrainMU-H runtime lacks official EEG encoder helper methods")

        patch_tokens, patch_info = encode_tokens(
            encoder,
            signals.to(self.device),
            output_layer=-1,
            channel_masks=None,
            time_masks=None,
        )
        features = scatter_tokens(encoder, patch_tokens, patch_info)
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise RuntimeError(f"expected encoder features [B,L,1024], got {tuple(features.shape)}")
        return features.mean(dim=1).float()
