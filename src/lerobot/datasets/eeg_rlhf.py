# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""On-the-fly EEG alignment for LeRobot action windows.

The dataset deliberately does not materialize a feature/token cache.  A sample
is a normal LeRobot frame whose action target is the *next* ``chunk_size``
frames.  Its EEG is read from the source ``.npz`` in ``__getitem__`` and the
two 2.5 second halves of the window around the action-window start are passed
through the supplied encoder.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from .lerobot_dataset import LeRobotDataset


EEG_TOKEN = "observation.eeg_rlhf.token"
EEG_WINDOW = "observation.eeg_rlhf.window"
EEG_VALID = "observation.eeg_rlhf.window_valid"
EEG_ANCHOR = "observation.eeg_rlhf.anchor_timestamp"
CHUNK_START = "observation.eeg_rlhf.chunk_start_frame"
CHUNK_END = "observation.eeg_rlhf.chunk_end_frame"


@dataclass(frozen=True)
class EEGWindowConfig:
    """Sampling contract for an EEG-assisted action window."""

    chunk_size: int = 50
    half_window_s: float = 2.5
    target_sfreq: float = 256.0
    target_samples: int = 640
    real_channels: int = 19
    padded_channels: int = 24

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.half_window_s <= 0 or self.target_sfreq <= 0 or self.target_samples <= 0:
            raise ValueError("EEG window duration, sampling rate, and sample count must be positive")
        if self.padded_channels < self.real_channels:
            raise ValueError("padded_channels must be >= real_channels")


def _video_on_from_rating(path: str | Path, trial: int) -> float:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    matches = [item for item in payload.get("trials", []) if int(item.get("trial", -1)) == int(trial)]
    if len(matches) != 1:
        raise ValueError(f"expected one timing record for trial {trial} in {path}")
    timing = matches[0].get("timing", {})
    for key in ("video_on", "video_on_pc", "video_start_pc"):
        if key in timing:
            return float(timing[key])
    raise KeyError(f"rating record {path} has no video_on timestamp")


def _manifest_records(
    manifest: str | Path | pd.DataFrame | Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    if isinstance(manifest, (str, Path)):
        table = pd.read_csv(manifest)
        records = table.to_dict("records")
    elif isinstance(manifest, pd.DataFrame):
        records = manifest.to_dict("records")
    else:
        records = [dict(item) for item in manifest]
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        dataset = str(
            record.get("lerobot_dataset", record.get("dataset", record.get("repo_id", "")))
        )
        if not dataset or "eeg_npz" not in record:
            raise ValueError("EEG manifest needs lerobot_dataset/dataset, lerobot_episode_index, and eeg_npz")
        if "lerobot_episode_index" not in record and "episode_index" not in record:
            raise ValueError("EEG manifest needs lerobot_episode_index/episode_index")
        episode = int(record.get("lerobot_episode_index", record.get("episode_index")))
        if "video_on" not in record:
            if "video_on_pc" in record:
                record["video_on"] = float(record["video_on_pc"])
            elif "rating_json" in record and "trial" in record:
                record["video_on"] = _video_on_from_rating(record["rating_json"], int(record["trial"]))
            else:
                raise ValueError("EEG manifest needs video_on (or rating_json + trial)")
        key = (dataset, episode)
        if key in result:
            raise ValueError(f"duplicate EEG manifest record for {key}")
        result[key] = {**record, "lerobot_dataset": dataset, "lerobot_episode_index": episode}
    return result


def _select_channels(channel_names: Sequence[str], limit: int) -> list[int]:
    excluded = ("EOG", "HEOG", "VEOG", "EMG", "ECG", "EKG", "GSR", "EDA", "RESP", "PULSE", "PPG", "TRIG")
    candidates = [
        (str(name).lower(), index)
        for index, name in enumerate(channel_names)
        if not any(marker in str(name).upper() for marker in excluded)
    ]
    candidates.sort()
    indices = [index for _, index in candidates[:limit]]
    if not indices:
        raise ValueError("EEG source contains no usable channels")
    return indices


def _prepare_segment(
    signals: np.ndarray,
    channel_names: Sequence[str],
    input_sfreq: float,
    config: EEGWindowConfig,
) -> Tensor:
    if signals.ndim != 2:
        raise ValueError(f"EEG segment must be [channels,time], got {signals.shape}")
    channels = _select_channels(channel_names, config.real_channels)
    x = torch.from_numpy(np.asarray(signals[channels], dtype=np.float32)).unsqueeze(0)
    if x.shape[-1] == 0:
        return torch.zeros(1, config.padded_channels, config.target_samples, dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError("EEG source contains NaN or Inf")
    x = (x - x.mean(dim=-1, keepdim=True)) / x.var(dim=-1, unbiased=False, keepdim=True).sqrt().clamp_min(1e-6)
    if float(input_sfreq) != config.target_sfreq:
        length = max(1, round(x.shape[-1] * config.target_sfreq / float(input_sfreq)))
        x = F.interpolate(x, size=length, mode="linear", align_corners=False)
    if x.shape[-1] >= config.target_samples:
        x = x[..., : config.target_samples]
    else:
        x = F.pad(x, (0, config.target_samples - x.shape[-1]))
    if x.shape[1] < config.padded_channels:
        x = F.pad(x, (0, 0, 0, config.padded_channels - x.shape[1]))
    return x.contiguous()


class EEGChunkWindowDataset(Dataset):
    """LeRobot dataset with a frame-stride EEG action window.

    Unlike a chunked manifest, this exposes one item for every frame.  Frame
    ``t`` targets actions ``[t, t + chunk_size)`` (the final window is padded by
    LeRobot's normal delta-timestamp logic), and EEG is anchored at timestamp
    ``t``.  ``encoder`` is called for both halves of the ``+/- half_window_s``
    interval.  Nothing produced by the encoder is written to disk.
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        manifest: str | Path | pd.DataFrame | Sequence[Mapping[str, Any]],
        *,
        config: EEGWindowConfig | None = None,
        encoder: Callable[[Tensor], Tensor] | None = None,
        include_raw_window: bool = False,
    ) -> None:
        self.dataset = dataset
        self.config = config or EEGWindowConfig()
        self.encoder = encoder
        self.include_raw_window = include_raw_window
        self.records = _manifest_records(manifest)
        if dataset.delta_timestamps is None or "action" not in dataset.delta_timestamps:
            raise ValueError(
                "dataset must be created with action delta timestamps for the next chunk_size frames; "
                "use resolve_delta_timestamps with a policy config"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def _load_eeg(self, record: Mapping[str, Any], anchor_s: float) -> tuple[Tensor, Tensor, Tensor]:
        source_path = Path(str(record["eeg_npz"]))
        preprocessed_path = record.get("eeg_preprocessed_npz")
        if preprocessed_path is None and "session" in record:
            candidate = source_path.parents[3] / "preprocess_all" / "sessions" / str(record["session"]) / "eeg_preprocessed.npz"
            if candidate.is_file():
                preprocessed_path = candidate
        if preprocessed_path is not None:
            source_path = Path(str(preprocessed_path))
        with np.load(source_path, allow_pickle=True) as source:
            data_key = "eeg_data" if "eeg_data" in source.files else "eeg_data_ch_time"
            data = np.asarray(source[data_key], dtype=np.float32)
            names = [str(value) for value in np.asarray(source["eeg_channel_names"]).reshape(-1)]
            sample_rate = float(np.asarray(source["eeg_sample_rate"]).reshape(-1)[0])
            timestamps = np.asarray(source["eeg_timestamps_pc"], dtype=np.float64)
        if data.ndim != 2 or timestamps.ndim != 1:
            raise ValueError("EEG npz must contain a 2-D signal array and matching timestamps")
        if data.shape[0] != timestamps.shape[0] and data.shape[1] == timestamps.shape[0]:
            data = data.T
        if data.shape[0] != timestamps.shape[0]:
            raise ValueError("EEG signal sample axis does not match timestamps")
        center = float(record["video_on"]) + anchor_s
        half = self.config.half_window_s
        left = int(np.searchsorted(timestamps, center - half, side="left"))
        middle = int(np.searchsorted(timestamps, center, side="left"))
        right = int(np.searchsorted(timestamps, center + half, side="right"))
        valid = (
            left < middle < right
            and center - half >= timestamps[0]
            and center + half <= timestamps[-1]
            and (middle - left) >= int(2.0 * sample_rate)
            and (right - middle) >= int(2.0 * sample_rate)
        )
        before = _prepare_segment(data[left:middle].T, names, sample_rate, self.config)
        after = _prepare_segment(data[middle:right].T, names, sample_rate, self.config)
        raw_windows = torch.stack((before, after), dim=1).squeeze(0)
        if self.encoder is None:
            tokens = raw_windows
        else:
            encoded = [self.encoder(part) for part in (before, after)]
            normalized: list[Tensor] = []
            for token in encoded:
                if not isinstance(token, Tensor):
                    raise TypeError("EEG encoder must return a torch.Tensor")
                if token.ndim == 3 and token.shape[0] == 1:
                    token = token.squeeze(0)
                if token.ndim == 1:
                    token = token.unsqueeze(0)
                if token.ndim != 2:
                    raise ValueError("EEG encoder output must be [tokens,dim] or [dim]")
                normalized.append(token.detach().float().mean(dim=0))
            if normalized[0].shape != normalized[1].shape:
                raise ValueError("EEG encoder output shape differs between before/after windows")
            # The +/-2.5 s interval is represented by one EEG token.  The
            # BrainMU encoder is fixed to 2.5 s, so encode both halves and
            # pool them only after encoding (never before the encoder).
            tokens = torch.stack(normalized, dim=0).mean(dim=0)
        return tokens, torch.tensor(bool(valid), dtype=torch.bool), raw_windows

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        raw = self.dataset.get_raw_item(index)
        episode = int(raw["episode_index"])
        dataset_name = str(self.dataset.repo_id)
        record = self.records.get((dataset_name, episode))
        if record is None:
            # Manifests created by a local dataset often use its basename rather
            # than the Hub id; accept that spelling without weakening episode checks.
            record = next(
                (value for (name, ep), value in self.records.items() if ep == episode and name in {dataset_name, Path(dataset_name).name}),
                None,
            )
        if record is None:
            raise KeyError(f"no EEG manifest record for dataset={dataset_name!r}, episode={episode}")
        anchor = float(raw["timestamp"])
        tokens, valid, raw_windows = self._load_eeg(record, anchor)
        start = int(raw.get("frame_index", raw.get("index", index)))
        item[EEG_TOKEN] = tokens
        item[EEG_VALID] = valid
        item[EEG_ANCHOR] = torch.tensor(anchor, dtype=torch.float32)
        item[CHUNK_START] = torch.tensor(start, dtype=torch.int64)
        item[CHUNK_END] = torch.tensor(start + self.config.chunk_size, dtype=torch.int64)
        if self.include_raw_window:
            item[EEG_WINDOW] = raw_windows
        return item


class EEGPreferencePairDataset(Dataset):
    """Zip two frame-stride EEG datasets into winner/loser preference pairs.

    Both sides are read lazily.  This class contains no tensors beyond the
    underlying LeRobot dataset objects and therefore cannot create a feature
    cache accidentally.
    """

    def __init__(self, winner: EEGChunkWindowDataset, loser: EEGChunkWindowDataset) -> None:
        if len(winner) != len(loser):
            raise ValueError(f"winner/loser datasets must have equal length, got {len(winner)} and {len(loser)}")
        self.winner = winner
        self.loser = loser

    def __len__(self) -> int:
        return len(self.winner)

    def __getitem__(self, index: int) -> dict[str, dict[str, Any]]:
        return {"winner": self.winner[index], "loser": self.loser[index]}


__all__ = [
    "CHUNK_END",
    "CHUNK_START",
    "EEG_ANCHOR",
    "EEGChunkWindowDataset",
    "EEGPreferencePairDataset",
    "EEG_TOKEN",
    "EEG_VALID",
    "EEG_WINDOW",
    "EEGWindowConfig",
]
