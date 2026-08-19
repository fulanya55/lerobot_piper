#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from bisect import bisect_right
from collections.abc import Callable
from copy import copy, deepcopy
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
import torch.utils
from torchvision.transforms.v2 import functional as tv_functional

from lerobot.policies.common.vla_utils import resize_with_pad_torch
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE

from .compute_stats import aggregate_stats
from .legacy_v21 import load_legacy_stats
from .lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


def _rename_keys(values: dict, rename_map: dict[str, str], *, context: str) -> dict:
    """Return a shallow key-renamed mapping and reject ambiguous aliases."""
    renamed = {}
    for key, value in values.items():
        target = rename_map.get(key, key)
        if target in renamed:
            raise ValueError(f"Feature rename collision for {target!r} in {context}.")
        renamed[target] = value
    return renamed


def _ordered_common_names(dataset_features: list[dict[str, dict]], key: str) -> list[str]:
    """Return the ordered intersection of a named vector feature across datasets."""
    names_by_dataset: list[list[str]] = []
    for features in dataset_features:
        if key not in features:
            raise ValueError(f"Feature {key!r} is missing from one or more datasets.")
        names = features[key].get("names")
        if not isinstance(names, list) or not names:
            raise ValueError(f"Feature {key!r} must define non-empty dimension names in every dataset.")
        if len(names) != len(set(names)):
            raise ValueError(f"Feature {key!r} contains duplicate dimension names: {names}")
        names_by_dataset.append(names)

    common = set(names_by_dataset[0])
    for names in names_by_dataset[1:]:
        common.intersection_update(names)
    ordered = [name for name in names_by_dataset[0] if name in common]
    if not ordered:
        raise ValueError(f"Feature {key!r} has no named dimensions common to every dataset.")
    return ordered


def _project_stats(stats: dict, indices_by_key: dict[str, list[int]]) -> dict:
    """Project vector statistics along their feature dimension."""
    projected = deepcopy(stats)
    for key, indices in indices_by_key.items():
        if key not in projected:
            continue
        for stat_name, value in projected[key].items():
            if stat_name == "count":
                continue
            array = np.asarray(value)
            projected[key][stat_name] = np.take(array, indices, axis=0)
    return projected


def _resize_feature_shape(feature: dict, image_size: tuple[int, int]) -> dict:
    feature = deepcopy(feature)
    shape = list(feature["shape"])
    names = feature.get("names") or []
    height, width = image_size
    if "height" in names and "width" in names:
        shape[names.index("height")] = height
        shape[names.index("width")] = width
    elif len(shape) == 3:
        # LeRobot image tensors are CHW at runtime; metadata without dimension names follows CHW.
        shape[-2:] = [height, width]
    feature["shape"] = tuple(shape)
    return feature


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A read-only virtual concatenation of multiple LeRobot datasets.

    Vector features such as ``observation.state`` and ``action`` can be projected to the
    ordered intersection of their dimension names. Projection happens after reading a sample
    and before DataLoader collation, so heterogeneous source tensors never need to be rewritten.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict[str, list[int]] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        delta_timestamps_by_repo: dict[str, dict[str, list[float]] | None] | None = None,
        tolerances_s: dict[str, float] | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        common_feature_keys: list[str] | None = None,
        feature_rename_map: dict[str, str] | None = None,
        image_size: tuple[int, int] | None = None,
        revision: str | None = None,
        return_uint8: bool = False,
        depth_output_unit: str = "mm",
        *,
        token: str | bool | None = None,
    ):
        super().__init__()
        if len(repo_ids) < 2:
            raise ValueError("MultiLeRobotDataset requires at least two repo ids.")
        if len(repo_ids) != len(set(repo_ids)):
            raise ValueError("MultiLeRobotDataset repo ids must be unique.")

        self.repo_ids = repo_ids
        self.repo_id = "+".join(repo_ids)
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s or dict.fromkeys(repo_ids, 0.0001)
        self.delta_timestamps = delta_timestamps
        self.image_size = image_size
        self.feature_rename_map = feature_rename_map or {}
        self._common_feature_keys = (
            [OBS_STATE, ACTION] if common_feature_keys is None else common_feature_keys
        )

        child_roots = {repo_id: self.root / repo_id if root is not None else None for repo_id in repo_ids}
        self._datasets = []
        for repo_id in repo_ids:
            repo_delta_timestamps = (
                delta_timestamps_by_repo.get(repo_id)
                if delta_timestamps_by_repo is not None
                else delta_timestamps
            )
            self._datasets.append(
                LeRobotDataset(
                    repo_id,
                    root=child_roots[repo_id],
                    episodes=episodes.get(repo_id) if episodes else None,
                    image_transforms=image_transforms,
                    delta_timestamps=repo_delta_timestamps,
                    tolerance_s=self.tolerances_s[repo_id],
                    download_videos=download_videos,
                    video_backend=video_backend,
                    revision=revision,
                    return_uint8=return_uint8,
                    depth_output_unit=depth_output_unit,
                    token=token,
                )
            )

        self._features_by_dataset = [
            _rename_keys(
                dataset.meta.features,
                self.feature_rename_map,
                context=f"features of {repo_id}",
            )
            for repo_id, dataset in zip(self.repo_ids, self._datasets, strict=True)
        ]
        self._lengths = [len(dataset) for dataset in self._datasets]
        self._cumulative_lengths = np.cumsum(self._lengths).tolist()
        self._feature_indices = self._build_feature_projection()
        self.disabled_features, features = self._build_common_features()
        self.meta = self._build_metadata(features)
        self._common_sample_keys = set(features) | {"task"}
        self._common_sample_keys.update(f"{key}_is_pad" for key in features)
        self.stats = self.meta.stats
        self.episodes = None
        self.image_transforms = None
        self.set_image_transforms(image_transforms)

    def _build_feature_projection(self) -> list[dict[str, list[int]]]:
        all_features = self._features_by_dataset
        indices_by_dataset: list[dict[str, list[int]]] = [{} for _ in self._datasets]
        for key in self._common_feature_keys:
            if not all(key in features for features in all_features):
                logger.warning("Skipping common-dimension projection for missing feature %s", key)
                continue
            selected_names = _ordered_common_names(all_features, key)
            for dataset_index, features in enumerate(all_features):
                source_names = features[key]["names"]
                indices_by_dataset[dataset_index][key] = [source_names.index(name) for name in selected_names]
            logger.info("Multi-dataset feature %s uses %d common named dimensions", key, len(selected_names))
        return indices_by_dataset

    def _projected_feature(self, dataset_index: int, key: str) -> dict:
        feature = deepcopy(self._features_by_dataset[dataset_index][key])
        indices = self._feature_indices[dataset_index].get(key)
        if indices is not None:
            names = feature["names"]
            feature["names"] = [names[index] for index in indices]
            feature["shape"] = (len(indices),)
        if self.image_size is not None and feature.get("dtype") in ("image", "video"):
            feature = _resize_feature_shape(feature, self.image_size)
        return feature

    def _build_common_features(self) -> tuple[set[str], dict[str, dict]]:
        feature_sets = [set(features) for features in self._features_by_dataset]
        common_keys = set.intersection(*feature_sets)
        disabled = set.union(*feature_sets).difference(common_keys)
        features: dict[str, dict] = {}

        for key in sorted(common_keys):
            projected = [self._projected_feature(index, key) for index in range(len(self._datasets))]
            reference = projected[0]
            compatible = all(
                feature.get("dtype") == reference.get("dtype")
                and tuple(feature.get("shape", ())) == tuple(reference.get("shape", ()))
                and feature.get("names") == reference.get("names")
                for feature in projected[1:]
            )
            if compatible:
                features[key] = reference
                continue
            if key in self._common_feature_keys:
                raise ValueError(f"Projected feature {key!r} is still incompatible across datasets.")
            disabled.add(key)
            logger.warning("Disabling incompatible multi-dataset feature %s", key)

        if not features:
            raise RuntimeError("Multiple datasets have no compatible features.")
        for repo_id, source_features in zip(self.repo_ids, self._features_by_dataset, strict=True):
            dropped = sorted(set(source_features).intersection(disabled))
            if dropped:
                logger.warning("Features disabled for %s: %s", repo_id, dropped)
        return disabled, features

    def _build_metadata(self, features: dict[str, dict]):
        # Reuse the established LeRobotDatasetMetadata interface as a read-only facade. Its
        # path-dependent methods are never used for the virtual dataset; properties such as
        # features, camera_keys and stats continue to work for policy/processors.
        meta = copy(self._datasets[0].meta)
        meta.info = deepcopy(self._datasets[0].meta.info)
        meta.repo_id = self.repo_id
        meta.root = self.root
        meta.info.features = features

        projected_stats = []
        for dataset_index, dataset in enumerate(self._datasets):
            child_stats = dataset.meta.stats
            if dataset.episodes is not None and dataset.meta.is_legacy_v21:
                selected_stats = load_legacy_stats(dataset.meta.root, dataset.episodes)
                if selected_stats is not None:
                    child_stats = selected_stats
            renamed_stats = _rename_keys(
                child_stats or {},
                self.feature_rename_map,
                context=f"stats of {self.repo_ids[dataset_index]}",
            )
            stats = _project_stats(renamed_stats, self._feature_indices[dataset_index])
            projected_stats.append({key: value for key, value in stats.items() if key in features})
        meta.stats = aggregate_stats(projected_stats)

        episode_rows = []
        frame_offset = 0
        global_episode_index = 0
        unique_tasks: list[str] = []
        seen_tasks: set[str] = set()
        for dataset_index, dataset in enumerate(self._datasets):
            selected = dataset.episodes
            episode_indices = selected if selected is not None else list(range(dataset.meta.total_episodes))
            for source_episode_index in episode_indices:
                source = dataset.meta.episodes[int(source_episode_index)]
                length = int(source["dataset_to_index"] - source["dataset_from_index"])
                tasks = list(source.get("tasks") or [])
                for task in tasks:
                    if task not in seen_tasks:
                        seen_tasks.add(task)
                        unique_tasks.append(task)
                episode_rows.append(
                    {
                        "episode_index": global_episode_index,
                        "dataset_index": dataset_index,
                        "source_episode_index": int(source_episode_index),
                        "dataset_from_index": frame_offset,
                        "dataset_to_index": frame_offset + length,
                        "length": length,
                        "tasks": tasks,
                    }
                )
                global_episode_index += 1
                frame_offset += length

        meta.episodes = datasets.Dataset.from_list(episode_rows)
        meta.tasks = pd.DataFrame(
            {"task_index": range(len(unique_tasks))}, index=pd.Index(unique_tasks, name="task")
        )
        meta.info.total_episodes = len(episode_rows)
        meta.info.total_frames = frame_offset
        meta.info.total_tasks = len(unique_tasks)
        meta.info.splits = {"train": f"0:{len(episode_rows)}"}
        if len({dataset.meta.fps for dataset in self._datasets}) > 1:
            logger.warning(
                "Mixed dataset frame rates %s; each child uses delta timestamps derived from its own FPS.",
                sorted({dataset.meta.fps for dataset in self._datasets}),
            )
        return meta

    @property
    def absolute_to_relative_idx(self) -> None:
        # Composite episode boundaries are already expressed in this dataset's contiguous index space.
        return None

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms
        for dataset in getattr(self, "_datasets", []):
            dataset.set_image_transforms(image_transforms)

    def clear_image_transforms(self) -> None:
        self.set_image_transforms(None)

    @property
    def repo_id_to_index(self) -> dict[str, int]:
        return {repo_id: index for index, repo_id in enumerate(self.repo_ids)}

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def video(self) -> bool:
        return bool(self.meta.video_keys)

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def camera_keys(self) -> list[str]:
        return self.meta.camera_keys

    @property
    def video_frame_keys(self) -> list[str]:
        return self.meta.video_keys

    @property
    def num_frames(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return self.meta.total_episodes

    @property
    def tolerance_s(self) -> float:
        return 1 / self.fps - 1e-4

    def __len__(self) -> int:
        return self._cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        dataset_index = bisect_right(self._cumulative_lengths, idx)
        start = self._cumulative_lengths[dataset_index - 1] if dataset_index > 0 else 0
        item = self._datasets[dataset_index][idx - start]
        item = _rename_keys(
            item,
            self.feature_rename_map,
            context=f"sample from {self.repo_ids[dataset_index]}",
        )

        for key, indices in self._feature_indices[dataset_index].items():
            if key in item:
                item[key] = item[key][..., indices]
        for key in self.disabled_features:
            item.pop(key, None)
        # Some legacy datasets contain undeclared parquet columns (for example
        # ``subtask_indices``) that are not present in every child dataset. Keep only
        # the common declared features and their runtime companions so default_collate
        # always sees a consistent schema. This filtering is local to the multi-dataset
        # facade and does not change single-dataset samples.
        item = {key: value for key, value in item.items() if key in self._common_sample_keys}
        if self.image_size is not None:
            for key in self.meta.camera_keys:
                if key in item and tuple(item[key].shape[-2:]) != self.image_size:
                    if key in self.meta.depth_keys:
                        item[key] = tv_functional.resize(item[key], list(self.image_size), antialias=True)
                    else:
                        item[key] = resize_with_pad_torch(
                            item[key].unsqueeze(0), *self.image_size
                        ).squeeze(0)
        item["dataset_index"] = torch.tensor(dataset_index)
        return item

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(repo_ids={self.repo_ids!r}, num_frames={self.num_frames}, "
            f"num_episodes={self.num_episodes}, features={list(self.features)!r})"
        )
