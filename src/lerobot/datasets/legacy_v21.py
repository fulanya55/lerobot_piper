#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Read-only compatibility helpers for LeRobot dataset format v2.1."""

from __future__ import annotations

import json
from pathlib import Path

import datasets
import pandas as pd

from .compute_stats import aggregate_stats
from .io_utils import cast_stats_to_numpy

V21 = "v2.1"
LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"


def _load_jsonlines(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_legacy_tasks(root: Path) -> pd.DataFrame:
    """Load v2.1 ``tasks.jsonl`` using the v3 in-memory DataFrame contract."""
    rows = sorted(_load_jsonlines(root / LEGACY_TASKS_PATH), key=lambda row: row["task_index"])
    return pd.DataFrame(
        {"task_index": [row["task_index"] for row in rows]},
        index=pd.Index([row["task"] for row in rows], name="task"),
    )


def load_legacy_episodes(root: Path, fps: int, video_keys: list[str]) -> datasets.Dataset:
    """Build the v3 episode-index view from v2.1 ``episodes.jsonl`` records."""
    rows = sorted(_load_jsonlines(root / LEGACY_EPISODES_PATH), key=lambda row: row["episode_index"])
    offset = 0
    adapted_rows = []
    for row in rows:
        length = int(row["length"])
        adapted = dict(row)
        adapted["dataset_from_index"] = offset
        adapted["dataset_to_index"] = offset + length
        for video_key in video_keys:
            adapted[f"videos/{video_key}/from_timestamp"] = 0.0
            adapted[f"videos/{video_key}/to_timestamp"] = max(0, length - 1) / fps
        adapted_rows.append(adapted)
        offset += length
    return datasets.Dataset.from_list(adapted_rows)


def load_legacy_stats(root: Path, episode_indices: list[int] | None = None) -> dict | None:
    """Aggregate selected v2.1 per-episode stats into the v3 global-stats contract."""
    path = root / LEGACY_EPISODES_STATS_PATH
    if not path.exists():
        return None
    rows = sorted(_load_jsonlines(path), key=lambda row: row["episode_index"])
    if episode_indices is not None:
        selected = set(episode_indices)
        rows = [row for row in rows if int(row["episode_index"]) in selected]
    if not rows:
        return None
    return aggregate_stats([cast_stats_to_numpy(row["stats"]) for row in rows])
