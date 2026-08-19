import json
from pathlib import Path

import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.multi_dataset import MultiLeRobotDataset


def _write_jsonlines(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_v21_dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test_robot",
        "total_episodes": 2,
        "total_frames": 4,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
            "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonlines(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "test task"}])
    _write_jsonlines(
        root / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["test task"], "length": 2},
            {"episode_index": 1, "tasks": ["test task"], "length": 2},
        ],
    )
    stats = []
    for episode_index, mean in enumerate((0.5, 2.5)):
        feature_stats = {
            "min": [mean - 0.5, mean - 0.5],
            "max": [mean + 0.5, mean + 0.5],
            "mean": [mean, mean],
            "std": [0.5, 0.5],
            "count": [2],
        }
        stats.append(
            {
                "episode_index": episode_index,
                "stats": {"observation.state": feature_stats, "action": feature_stats},
            }
        )
    _write_jsonlines(root / "meta" / "episodes_stats.jsonl", stats)

    for episode_index in range(2):
        start = episode_index * 2
        pd.DataFrame(
            {
                "observation.state": [[float(start), float(start)], [float(start + 1), float(start + 1)]],
                "action": [[float(start), float(start)], [float(start + 1), float(start + 1)]],
                "timestamp": [0.0, 0.1],
                "frame_index": [0, 1],
                "episode_index": [episode_index, episode_index],
                "index": [start, start + 1],
                "task_index": [0, 0],
            }
        ).to_parquet(root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet")


def test_load_v21_dataset_without_conversion(tmp_path: Path) -> None:
    _make_v21_dataset(tmp_path)

    dataset = LeRobotDataset("test/v21", root=tmp_path, episodes=[1])

    assert dataset.meta.is_legacy_v21
    assert dataset.meta.get_data_file_path(1) == Path("data/chunk-000/episode_000001.parquet")
    assert len(dataset) == 2
    assert dataset.num_episodes == 1
    assert dataset[0]["task"] == "test task"
    torch.testing.assert_close(dataset[0]["observation.state"], torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(
        torch.from_numpy(dataset.meta.stats["action"]["mean"]), torch.tensor([1.5, 1.5], dtype=torch.float64)
    )


def test_v21_delta_timestamps_use_legacy_episode_boundaries(tmp_path: Path) -> None:
    _make_v21_dataset(tmp_path)
    dataset = LeRobotDataset(
        "test/v21",
        root=tmp_path,
        delta_timestamps={"action": [-0.1, 0.0, 0.1]},
    )

    item = dataset[2]
    torch.testing.assert_close(item["action"], torch.tensor([[2.0, 2.0], [2.0, 2.0], [3.0, 3.0]]))
    assert item["action_is_pad"].tolist() == [True, False, False]


def test_multilerobot_v21_stats_respect_selected_episodes(tmp_path: Path) -> None:
    repo_ids = ["first", "second"]
    for repo_id in repo_ids:
        _make_v21_dataset(tmp_path / repo_id)

    dataset = MultiLeRobotDataset(
        repo_ids,
        root=tmp_path,
        episodes={repo_id: [1] for repo_id in repo_ids},
        download_videos=False,
    )

    torch.testing.assert_close(
        torch.from_numpy(dataset.meta.stats["action"]["mean"]),
        torch.tensor([2.5, 2.5], dtype=torch.float64),
    )
