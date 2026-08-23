import json
from pathlib import Path

import pandas as pd
import torch

from lerobot.datasets.dataset_tools import split_dataset
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


def _add_v21_fake_videos(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_videos"] = info["total_episodes"]
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info["features"]["observation.images.cam"] = {
        "dtype": "video",
        "shape": [3, 2, 2],
        "names": ["channels", "height", "width"],
        "info": {
            "video.height": 2,
            "video.width": 2,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 10,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    info_path.write_text(json.dumps(info), encoding="utf-8")

    video_dir = root / "videos" / "chunk-000" / "observation.images.cam"
    video_dir.mkdir(parents=True)
    for episode_index in range(2):
        (video_dir / f"episode_{episode_index:06d}.mp4").write_bytes(f"episode-{episode_index}".encode())


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


def test_split_v21_dataset_to_v30(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _make_v21_dataset(source_root)
    source = LeRobotDataset("test/v21", root=source_root)

    result = split_dataset(source, {"selected": [1]}, output_dir=tmp_path / "splits")["selected"]

    assert not result.meta.is_legacy_v21
    assert result.meta.total_episodes == 1
    assert result.meta.total_frames == 2
    assert result.meta.get_data_file_path(0) == Path("data/chunk-000/file-000.parquet")
    assert {int(index) for index in result.hf_dataset["episode_index"]} == {0}
    torch.testing.assert_close(result[0]["action"], torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(
        torch.from_numpy(result.meta.stats["action"]["mean"]),
        torch.tensor([2.5, 2.5], dtype=torch.float64),
    )


def test_split_v21_dataset_copies_per_episode_videos_without_reencoding(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _make_v21_dataset(source_root)
    _add_v21_fake_videos(source_root)
    source = LeRobotDataset("test/v21-video", root=source_root)

    result = split_dataset(source, {"selected": [1]}, output_dir=tmp_path / "splits")["selected"]

    relative_video_path = result.meta.get_video_file_path(0, "observation.images.cam")
    assert relative_video_path == Path("videos/observation.images.cam/chunk-000/file-000.mp4")
    assert (result.root / relative_video_path).read_bytes() == b"episode-1"
    assert result.meta.episodes[0]["videos/observation.images.cam/from_timestamp"] == 0.0
    assert result.meta.episodes[0]["videos/observation.images.cam/to_timestamp"] == 0.2
