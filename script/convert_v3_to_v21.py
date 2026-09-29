#!/usr/bin/env python3
"""Convert a local LeRobot v3 dataset to the legacy v2.1 layout in place.

The destination is the same dataset directory.  A sibling ``*.v3-backup``
directory is kept until the caller removes it.  Videos are split back into
one MP4 per episode because that is required by the v2.1 layout.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def probe_frames(path: Path) -> int:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
         "-of", "default=nw=1:nk=1", str(path)], text=True,
    ).strip()
    return int(out)


def episode_rows(root: Path) -> list[dict]:
    meta = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    return sorted(meta, key=lambda row: int(row["episode_index"]))


def encode_segment(src: Path, dst: Path, start: float, duration: float, width: int, height: int, fps: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, start):.6f}", "-i", str(src),
        "-t", f"{max(0.001, duration):.6f}", "-vf", f"scale={width}:{height}:flags=lanczos",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), "-g", "2", "-crf", "23", str(dst),
    ]
    subprocess.run(command, check=True)


def stats_for(table, columns: list[str]) -> dict:
    stats = {}
    for key in columns:
        values = np.asarray(table[key].to_pylist(), dtype=np.float64)
        if values.size == 0:
            continue
        values = values.reshape((len(values), -1))
        stats[key] = {
            "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "count": [len(values)],
        }
    return stats


def convert(root: Path) -> Path:
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError(f"需要 v3.0 数据集，当前是 {info.get('codebase_version')!r}")
    fps = int(info["fps"])
    rows = episode_rows(root)
    data = pq.read_table(root / "data/chunk-000/file-000.parquet")
    camera_info = {}
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        files = sorted((root / "videos" / key).glob("chunk-*/file-*.mp4"))
        camera_info[camera] = {int(p.stem.split("-")[1]): p for p in files}

    parent = root.parent
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.v21-", dir=parent))
    try:
        (staging / "meta").mkdir(parents=True)
        numeric = [k for k in data.column_names if not k.startswith("observation.images.") and k not in {"task_index"}]
        episodes_json = []
        stats_json = []
        tasks = {}
        for row in rows:
            ep = int(row["episode_index"])
            start, end = int(row["dataset_from_index"]), int(row["dataset_to_index"])
            length = end - start
            chunk = ep // int(info.get("chunks_size", 1000))
            ep_table = data.slice(start, length)
            ep_dir = staging / "data" / f"chunk-{chunk:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            keep = [k for k in data.column_names if not k.startswith("observation.images.")]
            pq.write_table(ep_table.select(keep), ep_dir / f"episode_{ep:06d}.parquet", compression="snappy")
            task_list = row.get("tasks") or ["dual-arm manipulation"]
            task_list = list(task_list)
            for task_index, task in enumerate(task_list):
                tasks.setdefault(task, len(tasks))
            episodes_json.append({"episode_index": ep, "tasks": task_list, "length": length})
            stats_json.append({"episode_index": ep, "stats": stats_for(ep_table, numeric)})
            for camera in CAMERAS:
                key = f"observation.images.{camera}"
                file_idx = int(row[f"videos/{key}/file_index"])
                src = camera_info[camera][file_idx]
                start_ts = float(row[f"videos/{key}/from_timestamp"])
                duration = max(1, length - 1) / fps
                shape = info["features"][key]["shape"]
                _, height, width = shape
                # The left stream was recorded at 640x480.  Normalize all
                # v2.1 videos to the requested common 960x540 profile.
                width, height = 960, 540
                dst = staging / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{ep:06d}.mp4"
                encode_segment(src, dst, start_ts, duration, width, height, fps)

        (staging / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episodes_json), encoding="utf-8"
        )
        (staging / "meta/episodes_stats.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in stats_json), encoding="utf-8"
        )
        (staging / "meta/tasks.jsonl").write_text(
            "".join(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n" for task, idx in tasks.items()), encoding="utf-8"
        )
        features = dict(info["features"])
        for key, feature in features.items():
            if feature.get("dtype") == "video":
                feature = dict(feature)
                feature["shape"] = [3, 540, 960]
                feature["info"] = dict(feature.get("info") or {})
                feature["info"].update({"video.height": 540, "video.width": 960, "video.fps": fps, "video.codec": "h264", "video.pix_fmt": "yuv420p"})
                features[key] = feature
        new_info = {
            "codebase_version": "v2.1", "robot_type": info.get("robot_type", "bi_piper"),
            "total_episodes": len(rows), "total_frames": int(data.num_rows), "total_tasks": len(tasks),
            "total_chunks": max(1, (len(rows) + 999) // 1000), "chunks_size": 1000, "fps": fps,
            "splits": {"train": f"0:{len(rows)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }
        (staging / "meta/info.json").write_text(json.dumps(new_info, indent=2, ensure_ascii=False), encoding="utf-8")
        backup = parent / f"{root.name}.v3-backup"
        if backup.exists():
            raise RuntimeError(f"备份目录已存在，请先处理：{backup}")
        root.rename(backup)
        staging.rename(root)
        return backup
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    args = parser.parse_args()
    root = Path(args.dataset_path).expanduser().resolve()
    print(f"开始转换：{root}", flush=True)
    backup = convert(root)
    print(f"转换完成：{root}（原 v3 备份：{backup}）", flush=True)


if __name__ == "__main__":
    main()
