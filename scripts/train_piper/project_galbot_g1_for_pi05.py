#!/usr/bin/env python3
"""Project Galbot G1's 38D vectors to a pi0.5-compatible action space.

The source dataset is left untouched. Parquet vectors and their per-episode
statistics are projected to the first ``target_dim`` named dimensions, while
only the selected camera streams are declared in the projected metadata. Video
files for the selected cameras are copied into the projection so workers do not
depend on intermittent network-mounted symlink resolution.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED_SOURCE_DIM = 38
REQUIRED_VECTOR_KEYS = ("action", "observation.state")
DEFAULT_CAMERA_KEYS = (
    "observation.images.image_head_left",
    "observation.images.image_arm_left",
    "observation.images.image_arm_right",
)


def _project_stats(value: Any, target_dim: int) -> Any:
    if isinstance(value, list):
        return value[:target_dim]
    return value


def _rewrite_episode_stats(
    path: Path, data_root: Path, target_dim: int, camera_keys: tuple[str, ...]
) -> None:
    lines = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            stats = record.get("stats", {})
            stats = {
                key: value
                for key, value in stats.items()
                if not key.startswith("observation.images.") or key in camera_keys
            }
            record["stats"] = stats
            episode_index = int(record["episode_index"])
            parquet_files = sorted(data_root.glob(f"*/episode_{episode_index:06d}.parquet"))
            if len(parquet_files) != 1:
                raise ValueError(f"Expected one parquet file for episode {episode_index}")
            table = pq.read_table(parquet_files[0], columns=list(REQUIRED_VECTOR_KEYS))
            for key in REQUIRED_VECTOR_KEYS:
                if key not in stats:
                    raise ValueError(f"Missing {key} statistics for episode {episode_index}")
                for stat_name, stat_value in stats[key].items():
                    if stat_name != "count":
                        stats[key][stat_name] = _project_stats(stat_value, target_dim)
                values = np.asarray(table[key].to_pylist(), dtype=np.float64)
                if values.shape != (table.num_rows, target_dim) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {key} vectors in episode {episode_index}")
                quantiles = (0.01, 0.10, 0.50, 0.90, 0.99)
                for q, values_at_q in zip(quantiles, np.quantile(values, quantiles, axis=0), strict=True):
                    stats[key][f"q{int(q * 100):02d}"] = values_at_q.tolist()
            lines.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rewrite_info(path: Path, target_dim: int, camera_keys: tuple[str, ...]) -> None:
    info = json.loads(path.read_text(encoding="utf-8"))
    features = info["features"]
    camera_features = {key: features[key] for key in camera_keys}
    # Keep all scalar features in their source order, then append the selected
    # cameras in the requested order.  The latter is also the order consumed by
    # the policy's visual-token stack.
    info["features"] = {
        key: value for key, value in features.items() if not key.startswith("observation.images.")
    }
    info["features"].update(camera_features)
    for key in REQUIRED_VECTOR_KEYS:
        feature = info["features"][key]
        feature["shape"] = [target_dim]
        feature["names"] = feature["names"][:target_dim]
    info["total_videos"] = info["total_episodes"] * sum(
        feature.get("dtype") == "video" for feature in info["features"].values()
    )
    path.write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def _project_parquet(source: Path, destination: Path, target_dim: int) -> None:
    table = pq.read_table(source)
    for key in REQUIRED_VECTOR_KEYS:
        index = table.column_names.index(key)
        values = [[float(item) for item in row[:target_dim]] for row in table[key].to_pylist()]
        projected = pa.array(values, type=pa.list_(pa.float32()))
        table = table.set_column(index, key, projected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="snappy")


def _is_ready(destination: Path, target_dim: int, camera_keys: tuple[str, ...]) -> bool:
    info_path = destination / "meta" / "info.json"
    stats_path = destination / "meta" / "episodes_stats.jsonl"
    if not info_path.exists():
        return False
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if not all(info["features"][key]["shape"] == [target_dim] for key in REQUIRED_VECTOR_KEYS):
            return False
        actual_camera_keys = tuple(
            key for key, feature in info["features"].items() if feature.get("dtype") in ("video", "image")
        )
        # Feature order is metadata-only; the policy accesses images by key. Accept
        # the same selected camera set regardless of CLI ordering.
        if set(actual_camera_keys) != set(camera_keys) or len(actual_camera_keys) != len(camera_keys):
            return False
        first = next(
            (json.loads(line) for line in stats_path.read_text(encoding="utf-8").splitlines() if line.strip()),
            None,
        )
        videos_root = destination / "videos"
        if videos_root.is_symlink():
            return False
        videos_ready = all(
            all((videos_root / "chunk-000" / key / f"episode_{episode:06d}.mp4").is_file() for episode in range(100))
            for key in camera_keys
        )
        return videos_ready and first is not None and all(
            all(name in first["stats"][key] for name in ("q01", "q99")) for key in REQUIRED_VECTOR_KEYS
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def project_dataset(
    source: Path,
    destination: Path,
    target_dim: int,
    camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS,
) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    source_info_path = source / "meta" / "info.json"
    if not source_info_path.exists():
        raise FileNotFoundError(f"Missing source metadata: {source_info_path}")

    source_info = json.loads(source_info_path.read_text(encoding="utf-8"))
    source_dims = [source_info["features"][key]["shape"][0] for key in REQUIRED_VECTOR_KEYS]
    if source_dims[0] != source_dims[1] or source_dims[0] != EXPECTED_SOURCE_DIM:
        raise ValueError(f"Expected both source vectors to be {EXPECTED_SOURCE_DIM}D, got {source_dims}")
    if not 1 <= target_dim <= source_dims[0]:
        raise ValueError(f"target_dim must be between 1 and {source_dims[0]}, got {target_dim}")
    source_camera_keys = tuple(
        key for key, feature in source_info["features"].items() if feature.get("dtype") in ("video", "image")
    )
    missing_camera_keys = [key for key in camera_keys if key not in source_camera_keys]
    if missing_camera_keys:
        raise ValueError(f"Requested camera keys are missing from the source dataset: {missing_camera_keys}")
    if len(set(camera_keys)) != len(camera_keys):
        raise ValueError(f"camera_keys contains duplicates: {camera_keys}")

    if _is_ready(destination, target_dim, camera_keys):
        print(f"Projected dataset already exists: {destination}")
        return
    if destination.exists():
        raise FileExistsError(
            f"Destination exists but is not a valid {target_dim}D projection: {destination}. "
            "Remove it manually before retrying."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        shutil.copytree(source / "meta", temp_root / "meta")
        (temp_root / "data").mkdir(parents=True, exist_ok=True)
        _rewrite_info(temp_root / "meta" / "info.json", target_dim, camera_keys)
        for source_parquet in sorted((source / "data").glob("*/*.parquet")):
            relative = source_parquet.relative_to(source / "data")
            destination_parquet = temp_root / "data" / relative
            _project_parquet(source_parquet, destination_parquet, target_dim)

        _rewrite_episode_stats(
            temp_root / "meta" / "episodes_stats.jsonl", temp_root / "data", target_dim, camera_keys
        )

        source_videos = source / "videos"
        if source_videos.exists():
            # Keep the selected videos local to the projected dataset. A symlink to the
            # network-mounted source can intermittently disappear inside DataLoader workers.
            destination_videos = temp_root / "videos"
            for camera_key in camera_keys:
                shutil.copytree(source_videos / "chunk-000" / camera_key, destination_videos / "chunk-000" / camera_key)
        os.replace(temp_root, destination)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    print(f"Created {target_dim}D projection: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--target-dim", type=int, default=23)
    parser.add_argument(
        "--camera-key",
        action="append",
        dest="camera_keys",
        default=None,
        help="Camera feature to retain; repeat for multiple cameras (default: head-left + both wrists).",
    )
    args = parser.parse_args()
    camera_keys = tuple(args.camera_keys) if args.camera_keys else DEFAULT_CAMERA_KEYS
    project_dataset(args.source, args.destination, args.target_dim, camera_keys)


if __name__ == "__main__":
    main()
