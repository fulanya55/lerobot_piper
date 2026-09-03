#!/usr/bin/env python3
"""Run queued HDF5-to-LeRobot conversions sequentially in the background."""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queue", help="FIFO containing: episode_idx<TAB>hdf5_path")
    source.add_argument("--resume-dir", help="Process staged episode_*.hdf5 files in index order")
    parser.add_argument("--converter", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--status-dir", required=True)
    parser.add_argument("--video-codec", choices=("h264", "hevc", "libsvtav1"), default="h264")
    parser.add_argument("--video-crf", type=int, default=23)
    return parser.parse_args()


def write_status(status_dir, episode_idx, suffix, text):
    path = status_dir / f"episode_{episode_idx}.{suffix}"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def queued_episodes(args):
    if args.resume_dir:
        resume_dir = Path(args.resume_dir).expanduser().resolve()
        candidates = []
        for path in resume_dir.glob("episode_*.hdf5"):
            match = re.fullmatch(r"episode_(\d+)\.hdf5", path.name)
            if match:
                candidates.append((int(match.group(1)), path.resolve()))
        yield from sorted(candidates)
        return

    queue_path = Path(args.queue).resolve()
    with queue_path.open("r", encoding="utf-8") as queue:
        for line in queue:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                episode_text, raw_text = line.split("\t", 1)
                yield int(episode_text), Path(raw_text).expanduser().resolve()
            except (ValueError, TypeError):
                raise RuntimeError(f"Invalid conversion queue entry: {line!r}") from None


def main():
    args = parse_args()
    converter = Path(args.converter).resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    log_dir = Path(args.log_dir).resolve()
    status_dir = Path(args.status_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    for episode_idx, raw_path in queued_episodes(args):
        log_path = log_dir / f"conversion_episode_{episode_idx}.log"
        command = [
            sys.executable,
            str(converter),
            "--input",
            str(raw_path),
            "--dataset-path",
            str(dataset_path),
            "--repo-id",
            args.repo_id,
            "--episode-idx",
            str(episode_idx),
            "--task",
            args.task,
            "--video-codec",
            args.video_codec,
            "--video-crf",
            str(args.video_crf),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)

        if result.returncode != 0:
            write_status(
                status_dir,
                episode_idx,
                "failed",
                f"episode={episode_idx}\nraw={raw_path}\nlog={log_path}\nexit_code={result.returncode}\n",
            )
            return result.returncode

        # Delete staging data only after the converter has returned success.
        raw_path.unlink(missing_ok=True)
        write_status(
            status_dir,
            episode_idx,
            "done",
            f"episode={episode_idx}\nlog={log_path}\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
