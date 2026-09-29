#!/usr/bin/env python
"""uv-side direct LeRobot writer for frames from ros_lerobot_stream.py."""
import argparse
import json
import pickle
import socket
import struct
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
JOINT_NAMES = [f"joint_{i}" for i in range(14)]


def recv_packet(stream):
    header = stream.read(8)
    if not header:
        return None
    size = struct.unpack("!Q", header)[0]
    return pickle.loads(stream.read(size))


def features_for(images):
    vector = {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES}
    features = {
        "observation.state": dict(vector),
        "observation.velocity": dict(vector),
        "observation.effort": dict(vector),
        # Keep the existing int64 feature schema, but store Unix seconds rather
        # than nanoseconds so Arrow metadata statistics remain representable.
        "observation.timestamps_ns": {"dtype": "int64", "shape": (7,), "names": [f"topic_{i}" for i in range(7)]},
        "action": dict(vector),
    }
    for key, image in zip(CAMERAS, images):
        h, w, c = image.shape
        features[f"observation.images.{key}"] = {
            "dtype": "video", "shape": (c, h, w), "names": ["channels", "height", "width"]
        }
    return features


def open_dataset(args, first):
    root = Path(args.dataset_path).expanduser().resolve()
    info = root / "meta" / "info.json"
    encoder = RGBEncoderConfig(vcodec=args.video_codec, pix_fmt="yuv420p", g=2, crf=args.video_crf)
    if info.exists():
        with info.open(encoding="utf-8") as f:
            metadata = json.load(f)
        if int(metadata["fps"]) != args.fps:
            raise RuntimeError(f"数据集 fps={metadata['fps']}，本次为 {args.fps}")
        dataset = LeRobotDataset.resume(
            repo_id=args.repo_id, root=root, streaming_encoding=True,
            encoder_queue_maxsize=90, encoder_threads=args.encoder_threads,
            rgb_encoder=encoder,
        )
        if dataset.num_episodes != args.episode_idx:
            raise RuntimeError(f"episode_idx={args.episode_idx}，下一个可写索引是 {dataset.num_episodes}")
        return dataset
    # A failed first run can leave the dataset root behind before metadata is
    # created.  LeRobotDataset.create() intentionally requires a fresh root,
    # so remove only that known-empty residue and retry creation.  A non-empty
    # partial dataset is left untouched and reported instead of overwriting it.
    if root.exists():
        entries = list(root.iterdir())
        if entries:
            raise RuntimeError(f"数据集目录已存在但缺少 meta/info.json，未覆盖：{root}")
        root.rmdir()
    return LeRobotDataset.create(
        repo_id=args.repo_id, root=root, fps=args.fps, features=features_for(first["images"]),
        robot_type="bi_piper", use_videos=True, streaming_encoding=True,
        encoder_queue_maxsize=90, encoder_threads=args.encoder_threads, rgb_encoder=encoder,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--task", default="dual-arm manipulation")
    p.add_argument("--episode-idx", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--video-codec", default="h264")
    p.add_argument("--video-crf", type=int, default=23)
    p.add_argument("--encoder-threads", type=int, default=2)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--episode-file")
    args = p.parse_args()
    path = Path(args.socket)
    path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    print(f"writer listening: {path}", flush=True)
    conn, _ = server.accept()
    stream = conn.makefile("rb")
    first = recv_packet(stream)
    while isinstance(first, dict) and "_control" in first:
        if first["_control"] == "shutdown":
            raise RuntimeError("录制尚未开始就收到 shutdown")
        first = recv_packet(stream)
    if first is None:
        raise RuntimeError("未收到任何帧")
    dataset = open_dataset(args, first)
    if args.episode_file:
        Path(args.episode_file).write_text(str(dataset.num_episodes), encoding="utf-8")
    count = 0
    try:
        frame = first
        while frame is not None:
            if isinstance(frame, dict) and "_control" in frame:
                control = frame["_control"]
                if control == "episode_end":
                    if count:
                        dataset.save_episode(parallel_encoding=False)
                        if args.episode_file:
                            Path(args.episode_file).write_text(str(dataset.num_episodes), encoding="utf-8")
                        print(f"\n保存 episode frames={count}", flush=True)
                        count = 0
                    frame = recv_packet(stream)
                    continue
                if control == "shutdown":
                    break
            dataset.add_frame({
                "observation.state": np.asarray(frame["qpos"], dtype=np.float32),
                "observation.velocity": np.asarray(frame["qvel"], dtype=np.float32),
                "observation.effort": np.asarray(frame["effort"], dtype=np.float32),
                "observation.timestamps_ns": np.asarray(frame["timestamps_ns"], dtype=np.int64) // 1_000_000_000,
                "action": np.asarray(frame["action"], dtype=np.float32),
                "observation.images.cam_high": frame["images"][0],
                "observation.images.cam_left_wrist": frame["images"][1],
                "observation.images.cam_right_wrist": frame["images"][2],
                "task": args.task,
            })
            count += 1
            if count % 30 == 0:
                print(f"\rwriter frames: {count}", end="", flush=True)
            frame = recv_packet(stream)
        if count:
            dataset.save_episode(parallel_encoding=False)
        dataset.finalize()
        print(f"\n保存完成：{args.dataset_path}", flush=True)
    finally:
        conn.close()
        server.close()
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
