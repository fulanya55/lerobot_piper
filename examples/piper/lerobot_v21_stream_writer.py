#!/usr/bin/env python3
"""Socket writer for direct PiPER recording in LeRobot v2.1 layout."""
from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
JOINT_NAMES = [f"{side}_{name}" for side in ("left", "right") for name in
               ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")]


def recv_packet(stream):
    header = stream.recv(8)
    if not header:
        return None
    size = struct.unpack("!Q", header)[0]
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.recv(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return pickle.loads(chunks)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def feature_stats(values) -> dict:
    array = np.asarray(values, dtype=np.float64).reshape((len(values), -1))
    return {"min": array.min(0).tolist(), "max": array.max(0).tolist(),
            "mean": array.mean(0).tolist(), "std": array.std(0).tolist(), "count": [len(array)]}


class Writer:
    def __init__(self, args, first):
        self.args = args
        self.root = Path(args.dataset_path).expanduser().resolve()
        self.info_path = self.root / "meta/info.json"
        self.first = first
        self.episode_index = args.episode_idx
        self.buffers = None
        self.containers = []
        self.streams = []
        self._open_dataset()

    def _open_dataset(self):
        if self.info_path.exists():
            self.info = json.loads(self.info_path.read_text(encoding="utf-8"))
            if self.info.get("codebase_version") != "v2.1":
                raise RuntimeError("目标数据集不是 LeRobot v2.1")
            self.episode_index = int(self.info["total_episodes"])
            return
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meta").mkdir(exist_ok=True)
        vector = {"dtype": "float32", "shape": [14], "names": JOINT_NAMES}
        features = {"observation.state": dict(vector), "observation.velocity": dict(vector),
                    "observation.effort": dict(vector),
                    "observation.timestamps_ns": {"dtype": "int64", "shape": [7], "names": [f"topic_{i}" for i in range(7)]},
                    "action": dict(vector)}
        for camera, image in zip(CAMERAS, self.first["images"], strict=True):
            height, width, channels = image.shape
            features[f"observation.images.{camera}"] = {
                "dtype": "video", "shape": [channels, height, width],
                "names": ["channels", "height", "width"],
                "info": {"video.height": height, "video.width": width, "video.fps": self.args.fps,
                         "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.channels": channels,
                         "has_audio": False},
            }
        for key, dtype in (("timestamp", "float32"), ("frame_index", "int64"),
                           ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
            features[key] = {"dtype": dtype, "shape": [1], "names": None}
        self.info = {"codebase_version": "v2.1", "robot_type": "bi_piper", "total_episodes": 0,
                     "total_frames": 0, "total_tasks": 1, "total_chunks": 1, "chunks_size": 1000,
                     "fps": self.args.fps, "splits": {"train": "0:0"},
                     "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                     "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                     "features": features}
        write_json(self.info_path, self.info)
        append_jsonl(self.root / "meta/tasks.jsonl", {"task_index": 0, "task": self.args.task})

    def begin(self):
        ep, chunk = self.episode_index, self.episode_index // 1000
        self.buffers = {key: [] for key in ("observation.state", "observation.velocity",
                                            "observation.effort", "observation.timestamps_ns", "action")}
        self.containers, self.streams = [], []
        for camera, image in zip(CAMERAS, self.first["images"], strict=True):
            key = f"observation.images.{camera}"
            path = self.root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{ep:06d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            container = av.open(str(path), "w", options={"movflags": "faststart"})
            stream = container.add_stream("libx264", rate=self.args.fps, options={"crf": str(self.args.video_crf)})
            stream.width, stream.height, stream.pix_fmt = image.shape[1], image.shape[0], "yuv420p"
            stream.gop_size = 2
            self.containers.append(container); self.streams.append(stream)

    def add(self, frame):
        if self.buffers is None:
            self.begin()
        for key, source in (("observation.state", "qpos"), ("observation.velocity", "qvel"),
                            ("observation.effort", "effort"), ("observation.timestamps_ns", "timestamps_ns"),
                            ("action", "action")):
            self.buffers[key].append(np.asarray(frame[source]))
        for image, container, stream in zip(frame["images"], self.containers, self.streams, strict=True):
            video_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(video_frame): container.mux(packet)

    def save(self):
        if not self.buffers or not self.buffers["action"]:
            return
        length, ep, chunk = len(self.buffers["action"]), self.episode_index, self.episode_index // 1000
        for container, stream in zip(self.containers, self.streams, strict=True):
            for packet in stream.encode(): container.mux(packet)
            container.close()
        start = int(self.info["total_frames"])
        table = {key: pa.array([value.tolist() for value in values]) for key, values in self.buffers.items()}
        table.update({"timestamp": pa.array(np.arange(length, dtype=np.float32) / self.args.fps),
                      "frame_index": pa.array(np.arange(length, dtype=np.int64)),
                      "episode_index": pa.array(np.full(length, ep, dtype=np.int64)),
                      "index": pa.array(np.arange(start, start + length, dtype=np.int64)),
                      "task_index": pa.array(np.zeros(length, dtype=np.int64))})
        path = self.root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(table), path, compression="snappy")
        stats = {key: feature_stats(values) for key, values in self.buffers.items()}
        append_jsonl(self.root / "meta/episodes.jsonl", {"episode_index": ep, "tasks": [self.args.task], "length": length})
        append_jsonl(self.root / "meta/episodes_stats.jsonl", {"episode_index": ep, "stats": stats})
        self.info["total_episodes"] = ep + 1; self.info["total_frames"] = start + length
        self.info["splits"] = {"train": f"0:{ep + 1}"}; write_json(self.info_path, self.info)
        self.episode_index += 1; self.buffers = None; self.containers = []; self.streams = []


def main():
    p = argparse.ArgumentParser(); p.add_argument("--socket", required=True); p.add_argument("--dataset-path", required=True)
    p.add_argument("--repo-id", default="local/piper_dual_arm"); p.add_argument("--task", default="dual-arm manipulation")
    p.add_argument("--episode-idx", type=int, default=0); p.add_argument("--fps", type=int, default=30)
    p.add_argument("--video-codec", default="h264"); p.add_argument("--video-crf", type=int, default=23)
    p.add_argument("--continuous", action="store_true"); p.add_argument("--episode-file"); args = p.parse_args()
    path = Path(args.socket); path.unlink(missing_ok=True); server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path)); server.listen(1); stream, _ = server.accept(); first = recv_packet(stream)
    if first is None: raise RuntimeError("未收到任何帧")
    writer = Writer(args, first)
    if args.episode_file: Path(args.episode_file).write_text(str(writer.episode_index), encoding="utf-8")
    frame = first
    try:
        while frame is not None:
            if isinstance(frame, dict) and "_control" in frame:
                if frame["_control"] == "episode_end":
                    writer.save()
                    if args.episode_file: Path(args.episode_file).write_text(str(writer.episode_index), encoding="utf-8")
                elif frame["_control"] == "shutdown": break
            else: writer.add(frame)
            frame = recv_packet(stream)
        writer.save()
    finally:
        stream.close(); server.close(); path.unlink(missing_ok=True)


if __name__ == "__main__": main()
