#!/usr/bin/env python3
"""Convert one staged PiPER HDF5 episode to a local LeRobot v2.1 dataset."""

import argparse
from functools import partial
import json
from pathlib import Path

EXPECTED_CODEBASE_VERSION = "v2.1"
IMAGE_WRITER_THREADS = 8
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
TIMESTAMP_NAMES = (
    "image_front",
    "image_left",
    "image_right",
    "master_left",
    "master_right",
    "puppet_left",
    "puppet_right",
)
JOINT_NAMES = tuple(
    f"{side}_{name}"
    for side in ("left", "right")
    for name in ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert staged ROS capture to LeRobot v2.1")
    parser.add_argument("--input", help="Temporary HDF5 episode")
    parser.add_argument("--dataset-path", required=True, help="LeRobot dataset root")
    parser.add_argument("--repo-id", default="local/piper_dual_arm", help="Logical local dataset id")
    parser.add_argument("--episode-idx", type=int, required=True, help="Expected episode index")
    parser.add_argument("--task", default="dual-arm manipulation", help="Natural-language task")
    parser.add_argument("--fps", type=int, help="Expected FPS (used with --validate-target)")
    parser.add_argument(
        "--video-codec", choices=("h264", "hevc", "libsvtav1"), default="h264",
        help="Video codec used for newly converted episodes (default: h264)",
    )
    parser.add_argument("--video-crf", type=int, default=23, help="Video CRF quality setting")
    parser.add_argument("--validate-target", action="store_true", help="Only validate the target/index")
    return parser.parse_args()


def dataset_features(height, width):
    vector = {"dtype": "float32", "shape": (14,), "names": list(JOINT_NAMES)}
    features = {
        "observation.state": dict(vector),
        "observation.velocity": dict(vector),
        "observation.effort": dict(vector),
        "observation.timestamps_ns": {
            "dtype": "int64",
            "shape": (len(TIMESTAMP_NAMES),),
            "names": list(TIMESTAMP_NAMES),
        },
        "action": dict(vector),
    }
    for camera in CAMERA_KEYS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return features


def load_or_create_dataset(path, repo_id, episode_idx, fps, features):
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != EXPECTED_CODEBASE_VERSION:
        raise RuntimeError(
            f"LeRobot 环境格式为 {CODEBASE_VERSION}，脚本要求 {EXPECTED_CODEBASE_VERSION}"
        )
    info_path = path / "meta/info.json"
    if path.exists():
        if not info_path.is_file():
            raise RuntimeError(f"目标路径已存在但不是 LeRobot 数据集：{path}")
        with info_path.open("r", encoding="utf-8") as stream:
            info = json.load(stream)
        if info.get("codebase_version") != CODEBASE_VERSION:
            raise RuntimeError(
                f"目标数据集版本是 {info.get('codebase_version')}，当前转换器要求 {CODEBASE_VERSION}"
            )
        dataset = LeRobotDataset(repo_id=repo_id, root=path)
        dataset.start_image_writer(num_threads=IMAGE_WRITER_THREADS)
        if dataset.num_episodes != episode_idx:
            raise RuntimeError(
                f"episode_idx={episode_idx} 与下一个可写索引 {dataset.num_episodes} 不一致；"
                "LeRobot v2.1 要求 episode 连续写入"
            )
        if dataset.fps != fps:
            raise RuntimeError(f"FPS 不一致：已有数据集 {dataset.fps}，本次采集 {fps}")
        expected = {key: (value["dtype"], tuple(value["shape"])) for key, value in features.items()}
        actual = {
            key: (dataset.features[key]["dtype"], tuple(dataset.features[key]["shape"]))
            for key in expected
            if key in dataset.features
        }
        if actual != expected:
            raise RuntimeError("已有数据集的特征或相机分辨率与本次采集不一致")
        return dataset

    if episode_idx != 0:
        raise RuntimeError("新数据集必须从 episode_idx=0 开始")
    path.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=path,
        robot_type="piper_dual_arm_ros",
        features=features,
        use_videos=True,
        image_writer_threads=IMAGE_WRITER_THREADS,
    )


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if args.episode_idx < 0:
        raise SystemExit("--episode-idx must be non-negative")

    if args.validate_target:
        info_path = dataset_path / "meta/info.json"
        if dataset_path.exists():
            if not info_path.is_file():
                raise RuntimeError(f"目标路径已存在但不是 LeRobot 数据集：{dataset_path}")
            with info_path.open("r", encoding="utf-8") as stream:
                info = json.load(stream)
            if info.get("codebase_version") != EXPECTED_CODEBASE_VERSION:
                raise RuntimeError(
                    f"目标数据集版本是 {info.get('codebase_version')}，当前要求 {EXPECTED_CODEBASE_VERSION}"
                )
            next_episode = int(info.get("total_episodes", -1))
            if next_episode != args.episode_idx:
                raise RuntimeError(
                    f"episode_idx={args.episode_idx} 与下一个可写索引 {next_episode} 不一致"
                )
            if args.fps is not None and int(info.get("fps", -1)) != args.fps:
                raise RuntimeError(f"FPS 不一致：已有数据集 {info.get('fps')}，本次设置 {args.fps}")
        elif args.episode_idx != 0:
            raise RuntimeError("新数据集必须从 episode_idx=0 开始")
        print(
            f"目标检查通过：{dataset_path}，episode={args.episode_idx}，"
            f"format={EXPECTED_CODEBASE_VERSION}"
        )
        return

    if args.input is None:
        raise SystemExit("转换时必须提供 --input")
    source_path = Path(args.input).expanduser().resolve()

    import h5py
    import numpy as np
    import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
    from lerobot.datasets.video_utils import encode_video_frames

    # LeRobot v2.1 defaults to software AV1. Bind its dataset-level encoder to
    # H.264 (or the explicitly requested codec) without modifying the installed
    # LeRobot package.
    lerobot_dataset_module.encode_video_frames = partial(
        encode_video_frames,
        vcodec=args.video_codec,
        pix_fmt="yuv420p",
        g=2,
        crf=args.video_crf,
    )

    with h5py.File(source_path, "r") as source:
        if source.attrs.get("format") != "piper_ros_capture_staging_v1":
            raise RuntimeError(f"不支持的临时文件格式：{source.attrs.get('format')}")
        num_frames = int(source.attrs["num_frames"])
        fps = int(source.attrs["fps"])
        if num_frames <= 0:
            raise RuntimeError("临时文件中没有帧")
        image_shape = source["images/cam_high"].shape[1:]
        if len(image_shape) != 3 or image_shape[2] != 3:
            raise RuntimeError(f"相机图像不是 HWC RGB：{image_shape}")
        height, width, _ = image_shape
        for camera in CAMERA_KEYS:
            if source[f"images/{camera}"].shape != (num_frames, height, width, 3):
                raise RuntimeError(f"{camera} 分辨率或帧数不一致")

        features = dataset_features(height, width)
        dataset = load_or_create_dataset(dataset_path, args.repo_id, args.episode_idx, fps, features)
        try:
            for index in range(num_frames):
                frame = {
                    "observation.state": np.asarray(source["qpos"][index], dtype=np.float32),
                    "observation.velocity": np.asarray(source["qvel"][index], dtype=np.float32),
                    "observation.effort": np.asarray(source["effort"][index], dtype=np.float32),
                    "observation.timestamps_ns": np.asarray(source["timestamps_ns"][index], dtype=np.int64),
                    "action": np.asarray(source["action"][index], dtype=np.float32),
                }
                for camera in CAMERA_KEYS:
                    frame[f"observation.images.{camera}"] = np.asarray(source[f"images/{camera}"][index])
                dataset.add_frame(frame, task=args.task, timestamp=index / fps)
                if (index + 1) % 30 == 0 or index + 1 == num_frames:
                    print(f"\r转换帧 {index + 1}/{num_frames}", end="", flush=True)
            print("\n正在编码三路 MP4 并写入 Parquet/metadata……", flush=True)
            dataset.save_episode()
        finally:
            if getattr(dataset, "image_writer", None) is not None:
                dataset.stop_image_writer()

    print(
        f"保存完成：{dataset_path}，episode={args.episode_idx}，frames={num_frames}，"
        f"format={EXPECTED_CODEBASE_VERSION}",
        flush=True,
    )


if __name__ == "__main__":
    main()
