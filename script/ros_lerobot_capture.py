#!/usr/bin/env python3
"""Capture synchronized PiPER/RealSense ROS messages into a temporary HDF5 file.

This process intentionally only depends on the ROS-capable ``aloha`` conda
environment.  A separate process converts the temporary file to LeRobot v2.1.
"""

import argparse
import select
import sys
import threading
import time
from pathlib import Path

import h5py
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState


TOPIC_LABELS = (
    "image_front",
    "image_left",
    "image_right",
    "master_left",
    "master_right",
    "puppet_left",
    "puppet_right",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Record synchronized ROS data for LeRobot conversion")
    parser.add_argument("--output", required=True, help="Temporary HDF5 output path")
    parser.add_argument("--timesteps", type=int, default=3000, help="Maximum frames in this episode")
    parser.add_argument("--fps", type=int, default=30, help="Recording rate")
    parser.add_argument("--sync-slop", type=float, default=0.10, help="Approximate ROS synchronization window (s)")
    parser.add_argument("--wait-timeout", type=float, default=60.0, help="Seconds to wait for the first synchronized frame")
    parser.add_argument("--stale-timeout", type=float, default=5.0, help="Abort if synchronized data stops for this many seconds")
    parser.add_argument("--img-front-topic", default="/camera_f/color/image_raw")
    parser.add_argument("--img-left-topic", default="/camera_l/color/image_raw")
    parser.add_argument("--img-right-topic", default="/camera_r/color/image_raw")
    parser.add_argument("--master-left-topic", default="/master/joint_left")
    parser.add_argument("--master-right-topic", default="/master/joint_right")
    parser.add_argument("--puppet-left-topic", default="/puppet/joint_left")
    parser.add_argument("--puppet-right-topic", default="/puppet/joint_right")
    return parser.parse_args()


class SynchronizedFrameSource:
    def __init__(self, args):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.latest = None
        self.sequence = 0

        topics = (
            (args.img_front_topic, Image),
            (args.img_left_topic, Image),
            (args.img_right_topic, Image),
            (args.master_left_topic, JointState),
            (args.master_right_topic, JointState),
            (args.puppet_left_topic, JointState),
            (args.puppet_right_topic, JointState),
        )
        self.subscribers = [message_filters.Subscriber(topic, msg_type) for topic, msg_type in topics]
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=90, slop=args.sync_slop, allow_headerless=False
        )
        self.synchronizer.registerCallback(self._callback)

    @staticmethod
    def _joint_vector(message, field, topic_label):
        values = np.asarray(getattr(message, field), dtype=np.float32)
        if values.shape != (7,):
            raise ValueError(f"{topic_label}.{field} expected 7 values, got shape {values.shape}")
        return values

    def _callback(self, image_front, image_left, image_right, master_left, master_right, puppet_left, puppet_right):
        try:
            # Force a single RGB convention for both LeRobot and the preview.
            images = [
                self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
                for message in (image_front, image_left, image_right)
            ]
            frame = {
                "images": [np.ascontiguousarray(image, dtype=np.uint8) for image in images],
                "qpos": np.concatenate(
                    [
                        self._joint_vector(puppet_left, "position", "puppet_left"),
                        self._joint_vector(puppet_right, "position", "puppet_right"),
                    ]
                ),
                "qvel": np.concatenate(
                    [
                        self._joint_vector(puppet_left, "velocity", "puppet_left"),
                        self._joint_vector(puppet_right, "velocity", "puppet_right"),
                    ]
                ),
                "effort": np.concatenate(
                    [
                        self._joint_vector(puppet_left, "effort", "puppet_left"),
                        self._joint_vector(puppet_right, "effort", "puppet_right"),
                    ]
                ),
                "action": np.concatenate(
                    [
                        self._joint_vector(master_left, "position", "master_left"),
                        self._joint_vector(master_right, "position", "master_right"),
                    ]
                ),
                "timestamps_ns": np.asarray(
                    [
                        image_front.header.stamp.to_nsec(),
                        image_left.header.stamp.to_nsec(),
                        image_right.header.stamp.to_nsec(),
                        master_left.header.stamp.to_nsec(),
                        master_right.header.stamp.to_nsec(),
                        puppet_left.header.stamp.to_nsec(),
                        puppet_right.header.stamp.to_nsec(),
                    ],
                    dtype=np.int64,
                ),
            }
        except Exception as exc:
            rospy.logwarn_throttle(2.0, f"Dropping invalid synchronized frame: {exc}")
            return

        with self.condition:
            self.latest = frame
            self.sequence += 1
            self.condition.notify_all()

    def wait_for_first(self, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.latest is None and not rospy.is_shutdown():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(timeout=min(remaining, 0.5))
            return self.latest

    def get_newer_than(self, previous_sequence):
        with self.lock:
            if self.latest is None or self.sequence == previous_sequence:
                return None, previous_sequence
            return self.latest, self.sequence


class HDF5EpisodeWriter:
    def __init__(self, path, max_frames, fps, first_frame):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "x")
        self.count = 0
        self.max_frames = max_frames

        self.file.attrs["format"] = "piper_ros_capture_staging_v1"
        self.file.attrs["fps"] = fps
        self.file.attrs["timestamp_labels"] = np.asarray(TOPIC_LABELS, dtype=h5py.string_dtype())

        image_group = self.file.create_group("images")
        for key, image in zip(("cam_high", "cam_left_wrist", "cam_right_wrist"), first_frame["images"]):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"{key} expected HWC RGB image, got {image.shape}")
            image_group.create_dataset(
                key,
                shape=(max_frames, *image.shape),
                maxshape=(None, *image.shape),
                chunks=(1, *image.shape),
                compression="lzf",
                dtype=np.uint8,
            )

        for key in ("qpos", "qvel", "effort", "action"):
            self.file.create_dataset(key, shape=(max_frames, 14), maxshape=(None, 14), chunks=(256, 14), dtype=np.float32)
        self.file.create_dataset(
            "timestamps_ns", shape=(max_frames, len(TOPIC_LABELS)), maxshape=(None, len(TOPIC_LABELS)),
            chunks=(256, len(TOPIC_LABELS)), dtype=np.int64
        )

    def append(self, frame):
        if self.count >= self.max_frames:
            raise RuntimeError("episode writer is full")
        for key, image in zip(("cam_high", "cam_left_wrist", "cam_right_wrist"), frame["images"]):
            self.file[f"images/{key}"][self.count] = image
        for key in ("qpos", "qvel", "effort", "action", "timestamps_ns"):
            self.file[key][self.count] = frame[key]
        self.count += 1
        if self.count % 30 == 0:
            self.file.flush()

    def close(self):
        if self.file is None:
            return
        for dataset in self.file.values():
            if isinstance(dataset, h5py.Dataset):
                dataset.resize(self.count, axis=0)
            elif isinstance(dataset, h5py.Group):
                for child in dataset.values():
                    child.resize(self.count, axis=0)
        self.file.attrs["num_frames"] = self.count
        self.file.flush()
        self.file.close()
        self.file = None


def enter_pressed():
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return False
    sys.stdin.readline()
    return True


def main():
    args = parse_args()
    if args.timesteps <= 0 or args.fps <= 0:
        raise SystemExit("--timesteps and --fps must be positive")
    if not sys.stdin.isatty():
        raise SystemExit("Interactive recording requires a terminal (stdin is not a TTY)")

    rospy.init_node("piper_lerobot_capture", anonymous=True, disable_signals=True)
    source = SynchronizedFrameSource(args)
    print("\n等待三个相机和四路关节话题同步……", flush=True)
    first_frame = source.wait_for_first(args.wait_timeout)
    if first_frame is None:
        raise SystemExit(f"{args.wait_timeout:.0f} 秒内未收到同步帧，请检查 ROS topics 和日志")

    print("数据已就绪。按 Enter 开始录制；录制中再次按 Enter 停止。", flush=True)
    input()

    writer = HDF5EpisodeWriter(args.output, args.timesteps, args.fps, first_frame)
    period = 1.0 / args.fps
    next_tick = time.monotonic()
    sequence = -1
    started = time.monotonic()
    last_frame_time = started
    print(f"● 正在录制（上限 {args.timesteps} 帧，{args.fps} FPS）……", flush=True)
    try:
        while writer.count < args.timesteps and not rospy.is_shutdown():
            if enter_pressed():
                print("收到停止按键。", flush=True)
                break
            frame, new_sequence = source.get_newer_than(sequence)
            if frame is not None:
                writer.append(frame)
                sequence = new_sequence
                last_frame_time = time.monotonic()
                print(f"\r已录制 {writer.count}/{args.timesteps} 帧", end="", flush=True)
            elif time.monotonic() - last_frame_time > args.stale_timeout:
                raise RuntimeError(
                    f"连续 {args.stale_timeout:.1f} 秒没有新的同步帧，请检查相机/双臂 topics"
                )
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，结束当前录制。", flush=True)
    finally:
        writer.close()

    elapsed = time.monotonic() - started
    print(f"\n临时采集完成：{writer.count} 帧，{elapsed:.1f} 秒，文件 {writer.path}", flush=True)
    if writer.count == 0:
        raise SystemExit("没有采集到有效帧，不执行 LeRobot 转换")


if __name__ == "__main__":
    main()
