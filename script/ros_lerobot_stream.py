#!/usr/bin/env python3
"""ROS-side producer for direct LeRobot recording.

The ROS Noetic environment sends synchronized frames to the uv-based writer
over a local Unix socket. No HDF5 staging is used.
"""
import argparse
import pickle
import select
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
import rospy

from ros_lerobot_capture import SynchronizedFrameSource


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", required=True)
    p.add_argument("--timesteps", type=int, default=3000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--wait-timeout", type=float, default=60.0)
    p.add_argument("--stale-timeout", type=float, default=5.0)
    p.add_argument("--sync-slop", type=float, default=0.10)
    p.add_argument("--auto-start", action="store_true")
    p.add_argument("--img-front-topic", default="/camera_f/color/image_raw")
    p.add_argument("--img-left-topic", default="/camera_l/color/image_raw")
    p.add_argument("--img-right-topic", default="/camera_r/color/image_raw")
    p.add_argument("--master-left-topic", default="/master/joint_left")
    p.add_argument("--master-right-topic", default="/master/joint_right")
    p.add_argument("--puppet-left-topic", default="/puppet/joint_left")
    p.add_argument("--puppet-right-topic", default="/puppet/joint_right")
    return p.parse_args()


def send_packet(stream, payload):
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    stream.sendall(struct.pack("!Q", len(data)) + data)


def enter_pressed():
    if not sys.stdin.isatty():
        return False
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if readable:
        sys.stdin.readline()
        return True
    return False


def main():
    args = parse_args()
    rospy.init_node("piper_lerobot_stream", anonymous=True, disable_signals=True)
    source = SynchronizedFrameSource(args)
    print("等待三路相机和四路关节话题同步……", flush=True)
    first = source.wait_for_first(args.wait_timeout)
    if first is None:
        raise SystemExit("等待同步帧超时")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + args.wait_timeout
    while True:
        try:
            sock.connect(args.socket)
            break
        except FileNotFoundError:
            if time.monotonic() > deadline:
                raise SystemExit("LeRobot writer socket 不存在")
            time.sleep(0.1)
        except ConnectionRefusedError:
            if time.monotonic() > deadline:
                raise SystemExit("LeRobot writer 尚未监听 socket")
            time.sleep(0.1)

    print("数据已就绪。按 Enter 开始录制；录制中再次按 Enter 停止。", flush=True)
    if not args.auto_start:
        input()
    stream = sock
    period = 1.0 / args.fps
    next_tick = time.monotonic()
    sequence = -1
    count = 0
    last_frame_time = time.monotonic()
    try:
        while count < args.timesteps and not rospy.is_shutdown():
            if count and enter_pressed():
                break
            frame, new_sequence = source.get_newer_than(sequence)
            if frame is not None:
                send_packet(stream, frame)
                sequence = new_sequence
                count += 1
                last_frame_time = time.monotonic()
                print(f"\r已采集 {count}/{args.timesteps} 帧", end="", flush=True)
            elif time.monotonic() - last_frame_time > args.stale_timeout:
                raise RuntimeError("同步数据停止更新")
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    finally:
        try:
            try:
                send_packet(stream, None)
            except BrokenPipeError:
                pass
        finally:
            stream.close()
    print(f"\n直接采集完成：{count} 帧", flush=True)


if __name__ == "__main__":
    main()
