#!/usr/bin/env python
"""Replay one LeRobot episode through the inference-tested BiPiper control path."""

import argparse
import logging
import time
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots.bi_piper import BiPiper, BiPiperConfig
from lerobot.scripts.lerobot_replay import process_replay_action
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a local LeRobot episode on dual PiPER and hold after completion"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/piper_replay")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--control-hz", type=float, default=150.0)
    parser.add_argument("--velocity", type=int, default=20)
    parser.add_argument("--can-left", default="can_left")
    parser.add_argument("--can-right", default="can_right")
    return parser.parse_args()


def hold_until_second_interrupt(robot: BiPiper, control_hz: float) -> None:
    logging.info("Measured-pose hold active. Press Ctrl+C again to close CAN and exit.")
    while True:
        robot.hold_current()
        precise_sleep(1.0 / control_hz)


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.control_hz <= 0:
        raise SystemExit("--fps and --control-hz must be positive")

    init_logging()
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root.expanduser().resolve(),
        episodes=[args.episode],
    )
    action_names = tuple(dataset.features[ACTION]["names"])
    actions = dataset.select_columns(ACTION)
    robot_config = BiPiperConfig(
        id="piper_dual_replay",
        can_left=args.can_left,
        can_right=args.can_right,
        camera_topics={},
        action_feature_names=action_names,
        velocity=args.velocity,
        policy_fps=args.fps,
        command_refresh_hz=args.control_hz,
        enable_keepalive_hz=10.0,
        recover_enable_loss=True,
        dry_run=False,
        keep_enabled_on_disconnect=True,
    )
    robot = BiPiper(robot_config)
    action_processor = make_default_robot_action_processor()
    connected = False
    replay_complete = False

    try:
        robot.connect()
        connected = True
        logging.info(
            "Replaying episode %d: %d frames at %.1f Hz (BiPiper refresh %.1f Hz)",
            args.episode,
            dataset.num_frames,
            args.fps,
            args.control_hz,
        )
        for index in range(dataset.num_frames):
            tick = time.perf_counter()
            observation = robot.get_observation()
            action = process_replay_action(
                actions[index][ACTION],
                action_names,
                observation,
                action_processor,
            )
            robot.send_action(action)
            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - tick), 0.0))
        replay_complete = True
        logging.info("Episode %d complete; all %d frames sent", args.episode, dataset.num_frames)
    except KeyboardInterrupt:
        logging.info("Replay interrupted; freezing the measured pose")
    except Exception:
        logging.exception("Replay failed; freezing the measured pose before exit")
        raise
    finally:
        if connected:
            try:
                hold_until_second_interrupt(robot, args.control_hz)
            except KeyboardInterrupt:
                logging.info(
                    "%s; closing CAN after final measured-pose command",
                    "Replay complete" if replay_complete else "Replay stopped",
                )
            finally:
                robot.disconnect()


if __name__ == "__main__":
    main()
