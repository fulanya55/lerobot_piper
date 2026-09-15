#!/usr/bin/env python
"""Run a local LeRobot policy directly through the replay-style PiPER loop.

The control path is deliberately synchronous:

    get_observation -> predict an action chunk -> send_action at policy FPS

No gRPC policy server and no interactive enable/live confirmation are used.
The process owns both CAN interfaces and the ROS camera subscribers while it
is running, just like ``replay_episode.py``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from async_policy_client import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_INFO,
    DEFAULT_TASK,
    load_checkpoint_contract,
    load_dataset_camera_shape,
    load_dataset_fps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly run a local Pi0/Pi0.5 checkpoint on dual PiPER"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--task", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--velocity", type=int, default=30)
    parser.add_argument("--can-left", default="can_left")
    parser.add_argument("--can-right", default="can_right")
    parser.add_argument("--max-policy-actions", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch.compile (the checkpoint default is intentionally not used automatically).",
    )
    return parser.parse_args()


def _rename_map(preprocessor) -> dict[str, str]:
    return next((dict(step.rename_map) for step in preprocessor.steps if hasattr(step, "rename_map")), {})


def _load_policy(checkpoint: Path, device: str, compile_model: bool):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not hasattr(config, "device"):
        raise RuntimeError(f"Policy config has no device field: {checkpoint}")
    config.device = device
    if hasattr(config, "compile_model"):
        config.compile_model = compile_model
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
    )
    device_override = {"device": device}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": device_override},
    )
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    return policy, preprocessor, postprocessor, _rename_map(preprocessor)


def _predict_chunk(policy, preprocessor, postprocessor, raw_observation, robot, rename_map, task):
    from lerobot.async_inference.helpers import (
        map_robot_keys_to_lerobot_features,
        raw_observation_to_observation,
    )

    raw_observation = dict(raw_observation)
    raw_observation["task"] = task
    lerobot_features = map_robot_keys_to_lerobot_features(robot)
    observation = raw_observation_to_observation(
        raw_observation,
        lerobot_features,
        policy.config.image_features,
        observation_rename_map=rename_map,
    )
    observation = preprocessor(observation)
    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(observation)
    if action_chunk.ndim == 2:
        action_chunk = action_chunk.unsqueeze(0)

    processed = []
    for index in range(action_chunk.shape[1]):
        processed.append(postprocessor(action_chunk[:, index, :]).squeeze(0).detach().cpu())
    return processed


def _hold_until_second_interrupt(robot, control_hz: float) -> None:
    logging.info("Measured-pose hold active. Press Ctrl+C again to close CAN and exit.")
    while True:
        robot.hold_current()
        time.sleep(1.0 / control_hz)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_info = args.dataset_info.expanduser().resolve()
    task = args.task
    if task is None and checkpoint == DEFAULT_CHECKPOINT.resolve():
        task = DEFAULT_TASK
    if not task:
        raise SystemExit("--task is required for a checkpoint other than the default model")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"CUDA device is unavailable: {args.device}")
    if args.max_policy_actions is not None and args.max_policy_actions <= 0:
        raise SystemExit("--max-policy-actions must be positive")

    contract = load_checkpoint_contract(checkpoint)
    dataset_fps = load_dataset_fps(dataset_info, contract.action_feature_names)
    fps = dataset_fps if args.fps is None else args.fps
    if fps <= 0 or fps != dataset_fps:
        raise SystemExit(f"--fps must equal dataset FPS ({dataset_fps})")
    height, width = load_dataset_camera_shape(dataset_info)
    height = args.camera_height or height
    width = args.camera_width or width

    from lerobot.robots.bi_piper import BiPiper, BiPiperConfig
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import init_logging

    init_logging()
    robot_config = BiPiperConfig(
        id="piper_dual_direct_inference",
        can_left=args.can_left,
        can_right=args.can_right,
        camera_topics=contract.camera_topics,
        action_feature_names=contract.action_feature_names,
        gripper_action_unit=contract.gripper_action_unit,
        image_height=height,
        image_width=width,
        velocity=args.velocity,
        policy_fps=fps,
        recover_enable_loss=True,
        enable_keepalive_hz=20.0,
        dry_run=False,
        keep_enabled_on_disconnect=True,
    )

    logging.info(
        "Direct inference: checkpoint=%s type=%s, chunk=%d, fps=%.1f, camera=%dx%d; CAN commands are live",
        checkpoint,
        contract.policy_type,
        contract.chunk_size,
        fps,
        width,
        height,
    )
    robot = BiPiper(robot_config)
    connected = False
    policy_actions = 0
    completed = False
    try:
        robot.connect()
        connected = True
        logging.info(
            "Connected and holding measured pose. Loading policy on %s; "
            "the arms remain enabled during model loading.",
            args.device,
        )
        policy, preprocessor, postprocessor, rename_map = _load_policy(
            checkpoint, args.device, args.compile_model
        )
        logging.info("Policy ready. Starting replay-style direct control; press Ctrl+C to stop.")
        while args.max_policy_actions is None or policy_actions < args.max_policy_actions:
            try:
                observation = robot.get_observation()
                chunk_started = time.perf_counter()
                chunk = _predict_chunk(
                    policy, preprocessor, postprocessor, observation, robot, rename_map, task
                )
                logging.info(
                    "Predicted action chunk: %d actions in %.3fs (sent=%d)",
                    len(chunk),
                    time.perf_counter() - chunk_started,
                    policy_actions,
                )
                for action_tensor in chunk:
                    tick = time.perf_counter()
                    action = {
                        name: float(action_tensor[index])
                        for index, name in enumerate(contract.action_feature_names)
                    }
                    robot.send_action(action)
                    policy_actions += 1
                    if policy_actions == 1 or policy_actions % 10 == 0:
                        logging.info("Live policy action sent: %d", policy_actions)
                    if args.max_policy_actions is not None and policy_actions >= args.max_policy_actions:
                        break
                    precise_sleep(max(1.0 / fps - (time.perf_counter() - tick), 0.0))
                if args.max_policy_actions is not None and policy_actions >= args.max_policy_actions:
                    completed = True
                    break
            except Exception as exc:
                # A dropped frame, transient CUDA/ROS error, or recovered CAN
                # status must not tear down the enable owner.  Hold the
                # measured pose and retry the next chunk instead of entering
                # the final shutdown path.
                logging.exception("Policy/control tick failed; holding and retrying: %s", exc)
                robot.hold_current()
                time.sleep(0.5)
        logging.info("Direct inference stopped after %d policy actions", policy_actions)
    except KeyboardInterrupt:
        logging.info("Inference interrupted after %d policy actions; holding measured pose", policy_actions)
    except Exception:
        logging.exception("Direct inference failed after %d policy actions", policy_actions)
        raise
    finally:
        if connected:
            try:
                _hold_until_second_interrupt(robot, robot_config.command_refresh_hz)
            except KeyboardInterrupt:
                logging.info("%s; closing CAN", "Inference complete" if completed else "Inference stopped")
            finally:
                robot.disconnect()


if __name__ == "__main__":
    main()
