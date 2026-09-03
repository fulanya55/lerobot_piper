#!/usr/bin/env python
"""Deploy a local Pi0/Pi0.5 checkpoint on two PiPER arms through direct CAN.

The entrypoint is model-name agnostic: the policy type is loaded from the
checkpoint configuration rather than selected by this filename.

ROS remains the owner of the three RealSense streams. ``piper-sdk`` is the
only owner of ``can_left`` and ``can_right``. The default mode is read-only;
enabling and policy execution both require explicit acknowledgements.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHECKPOINT = Path("/home/agilex/wxwu/model/PLATE_THE_TUBE_PI05_bs192/14k/pretrained_model")
DEFAULT_DATASET_INFO = Path("/home/agilex/wxwu/data/PLACE_THE_TEST_TUBE/meta/info.json")
DEFAULT_TASK = "Place the test tube on the test tube rack on the desk with the gripper."
DEFAULT_V21_CONVERTER = Path("/home/agilex/wxwu/script/hdf5_to_lerobot_v2.py")
DEFAULT_V21_PYTHON = Path("/home/agilex/miniconda3/envs/lerobot/bin/python")
METER_ACTION_NAMES = tuple(
    f"{side}_{name}"
    for side in ("left", "right")
    for name in ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")
)
LEGACY_SCALE_ACTION_NAMES = tuple(
    f"{side}_{name}"
    for side in ("left", "right")
    for name in (
        "arm_joint_1_rad",
        "arm_joint_2_rad",
        "arm_joint_3_rad",
        "arm_joint_4_rad",
        "arm_joint_5_rad",
        "arm_joint_6_rad",
        "gripper_open_scale",
    )
)
LEGACY_OPEN_ACTION_NAMES = tuple(name.replace("_scale", "") for name in LEGACY_SCALE_ACTION_NAMES)
EXPECTED_ACTION_NAMES = METER_ACTION_NAMES  # Backward-compatible import used by deployment tests.
SUPPORTED_ACTION_SCHEMAS = {
    METER_ACTION_NAMES: "meters",
    LEGACY_SCALE_ACTION_NAMES: "open_scale",
    LEGACY_OPEN_ACTION_NAMES: "open_scale",
}
EXPECTED_POLICY_IMAGES = {
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
}
CAMERA_TOPICS = {
    "cam_high": "/camera_f/color/image_raw",
    "cam_high_rgb": "/camera_f/color/image_raw",
    "cam_left_wrist": "/camera_l/color/image_raw",
    "cam_left_wrist_rgb": "/camera_l/color/image_raw",
    "cam_right_wrist": "/camera_r/color/image_raw",
    "cam_right_wrist_rgb": "/camera_r/color/image_raw",
    "base_0_rgb": "/camera_f/color/image_raw",
    "left_wrist_0_rgb": "/camera_l/color/image_raw",
    "right_wrist_0_rgb": "/camera_r/color/image_raw",
}


@dataclass(frozen=True)
class CheckpointContract:
    policy_type: str
    chunk_size: int
    n_action_steps: int
    action_dim: int
    camera_topics: dict[str, str]
    training_repo_id: str
    checkpoint_compile_model: bool
    action_feature_names: tuple[str, ...]
    gripper_action_unit: str
    state_dim: int
    has_train_config: bool


@dataclass(frozen=True)
class RecordingPlan:
    dataset_path: Path
    raw_path: Path
    episode_idx: int
    converter_command: tuple[str, ...]


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file is missing: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint JSON must contain an object: {path}")
    return value


def _require_processor_payload(
    checkpoint: Path,
    processor_config: dict,
    *,
    config_filename: str,
    registry_name: str,
) -> None:
    indexes = [
        index
        for index, step in enumerate(processor_config.get("steps", []))
        if step.get("registry_name") == registry_name
    ]
    if len(indexes) != 1:
        raise ValueError(f"{config_filename} must contain exactly one {registry_name!r} step")
    payload = checkpoint / f"{Path(config_filename).stem}_step_{indexes[0]}_{registry_name}.safetensors"
    if not payload.is_file() or payload.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint processor payload is missing or empty: {payload}")


def load_checkpoint_contract(checkpoint: Path) -> CheckpointContract:
    """Validate the deployment-relevant, real (unpadded) checkpoint interface."""
    checkpoint = checkpoint.expanduser().resolve()
    config = _read_json(checkpoint / "config.json")
    preprocessor = _read_json(checkpoint / "policy_preprocessor.json")
    postprocessor = _read_json(checkpoint / "policy_postprocessor.json")

    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint payload is missing or empty: {model_path}")
    _require_processor_payload(
        checkpoint,
        preprocessor,
        config_filename="policy_preprocessor.json",
        registry_name="normalizer_processor",
    )
    _require_processor_payload(
        checkpoint,
        postprocessor,
        config_filename="policy_postprocessor.json",
        registry_name="unnormalizer_processor",
    )

    policy_type = config.get("type")
    if policy_type not in {"pi0", "pi05"}:
        raise ValueError(f"Expected a Pi0/Pi0.5 checkpoint, found policy type {policy_type!r}")

    output_shape = config.get("output_features", {}).get("action", {}).get("shape")
    if output_shape != [14]:
        raise ValueError(f"PiPER deployment requires a real 14D action, checkpoint declares {output_shape}")
    if config.get("use_relative_actions") is not False:
        raise ValueError("PiPER deployment requires absolute joint-position actions")
    action_feature_names = tuple(config.get("action_feature_names") or ())
    gripper_action_unit = SUPPORTED_ACTION_SCHEMAS.get(action_feature_names)
    if gripper_action_unit is None:
        raise ValueError(
            "Checkpoint action names/order or units do not match a supported dual-PiPER 14D schema"
        )

    train_config_path = checkpoint / "train_config.json"
    train_config = _read_json(train_config_path) if train_config_path.is_file() else None
    if train_config is not None:
        train_policy = train_config.get("policy")
        if not isinstance(train_policy, dict):
            raise ValueError("train_config.json is missing the serialized training policy configuration")
        if train_policy != config:
            mismatched_keys = sorted(
                key for key in set(train_policy) | set(config) if train_policy.get(key) != config.get(key)
            )
            raise ValueError(f"Training/exported policy configuration mismatch: {mismatched_keys}")

    input_features = config.get("input_features", {})
    state_shape = input_features.get("observation.state", {}).get("shape")
    if state_shape not in ([14], [32]):
        raise ValueError(f"Expected a real or padded dual-PiPER state interface, found {state_shape}")
    state_dim = int(state_shape[0])
    image_features = {key for key in input_features if key.startswith("observation.images.")}
    if image_features != EXPECTED_POLICY_IMAGES:
        raise ValueError(f"Unexpected checkpoint camera contract: {sorted(image_features)}")

    rename_steps = [
        step
        for step in preprocessor.get("steps", [])
        if step.get("registry_name") == "rename_observations_processor"
    ]
    if len(rename_steps) > 1:
        raise ValueError("Checkpoint contains multiple observation rename processors")
    rename_map = rename_steps[0].get("config", {}).get("rename_map", {}) if rename_steps else {}
    if rename_map and set(rename_map.values()) != EXPECTED_POLICY_IMAGES:
        raise ValueError(f"Checkpoint camera rename map is incomplete: {rename_map}")
    if train_config is not None and train_config.get("rename_map", {}) != rename_map:
        raise ValueError("Training and checkpoint camera rename maps do not match")

    camera_topics = {}
    camera_source_keys = rename_map or {key: key for key in EXPECTED_POLICY_IMAGES}
    for source_key in camera_source_keys:
        prefix = "observation.images."
        if not source_key.startswith(prefix):
            raise ValueError(f"Unexpected camera source key: {source_key}")
        camera_name = source_key.removeprefix(prefix)
        if camera_name not in CAMERA_TOPICS:
            raise ValueError(f"No ROS topic is defined for checkpoint camera {camera_name!r}")
        camera_topics[camera_name] = CAMERA_TOPICS[camera_name]

    chunk_size = int(config.get("chunk_size", 0))
    n_action_steps = int(config.get("n_action_steps", 0))
    if chunk_size <= 0 or not 0 < n_action_steps <= chunk_size:
        raise ValueError(
            f"Invalid checkpoint action horizon: chunk_size={chunk_size}, n_action_steps={n_action_steps}"
        )
    training_dataset = train_config.get("dataset", {}) if train_config is not None else {}
    training_repo_id = str(training_dataset.get("repo_id", "<not recorded>"))
    return CheckpointContract(
        policy_type=policy_type,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        action_dim=14,
        camera_topics=camera_topics,
        training_repo_id=training_repo_id,
        checkpoint_compile_model=bool(config.get("compile_model", False)),
        action_feature_names=action_feature_names,
        gripper_action_unit=gripper_action_unit,
        state_dim=state_dim,
        has_train_config=train_config is not None,
    )


def load_dataset_fps(dataset_info_path: Path, expected_action_names: tuple[str, ...]) -> int:
    """Read the control frequency and real 14D schema from dataset metadata."""
    dataset_info = _read_json(dataset_info_path.expanduser().resolve())
    fps = int(dataset_info.get("fps", 0))
    if fps <= 0:
        raise ValueError(f"Dataset metadata has an invalid fps: {dataset_info.get('fps')!r}")

    features = dataset_info.get("features", {})
    action = features.get("action", {})
    state = features.get("observation.state", {})
    if action.get("shape") != [14] or tuple(action.get("names") or ()) != expected_action_names:
        raise ValueError("Dataset action metadata does not match the expected dual-PiPER 14D schema")
    if state.get("shape") != [14] or tuple(state.get("names") or ()) != expected_action_names:
        raise ValueError("Dataset state metadata does not match the expected dual-PiPER 14D schema")

    camera_keys = {
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    }
    for key in camera_keys:
        video_fps = features.get(key, {}).get("info", {}).get("video.fps")
        if video_fps != fps:
            raise ValueError(f"Dataset camera {key!r} fps={video_fps!r} does not match dataset fps={fps}")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dataset-info",
        type=Path,
        default=DEFAULT_DATASET_INFO,
        help=(
            "Training dataset meta/info.json used to recover and validate FPS/schema. "
            f"Default: {DEFAULT_DATASET_INFO}"
        ),
    )
    parser.add_argument("--server-address", default="127.0.0.1:18080")
    parser.add_argument(
        "--task",
        default=None,
        help="Policy instruction. Required for non-default checkpoints; never inferred from model type.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Optional assertion; must equal the FPS read from --dataset-info.",
    )
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=None,
        help="Actions requested/executed per inference. Default: checkpoint chunk_size (50).",
    )
    parser.add_argument(
        "--max-policy-actions",
        type=int,
        default=None,
        help=(
            "Stop after this many real policy commands and send a final hold. "
            "Physically support both arms before CAN closes because controller-side enable may be lost."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("validate", "observe", "hold", "execute"),
        default="observe",
        help=(
            "validate=config only; observe=dry-run inference; "
            "hold=enable/current-pose hold only; execute=real policy actions."
        ),
    )
    parser.add_argument("--can-left", default="can_left")
    parser.add_argument("--can-right", default="can_right")
    parser.add_argument("--velocity", type=int, default=30)
    parser.add_argument(
        "--gripper-range-m",
        type=float,
        default=0.08,
        help="Physical full-open travel used for legacy 0-1 gripper scale conversion.",
    )
    parser.add_argument("--max-joint-step-rad", type=float, default=0.05)
    parser.add_argument("--max-gripper-step-m", type=float, default=0.005)
    parser.add_argument(
        "--prefetch-ratio",
        type=float,
        default=0.5,
        help="Request one new chunk when the queued fraction reaches this value.",
    )
    parser.add_argument(
        "--inference-request-timeout-s",
        type=float,
        default=3.0,
        help="Retry a lost inference request after this timeout; only one request is normally in flight.",
    )
    parser.add_argument(
        "--action-commit-steps",
        type=int,
        default=5,
        help="Keep this many immediate overlapping actions from the current chunk.",
    )
    parser.add_argument(
        "--action-blend-steps",
        type=int,
        default=10,
        help="Smoothstep cross-fade length for overlapping old/new chunks.",
    )
    parser.add_argument(
        "--trajectory-smoothing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply stateful velocity/acceleration limiting before the measured-position safety envelope. "
            "Default: disabled, matching the original per-step-limited execution speed."
        ),
    )
    parser.add_argument("--max-joint-velocity-rad-s", type=float, default=1.0)
    parser.add_argument("--max-joint-acceleration-rad-s2", type=float, default=4.0)
    parser.add_argument("--max-gripper-velocity-m-s", type=float, default=0.08)
    parser.add_argument("--max-gripper-acceleration-m-s2", type=float, default=0.4)
    parser.add_argument(
        "--telemetry-path",
        type=Path,
        default=None,
        help="JSONL action telemetry path. Default: logs/piper_actions_<timestamp>.jsonl.",
    )
    parser.add_argument(
        "--record-dataset-path",
        type=Path,
        default=None,
        help=(
            "Append this execute rollout as one LeRobot v2.1 episode. Recording starts with the "
            "first executed action and Enter stops before the next action."
        ),
    )
    parser.add_argument(
        "--record-episode-idx",
        type=int,
        default=None,
        help="Expected episode index. Default: read the next index from dataset metadata.",
    )
    parser.add_argument("--record-repo-id", default="local/piper_inference")
    parser.add_argument(
        "--record-staging-dir",
        type=Path,
        default=None,
        help="Raw HDF5 directory. Default: a hidden directory next to the target dataset.",
    )
    parser.add_argument("--record-converter", type=Path, default=DEFAULT_V21_CONVERTER)
    parser.add_argument("--record-converter-python", type=Path, default=DEFAULT_V21_PYTHON)
    parser.add_argument("--record-video-codec", choices=("h264", "hevc", "libsvtav1"), default="h264")
    parser.add_argument("--record-video-crf", type=int, default=23)
    parser.add_argument(
        "--confirm-enable",
        action="store_true",
        help="Required for hold/execute: acknowledges that both arms will enable and hold their measured pose.",
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required for execute: acknowledges that postprocessed policy joint targets will move the arms.",
    )
    return parser.parse_args()


def _validate_args(
    args: argparse.Namespace, contract: CheckpointContract, training_fps: int | None
) -> tuple[int, int]:
    if training_fps is None and args.fps is None:
        raise SystemExit(
            "Provide --dataset-info or an explicit --fps; policy config does not store dataset FPS"
        )
    fps = training_fps if args.fps is None else args.fps
    assert fps is not None
    if fps <= 0:
        raise SystemExit("--fps must be positive")
    if training_fps is not None and fps != training_fps:
        raise SystemExit(f"Refusing FPS mismatch: dataset metadata says {training_fps}, but --fps={fps}")
    if args.mode in {"hold", "execute"} and not args.confirm_enable:
        raise SystemExit(f"Refusing {args.mode} mode without --confirm-enable")
    if args.mode == "execute" and not args.confirm_live:
        raise SystemExit("Refusing policy execution without --confirm-live")
    if args.max_policy_actions is not None and args.max_policy_actions <= 0:
        raise SystemExit("--max-policy-actions must be positive")
    if args.max_policy_actions is not None and args.mode != "execute":
        raise SystemExit("--max-policy-actions is only meaningful in execute mode")
    if args.record_dataset_path is not None and args.mode != "execute":
        raise SystemExit("--record-dataset-path is only supported in execute mode")
    if args.record_episode_idx is not None and args.record_episode_idx < 0:
        raise SystemExit("--record-episode-idx must be non-negative")
    if args.record_episode_idx is not None and args.record_dataset_path is None:
        raise SystemExit("--record-episode-idx requires --record-dataset-path")
    if args.record_video_crf < 0:
        raise SystemExit("--record-video-crf must be non-negative")
    if args.gripper_range_m <= 0:
        raise SystemExit("--gripper-range-m must be positive")
    if not 0 <= args.prefetch_ratio <= 1:
        raise SystemExit("--prefetch-ratio must be in [0, 1]")
    if args.inference_request_timeout_s <= 0:
        raise SystemExit("--inference-request-timeout-s must be positive")
    if args.action_commit_steps < 0 or args.action_blend_steps < 0:
        raise SystemExit("--action-commit-steps and --action-blend-steps must be non-negative")
    trajectory_limits = (
        args.max_joint_velocity_rad_s,
        args.max_joint_acceleration_rad_s2,
        args.max_gripper_velocity_m_s,
        args.max_gripper_acceleration_m_s2,
    )
    if any(value <= 0 for value in trajectory_limits):
        raise SystemExit("Trajectory velocity/acceleration limits must be positive")

    actions_per_chunk = contract.chunk_size if args.actions_per_chunk is None else args.actions_per_chunk
    if not 0 < actions_per_chunk <= contract.chunk_size:
        raise SystemExit(f"--actions-per-chunk must be in [1, {contract.chunk_size}] for this checkpoint")
    return actions_per_chunk, fps


def _prepare_recording(args: argparse.Namespace, *, fps: int, task: str) -> RecordingPlan | None:
    if args.record_dataset_path is None:
        return None
    if not sys.stdin.isatty():
        raise SystemExit("Interactive inference recording requires a terminal (stdin is not a TTY)")

    dataset_path = args.record_dataset_path.expanduser().resolve()
    info_path = dataset_path / "meta/info.json"
    next_episode_idx = 0
    if info_path.is_file():
        info = _read_json(info_path)
        next_episode_idx = int(info.get("total_episodes", -1))
        if next_episode_idx < 0:
            raise SystemExit(f"Dataset metadata has an invalid total_episodes value: {info_path}")
    episode_idx = next_episode_idx if args.record_episode_idx is None else args.record_episode_idx

    staging_dir = (
        args.record_staging_dir.expanduser().resolve()
        if args.record_staging_dir is not None
        else dataset_path.parent / f".{dataset_path.name}_inference_staging"
    )
    raw_path = staging_dir / f"episode_{episode_idx}.hdf5"
    if raw_path.exists():
        raise SystemExit(
            f"Refusing to overwrite pending inference data: {raw_path}. "
            "Convert or move that HDF5 before recording this episode again."
        )

    converter = args.record_converter.expanduser().resolve()
    converter_python = args.record_converter_python.expanduser().resolve()
    for required in (converter, converter_python):
        if not required.is_file():
            raise SystemExit(f"Inference dataset conversion dependency is missing: {required}")

    validate_command = (
        str(converter_python),
        str(converter),
        "--validate-target",
        "--dataset-path",
        str(dataset_path),
        "--repo-id",
        args.record_repo_id,
        "--episode-idx",
        str(episode_idx),
        "--fps",
        str(fps),
    )
    validation = subprocess.run(validate_command, check=False)
    if validation.returncode != 0:
        raise SystemExit("LeRobot v2.1 target validation failed; inference was not started")

    converter_command = (
        str(converter_python),
        str(converter),
        "--input",
        str(raw_path),
        "--dataset-path",
        str(dataset_path),
        "--repo-id",
        args.record_repo_id,
        "--episode-idx",
        str(episode_idx),
        "--task",
        task,
        "--video-codec",
        args.record_video_codec,
        "--video-crf",
        str(args.record_video_crf),
    )
    return RecordingPlan(
        dataset_path=dataset_path,
        raw_path=raw_path,
        episode_idx=episode_idx,
        converter_command=converter_command,
    )


def _convert_recording(plan: RecordingPlan) -> None:
    print(
        "[PiPER client] Robot connection is closed; converting the staged episode to LeRobot v2.1...",
        flush=True,
    )
    result = subprocess.run(plan.converter_command, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Inference recording conversion failed (exit={result.returncode}); raw HDF5 retained at "
            f"{plan.raw_path}"
        )
    plan.raw_path.unlink()
    print(
        f"[PiPER client] Dataset saved: {plan.dataset_path} (episode {plan.episode_idx}); "
        "validated raw HDF5 removed after successful conversion.",
        flush=True,
    )


def _hold_only(robot_config) -> None:
    from lerobot.robots.bi_piper import BiPiper

    robot = BiPiper(robot_config)
    connected = False
    try:
        robot.connect()
        connected = True
        print(
            "[PiPER client] HOLD mode active: all 12 drivers are enabled; no policy is loaded or executed. "
            "Current-pose commands are refreshed continuously. Press Ctrl+C only after physically "
            "supporting both arms; closing CAN may cause controller-side enable loss.",
            flush=True,
        )
        while True:
            robot.hold_current()
            time.sleep(1.0 / robot_config.policy_fps)
    except KeyboardInterrupt:
        pass
    finally:
        if connected:
            robot.disconnect()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    contract = load_checkpoint_contract(checkpoint)
    task = args.task
    if task is None and checkpoint == DEFAULT_CHECKPOINT.resolve():
        task = DEFAULT_TASK
    if args.mode not in {"validate", "hold"} and not task:
        raise SystemExit("Provide --task for this checkpoint; the instruction is dataset-specific")
    training_fps = (
        load_dataset_fps(args.dataset_info, contract.action_feature_names)
        if args.dataset_info is not None
        else None
    )
    actions_per_chunk, fps = _validate_args(args, contract, training_fps)
    recording_plan = _prepare_recording(args, fps=fps, task=task or "")

    print(
        "[PiPER client] Checkpoint contract: "
        f"type={contract.policy_type}, action={contract.action_dim}D absolute joints, "
        f"trained_chunk={contract.chunk_size}, configured_n_action_steps={contract.n_action_steps}, "
        f"deployment_chunk={actions_per_chunk}, fps={fps}, "
        f"training_dataset={contract.training_repo_id}, "
        f"train_config={'present' if contract.has_train_config else 'missing'}, "
        f"checkpoint_compile_model={contract.checkpoint_compile_model}, "
        f"gripper_unit={contract.gripper_action_unit}",
        flush=True,
    )
    if actions_per_chunk != contract.chunk_size:
        print(
            "[PiPER client] Trial override active: inference still predicts the full checkpoint chunk, but only "
            f"the first {actions_per_chunk} actions are returned to the client.",
            flush=True,
        )

    if args.mode == "validate":
        print(
            "[PiPER client] VALIDATE mode complete; ROS, CAN, and policy weights were not opened.", flush=True
        )
        return

    print("[PiPER client] Loading LeRobot client modules...", flush=True)
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.robots.bi_piper import BiPiperConfig

    if recording_plan is not None:
        from lerobot.robots.bi_piper.inference_episode_recorder import (
            BufferedInferenceEpisodeRecorder,
        )

    robot_config = BiPiperConfig(
        id="piper_dual",
        can_left=args.can_left,
        can_right=args.can_right,
        camera_topics=contract.camera_topics,
        action_feature_names=contract.action_feature_names,
        gripper_action_unit=contract.gripper_action_unit,
        gripper_upper_m=args.gripper_range_m,
        velocity=args.velocity,
        max_joint_step_rad=args.max_joint_step_rad,
        max_gripper_step_m=args.max_gripper_step_m,
        trajectory_smoothing=args.trajectory_smoothing,
        policy_fps=fps,
        max_joint_velocity_rad_s=args.max_joint_velocity_rad_s,
        max_joint_acceleration_rad_s2=args.max_joint_acceleration_rad_s2,
        max_gripper_velocity_m_s=args.max_gripper_velocity_m_s,
        max_gripper_acceleration_m_s2=args.max_gripper_acceleration_m_s2,
        dry_run=args.mode == "observe",
        keep_enabled_on_disconnect=True,
    )
    print(
        "[PiPER client] Motion config: "
        f"velocity={args.velocity}, joint_step={args.max_joint_step_rad} rad, "
        f"gripper_step={args.max_gripper_step_m} m, "
        f"trajectory_smoothing={args.trajectory_smoothing}",
        flush=True,
    )
    if recording_plan is not None:
        print(
            "[PiPER client] Recording config: "
            f"dataset={recording_plan.dataset_path}, episode={recording_plan.episode_idx}, "
            f"raw={recording_plan.raw_path}",
            flush=True,
        )
    if args.mode == "hold":
        _hold_only(robot_config)
        return

    telemetry_path = args.telemetry_path
    if telemetry_path is None:
        telemetry_path = Path("logs") / f"piper_actions_{int(time.time())}.jsonl"
    client_config = RobotClientConfig(
        robot=robot_config,
        server_address=args.server_address,
        policy_device="cuda",
        client_device="cpu",
        policy_type=contract.policy_type,
        pretrained_name_or_path=str(checkpoint),
        actions_per_chunk=actions_per_chunk,
        task=task,
        fps=fps,
        chunk_size_threshold=args.prefetch_ratio,
        aggregate_fn_name="latest_only",
        inference_request_timeout_s=args.inference_request_timeout_s,
        force_inference_on_prefetch=True,
        action_commit_steps=args.action_commit_steps,
        action_blend_steps=args.action_blend_steps,
        action_telemetry_path=str(telemetry_path),
    )
    print("[PiPER client] Connecting to ROS cameras and direct PiPER CAN feedback...", flush=True)
    client = RobotClient(client_config)
    print(f"[PiPER client] Cameras and PiPER SDK ready. Mode: {args.mode.upper()}", flush=True)
    print(
        "[PiPER client] Smooth execution: "
        f"prefetch={args.prefetch_ratio:.2f}, one_request_in_flight=true, "
        f"commit={args.action_commit_steps}, blend={args.action_blend_steps}, "
        f"trajectory_limiter={args.trajectory_smoothing}, telemetry={telemetry_path}",
        flush=True,
    )
    print(
        "[PiPER client] Requesting checkpoint load from policy server "
        "(the first 16 GB checkpoint load can take several minutes)...",
        flush=True,
    )
    if not client.start():
        client.stop()
        raise RuntimeError(f"Unable to connect to policy server at {args.server_address}")
    print("[PiPER client] Policy ready. Starting inference loop.", flush=True)

    recorder = None
    stop_event = threading.Event()
    if recording_plan is not None:
        recorder = BufferedInferenceEpisodeRecorder(
            recording_plan.raw_path,
            fps=fps,
            episode_idx=recording_plan.episode_idx,
            action_feature_names=contract.action_feature_names,
            camera_topics=contract.camera_topics,
            gripper_action_unit=contract.gripper_action_unit,
            gripper_lower_m=robot_config.gripper_lower_m,
            gripper_upper_m=robot_config.gripper_upper_m,
        )
        print(
            f"[PiPER client] Recording episode {recording_plan.episode_idx} from the first executed "
            f"action. Press Enter to stop inference and save to {recording_plan.dataset_path}.",
            flush=True,
        )

        def wait_for_stop_key() -> None:
            try:
                input()
            except EOFError:
                return
            stop_event.set()

        threading.Thread(target=wait_for_stop_key, name="piper-record-stop-key", daemon=True).start()

    action_receiver = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver.start()
    try:
        client.control_loop(
            task=task,
            max_actions=args.max_policy_actions,
            frame_callback=(
                None
                if recorder is None
                else lambda observation, action, timestamps, action_time_ns: recorder.append(
                    observation,
                    action,
                    timestamps,
                    action_time_ns=action_time_ns,
                )
            ),
            stop_event=stop_event if recorder is not None else None,
        )
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            client.stop()
            action_receiver.join()
        finally:
            if recorder is not None:
                recorder.close()

    if recording_plan is not None and recorder is not None:
        if recorder.count == 0:
            print(
                "[PiPER client] No policy action was executed; no dataset episode was created.",
                flush=True,
            )
        else:
            print(
                f"[PiPER client] Staged {recorder.count} synchronized action frames at "
                f"{recording_plan.raw_path}.",
                flush=True,
            )
            _convert_recording(recording_plan)


if __name__ == "__main__":
    main()
