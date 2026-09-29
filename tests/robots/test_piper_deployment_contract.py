import json
import runpy
from pathlib import Path

import pytest

CLIENT = Path(__file__).parents[2] / "examples" / "piper" / "async_policy_client.py"
CLIENT_GLOBALS = runpy.run_path(str(CLIENT))
EXPECTED_ACTION_NAMES = CLIENT_GLOBALS["EXPECTED_ACTION_NAMES"]
LEGACY_SCALE_ACTION_NAMES = CLIENT_GLOBALS["LEGACY_SCALE_ACTION_NAMES"]
DEFAULT_DATASET_INFO = CLIENT_GLOBALS["DEFAULT_DATASET_INFO"]
load_checkpoint_contract = CLIENT_GLOBALS["load_checkpoint_contract"]
load_dataset_fps = CLIENT_GLOBALS["load_dataset_fps"]
parse_args = CLIENT_GLOBALS["parse_args"]


def _write_checkpoint(
    root: Path,
    *,
    action_shape: list[int] | None = None,
    policy_type: str = "pi05",
    action_names: tuple[str, ...] = EXPECTED_ACTION_NAMES,
    state_shape: list[int] | None = None,
    rename_map: dict[str, str] | None = None,
    include_train_config: bool = True,
) -> Path:
    root.mkdir()
    rename_map = (
        rename_map
        if rename_map is not None
        else {
            "observation.images.cam_high": "observation.images.base_0_rgb",
            "observation.images.cam_left_wrist": "observation.images.left_wrist_0_rgb",
            "observation.images.cam_right_wrist": "observation.images.right_wrist_0_rgb",
        }
    )
    config = {
        "type": policy_type,
        "chunk_size": 50,
        "n_action_steps": 10,
        "max_action_dim": 32,
        "use_relative_actions": False,
        "action_feature_names": list(action_names),
        "input_features": {
            "observation.images.base_0_rgb": {"shape": [3, 224, 224]},
            "observation.images.left_wrist_0_rgb": {"shape": [3, 224, 224]},
            "observation.images.right_wrist_0_rgb": {"shape": [3, 224, 224]},
            "observation.state": {"shape": state_shape or [32]},
        },
        "output_features": {"action": {"shape": action_shape or [14]}},
    }
    preprocessor = {
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": rename_map},
            },
            {"registry_name": "normalizer_processor", "config": {}},
        ]
    }
    postprocessor = {"steps": [{"registry_name": "unnormalizer_processor", "config": {}}]}
    train_config = {
        "dataset": {"repo_id": "test_dataset"},
        "policy": dict(config),
        "rename_map": preprocessor["steps"][0]["config"]["rename_map"],
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if include_train_config:
        (root / "train_config.json").write_text(json.dumps(train_config), encoding="utf-8")
    (root / "policy_preprocessor.json").write_text(json.dumps(preprocessor), encoding="utf-8")
    (root / "policy_postprocessor.json").write_text(json.dumps(postprocessor), encoding="utf-8")
    for filename in (
        "model.safetensors",
        "policy_preprocessor_step_1_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        (root / filename).write_bytes(b"test")
    return root


def test_pi05_checkpoint_contract_maps_training_camera_names(tmp_path):
    contract = load_checkpoint_contract(_write_checkpoint(tmp_path / "checkpoint"))

    assert contract.policy_type == "pi05"
    assert contract.action_dim == 14
    assert contract.chunk_size == 50
    assert contract.n_action_steps == 10
    assert contract.training_repo_id == "test_dataset"
    assert contract.checkpoint_compile_model is False
    assert contract.gripper_action_unit == "meters"
    assert contract.action_feature_names == EXPECTED_ACTION_NAMES
    assert contract.camera_topics == {
        "cam_high": "/camera_f/color/image_raw",
        "cam_left_wrist": "/camera_l/color/image_raw",
        "cam_right_wrist": "/camera_r/color/image_raw",
    }


def test_checkpoint_contract_rejects_non_joint_action_dimension(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint", action_shape=[32])

    with pytest.raises(ValueError, match="requires a real 14D action"):
        load_checkpoint_contract(checkpoint)


def test_pi0_legacy_scale_contract_selects_unit_adapter(tmp_path):
    checkpoint = _write_checkpoint(
        tmp_path / "checkpoint",
        policy_type="pi0",
        action_names=LEGACY_SCALE_ACTION_NAMES,
    )

    contract = load_checkpoint_contract(checkpoint)

    assert contract.policy_type == "pi0"
    assert contract.gripper_action_unit == "open_scale"
    assert contract.n_action_steps == 10


def test_pi0_checkpoint_without_train_config_uses_direct_policy_camera_names(tmp_path):
    checkpoint = _write_checkpoint(
        tmp_path / "checkpoint",
        policy_type="pi0",
        action_names=LEGACY_SCALE_ACTION_NAMES,
        state_shape=[14],
        rename_map={},
        include_train_config=False,
    )

    contract = load_checkpoint_contract(checkpoint)

    assert contract.has_train_config is False
    assert contract.training_repo_id == "<not recorded>"
    assert contract.camera_topics == {
        "base_0_rgb": "/camera_f/color/image_raw",
        "left_wrist_0_rgb": "/camera_l/color/image_raw",
        "right_wrist_0_rgb": "/camera_r/color/image_raw",
    }


def test_dataset_fps_and_joint_schema_are_read_from_metadata(tmp_path):
    features = {
        "observation.state": {"shape": [14], "names": list(EXPECTED_ACTION_NAMES)},
        "action": {"shape": [14], "names": list(EXPECTED_ACTION_NAMES)},
    }
    for key in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        features[key] = {"info": {"video.fps": 30}}
    info_path = tmp_path / "info.json"
    info_path.write_text(json.dumps({"fps": 30, "features": features}), encoding="utf-8")

    assert load_dataset_fps(info_path, EXPECTED_ACTION_NAMES) == 30


def test_inference_cli_defaults_to_original_speed_behavior(monkeypatch):
    monkeypatch.setattr("sys.argv", [str(CLIENT)])

    args = parse_args()

    assert args.velocity == 30
    assert args.max_joint_step_rad == 0.05
    assert args.max_gripper_step_m == 0.005
    assert args.trajectory_smoothing is False
    assert args.dataset_info == DEFAULT_DATASET_INFO


def test_inference_cli_speed_and_recording_data_are_configurable(monkeypatch, tmp_path):
    dataset_path = tmp_path / "inference_dataset"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(CLIENT),
            "--velocity",
            "45",
            "--max-joint-step-rad",
            "0.08",
            "--trajectory-smoothing",
            "--record-dataset-path",
            str(dataset_path),
            "--record-repo-id",
            "local/custom",
            "--record-episode-idx",
            "7",
        ],
    )

    args = parse_args()

    assert args.velocity == 45
    assert args.max_joint_step_rad == 0.08
    assert args.trajectory_smoothing is True
    assert args.record_dataset_path == dataset_path
    assert args.record_repo_id == "local/custom"
    assert args.record_episode_idx == 7
