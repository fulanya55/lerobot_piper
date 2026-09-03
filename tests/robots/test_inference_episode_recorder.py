from __future__ import annotations

import h5py
import numpy as np

from lerobot.robots.bi_piper.inference_episode_recorder import (
    BufferedInferenceEpisodeRecorder,
    InferenceEpisodeRecorder,
)


def test_inference_episode_recorder_matches_v21_staging_contract(tmp_path):
    action_names = tuple(
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
    camera_topics = {
        "base_0_rgb": "/camera_f/color/image_raw",
        "left_wrist_0_rgb": "/camera_l/color/image_raw",
        "right_wrist_0_rgb": "/camera_r/color/image_raw",
    }
    path = tmp_path / "episode_0.hdf5"
    recorder = InferenceEpisodeRecorder(
        path,
        fps=30,
        episode_idx=0,
        action_feature_names=action_names,
        camera_topics=camera_topics,
        gripper_action_unit="open_scale",
        gripper_lower_m=0.0,
        gripper_upper_m=0.08,
        allocation_frames=1,
    )

    def frame(value: float):
        observation = dict.fromkeys(action_names, value)
        observation.update(
            {
                "base_0_rgb": np.full((4, 6, 3), 10, dtype=np.uint8),
                "left_wrist_0_rgb": np.full((4, 6, 3), 20, dtype=np.uint8),
                "right_wrist_0_rgb": np.full((4, 6, 3), 30, dtype=np.uint8),
            }
        )
        action = dict.fromkeys(action_names, value + 0.1)
        return observation, action

    first_observation, first_action = frame(0.5)
    recorder.append(
        first_observation,
        first_action,
        {"base_0_rgb": 1, "left_wrist_0_rgb": 2, "right_wrist_0_rgb": 3, "state": 10},
        action_time_ns=20,
    )
    second_observation, second_action = frame(0.6)
    recorder.append(
        second_observation,
        second_action,
        {"base_0_rgb": 4, "left_wrist_0_rgb": 5, "right_wrist_0_rgb": 6, "state": 1_000_000_010},
        action_time_ns=1_000_000_020,
    )
    recorder.close()

    with h5py.File(path, "r") as episode:
        assert episode.attrs["format"] == "piper_ros_capture_staging_v1"
        assert episode.attrs["num_frames"] == 2
        assert episode.attrs["action_source"] == "post_safety_command"
        assert episode["qpos"].shape == (2, 14)
        assert episode["images/cam_high"].shape == (2, 4, 6, 3)
        np.testing.assert_allclose(episode["qpos"][:, [6, 13]], [[0.04, 0.04], [0.048, 0.048]])
        np.testing.assert_allclose(episode["action"][:, [6, 13]], [[0.048, 0.048], [0.056, 0.056]])
        np.testing.assert_allclose(episode["qvel"][0], 0)
        np.testing.assert_allclose(episode["qvel"][1, :6], 0.1, atol=1e-7)
        np.testing.assert_allclose(episode["qvel"][1, [6, 13]], 0.008, atol=1e-7)
        np.testing.assert_allclose(episode["effort"][:], 0)
        np.testing.assert_array_equal(
            episode["timestamps_ns"][:],
            [
                [1, 2, 3, 20, 20, 10, 10],
                [4, 5, 6, 1_000_000_020, 1_000_000_020, 1_000_000_010, 1_000_000_010],
            ],
        )


def test_buffered_recorder_drains_all_frames_before_close(tmp_path):
    action_names = tuple(
        f"{side}_{name}"
        for side in ("left", "right")
        for name in (
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper",
        )
    )
    camera_topics = {
        "cam_high": "/camera_f/color/image_raw",
        "cam_left_wrist": "/camera_l/color/image_raw",
        "cam_right_wrist": "/camera_r/color/image_raw",
    }
    path = tmp_path / "buffered.hdf5"
    recorder = BufferedInferenceEpisodeRecorder(
        path,
        fps=30,
        episode_idx=0,
        action_feature_names=action_names,
        camera_topics=camera_topics,
        gripper_action_unit="meters",
        gripper_lower_m=0.0,
        gripper_upper_m=0.08,
        allocation_frames=1,
        queue_frames=1,
    )
    for index in range(3):
        observation = dict.fromkeys(action_names, float(index))
        observation.update({camera: np.full((4, 6, 3), index, dtype=np.uint8) for camera in camera_topics})
        recorder.append(
            observation,
            dict.fromkeys(action_names, float(index + 1)),
            {**dict.fromkeys(camera_topics, index), "state": index + 1},
            action_time_ns=index + 2,
        )
    recorder.close()

    assert recorder.count == 3
    with h5py.File(path, "r") as episode:
        assert episode.attrs["num_frames"] == 3
        np.testing.assert_allclose(episode["qpos"][:, 0], [0, 1, 2])
