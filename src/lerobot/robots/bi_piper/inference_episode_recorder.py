#!/usr/bin/env python
"""Stage one direct-CAN PiPER inference rollout for LeRobot v2.1 conversion."""

from __future__ import annotations

import time
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

import numpy as np

CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CAMERA_TOPICS = {
    "cam_high": "/camera_f/color/image_raw",
    "cam_left_wrist": "/camera_l/color/image_raw",
    "cam_right_wrist": "/camera_r/color/image_raw",
}
TIMESTAMP_LABELS = (
    "image_front",
    "image_left",
    "image_right",
    "master_left",
    "master_right",
    "puppet_left",
    "puppet_right",
)
_STOP_WRITER = object()


class InferenceEpisodeRecorder:
    """Write observations and post-safety commands using the collector's HDF5 schema.

    The direct SDK does not expose joint velocity or effort in the interface used
    by this deployment. Velocity is therefore computed from consecutive measured
    states and effort is explicitly zero-filled. These provenance facts are kept
    as HDF5 attributes rather than silently presenting both fields as SDK data.
    """

    def __init__(
        self,
        path: Path,
        *,
        fps: int,
        episode_idx: int,
        action_feature_names: tuple[str, ...],
        camera_topics: dict[str, str],
        gripper_action_unit: str,
        gripper_lower_m: float,
        gripper_upper_m: float,
        allocation_frames: int = 300,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if episode_idx < 0:
            raise ValueError("episode_idx must be non-negative")
        if len(action_feature_names) != 14:
            raise ValueError("PiPER inference recording requires 14 state/action features")
        if gripper_action_unit not in {"meters", "open_scale"}:
            raise ValueError(f"Unsupported gripper action unit: {gripper_action_unit}")
        if gripper_lower_m >= gripper_upper_m:
            raise ValueError("gripper_lower_m must be smaller than gripper_upper_m")
        if allocation_frames <= 0:
            raise ValueError("allocation_frames must be positive")

        source_by_topic = {topic: name for name, topic in camera_topics.items()}
        missing_topics = [topic for topic in CAMERA_TOPICS.values() if topic not in source_by_topic]
        if missing_topics:
            raise ValueError(f"Camera topics do not cover the three recording streams: {missing_topics}")

        self.path = path.expanduser().resolve()
        self.fps = fps
        self.episode_idx = episode_idx
        self.action_feature_names = action_feature_names
        self.gripper_action_unit = gripper_action_unit
        self.gripper_lower_m = gripper_lower_m
        self.gripper_upper_m = gripper_upper_m
        self.allocation_frames = allocation_frames
        self.camera_source_keys = {
            dataset_key: source_by_topic[topic] for dataset_key, topic in CAMERA_TOPICS.items()
        }
        self.count = 0
        self._capacity = 0
        self._file: Any | None = None
        self._previous_state: np.ndarray | None = None
        self._previous_state_time_ns: int | None = None

    def _policy_to_physical(self, values: dict[str, Any]) -> np.ndarray:
        vector = np.asarray([values[name] for name in self.action_feature_names], dtype=np.float32)
        if vector.shape != (14,) or not np.isfinite(vector).all():
            raise ValueError("Recorded PiPER state/action must be a finite 14D vector")
        if self.gripper_action_unit == "open_scale":
            span = self.gripper_upper_m - self.gripper_lower_m
            vector[[6, 13]] = self.gripper_lower_m + vector[[6, 13]] * span
        return vector

    def _open(self, images: list[np.ndarray]) -> None:
        import h5py

        expected_image_shape = images[0].shape
        for key, image in zip(CAMERA_KEYS, images, strict=True):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"{key} expected an HWC RGB image, got {image.shape}")
            if image.shape != expected_image_shape:
                raise ValueError(
                    f"All three cameras must have one resolution; {key} has {image.shape}, "
                    f"expected {expected_image_shape}"
                )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._capacity = self.allocation_frames
        self._file = h5py.File(self.path, "x")
        self._file.attrs["format"] = "piper_ros_capture_staging_v1"
        self._file.attrs["fps"] = self.fps
        self._file.attrs["episode_idx"] = self.episode_idx
        self._file.attrs["capture_source"] = "piper_direct_can_inference"
        self._file.attrs["action_source"] = "post_safety_command"
        self._file.attrs["qvel_source"] = "finite_difference_from_qpos"
        self._file.attrs["effort_source"] = "unavailable_zero_filled"
        self._file.attrs["timestamp_labels"] = np.asarray(TIMESTAMP_LABELS, dtype=h5py.string_dtype())

        image_group = self._file.create_group("images")
        for key, image in zip(CAMERA_KEYS, images, strict=True):
            image_group.create_dataset(
                key,
                shape=(self._capacity, *image.shape),
                maxshape=(None, *image.shape),
                chunks=(1, *image.shape),
                compression="lzf",
                dtype=np.uint8,
            )
        for key in ("qpos", "qvel", "effort", "action"):
            self._file.create_dataset(
                key,
                shape=(self._capacity, 14),
                maxshape=(None, 14),
                chunks=(min(256, self._capacity), 14),
                dtype=np.float32,
            )
        self._file.create_dataset(
            "timestamps_ns",
            shape=(self._capacity, len(TIMESTAMP_LABELS)),
            maxshape=(None, len(TIMESTAMP_LABELS)),
            chunks=(min(256, self._capacity), len(TIMESTAMP_LABELS)),
            dtype=np.int64,
        )

    def _grow(self) -> None:
        assert self._file is not None
        self._capacity += self.allocation_frames
        for dataset in self._file.values():
            if hasattr(dataset, "resize"):
                dataset.resize(self._capacity, axis=0)
            else:
                for child in dataset.values():
                    child.resize(self._capacity, axis=0)

    def append(
        self,
        observation: dict[str, Any],
        performed_action: dict[str, Any],
        source_timestamps_ns: dict[str, int] | None,
        *,
        action_time_ns: int | None = None,
    ) -> None:
        images = [
            np.ascontiguousarray(observation[self.camera_source_keys[key]], dtype=np.uint8)
            for key in CAMERA_KEYS
        ]
        state = self._policy_to_physical(observation)
        action = self._policy_to_physical(performed_action)
        timestamps = source_timestamps_ns or {}
        state_time_ns = int(timestamps.get("state", time.time_ns()))
        command_time_ns = time.time_ns() if action_time_ns is None else int(action_time_ns)

        if self._previous_state is None or self._previous_state_time_ns is None:
            velocity = np.zeros(14, dtype=np.float32)
        else:
            dt = (state_time_ns - self._previous_state_time_ns) / 1_000_000_000
            velocity = (
                np.zeros(14, dtype=np.float32)
                if dt <= 0
                else np.asarray((state - self._previous_state) / dt, dtype=np.float32)
            )
        self._previous_state = state.copy()
        self._previous_state_time_ns = state_time_ns

        if self._file is None:
            self._open(images)
        assert self._file is not None
        if self.count >= self._capacity:
            self._grow()

        for key, image in zip(CAMERA_KEYS, images, strict=True):
            target = self._file[f"images/{key}"]
            if image.shape != target.shape[1:]:
                raise ValueError(f"{key} image shape changed from {target.shape[1:]} to {image.shape}")
            target[self.count] = image
        self._file["qpos"][self.count] = state
        self._file["qvel"][self.count] = velocity
        self._file["effort"][self.count] = np.zeros(14, dtype=np.float32)
        self._file["action"][self.count] = action
        self._file["timestamps_ns"][self.count] = np.asarray(
            [int(timestamps.get(self.camera_source_keys[key], state_time_ns)) for key in CAMERA_KEYS]
            + [command_time_ns, command_time_ns, state_time_ns, state_time_ns],
            dtype=np.int64,
        )
        self.count += 1
        if self.count % self.fps == 0:
            self._file.flush()

    def close(self) -> None:
        if self._file is None:
            return
        for dataset in self._file.values():
            if hasattr(dataset, "resize"):
                dataset.resize(self.count, axis=0)
            else:
                for child in dataset.values():
                    child.resize(self.count, axis=0)
        self._file.attrs["num_frames"] = self.count
        self._file.flush()
        self._file.close()
        self._file = None

    def __enter__(self) -> InferenceEpisodeRecorder:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


class BufferedInferenceEpisodeRecorder:
    """Move HDF5 compression and writes off the robot's 30 Hz control thread."""

    def __init__(self, *args: Any, queue_frames: int = 30, **kwargs: Any) -> None:
        if queue_frames <= 0:
            raise ValueError("queue_frames must be positive")
        self.writer = InferenceEpisodeRecorder(*args, **kwargs)
        self._queue: Queue[Any] = Queue(maxsize=queue_frames)
        self._error: BaseException | None = None
        self._closed = False
        self._thread = Thread(target=self._write_loop, name="piper-inference-hdf5", daemon=True)
        self._thread.start()

    @property
    def count(self) -> int:
        return self.writer.count

    @property
    def path(self) -> Path:
        return self.writer.path

    def _write_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP_WRITER:
                    return
                if self._error is None:
                    try:
                        observation, performed_action, timestamps, action_time_ns = item
                        self.writer.append(
                            observation,
                            performed_action,
                            timestamps,
                            action_time_ns=action_time_ns,
                        )
                    except BaseException as exc:
                        self._error = exc
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Inference HDF5 writer failed") from self._error

    def append(
        self,
        observation: dict[str, Any],
        performed_action: dict[str, Any],
        source_timestamps_ns: dict[str, int] | None,
        *,
        action_time_ns: int | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Inference HDF5 writer is already closed")
        self._raise_if_failed()
        self._queue.put((observation, performed_action, source_timestamps_ns, action_time_ns))
        self._raise_if_failed()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP_WRITER)
        self._thread.join()
        try:
            self.writer.close()
        finally:
            self._raise_if_failed()

    def __enter__(self) -> BufferedInferenceEpisodeRecorder:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
