#!/usr/bin/env python

import contextlib
import logging
import os
import sys
import threading
import time
import xmlrpc.client
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_bi_piper import BiPiperConfig
from .piper_sdk_arm import PiperSDKArm, model_to_raw, raw_to_model

logger = logging.getLogger(__name__)

JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7)) + ("gripper",)
LEFT_FEATURES = tuple(f"left_{name}" for name in JOINT_NAMES)
RIGHT_FEATURES = tuple(f"right_{name}" for name in JOINT_NAMES)
ACTION_FEATURES = LEFT_FEATURES + RIGHT_FEATURES


class BiPiper(Robot):
    """Control two PiPER arms directly through piper-sdk while reading ROS cameras.

    This adapter is the sole owner of ``can_left`` and ``can_right``. PiPER ROS
    control/feedback nodes must not run at the same time. The three RealSense
    cameras remain ROS topics so the checkpoint input contract is unchanged.
    """

    config_class = BiPiperConfig
    name = "bi_piper"
    # Backward-compatible conversion helpers for deployment scripts and tests.
    raw_to_model = staticmethod(raw_to_model)
    model_to_raw = staticmethod(model_to_raw)

    def __init__(self, config: BiPiperConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._command_lock = threading.RLock()
        self._image_lock = threading.RLock()
        self._arms: dict[str, PiperSDKArm] = {}
        self._images: dict[str, np.ndarray] = {}
        self._image_times: dict[str, float] = {}
        self._image_source_times_ns: dict[str, int] = {}
        self._subscribers: list[Any] = []
        self._rospy: Any = None
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._fatal_error: str | None = None
        self.last_requested_action: dict[str, float] | None = None
        self.last_sent_action: dict[str, float] | None = None
        self.last_measured_action: dict[str, float] | None = None
        self.last_sent_physical_action: np.ndarray | None = None
        self.last_observation_timestamps_ns: dict[str, int] | None = None
        self._trajectory_position: np.ndarray | None = None
        self._trajectory_velocity: np.ndarray | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        state_features = dict.fromkeys(self.config.action_feature_names, float)
        image_features = dict.fromkeys(
            self.config.camera_topics, (self.config.image_height, self.config.image_width, 3)
        )
        return {**state_features, **image_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self.config.action_feature_names, float)

    @property
    def is_connected(self) -> bool:
        if not self._connected or len(self._arms) != 2:
            return False
        return all(arm.is_connected for arm in self._arms.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @staticmethod
    def _import_ros() -> tuple[Any, Any, Any]:
        for path in ("/opt/ros/noetic/lib/python3/dist-packages", "/usr/lib/python3/dist-packages"):
            if path not in sys.path:
                sys.path.append(path)

        try:
            from rosgraph.roslogging import RospyLogger

            RospyLogger.findCaller = logging.Logger.findCaller
            import rosnode
            import rospy
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise ImportError(
                "Direct PiPER deployment still requires ROS Noetic for RealSense images"
            ) from exc
        return rospy, rosnode, Image

    @staticmethod
    def decode_ros_image(message: Any) -> np.ndarray:
        encoding = message.encoding.lower()
        channels_by_encoding = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
        if encoding not in channels_by_encoding:
            raise ValueError(f"Unsupported ROS image encoding: {message.encoding!r}")

        channels = channels_by_encoding[encoding]
        row_bytes = int(message.width) * channels
        if int(message.step) < row_bytes:
            raise ValueError("ROS image step is smaller than the expected row size")
        flat = np.frombuffer(message.data, dtype=np.uint8)
        expected = int(message.height) * int(message.step)
        if flat.size < expected:
            raise ValueError(f"ROS image payload has {flat.size} bytes, expected at least {expected}")

        rows = flat[:expected].reshape(int(message.height), int(message.step))[:, :row_bytes]
        image = rows.reshape(int(message.height), int(message.width), channels)
        if encoding == "mono8":
            image = np.repeat(image, 3, axis=2)
        elif encoding in {"rgba8", "bgra8"}:
            image = image[:, :, :3]
        if encoding in {"bgr8", "bgra8"}:
            image = image[:, :, ::-1]
        return np.ascontiguousarray(image)

    def _image_callback(self, message: Any, name: str) -> None:
        try:
            image = self.decode_ros_image(message)
        except (TypeError, ValueError):
            logger.exception("Failed to decode ROS camera frame %s", name)
            return
        if image.shape != (self.config.image_height, self.config.image_width, 3):
            logger.error("Ignoring %s image with unexpected shape %s", name, image.shape)
            return
        with self._image_lock:
            self._images[name] = image
            self._image_times[name] = time.monotonic()
            try:
                self._image_source_times_ns[name] = int(message.header.stamp.to_nsec())
            except (AttributeError, TypeError, ValueError):
                self._image_source_times_ns[name] = time.time_ns()

    def _check_exclusive_can_ownership(self, rosnode: Any) -> None:
        if not self.config.reject_piper_ros_nodes:
            return
        nodes = rosnode.get_node_names()
        conflicts = sorted(
            node
            for node in nodes
            if any(fragment in node for fragment in self.config.conflicting_ros_node_fragments)
        )
        if conflicts:
            raise RuntimeError(
                "Direct PiPER control requires exclusive CAN ownership. Stop the PiPER ROS nodes first: "
                + ", ".join(conflicts)
            )

    def _check_exclusive_can_ownership_without_ros(self) -> None:
        """Reject known PiPER ROS CAN owners without requiring a ROS node.

        A replay-only adapter has no camera topics and therefore should not
        require a ROS master.  If a master is available, query its system state
        directly so the same arm-node exclusion still applies.
        """
        if not self.config.reject_piper_ros_nodes:
            return
        master_uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
        try:
            code, _, state = xmlrpc.client.ServerProxy(master_uri).getSystemState(
                "/lerobot_bi_piper_direct_replay"
            )
        except (OSError, xmlrpc.client.Error):
            return
        if code != 1:
            return
        nodes = {
            node
            for registrations in state
            for _, registered_nodes in registrations
            for node in registered_nodes
        }
        conflicts = sorted(
            node
            for node in nodes
            if any(fragment in node for fragment in self.config.conflicting_ros_node_fragments)
        )
        if conflicts:
            raise RuntimeError(
                "Direct PiPER control requires exclusive CAN ownership. Stop the PiPER ROS nodes first: "
                + ", ".join(conflicts)
            )

    def _read_side(self, side: str) -> np.ndarray:
        try:
            return self._arms[side].state()
        except RuntimeError as exc:
            raise DeviceNotConnectedError(f"{side} arm: {exc}") from exc

    def _read_both(self) -> np.ndarray:
        return np.concatenate((self._read_side("left"), self._read_side("right")))

    def _physical_to_policy_units(self, values: np.ndarray) -> np.ndarray:
        converted = np.asarray(values, dtype=np.float64).copy()
        if self.config.gripper_action_unit == "open_scale":
            span = self.config.gripper_upper_m - self.config.gripper_lower_m
            converted[[6, 13]] = (converted[[6, 13]] - self.config.gripper_lower_m) / span
        return converted

    def _policy_to_physical_units(self, values: np.ndarray) -> np.ndarray:
        converted = np.asarray(values, dtype=np.float64).copy()
        if self.config.gripper_action_unit == "open_scale":
            span = self.config.gripper_upper_m - self.config.gripper_lower_m
            converted[[6, 13]] = self.config.gripper_lower_m + converted[[6, 13]] * span
        return converted

    def _enable_status(self) -> dict[str, list[bool]]:
        return {side: arm.enable_status() for side, arm in self._arms.items()}

    @staticmethod
    def _all_enabled(status: dict[str, list[bool]]) -> bool:
        return bool(status) and all(len(values) == 6 and all(values) for values in status.values())

    def _confirmed_enable_loss_locked(self) -> tuple[bool, dict[str, list[bool]]]:
        """Filter a one-frame SDK status glitch without masking sustained enable loss."""
        status: dict[str, list[bool]] = {}
        for attempt in range(self.config.enable_loss_confirmations):
            status = self._enable_status()
            if self._all_enabled(status):
                return False, status
            if attempt + 1 < self.config.enable_loss_confirmations:
                time.sleep(self.config.enable_loss_confirmation_interval_s)
        return True, status

    def _send_locked(self, left: np.ndarray, right: np.ndarray) -> None:
        self._arms["left"].command(left)
        self._arms["right"].command(right)

    def _hold_current_locked(self) -> None:
        current = self._read_both()
        self._send_locked(current[:7], current[7:])
        self._reset_trajectory(current)

    def _reset_trajectory(self, current: np.ndarray | None = None) -> None:
        self._trajectory_position = None if current is None else np.asarray(current, dtype=np.float64).copy()
        self._trajectory_velocity = None if current is None else np.zeros(14, dtype=np.float64)

    def _ensure_enabled_locked(self, *, hold_after: bool) -> None:
        deadline = time.monotonic() + self.config.enable_timeout_s
        last_status: dict[str, list[bool]] = {}
        while time.monotonic() < deadline:
            last_status = self._enable_status()
            if self._all_enabled(last_status):
                if hold_after:
                    self._hold_current_locked()
                return
            for side, arm in self._arms.items():
                values = last_status.get(side, [])
                if len(values) != 6 or not all(values):
                    arm.request_enable()
            time.sleep(self.config.enable_retry_interval_s)
        raise RuntimeError(f"PiPER motor enable timed out; driver status: {last_status}")

    def _watchdog_loop(self) -> None:
        interval = 1.0 / self.config.enable_watchdog_hz
        while not self._watchdog_stop.wait(interval):
            try:
                with self._command_lock:
                    enable_lost, status = self._confirmed_enable_loss_locked()
                    if enable_lost:
                        logger.critical(
                            "PiPER enable lost; recovering at current pose and aborting this rollout"
                        )
                        self._ensure_enabled_locked(hold_after=True)
                        self._fatal_error = (
                            "PiPER motor enable was lost during inference. The arms were re-enabled at the "
                            "measured pose; inspect the hardware and restart the client."
                        )
                        return
            except Exception as exc:
                self._fatal_error = f"PiPER enable watchdog failed: {exc}"
                logger.exception(self._fatal_error)
                return

    def _release_ros_handles(self) -> None:
        for handle in self._subscribers:
            with contextlib.suppress(Exception):
                handle.unregister()
        self._subscribers.clear()

    def _disconnect_arms(self) -> None:
        for arm in self._arms.values():
            with contextlib.suppress(Exception):
                arm.disconnect()
        self._arms.clear()

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self._fatal_error = None
        self.last_observation_timestamps_ns = None
        self._reset_trajectory()
        with self._image_lock:
            self._images.clear()
            self._image_times.clear()
            self._image_source_times_ns.clear()
        if self.config.camera_topics:
            rospy, rosnode, image_cls = self._import_ros()
            self._rospy = rospy
            if not rospy.core.is_initialized():
                rospy.init_node(self.config.ros_node_name, anonymous=True, disable_signals=True)
            self._check_exclusive_can_ownership(rosnode)

            for name, topic in self.config.camera_topics.items():
                self._subscribers.append(
                    rospy.Subscriber(
                        topic,
                        image_cls,
                        self._image_callback,
                        callback_args=name,
                        queue_size=self.config.subscriber_queue_size,
                        tcp_nodelay=True,
                    )
                )
        else:
            self._check_exclusive_can_ownership_without_ros()

        try:
            self._arms = {
                "left": PiperSDKArm(
                    self.config.can_left,
                    velocity=self.config.velocity,
                    gripper_effort=self.config.gripper_effort,
                    max_feedback_age_s=self.config.max_feedback_age_s,
                ),
                "right": PiperSDKArm(
                    self.config.can_right,
                    velocity=self.config.velocity,
                    gripper_effort=self.config.gripper_effort,
                    max_feedback_age_s=self.config.max_feedback_age_s,
                ),
            }
            for arm in self._arms.values():
                arm.connect()

            deadline = time.monotonic() + self.config.connect_timeout_s
            while time.monotonic() < deadline:
                try:
                    self._read_both()
                    break
                except DeviceNotConnectedError:
                    pass
                time.sleep(0.05)
            else:
                raise TimeoutError("Timed out waiting for direct PiPER joint feedback")

            if not self.config.dry_run:
                with self._command_lock:
                    self._ensure_enabled_locked(hold_after=True)

            if self.config.camera_topics:
                deadline = time.monotonic() + self.config.connect_timeout_s
                while time.monotonic() < deadline:
                    with self._image_lock:
                        cameras_ready = set(self._images) == set(self.config.camera_topics)
                    if cameras_ready:
                        break
                    time.sleep(0.05)
                else:
                    raise TimeoutError("Timed out waiting for the three ROS RealSense RGB streams")

            self._connected = True
            if not self.config.dry_run:
                self._watchdog_stop.clear()
                self._watchdog_thread = threading.Thread(
                    target=self._watchdog_loop, name="bi-piper-enable-watchdog", daemon=True
                )
                self._watchdog_thread.start()
        except Exception:
            self._release_ros_handles()
            self._disconnect_arms()
            raise

        mode = "DRY-RUN (direct CAN commands suppressed)" if self.config.dry_run else "LIVE DIRECT CAN"
        logger.info("BiPiper connected in %s mode", mode)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self._fatal_error is not None:
            raise DeviceNotConnectedError(self._fatal_error)
        with self._command_lock:
            state = self._read_both()
            state_time_ns = time.time_ns()
        with self._image_lock:
            images = {name: image.copy() for name, image in self._images.items()}
            image_times = dict(self._image_times)
            image_source_times_ns = dict(self._image_source_times_ns)
        now = time.monotonic()
        stale_images = [
            name for name, stamp in image_times.items() if now - stamp > self.config.max_image_age_s
        ]
        if stale_images:
            raise DeviceNotConnectedError(f"Stale PiPER camera data: {', '.join(stale_images)}")

        observation: RobotObservation = {
            name: float(value)
            for name, value in zip(
                self.config.action_feature_names,
                self._physical_to_policy_units(state),
                strict=True,
            )
        }
        observation.update(images)
        self.last_observation_timestamps_ns = {
            **image_source_times_ns,
            "state": state_time_ns,
        }
        return observation

    def _clip_action(self, requested: np.ndarray, current: np.ndarray) -> np.ndarray:
        lower_one_arm = np.asarray((*self.config.joint_lower_limits_rad, self.config.gripper_lower_m))
        upper_one_arm = np.asarray((*self.config.joint_upper_limits_rad, self.config.gripper_upper_m))
        lower = np.tile(lower_one_arm, 2)
        upper = np.tile(upper_one_arm, 2)
        bounded = np.clip(requested, lower, upper)
        slew_one_arm = np.asarray(
            (*(self.config.max_joint_step_rad for _ in range(6)), self.config.max_gripper_step_m)
        )
        slew = np.tile(slew_one_arm, 2)
        if not self.config.trajectory_smoothing:
            return np.clip(bounded, current - slew, current + slew)

        if self._trajectory_position is None or self._trajectory_velocity is None:
            self._reset_trajectory(current)
        assert self._trajectory_position is not None
        assert self._trajectory_velocity is not None

        dt = 1.0 / self.config.policy_fps
        velocity_one_arm = np.asarray(
            (*(self.config.max_joint_velocity_rad_s for _ in range(6)), self.config.max_gripper_velocity_m_s)
        )
        acceleration_one_arm = np.asarray(
            (
                *(self.config.max_joint_acceleration_rad_s2 for _ in range(6)),
                self.config.max_gripper_acceleration_m_s2,
            )
        )
        max_velocity = np.tile(velocity_one_arm, 2)
        max_acceleration = np.tile(acceleration_one_arm, 2)

        error = bounded - self._trajectory_position
        desired_velocity = np.clip(error / dt, -max_velocity, max_velocity)
        velocity = np.clip(
            desired_velocity,
            self._trajectory_velocity - max_acceleration * dt,
            self._trajectory_velocity + max_acceleration * dt,
        )
        step = velocity * dt
        # Snap only when moving toward the target would pass it. If a newly
        # requested target is behind the current motion, retain the limited
        # velocity and decelerate across subsequent frames instead of jumping.
        overshoot = (step * error >= 0) & (np.abs(step) > np.abs(error))
        step[overshoot] = error[overshoot]

        candidate = self._trajectory_position + step
        # Measured-relative clipping is retained only as the final hardware
        # safety envelope. Under normal tracking it does not shape the path.
        safe = np.clip(candidate, current - slew, current + slew)
        safe = np.clip(safe, lower, upper)
        self._trajectory_velocity = (safe - self._trajectory_position) / dt
        self._trajectory_position = safe.copy()
        return safe

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self._fatal_error is not None:
            raise DeviceNotConnectedError(self._fatal_error)
        missing = [name for name in self.config.action_feature_names if name not in action]
        if missing:
            raise ValueError(f"PiPER action is missing features: {missing}")
        requested_policy = np.asarray(
            [action[name] for name in self.config.action_feature_names], dtype=np.float64
        )
        if not np.isfinite(requested_policy).all():
            raise ValueError("PiPER action contains NaN or infinity")
        requested_physical = self._policy_to_physical_units(requested_policy)

        with self._command_lock:
            if self._fatal_error is not None:
                raise DeviceNotConnectedError(self._fatal_error)
            current = self._read_both()
            measured_policy = self._physical_to_policy_units(current)
            self.last_measured_action = {
                name: float(value)
                for name, value in zip(self.config.action_feature_names, measured_policy, strict=True)
            }
            safe_physical = self._clip_action(requested_physical, current)
            if not self.config.dry_run:
                enable_lost, _ = self._confirmed_enable_loss_locked()
                if enable_lost:
                    self._reset_trajectory(current)
                    self._ensure_enabled_locked(hold_after=True)
                    self._fatal_error = (
                        "PiPER motor enable was lost before an action. The arms were re-enabled at the "
                        "measured pose; the old action queue must not continue."
                    )
                    raise DeviceNotConnectedError(self._fatal_error)
                self._send_locked(safe_physical[:7], safe_physical[7:])

        safe_policy = self._physical_to_policy_units(safe_physical)
        sent = {
            name: float(value)
            for name, value in zip(self.config.action_feature_names, safe_policy, strict=True)
        }
        self.last_requested_action = {
            name: float(value)
            for name, value in zip(self.config.action_feature_names, requested_policy, strict=True)
        }
        self.last_sent_action = sent
        self.last_sent_physical_action = safe_physical.copy()
        return sent

    @check_if_not_connected
    def hold_current(self) -> RobotAction:
        """Continuously callable measured-pose hold used between live actions.

        A single MOVE J target is not a durable holding guarantee on every
        controller firmware. Callers that need the arm to remain supported
        should invoke this method periodically while the CAN connection stays
        open.
        """
        with self._command_lock:
            enable_lost, _ = self._confirmed_enable_loss_locked()
            if enable_lost:
                self._ensure_enabled_locked(hold_after=False)
            current = self._read_both()
            self._send_locked(current[:7], current[7:])
            self._reset_trajectory(current)
        policy = self._physical_to_policy_units(current)
        return {
            name: float(value) for name, value in zip(self.config.action_feature_names, policy, strict=True)
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=2.0)
            self._watchdog_thread = None

        try:
            with self._command_lock:
                if not self.config.dry_run:
                    with contextlib.suppress(Exception):
                        if self._all_enabled(self._enable_status()):
                            self._hold_current_locked()
                if not self.config.keep_enabled_on_disconnect:
                    for arm in self._arms.values():
                        with contextlib.suppress(Exception):
                            arm.disconnect(disable=True)
        finally:
            self._release_ros_handles()
            self._disconnect_arms()
            self._connected = False
        logger.info(
            "BiPiper disconnected; %s",
            (
                "DisableArm was not sent, but CAN closure may still cause controller-side enable loss"
                if self.config.keep_enabled_on_disconnect
                else "DisableArm was requested"
            ),
        )
