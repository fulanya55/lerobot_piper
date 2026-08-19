#!/usr/bin/env python
"""Framework-independent PiPER SDK transport.

This module deliberately has no LeRobot or ROS imports. A deployment adapter
for another policy framework (for example StarVLA) can reuse ``PiperSDKArm``
without pulling in the async client or camera stack.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

RAD_TO_001_DEG = 180000.0 / math.pi
M_TO_001_MM = 1_000_000.0


def raw_to_model(raw: np.ndarray) -> np.ndarray:
    """Convert SDK joint/gripper units to radians and metres."""
    model = np.asarray(raw, dtype=np.float64).copy()
    model[:6] /= RAD_TO_001_DEG
    model[6] /= M_TO_001_MM
    return model


def model_to_raw(model: np.ndarray) -> np.ndarray:
    """Convert radians and metres to the integer units expected by the SDK."""
    raw = np.asarray(model, dtype=np.float64).copy()
    if raw.shape != (7,):
        raise ValueError(f"Expected a seven-element PiPER target, got shape {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("PiPER target contains NaN or infinity")
    raw[:6] = np.rint(raw[:6] * RAD_TO_001_DEG)
    raw[6] = np.rint(raw[6] * M_TO_001_MM)
    return raw.astype(np.int64)


def import_piper_interface() -> type[Any]:
    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise ImportError("Install piper-sdk==0.6.1 in the deployment environment") from exc
    return C_PiperInterface_V2


class PiperSDKArm:
    """One PiPER arm controlled through piper-sdk's MOVE J interface."""

    def __init__(
        self,
        can_name: str,
        *,
        velocity: int,
        gripper_effort: int,
        max_feedback_age_s: float,
        interface_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.can_name = can_name
        self.velocity = velocity
        self.gripper_effort = gripper_effort
        self.max_feedback_age_s = max_feedback_age_s
        self._interface_factory = interface_factory or import_piper_interface()
        self._interface: Any | None = None
        self._last_feedback_stamp: float | None = None
        self._last_feedback_seen_at = 0.0

    @property
    def is_connected(self) -> bool:
        return self._interface is not None and bool(self._interface.get_connect_status())

    def connect(self) -> None:
        if self._interface is not None:
            raise RuntimeError(f"PiPER interface {self.can_name} is already open")
        interface = self._interface_factory(self.can_name)
        interface.ConnectPort()
        self._interface = interface
        self._last_feedback_stamp = None
        self._last_feedback_seen_at = 0.0

    def disconnect(self, *, disable: bool = False) -> None:
        if self._interface is None:
            return
        try:
            if disable:
                self._interface.DisableArm(7)
        finally:
            self._interface.DisconnectPort()
            self._interface = None
            self._last_feedback_stamp = None
            self._last_feedback_seen_at = 0.0

    def _require_interface(self) -> Any:
        if self._interface is None:
            raise RuntimeError(f"PiPER interface {self.can_name} is not connected")
        return self._interface

    def state(self) -> np.ndarray:
        """Read a fresh seven-element [joint radians, gripper metres] state."""
        interface = self._require_interface()
        joint_msg = interface.GetArmJointMsgs()
        joint = joint_msg.joint_state
        stamp = float(joint_msg.time_stamp)
        now = time.monotonic()
        if stamp <= 0:
            raise RuntimeError(f"No PiPER feedback received on {self.can_name}")
        if self._last_feedback_stamp != stamp:
            self._last_feedback_stamp = stamp
            self._last_feedback_seen_at = now
        if now - self._last_feedback_seen_at > self.max_feedback_age_s:
            raise RuntimeError(f"Stale PiPER feedback on {self.can_name}")

        gripper_msg = interface.GetArmGripperMsgs()
        raw = np.asarray(
            [
                joint.joint_1,
                joint.joint_2,
                joint.joint_3,
                joint.joint_4,
                joint.joint_5,
                joint.joint_6,
                gripper_msg.gripper_state.grippers_angle,
            ],
            dtype=np.float64,
        )
        return raw_to_model(raw)

    def enable_status(self) -> list[bool]:
        return [bool(value) for value in self._require_interface().GetArmEnableStatus()]

    def is_enabled(self) -> bool:
        status = self.enable_status()
        return len(status) == 6 and all(status)

    def request_enable(self) -> bool:
        """Send one enable request; callers own retry and cross-arm coordination."""
        return bool(self._require_interface().EnablePiper())

    def enable(self, timeout_s: float, retry_interval_s: float, *, hold_after: bool = True) -> None:
        self._require_interface()
        deadline = time.monotonic() + timeout_s
        last_status: list[bool] = []
        while time.monotonic() < deadline:
            last_status = self.enable_status()
            if len(last_status) == 6 and all(last_status):
                if hold_after:
                    self.hold_current()
                return
            self.request_enable()
            time.sleep(retry_interval_s)
        raise RuntimeError(f"PiPER motor enable timed out on {self.can_name}: {last_status}")

    def command(self, target: np.ndarray) -> np.ndarray:
        """Send one absolute MOVE J target and return the quantized model-space target."""
        interface = self._require_interface()
        raw = model_to_raw(target)
        interface.MotionCtrl_2(0x01, 0x01, self.velocity, 0x00)
        interface.JointCtrl(*(int(value) for value in raw[:6]))
        interface.GripperCtrl(int(raw[6]), self.gripper_effort, 0x01, 0x00)
        return raw_to_model(raw)

    def hold_current(self) -> np.ndarray:
        current = self.state()
        self.command(current)
        return current
