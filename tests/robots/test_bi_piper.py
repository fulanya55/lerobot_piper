# ruff: noqa: N802

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.bi_piper import BiPiper, BiPiperConfig
from lerobot.robots.bi_piper.bi_piper import ACTION_FEATURES
from lerobot.robots.bi_piper.piper_sdk_arm import PiperSDKArm
from lerobot.utils.errors import DeviceNotConnectedError


class FakeInterface:
    def __init__(self, raw: list[int], enabled: bool = True):
        self.raw = raw
        self.enabled = [enabled] * 6
        self.calls = []
        self.connected = True
        self.stamp = 1.0

    def get_connect_status(self):
        return self.connected

    def ConnectPort(self):
        self.calls.append(("connect",))
        self.connected = True

    def GetArmJointMsgs(self):
        joint = SimpleNamespace(**{f"joint_{i + 1}": self.raw[i] for i in range(6)})
        return SimpleNamespace(joint_state=joint, time_stamp=self.stamp)

    def GetArmGripperMsgs(self):
        gripper = SimpleNamespace(grippers_angle=self.raw[6])
        return SimpleNamespace(gripper_state=gripper, time_stamp=self.stamp)

    def GetArmEnableStatus(self):
        return list(self.enabled)

    def EnablePiper(self):
        self.calls.append(("enable",))
        self.enabled = [True] * 6
        return True

    def MotionCtrl_2(self, *args):
        self.calls.append(("motion", *args))

    def JointCtrl(self, *args):
        self.calls.append(("joint", *args))

    def GripperCtrl(self, *args):
        self.calls.append(("gripper", *args))

    def DisableArm(self, *args):
        self.calls.append(("disable", *args))

    def DisconnectPort(self):
        self.calls.append(("disconnect",))
        self.connected = False


def make_sdk_arm(raw: list[int], *, enabled: bool = True) -> PiperSDKArm:
    interface = FakeInterface(raw, enabled=enabled)
    arm = PiperSDKArm(
        "fake_can",
        velocity=30,
        gripper_effort=1000,
        max_feedback_age_s=0.5,
        interface_factory=lambda _: interface,
    )
    arm.connect()
    return arm


def make_connected_robot(*, dry_run: bool = False, enabled: bool = True):
    config = BiPiperConfig(id="test", dry_run=dry_run, enable_retry_interval_s=0.001)
    robot = BiPiper(config)
    raw = [0, 90000, -90000, 0, 0, 0, 40000]
    robot._arms = {
        "left": make_sdk_arm(raw.copy(), enabled=enabled),
        "right": make_sdk_arm(raw.copy(), enabled=enabled),
    }
    robot._connected = True
    return robot


def test_features_match_checkpoint_order():
    robot = BiPiper(BiPiperConfig(id="test"))

    assert list(robot.action_features) == list(ACTION_FEATURES)
    assert list(robot.observation_features)[:14] == list(ACTION_FEATURES)


def test_sdk_unit_roundtrip():
    model = np.array([0.0, np.pi / 2, -np.pi / 2, 0.1, -0.2, 0.3, 0.04])

    raw = BiPiper.model_to_raw(model)
    restored = BiPiper.raw_to_model(raw)

    np.testing.assert_allclose(restored, model, atol=np.pi / 180000)
    np.testing.assert_array_equal(raw[:3], [0, 90000, -90000])
    assert raw[6] == 40000


@pytest.mark.parametrize(
    "target",
    [np.zeros(6), np.zeros(8), np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0])],
)
def test_sdk_transport_rejects_invalid_targets(target):
    arm = make_sdk_arm([0] * 7)

    with pytest.raises(ValueError):
        arm.command(target)


def test_action_is_clipped_to_joint_limits_and_measured_slew():
    robot = make_connected_robot(dry_run=True)
    current = robot._read_both()
    requested = current + 100.0
    action = {name: float(value) for name, value in zip(ACTION_FEATURES, requested, strict=True)}

    sent = robot.send_action(action)
    sent_values = np.asarray([sent[name] for name in ACTION_FEATURES])

    expected_step = np.tile(np.asarray([0.05] * 6 + [0.005]), 2)
    np.testing.assert_allclose(sent_values, current + expected_step)


def test_partial_or_non_finite_action_is_rejected():
    robot = make_connected_robot(dry_run=True)

    with pytest.raises(ValueError, match="missing features"):
        robot.send_action({ACTION_FEATURES[0]: 0.0})

    current = robot._read_both()
    action = {name: float(value) for name, value in zip(ACTION_FEATURES, current, strict=True)}
    action[ACTION_FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        robot.send_action(action)


def test_send_action_uses_move_j_joint_and_gripper_for_both_arms():
    robot = make_connected_robot()
    current = robot._read_both()
    action = {name: float(value) for name, value in zip(ACTION_FEATURES, current, strict=True)}

    sent = robot.send_action(action)

    assert sent == action
    for arm in robot._arms.values():
        calls = arm._interface.calls
        assert ("motion", 0x01, 0x01, 30, 0x00) in calls
        assert ("joint", 0, 90000, -90000, 0, 0, 0) in calls
        assert ("gripper", 40000, 1000, 0x01, 0x00) in calls


def test_lost_enable_recovers_hold_then_aborts_old_action_queue():
    robot = make_connected_robot(enabled=False)
    current = robot._read_both()
    action = {name: float(value) for name, value in zip(ACTION_FEATURES, current, strict=True)}

    with pytest.raises(DeviceNotConnectedError, match="old action queue"):
        robot.send_action(action)

    for arm in robot._arms.values():
        assert ("enable",) in arm._interface.calls
        assert any(call[0] == "joint" for call in arm._interface.calls)


def test_single_enable_status_glitch_does_not_abort_action(monkeypatch):
    robot = make_connected_robot()
    current = robot._read_both()
    action = {name: float(value) for name, value in zip(ACTION_FEATURES, current, strict=True)}
    calls = 0

    def transient_status():
        nonlocal calls
        calls += 1
        enabled = calls > 1
        return {"left": [enabled] * 6, "right": [enabled] * 6}

    monkeypatch.setattr(robot, "_enable_status", transient_status)

    assert robot.send_action(action) == action
    assert robot._fatal_error is None


def test_disconnect_holds_and_does_not_disable_by_default():
    robot = make_connected_robot()
    interfaces = [arm._interface for arm in robot._arms.values()]

    robot.disconnect()

    for interface in interfaces:
        assert any(call[0] == "joint" for call in interface.calls)
        assert not any(call[0] == "disable" for call in interface.calls)
        assert interface.calls[-1] == ("disconnect",)
