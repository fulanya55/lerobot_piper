#!/usr/bin/env python

from dataclasses import dataclass, field

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_piper")
@dataclass
class BiPiperConfig(RobotConfig):
    """Direct-SDK dual-arm PiPER configuration for the AgileX IPC."""

    can_left: str = "can_left"
    can_right: str = "can_right"
    camera_topics: dict[str, str] = field(
        default_factory=lambda: {
            "cam_high": "/camera_f/color/image_raw",
            "cam_left_wrist": "/camera_l/color/image_raw",
            "cam_right_wrist": "/camera_r/color/image_raw",
        }
    )
    image_height: int = 480
    image_width: int = 640
    ros_node_name: str = "lerobot_bi_piper_direct_client"
    subscriber_queue_size: int = 1
    connect_timeout_s: float = 15.0
    max_feedback_age_s: float = 0.5
    max_image_age_s: float = 1.0

    # Policy-facing 14D schema. The transport converts legacy normalized
    # gripper values to/from the SDK's physical metre representation.
    action_feature_names: tuple[str, ...] = field(
        default_factory=lambda: tuple(
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
    )
    gripper_action_unit: str = "meters"

    # Direct SDK control settings, matching PCD-LeRobot's MOVE J path.
    velocity: int = 30
    # Keep the MOVE_J/JointCtrl stream alive between policy or replay frames,
    # matching the inference controller's high-rate command loop.
    command_refresh_hz: float = 150.0
    # Periodically repeat EnableArm(7), as required by controllers that lose
    # joint enable while still accepting gripper commands.
    enable_keepalive_hz: float = 10.0
    # Inference fails closed on sustained enable loss. Dataset replay can opt
    # into re-enable-at-current-pose and continue with the current frame.
    recover_enable_loss: bool = False
    gripper_effort: int = 1000
    enable_timeout_s: float = 10.0
    enable_retry_interval_s: float = 0.1
    enable_watchdog_hz: float = 10.0
    enable_loss_confirmations: int = 3
    enable_loss_confirmation_interval_s: float = 0.02
    # True suppresses an explicit DisableArm command. It cannot guarantee that
    # controller firmware will retain enable after the CAN socket closes.
    keep_enabled_on_disconnect: bool = True

    # The default is observation-only. Live mode is selected explicitly by the client CLI.
    dry_run: bool = True

    # Limits documented by piper-sdk 0.6.1 JointCtrl, with gripper in metres.
    joint_lower_limits_rad: tuple[float, ...] = (-2.6179, 0.0, -2.967, -1.745, -1.22, -2.09439)
    joint_upper_limits_rad: tuple[float, ...] = (2.6179, 3.14, 0.0, 1.745, 1.22, 2.09439)
    gripper_lower_m: float = 0.0
    gripper_upper_m: float = 0.08

    # Clip every target relative to the latest measured position.
    max_joint_step_rad: float = 0.05
    max_gripper_step_m: float = 0.005

    # Optional stateful trajectory limiter. The historical measured-position
    # step limits remain as a final safety envelope; these limits additionally
    # keep commanded velocity and acceleration continuous across policy/chunk
    # changes.
    trajectory_smoothing: bool = False
    policy_fps: float = 30.0
    max_joint_velocity_rad_s: float = 1.0
    max_joint_acceleration_rad_s2: float = 4.0
    max_gripper_velocity_m_s: float = 0.08
    max_gripper_acceleration_m_s2: float = 0.4

    # Direct control must be the only process owning the two CAN interfaces.
    reject_piper_ros_nodes: bool = True
    conflicting_ros_node_fragments: tuple[str, ...] = ("/piper_left_", "/piper_right_")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.can_left == self.can_right:
            raise ValueError("can_left and can_right must identify different CAN interfaces")
        if len(self.joint_lower_limits_rad) != 6 or len(self.joint_upper_limits_rad) != 6:
            raise ValueError("PiPER joint limit tuples must each contain six values")
        if any(
            lower >= upper
            for lower, upper in zip(self.joint_lower_limits_rad, self.joint_upper_limits_rad, strict=True)
        ):
            raise ValueError("Every PiPER lower joint limit must be smaller than its upper limit")
        if self.gripper_lower_m >= self.gripper_upper_m:
            raise ValueError("gripper_lower_m must be smaller than gripper_upper_m")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("Image dimensions must be positive")
        if len(self.camera_topics) not in {0, 3}:
            raise ValueError(
                "BiPiper requires either three RGB camera topics for inference or no cameras for direct replay"
            )
        if len(self.action_feature_names) != 14 or len(set(self.action_feature_names)) != 14:
            raise ValueError("action_feature_names must contain 14 unique dual-arm features")
        if self.gripper_action_unit not in {"meters", "open_scale"}:
            raise ValueError("gripper_action_unit must be 'meters' or 'open_scale'")
        if not 1 <= self.velocity <= 100:
            raise ValueError("velocity must be in [1, 100]")
        if self.command_refresh_hz <= 0:
            raise ValueError("command_refresh_hz must be positive")
        if self.enable_keepalive_hz <= 0:
            raise ValueError("enable_keepalive_hz must be positive")
        if not 0 <= self.gripper_effort <= 5000:
            raise ValueError("gripper_effort must be in [0, 5000]")
        if self.enable_timeout_s <= 0 or self.enable_retry_interval_s <= 0:
            raise ValueError("Enable timeout and retry interval must be positive")
        if self.enable_watchdog_hz <= 0:
            raise ValueError("enable_watchdog_hz must be positive")
        if self.enable_loss_confirmations <= 0 or self.enable_loss_confirmation_interval_s <= 0:
            raise ValueError("Enable-loss confirmation settings must be positive")
        if self.max_joint_step_rad <= 0 or self.max_gripper_step_m <= 0:
            raise ValueError("Action slew limits must be positive")
        if self.policy_fps <= 0:
            raise ValueError("policy_fps must be positive")
        if self.max_joint_velocity_rad_s <= 0 or self.max_joint_acceleration_rad_s2 <= 0:
            raise ValueError("Joint trajectory velocity/acceleration limits must be positive")
        if self.max_gripper_velocity_m_s <= 0 or self.max_gripper_acceleration_m_s2 <= 0:
            raise ValueError("Gripper trajectory velocity/acceleration limits must be positive")
