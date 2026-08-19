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
            "cam_high_rgb": "/camera_f/color/image_raw",
            "cam_left_wrist_rgb": "/camera_l/color/image_raw",
            "cam_right_wrist_rgb": "/camera_r/color/image_raw",
        }
    )
    image_height: int = 480
    image_width: int = 640
    ros_node_name: str = "lerobot_bi_piper_direct_client"
    subscriber_queue_size: int = 1
    connect_timeout_s: float = 15.0
    max_feedback_age_s: float = 0.5
    max_image_age_s: float = 1.0

    # Direct SDK control settings, matching PCD-LeRobot's MOVE J path.
    velocity: int = 30
    gripper_effort: int = 1000
    enable_timeout_s: float = 10.0
    enable_retry_interval_s: float = 0.1
    enable_watchdog_hz: float = 10.0
    enable_loss_confirmations: int = 3
    enable_loss_confirmation_interval_s: float = 0.02
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
        if len(self.camera_topics) != 3:
            raise ValueError("This Pi0 checkpoint requires exactly three RGB camera topics")
        if not 1 <= self.velocity <= 100:
            raise ValueError("velocity must be in [1, 100]")
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
