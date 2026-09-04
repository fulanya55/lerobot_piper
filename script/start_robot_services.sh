#!/usr/bin/env bash
set -Eeuo pipefail

PIPER_WS="/home/agilex/cobot_magic/Piper_ros_private-ros-noetic"
CAMERA_WS="/home/agilex/cobot_magic/camera_ws"
CONDA_SH="/home/agilex/miniconda3/etc/profile.d/conda.sh"
LOG_DIR="${PIPER_SERVICE_LOG_DIR:-/tmp/piper_robot_services}"
mkdir -p "$LOG_DIR"
ROS_SETUP="source /opt/ros/noetic/setup.bash; source '$CAMERA_WS/devel/setup.bash'; source '$PIPER_WS/devel/setup.bash'"
PIDS=()
cleanup(){ for p in "${PIDS[@]:-}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# The USB adapters come back as can0 (left, USB 1-12) and can1 (right, USB
# 1-13) after a reboot. Restore the names expected by the PiPER ROS launch
# without requiring an interactive sudo prompt when the host allows ip-link
# operations through sudoers.
if ! ip link show can_left >/dev/null 2>&1 && ip link show can0 >/dev/null 2>&1; then
  sudo -n ip link set can0 down && sudo -n ip link set can0 name can_left || true
fi
if ! ip link show can_right >/dev/null 2>&1 && ip link show can1 >/dev/null 2>&1; then
  sudo -n ip link set can1 down && sudo -n ip link set can1 name can_right || true
fi
for can in can_left can_right; do
  if ip link show "$can" >/dev/null 2>&1; then
    sudo -n ip link set "$can" down || true
    sudo -n ip link set "$can" type can bitrate 1000000 || true
    sudo -n ip link set "$can" up || true
    ip -details link show "$can" | grep -E '^[0-9]+:|bitrate' >>"$LOG_DIR/services.log" 2>&1 || true
  fi
done

if ! bash -lc "$ROS_SETUP; rosnode list" >/dev/null 2>&1; then
  setsid bash -lc "$ROS_SETUP; exec roscore" >"$LOG_DIR/roscore.log" 2>&1 & PIDS+=("$!")
  for _ in $(seq 1 50); do bash -lc "$ROS_SETUP; rosnode list" >/dev/null 2>&1 && break; sleep .2; done
fi

if [[ "${PIPER_CAMERA_RESOLUTION:-960x540}" == "960x540" ]]; then
  CAMERA_CMD="roslaunch /home/agilex/cobot_magic/xyx_piper_right/launch/three_cameras_60hz.launch"
else
  CAMERA_CMD="roslaunch realsense2_camera multi_camera.launch"
fi
setsid bash -lc "$ROS_SETUP; exec $CAMERA_CMD" >"$LOG_DIR/cameras.log" 2>&1 & PIDS+=("$!")
setsid bash -lc "source '$CONDA_SH'; conda activate aloha; $ROS_SETUP; exec roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=true" >"$LOG_DIR/arms.log" 2>&1 & PIDS+=("$!")

echo "services_started resolution=${PIPER_CAMERA_RESOLUTION:-960x540} log_dir=$LOG_DIR"
wait
