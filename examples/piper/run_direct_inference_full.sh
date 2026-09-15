#!/usr/bin/env bash
set -euo pipefail

# Complete direct-control launcher. It starts only roscore/RealSense when they
# are not already running, then execs the replay-style local policy controller.
# No gRPC policy server and no confirmation flags are involved.

readonly REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly CAMERA_LAUNCH="${PIPER_CAMERA_LAUNCH:-/home/agilex/cobot_magic/tmp/three_cameras_60hz.launch}"
readonly LOG_DIR="${PIPER_DIRECT_LOG_DIR:-/tmp/lerobot-piper-direct}"
readonly TASK='Pick up the pen cap and pen body, attach the cap to the body, then place the assembled pen into the pen holder.'

mkdir -p "$LOG_DIR"
source /opt/ros/noetic/setup.bash
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash

owned_pids=()
owned_names=()
cleanup() {
  local i pid
  for ((i=${#owned_pids[@]}-1; i>=0; i--)); do
    pid="${owned_pids[$i]}"
    kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

can_ready() {
  local name="$1"
  ip link show "$name" >/dev/null 2>&1 \
    && ip link show "$name" | head -1 | grep -q 'UP' \
    && ip -details link show "$name" | grep -q 'bitrate 1000000'
}
if ! can_ready can_left || ! can_ready can_right; then
  echo "CAN is not ready; configuring can_left/can_right..."
  sudo bash /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/can_config.sh
fi
can_ready can_left && can_ready can_right || { echo 'CAN preflight failed' >&2; exit 1; }

if ! rosnode list >/dev/null 2>&1; then
  roscore >"$LOG_DIR/roscore.log" 2>&1 &
  owned_pids+=("$!"); owned_names+=(roscore)
  for _ in $(seq 1 60); do rosnode list >/dev/null 2>&1 && break; sleep 0.2; done
fi
rosnode list >/dev/null 2>&1 || { echo "roscore failed; see $LOG_DIR/roscore.log" >&2; exit 1; }

camera_ready() {
  timeout 2 rostopic echo -n 1 "$1" >/dev/null 2>&1
}
if ! camera_ready /camera_f/color/image_raw || \
   ! camera_ready /camera_l/color/image_raw || \
   ! camera_ready /camera_r/color/image_raw; then
  roslaunch "$CAMERA_LAUNCH" >"$LOG_DIR/cameras.log" 2>&1 &
  owned_pids+=("$!"); owned_names+=(cameras)
  for _ in $(seq 1 90); do
    camera_ready /camera_f/color/image_raw && camera_ready /camera_l/color/image_raw && camera_ready /camera_r/color/image_raw && break
    sleep 1
  done
fi
camera_ready /camera_f/color/image_raw && camera_ready /camera_l/color/image_raw && camera_ready /camera_r/color/image_raw || {
  echo "camera streams failed; see $LOG_DIR/cameras.log" >&2; exit 1;
}

cd "$REPO_DIR"
exec uv run --frozen --extra pi --extra async python examples/piper/direct_inference.py \
  --checkpoint /home/agilex/wxwu/model/pretrained_model \
  --dataset-info /home/agilex/wxwu/data/ATTACH_CAP_TO_PEN_1/meta/info.json \
  --task "$TASK" \
  --device cuda --fps 30 --velocity 30
