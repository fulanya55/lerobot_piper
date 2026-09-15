#!/usr/bin/env bash
set -euo pipefail

# Hardware-side process for the split deployment. This script owns CAN and
# ROS/cameras; run_policy_server.sh owns only the GPU policy server.
readonly REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly CAMERA_LAUNCH="${PIPER_CAMERA_LAUNCH:-/home/agilex/cobot_magic/tmp/three_cameras_60hz.launch}"
readonly LOG_DIR="${PIPER_DIRECT_LOG_DIR:-/tmp/lerobot-piper-client}"
readonly TASK='Pick up the pen cap and pen body, attach the cap to the body, then place the assembled pen into the pen holder.'

mkdir -p "$LOG_DIR"
source /opt/ros/noetic/setup.bash
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash

owned_pids=()
cleanup() {
  local pid
  for pid in "${owned_pids[@]}"; do
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
  sudo bash /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/can_config.sh
fi
can_ready can_left && can_ready can_right || { echo 'CAN preflight failed' >&2; exit 1; }

if ! rosnode list >/dev/null 2>&1; then
  roscore >"$LOG_DIR/roscore.log" 2>&1 &
  owned_pids+=("$!")
  for _ in $(seq 1 60); do rosnode list >/dev/null 2>&1 && break; sleep 0.2; done
fi
rosnode list >/dev/null 2>&1 || { echo "roscore failed; see $LOG_DIR/roscore.log" >&2; exit 1; }

camera_ready() { timeout 2 rostopic echo -n 1 "$1" >/dev/null 2>&1; }
if ! camera_ready /camera_f/color/image_raw || \
   ! camera_ready /camera_l/color/image_raw || \
   ! camera_ready /camera_r/color/image_raw; then
  roslaunch "$CAMERA_LAUNCH" >"$LOG_DIR/cameras.log" 2>&1 &
  owned_pids+=("$!")
  for _ in $(seq 1 90); do
    camera_ready /camera_f/color/image_raw && \
      camera_ready /camera_l/color/image_raw && \
      camera_ready /camera_r/color/image_raw && break
    sleep 1
  done
fi
camera_ready /camera_f/color/image_raw && \
  camera_ready /camera_l/color/image_raw && \
  camera_ready /camera_r/color/image_raw || {
    echo "camera streams failed; see $LOG_DIR/cameras.log" >&2
    exit 1
  }

cd "$REPO_DIR"
client_args=(
  --mode execute
  --direct-live
  --checkpoint /home/agilex/wxwu/model/pretrained_model
  --dataset-info /home/agilex/wxwu/data/ATTACH_CAP_TO_PEN_1/meta/info.json
  --task "$TASK"
  --server-address "${PIPER_SERVER_ADDRESS:-127.0.0.1:18080}"
  --fps 30
  --actions-per-chunk 50
  --velocity 30
  --no-trajectory-smoothing
)
if [[ -n "${PIPER_MAX_POLICY_ACTIONS:-}" ]]; then
  client_args+=(--max-policy-actions "$PIPER_MAX_POLICY_ACTIONS")
fi

exec uv run --frozen --extra pi --extra async python examples/piper/async_policy_client.py \
  "${client_args[@]}" "$@"
