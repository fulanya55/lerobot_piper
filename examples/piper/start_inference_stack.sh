#!/usr/bin/env bash
set -euo pipefail

# Start only the infrastructure used by direct-SDK inference. In particular,
# this script never starts the PiPER ROS arm-control nodes and never enables an arm.

readonly REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly CAMERA_WS="/home/agilex/cobot_magic/camera_ws"
readonly CAN_CONFIG="/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/can_config.sh"
readonly SERVER_HOST="${PIPER_SERVER_HOST:-127.0.0.1}"
readonly SERVER_PORT="${PIPER_SERVER_PORT:-18080}"
readonly FPS="${PIPER_FPS:-30}"
readonly LOG_DIR="${PIPER_LOG_DIR:-/tmp/lerobot-piper-inference}"

mkdir -p "$LOG_DIR"

owned_pids=()
owned_names=()

stop_owned_processes() {
    local index pid
    for ((index=${#owned_pids[@]} - 1; index >= 0; index--)); do
        pid="${owned_pids[$index]}"
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping ${owned_names[$index]} (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap stop_owned_processes EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

register_process() {
    owned_pids+=("$1")
    owned_names+=("$2")
}

can_ready() {
    local interface="$1"
    [[ -e "/sys/class/net/$interface" ]] || return 1
    ip link show "$interface" | head -n 1 | grep -q "UP" || return 1
    ip -details link show "$interface" | grep -q "bitrate 1000000"
}

echo "[1/4] Checking can_left/can_right..."
if ! can_ready can_left || ! can_ready can_right; then
    echo "CAN names/bitrate are not ready; running the verified PiPER CAN configuration."
    echo "This may ask for the sudo password. Physically support both arms before continuing."
    sudo bash "$CAN_CONFIG"
fi
if ! can_ready can_left || ! can_ready can_right; then
    echo "CAN preflight failed: both can_left and can_right must be UP at 1 Mbit/s." >&2
    exit 1
fi

source /opt/ros/noetic/setup.bash
source "$CAMERA_WS/devel/setup.bash"

echo "[2/4] Checking ROS master..."
if rosnode list >/dev/null 2>&1; then
    echo "Using the existing ROS master."
else
    roscore >"$LOG_DIR/roscore.log" 2>&1 &
    roscore_pid=$!
    register_process "$roscore_pid" roscore
    for _ in $(seq 1 50); do
        rosnode list >/dev/null 2>&1 && break
        sleep 0.2
    done
    if ! rosnode list >/dev/null 2>&1; then
        echo "roscore failed to become ready; see $LOG_DIR/roscore.log" >&2
        exit 1
    fi
fi

conflicting_nodes="$(rosnode list 2>/dev/null | grep -E 'piper_(left|right)|/piper_left_|/piper_right_' || true)"
if [[ -n "$conflicting_nodes" ]]; then
    echo "Refusing direct-CAN inference while PiPER ROS arm nodes are running:" >&2
    echo "$conflicting_nodes" >&2
    exit 1
fi

camera_frame_ready() {
    timeout 2 rostopic echo -n 1 "$1" >/dev/null 2>&1
}

all_cameras_ready() {
    camera_frame_ready /camera_f/color/image_raw \
        && camera_frame_ready /camera_l/color/image_raw \
        && camera_frame_ready /camera_r/color/image_raw
}

echo "[3/4] Checking three RealSense RGB streams..."
if all_cameras_ready; then
    echo "Using the existing three-camera launch."
else
    existing_camera_nodes="$(rosnode list 2>/dev/null | grep -E '/camera_[lrf]' || true)"
    if [[ -n "$existing_camera_nodes" ]]; then
        echo "Some camera nodes already exist but all three RGB streams are not healthy:" >&2
        echo "$existing_camera_nodes" >&2
        echo "Stop the partial camera launch before retrying." >&2
        exit 1
    fi
    roslaunch realsense2_camera multi_camera.launch >"$LOG_DIR/cameras.log" 2>&1 &
    camera_pid=$!
    register_process "$camera_pid" cameras
    cameras_ready=false
    for _ in $(seq 1 30); do
        if all_cameras_ready; then
            cameras_ready=true
            break
        fi
        if ! kill -0 "$camera_pid" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if [[ "$cameras_ready" != true ]]; then
        echo "Three-camera preflight failed; see $LOG_DIR/cameras.log" >&2
        exit 1
    fi
fi

server_ready() {
    (exec 3<>"/dev/tcp/$SERVER_HOST/$SERVER_PORT") >/dev/null 2>&1
}

echo "[4/4] Checking LeRobot policy server..."
if server_ready; then
    echo "Using the existing policy server at $SERVER_HOST:$SERVER_PORT."
else
    (
        cd "$REPO_DIR"
        exec uv run --frozen python examples/piper/policy_server.py \
            --host "$SERVER_HOST" \
            --port "$SERVER_PORT" \
            --fps "$FPS" \
            --no-compile-model
    ) >"$LOG_DIR/policy_server.log" 2>&1 &
    server_pid=$!
    register_process "$server_pid" policy_server
    # Importing LeRobot can take longer than 10 seconds on the first run.
    for _ in $(seq 1 600); do
        server_ready && break
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 0.1
    done
    if ! server_ready; then
        echo "Policy server failed to listen; see $LOG_DIR/policy_server.log" >&2
        exit 1
    fi
fi

echo
echo "Inference prerequisites are ready."
echo "Logs: $LOG_DIR"
echo "Run examples/piper/run_pi05_inference.sh in another terminal."
echo "Keep this terminal open; Ctrl+C stops only processes started by this script."

while true; do
    for index in "${!owned_pids[@]}"; do
        if ! kill -0 "${owned_pids[$index]}" 2>/dev/null; then
            echo "${owned_names[$index]} exited unexpectedly; inspect $LOG_DIR." >&2
            exit 1
        fi
    done
    sleep 1
done
