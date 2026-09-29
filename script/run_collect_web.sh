#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_SCRIPT="$ROOT_DIR/script/start_robot_services.sh"
PREVIEW_SCRIPT="$ROOT_DIR/script/ros_preview_server.py"
STREAM_SCRIPT="$ROOT_DIR/script/ros_lerobot_stream.py"
DIRECT_SCRIPT="$ROOT_DIR/script/direct_collect.sh"
WEB_SCRIPT="$ROOT_DIR/script/collect_web.py"
LAUNCH_FILE="/home/agilex/cobot_magic/tmp/three_cameras_60hz.launch"
ALOHA_PYTHON="/home/agilex/miniconda3/envs/aloha/bin/python"
LOG_DIR="/tmp/piper_robot_services"
ROS_SETUP="source /opt/ros/noetic/setup.bash; source /home/agilex/cobot_magic/camera_ws/devel/setup.bash; source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash"
SERVICE_PID=""
PREVIEW_PID=""
WEB_PID=""
CLEANED=false
MODE="${1:-start}"

matching_pids() {
  local pid cmd
  for proc in /proc/[0-9]*; do
    pid="${proc##*/}"
    [[ "$pid" != "$$" && "$pid" != "$PPID" ]] || continue
    [[ -r "$proc/cmdline" ]] || continue
    cmd="$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)"
    case "$cmd" in
      *"$SERVICE_SCRIPT"*|*"$PREVIEW_SCRIPT"*|*"$DIRECT_SCRIPT"*|*"$STREAM_SCRIPT"*|*"$WEB_SCRIPT"*|*"collect_web.py"*|\
      *"/opt/ros/noetic/bin/roscore"*|*"roslaunch piper start_ms_piper.launch"*|\
      *"roslaunch $LAUNCH_FILE"*|*"roslaunch realsense2_camera multi_camera.launch"*)
        printf '%s\n' "$pid"
        ;;
    esac
  done
}

stop_old() {
  local pid pgid own_pgid
  own_pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  declare -A groups=()
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
    [[ -n "$pgid" && "$pgid" != "$own_pgid" ]] && groups["$pgid"]=1
  done < <(matching_pids)
  for pgid in "${!groups[@]}"; do kill -TERM -- "-$pgid" 2>/dev/null || true; done
  for _ in $(seq 1 50); do
    [[ -z "$(matching_pids)" ]] && break
    sleep .1
  done
  while read -r pid; do
    [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
  done < <(matching_pids)
  find /tmp -maxdepth 1 \( -name 'piper_collect_control.*' -o -name 'piper_lerobot_stream.*' \) -type s -delete 2>/dev/null || true
  find /tmp -maxdepth 1 \( -name 'piper_collect_control.*' -o -name 'piper_lerobot_stream.*' \) -type f -delete 2>/dev/null || true
}

cleanup() {
  [[ "$CLEANED" == false ]] || return
  CLEANED=true
  trap - EXIT INT TERM HUP
  for pid in "$WEB_PID" "$PREVIEW_PID" "$SERVICE_PID"; do
    [[ -n "$pid" ]] && kill -TERM -- "-$pid" 2>/dev/null || true
  done
  sleep 1
  stop_old
}
trap cleanup EXIT INT TERM HUP

mkdir -p "$LOG_DIR"
stop_old
if [[ "$MODE" == "stop" ]]; then
  CLEANED=true
  trap - EXIT INT TERM HUP
  echo "[停止] 网页、采集、预览和 ROS 服务已全部清理"
  exit 0
fi
[[ "$MODE" == "start" || "$MODE" == "restart" ]] || { echo "用法: $0 [start|restart|stop]" >&2; exit 2; }
echo "[启动] 旧网页、采集、预览和 ROS 服务已清理"

setsid env PIPER_CAMERA_RESOLUTION=960x540 PIPER_SERVICE_LOG_DIR="$LOG_DIR" \
  "$SERVICE_SCRIPT" >>"$LOG_DIR/services.log" 2>&1 &
SERVICE_PID=$!

for _ in $(seq 1 120); do
  kill -0 "$SERVICE_PID" 2>/dev/null || { echo "[错误] ROS 服务脚本已退出，请查看 $LOG_DIR/services.log" >&2; exit 1; }
  bash -lc "$ROS_SETUP; rosnode list >/dev/null 2>&1" && break
  sleep .5
done
bash -lc "$ROS_SETUP; rosnode list >/dev/null 2>&1" || { echo "[错误] ROS master 启动超时" >&2; exit 1; }

REQUIRED_TOPICS=(/camera_f/color/image_raw /camera_l/color/image_raw /camera_r/color/image_raw /master/joint_left /master/joint_right /puppet/joint_left /puppet/joint_right)
for topic in "${REQUIRED_TOPICS[@]}"; do
  if ! bash -lc "$ROS_SETUP; timeout 12s rostopic echo -n 1 '$topic' >/dev/null 2>&1"; then
    echo "[错误] 话题无实际消息：$topic" >&2
    exit 1
  fi
done

setsid bash -lc "$ROS_SETUP; exec '$ALOHA_PYTHON' '$PREVIEW_SCRIPT'" >"$LOG_DIR/preview.log" 2>&1 &
PREVIEW_PID=$!
for _ in $(seq 1 30); do
  ss -ltn '( sport = :8766 )' 2>/dev/null | grep -q LISTEN && break
  sleep .2
done
ss -ltn '( sport = :8766 )' 2>/dev/null | grep -q LISTEN || { echo "[错误] 预览服务启动失败" >&2; exit 1; }

echo "[启动] ROS、七路话题和相机预览已就绪"
cd "$ROOT_DIR"
setsid env PIPER_COLLECT_SUPERVISED=1 uv run python script/collect_web.py &
WEB_PID=$!
wait "$WEB_PID"
