#!/usr/bin/env bash
set -Eeuo pipefail

# Direct ROS -> LeRobot v3 collector. MP4 and Parquet are written while recording.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPER_WS="/home/agilex/cobot_magic/Piper_ros_private-ros-noetic"
CAMERA_WS="/home/agilex/cobot_magic/camera_ws"
CONDA_SH="/home/agilex/miniconda3/etc/profile.d/conda.sh"
ALOHA_PYTHON="/home/agilex/miniconda3/envs/aloha/bin/python"
LEROBOT_DIR="$ROOT_DIR/lerobot_piper"
DATASET_PATH="$ROOT_DIR/data/piper_lerobot_direct"
REPO_ID="local/piper_dual_arm"
TASK="dual-arm manipulation"
EPISODE_IDX=0
TIMESTEPS=3000
FPS=30
VIDEO_CODEC=h264
VIDEO_CRF=23
SKIP_CAN=false
NO_PREVIEW=true
AUTO_START=false
YES=false

usage() {
  cat <<'EOF'
用法: script/direct_collect.sh [选项]
  --dataset-path PATH   LeRobot v3 数据集目录
  --repo-id ID          数据集逻辑 ID
  --episode-idx N       episode 索引
  --timesteps N         最多帧数
  --fps N               采集帧率
  --camera-resolution R 960x540（默认）或 640x480
  --video-codec CODE    h264/hevc/libsvtav1（默认 h264）
  --video-crf N         编码质量（默认 23）
  --skip-can            不执行 can_config.sh
  --no-preview          不启动相机预览窗口
  --auto-start          不等待 Enter，收到首帧后立即录制
  --yes                 跳过人工确认
EOF
}
CAMERA_RES=960x540
while (($#)); do
  case "$1" in
    --dataset-path) DATASET_PATH="$2"; shift 2;;
    --repo-id) REPO_ID="$2"; shift 2;;
    --task) TASK="$2"; shift 2;;
    --episode-idx) EPISODE_IDX="$2"; shift 2;;
    --timesteps) TIMESTEPS="$2"; shift 2;;
    --fps) FPS="$2"; shift 2;;
    --camera-resolution) CAMERA_RES="$2"; shift 2;;
    --video-codec) VIDEO_CODEC="$2"; shift 2;;
    --video-crf) VIDEO_CRF="$2"; shift 2;;
    --skip-can) SKIP_CAN=true; shift;;
    --no-preview) NO_PREVIEW=true; shift;;
    --auto-start) AUTO_START=true; shift;;
    --yes) YES=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ "$CAMERA_RES" == 960x540 || "$CAMERA_RES" == 640x480 ]] || { echo "仅支持 960x540 或 640x480" >&2; exit 2; }
[[ "$EPISODE_IDX" =~ ^[0-9]+$ && "$TIMESTEPS" =~ ^[1-9][0-9]*$ && "$FPS" =~ ^[1-9][0-9]*$ ]] || exit 2
mkdir -p "$(dirname -- "$DATASET_PATH")"
[[ -x "$ALOHA_PYTHON" && -f "$PIPER_WS/devel/setup.bash" && -f "$CAMERA_WS/devel/setup.bash" ]] || { echo "ROS 环境不完整" >&2; exit 1; }

if [[ "$YES" != true ]]; then
  echo "请确认急停可触达、机械臂周围无人和障碍物，并可靠支撑机械臂。"
  read -r -p "输入 ENABLE 继续：" ok
  [[ "$ok" == ENABLE ]] || exit 1
fi
if [[ "$SKIP_CAN" != true ]]; then sudo bash "$PIPER_WS/can_config.sh"; fi

SOCKET_PATH="$(mktemp -u /tmp/piper_lerobot_stream.XXXXXX)"
LOG_DIR="$(mktemp -d /tmp/piper_direct_logs.XXXXXX)"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  rm -f "$SOCKET_PATH"
}
trap cleanup EXIT INT TERM
ROS_SETUP="source /opt/ros/noetic/setup.bash; source '$CAMERA_WS/devel/setup.bash'; source '$PIPER_WS/devel/setup.bash'"
REQUIRED_TOPICS=(/camera_f/color/image_raw /camera_l/color/image_raw /camera_r/color/image_raw /master/joint_left /master/joint_right /puppet/joint_left /puppet/joint_right)
existing_topics="$(bash -lc "$ROS_SETUP; rostopic list" 2>/dev/null || true)"
SERVICES_READY=true
for topic in "${REQUIRED_TOPICS[@]}"; do grep -Fxq "$topic" <<<"$existing_topics" || SERVICES_READY=false; done
if [[ "$CAMERA_RES" == 960x540 ]]; then
  CAMERA_LAUNCH="roslaunch '$ROOT_DIR/../cobot_magic/xyx_piper_right/launch/three_cameras_60hz.launch'"
else
  CAMERA_LAUNCH="roslaunch realsense2_camera multi_camera.launch"
fi
if [[ "$SERVICES_READY" != true ]] && ! bash -lc "$ROS_SETUP; rosnode list" >/dev/null 2>&1; then
  setsid bash -lc "$ROS_SETUP; exec roscore" >"$LOG_DIR/roscore.log" 2>&1 & PIDS+=("$!")
  for _ in $(seq 1 40); do bash -lc "$ROS_SETUP; rosnode list" >/dev/null 2>&1 && break; sleep .25; done
fi
if [[ "$SERVICES_READY" != true ]]; then
  setsid bash -lc "$ROS_SETUP; exec $CAMERA_LAUNCH" >"$LOG_DIR/cameras.log" 2>&1 & PIDS+=("$!")
  setsid bash -lc "source '$CONDA_SH'; conda activate aloha; $ROS_SETUP; exec roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=true" >"$LOG_DIR/arms.log" 2>&1 & PIDS+=("$!")
fi
setsid bash -lc "cd '$LEROBOT_DIR'; UV_CACHE_DIR=/tmp/wxwu-uv-cache uv run --frozen --extra dataset python examples/piper/lerobot_stream_writer.py --socket '$SOCKET_PATH' --dataset-path '$DATASET_PATH' --repo-id '$REPO_ID' --task '$TASK' --episode-idx '$EPISODE_IDX' --fps '$FPS' --video-codec '$VIDEO_CODEC' --video-crf '$VIDEO_CRF'" >"$LOG_DIR/writer.log" 2>&1 & PIDS+=("$!")
for _ in $(seq 1 40); do [[ -S "$SOCKET_PATH" ]] && break; sleep .1; done
echo "等待 ROS 话题……日志：$LOG_DIR"
sleep 4
STREAM_ARGS=(--socket "$SOCKET_PATH" --timesteps "$TIMESTEPS" --fps "$FPS" --sync-slop 0.10)
[[ "$AUTO_START" == true ]] && STREAM_ARGS+=(--auto-start)
"$ALOHA_PYTHON" "$ROOT_DIR/lerobot_piper/script/ros_lerobot_stream.py" "${STREAM_ARGS[@]}"
echo "采集完成，数据集已直接写入：$DATASET_PATH"
