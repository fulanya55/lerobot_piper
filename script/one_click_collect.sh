#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--direct" ]]; then
    shift
    exec "$SCRIPT_DIR/direct_collect.sh" "$@"
fi
PIPER_WS="/home/agilex/cobot_magic/Piper_ros_private-ros-noetic"
CAMERA_WS="/home/agilex/cobot_magic/camera_ws"
CONDA_SH="/home/agilex/miniconda3/etc/profile.d/conda.sh"
ALOHA_PYTHON="/home/agilex/miniconda3/envs/aloha/bin/python"
LEROBOT_PYTHON="/home/agilex/miniconda3/envs/lerobot/bin/python"

DATASET_PATH="/home/agilex/wxwu/data/piper_lerobot_v2"
REPO_ID="local/piper_dual_arm"
TASK="dual-arm manipulation"
TIMESTEPS=3000
EPISODE_IDX=0
FPS=30
VIDEO_CODEC="h264"
VIDEO_CRF=23
SKIP_CAN=false
NO_PREVIEW=false
YES=false

usage() {
    sed -n '/^# PiPER 双臂/,/^$/ { s/^# \{0,1\}//; p; }' "$0"
}

# PiPER 双臂 + 三路 RealSense + LeRobot v2.1 一键采集
#
# 用法：
#   ./one_click_collect.sh [选项]
#
# 常用选项：
#   --dataset-path PATH  LeRobot 数据集根目录
#   --timesteps N        单个 episode 最大帧数（默认 3000）
#   --episode-idx N      本次 episode 索引（必须连续，默认 0）
#   --task TEXT          任务描述
#   --repo-id ID         本地数据集逻辑 ID
#   --fps N              采集 FPS（默认 30）
#   --skip-can           跳过 sudo can_config.sh
#   --no-preview         不打开三相机拼接窗口
#   --yes                跳过启动 auto_enable 前的人工安全确认
#   -h, --help           显示帮助

while (($#)); do
    case "$1" in
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --timesteps) TIMESTEPS="$2"; shift 2 ;;
        --episode-idx) EPISODE_IDX="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --repo-id) REPO_ID="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --skip-can) SKIP_CAN=true; shift ;;
        --no-preview) NO_PREVIEW=true; shift ;;
        --yes) YES=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$TIMESTEPS" =~ ^[1-9][0-9]*$ ]] || { echo "--timesteps 必须是正整数" >&2; exit 2; }
[[ "$EPISODE_IDX" =~ ^[0-9]+$ ]] || { echo "--episode-idx 必须是非负整数" >&2; exit 2; }
[[ "$FPS" =~ ^[1-9][0-9]*$ ]] || { echo "--fps 必须是正整数" >&2; exit 2; }
TIMESTEPS=$((10#$TIMESTEPS))
EPISODE_IDX=$((10#$EPISODE_IDX))
FPS=$((10#$FPS))

for required in \
    "$PIPER_WS/devel/setup.bash" \
    "$CAMERA_WS/devel/setup.bash" \
    "$PIPER_WS/can_config.sh" \
    "$CONDA_SH" \
    "$ALOHA_PYTHON" \
    "$LEROBOT_PYTHON" \
    "$SCRIPT_DIR/ros_lerobot_capture.py" \
    "$SCRIPT_DIR/hdf5_to_lerobot_v2.py" \
    "$SCRIPT_DIR/lerobot_conversion_worker.py"; do
    [[ -e "$required" ]] || { echo "缺少必需文件：$required" >&2; exit 1; }
done

validate_target() {
    "$LEROBOT_PYTHON" "$SCRIPT_DIR/hdf5_to_lerobot_v2.py" \
        --validate-target \
        --dataset-path "$DATASET_PATH" \
        --repo-id "$REPO_ID" \
        --episode-idx "$EPISODE_IDX" \
        --fps "$FPS"
}

validate_target

ROS_SETUP="source /opt/ros/noetic/setup.bash; source '$CAMERA_WS/devel/setup.bash'; source '$PIPER_WS/devel/setup.bash'"
REQUIRED_TOPICS=(
    /camera_f/color/image_raw /camera_l/color/image_raw /camera_r/color/image_raw
    /master/joint_left /master/joint_right /puppet/joint_left /puppet/joint_right
)
ROS_ALREADY_RUNNING=false
if bash -lc "$ROS_SETUP; rosnode list" >/dev/null 2>&1; then
    existing_topics="$(bash -lc "$ROS_SETUP; rostopic list" 2>/dev/null || true)"
    for topic in "${REQUIRED_TOPICS[@]}"; do
        if grep -Fxq "$topic" <<<"$existing_topics"; then
            echo "检测到已有采集话题 $topic；为避免重复连接相机/CAN，本次启动已取消。" >&2
            echo "请先停止现有 PiPER/RealSense 节点后重试。" >&2
            exit 1
        fi
    done
    ROS_ALREADY_RUNNING=true
    echo "检测到已有 ROS master，将复用它（脚本退出时不会关闭该 master）。"
fi

if [[ "$YES" != true ]]; then
    echo
    echo "安全确认：即将以采集模式 mode=0 启动双臂，并传入 auto_enable=true。"
    echo "注意：当前 piper_start_ms_node.py 仅在 mode=1 执行自动使能；mode=0 下该参数不会实际使能。"
    echo "请确认急停可触达、机械臂周围无人和障碍物、机械臂已被可靠支撑。"
    read -r -p "确认后输入 ENABLE 继续：" confirmation
    [[ "$confirmation" == "ENABLE" ]] || { echo "已取消。"; exit 1; }
fi

if [[ "$SKIP_CAN" != true ]]; then
    echo "配置 can_left/can_right（可能要求 sudo 密码）……"
    sudo bash "$PIPER_WS/can_config.sh"
fi

mkdir -p "$(dirname -- "$DATASET_PATH")"
STAGING_DIR="$(mktemp -d "$(dirname -- "$DATASET_PATH")/.piper_capture.XXXXXX")"
RAW_FILE=""
LOG_DIR="$STAGING_DIR/logs"
CONVERSION_STATUS_DIR="$STAGING_DIR/conversion_status"
mkdir -p "$LOG_DIR"
mkdir -p "$CONVERSION_STATUS_DIR"

declare -a CHILD_PIDS=()
declare -A SERVICE_PIDS=()
SESSION_OK=false
CONVERTER_PID=""

stop_children() {
    local pid
    for pid in "${CHILD_PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM -- "-$pid" 2>/dev/null || true
        fi
    done
    for pid in "${CHILD_PIDS[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
}

report_conversion_results() {
    local marker episode
    for marker in "$CONVERSION_STATUS_DIR"/episode_*.done; do
        [[ -e "$marker" ]] || continue
        episode="$(sed -n 's/^episode=//p' "$marker")"
        echo "✓ 转换完成：episode $episode；对应 HDF5 已删除。"
        rm -f -- "$marker"
    done
}

show_conversion_failure() {
    local marker key value log_path=""
    for marker in "$CONVERSION_STATUS_DIR"/episode_*.failed; do
        [[ -e "$marker" ]] || continue
        echo "转换失败：" >&2
        while IFS='=' read -r key value; do
            echo "  $key=$value" >&2
            [[ "$key" == "log" ]] && log_path="$value"
        done < "$marker"
        if [[ -n "$log_path" && -f "$log_path" ]]; then
            echo "转换日志末尾：" >&2
            tail -n 40 "$log_path" >&2 || true
        fi
    done
}

finish_conversion_worker() {
    local status=0
    if [[ -n "$CONVERTER_PID" ]]; then
        echo "等待 LeRobot 转换完成……"
        wait "$CONVERTER_PID" || status=$?
        CONVERTER_PID=""
    fi
    report_conversion_results
    if ((status != 0)); then
        show_conversion_failure
        return "$status"
    fi
    return 0
}

start_deferred_conversion() {
    echo "采集服务已停止；开始按 episode 顺序转换暂存 HDF5。"
    "$LEROBOT_PYTHON" "$SCRIPT_DIR/lerobot_conversion_worker.py" \
        --resume-dir "$STAGING_DIR" \
        --converter "$SCRIPT_DIR/hdf5_to_lerobot_v2.py" \
        --dataset-path "$DATASET_PATH" \
        --repo-id "$REPO_ID" \
        --task "$TASK" \
        --log-dir "$LOG_DIR" \
        --status-dir "$CONVERSION_STATUS_DIR" \
        --video-codec "$VIDEO_CODEC" \
        --video-crf "$VIDEO_CRF" &
    CONVERTER_PID=$!
    echo "LeRobot H.264 转换已启动 (pid=$CONVERTER_PID)。"
}

cleanup() {
    stop_children
    if [[ -n "$CONVERTER_PID" ]]; then
        finish_conversion_worker || true
    fi
    if [[ "$SESSION_OK" == true ]]; then
        rm -rf -- "$STAGING_DIR"
    else
        echo
        echo "会话异常结束；临时数据/日志保留在：$STAGING_DIR" >&2
    fi
}
trap cleanup EXIT

start_group() {
    local name="$1"
    local command="$2"
    setsid bash -lc "$command" >"$LOG_DIR/$name.log" 2>&1 &
    local pid=$!
    CHILD_PIDS+=("$pid")
    SERVICE_PIDS["$name"]="$pid"
    echo "已启动 $name (pid=$pid, log=$LOG_DIR/$name.log)"
}

if [[ "$ROS_ALREADY_RUNNING" != true ]]; then
    start_group roscore "$ROS_SETUP; exec roscore"

    echo -n "等待 ROS master"
    for _ in $(seq 1 40); do
        if bash -lc "$ROS_SETUP; rosnode list" >/dev/null 2>&1; then
            echo "：就绪"
            break
        fi
        echo -n "."
        sleep 0.25
    done
    if ! bash -lc "$ROS_SETUP; rosnode list" >/dev/null 2>&1; then
        echo "：超时" >&2
        exit 1
    fi
fi

start_group cameras "$ROS_SETUP; exec roslaunch realsense2_camera multi_camera.launch"
start_group dual_arms "source '$CONDA_SH'; conda activate aloha; $ROS_SETUP; exec roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=true"

echo "等待相机和双臂话题……"
topics_ready=false
for _ in $(seq 1 120); do
    for service in cameras dual_arms; do
        service_pid="${SERVICE_PIDS[$service]}"
        if ! kill -0 "$service_pid" 2>/dev/null; then
            echo "$service 启动进程已退出，日志如下：" >&2
            tail -n 40 "$LOG_DIR/$service.log" >&2 || true
            exit 1
        fi
    done

    topic_list="$(bash -lc "$ROS_SETUP; rostopic list" 2>/dev/null || true)"
    missing_topics=()
    for topic in "${REQUIRED_TOPICS[@]}"; do
        if ! grep -Fxq "$topic" <<<"$topic_list"; then
            missing_topics+=("$topic")
        fi
    done
    if ((${#missing_topics[@]} == 0)); then
        topics_ready=true
        echo "相机和双臂话题：就绪"
        break
    fi
    if ((_ % 10 == 0)); then
        echo "仍缺少：${missing_topics[*]}"
    fi
    sleep 0.5
done
if [[ "$topics_ready" != true ]]; then
    echo "：超时，请查看 $LOG_DIR" >&2
    exit 1
fi

if [[ "$NO_PREVIEW" != true ]]; then
    start_group camera_preview "source '$CONDA_SH'; conda activate aloha; $ROS_SETUP; exec python '$SCRIPT_DIR/camera_mosaic.py'"
fi

echo
echo "数据集：$DATASET_PATH"
echo "基础服务已全部启动；后续 episode 不再重启 CAN、ROS、相机或双臂。"
echo "输入 q 可结束整个采集会话并关闭本脚本启动的服务。"

source /opt/ros/noetic/setup.bash
source "$CAMERA_WS/devel/setup.bash"
source "$PIPER_WS/devel/setup.bash"
export HF_HOME="$STAGING_DIR/hf_home"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
NEXT_EPISODE_IDX="$EPISODE_IDX"

prompt_episode_settings() {
    local value
    echo
    if ! read -r -p "episode_idx [$EPISODE_IDX]（q 结束）: " value; then
        return 1
    fi
    [[ "$value" == "q" || "$value" == "Q" ]] && return 1
    if [[ -n "$value" ]]; then
        if [[ ! "$value" =~ ^[0-9]+$ ]]; then
            echo "episode_idx 必须是非负整数。"
            return 2
        fi
        EPISODE_IDX=$((10#$value))
    fi

    if ! read -r -p "timesteps [$TIMESTEPS]（q 结束）: " value; then
        return 1
    fi
    [[ "$value" == "q" || "$value" == "Q" ]] && return 1
    if [[ -n "$value" ]]; then
        if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
            echo "timesteps 必须是正整数。"
            return 2
        fi
        TIMESTEPS=$((10#$value))
    fi
    return 0
}

while true; do
    prompt_status=0
    prompt_episode_settings || prompt_status=$?
    if [[ "$prompt_status" == 1 ]]; then
        echo "结束采集；先关闭相机/双臂服务，再统一转换暂存 HDF5。"
        stop_children
        CHILD_PIDS=()
        start_deferred_conversion
        if ! finish_conversion_worker; then
            echo "存在转换失败；失败及尚未转换的 HDF5 和日志已保留。" >&2
            exit 1
        fi
        SESSION_OK=true
        echo "所有 episode 均已转换完成，结束采集会话。"
        break
    fi
    if [[ "$prompt_status" != 0 ]]; then
        continue
    fi
    if ((EPISODE_IDX != NEXT_EPISODE_IDX)); then
        echo "本次采集要求 episode 连续；下一条必须是 $NEXT_EPISODE_IDX。" >&2
        EPISODE_IDX="$NEXT_EPISODE_IDX"
        continue
    fi

    RAW_FILE="$STAGING_DIR/episode_${EPISODE_IDX}.hdf5"
    "$ALOHA_PYTHON" "$SCRIPT_DIR/ros_lerobot_capture.py" \
        --output "$RAW_FILE" \
        --timesteps "$TIMESTEPS" \
        --fps "$FPS"

    echo "HDF5 已暂存：$RAW_FILE"
    RAW_FILE=""
    NEXT_EPISODE_IDX=$((EPISODE_IDX + 1))
    EPISODE_IDX="$NEXT_EPISODE_IDX"
    echo "可以立即开始下一条采集；所有转换将在结束采集后串行执行。"
done
