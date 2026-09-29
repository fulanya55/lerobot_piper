# 直接 LeRobot 采集

`direct_collect.sh` 将三路 RealSense 与双臂关节同步后，通过 Unix socket 送入
`lerobot_piper` 的 uv 环境。写入端使用 `LeRobotDataset` 的
`streaming_encoding=True`，因此录制过程中直接生成：

- `videos/observation.images.*/*/*.mp4`
- `data/*/*.parquet`（action、state、velocity、effort、时间戳）

当前连接的三台 D435 使用 `960x540@60`；LeRobot 保存时每个相机保留自己的分辨率。

命令行采集（总是使用项目 uv 环境）：

```bash
./lerobot_piper/script/direct_collect.sh --dataset-path /home/agilex/wxwu/data/piper_lerobot_direct \
  --repo-id local/piper_dual_arm --episode-idx 0 --timesteps 3000 \
  --camera-resolution 960x540
```

网页控制台：

```bash
cd /home/agilex/wxwu/lerobot_piper
UV_CACHE_DIR=/tmp/wxwu-uv-cache uv run --frozen --extra dataset \
  python script/collect_web.py
```

浏览器打开 `http://127.0.0.1:8765`。点击“开始采集”会自动启动 ROS、相机、双臂
和直接写入进程；完成后页面会列出可播放的 MP4。网页启动默认跳过确认并自动开始，
使用前请确认急停可触达且机械臂已经可靠支撑。
