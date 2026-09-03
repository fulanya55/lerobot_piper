# `script/`

脚本的统一操作说明在 [《PiPER 双臂推理与数据采集操作手册》](../推理与采集操作手册.md)。

常用入口：

- `one_click_collect.sh`：ROS + 三路 RealSense + 双臂主从采集。
- `direct_collect.sh`：推荐入口；直接写入 LeRobot v3（流式 MP4 + Parquet），默认 960x540。
- `collect_web.py`：本地网页控制台与录制后预览。
- `ros_lerobot_capture.py`：采集单条原始 HDF5。
- `hdf5_to_lerobot_v2.py`：转换为 LeRobot v2.1。
- `lerobot_conversion_worker.py`：会话结束后按 episode 顺序续转。
- `camera_mosaic.py`：三路相机预览。

直接采集请使用项目 uv 环境中的写入端，详见 [直接采集说明](README_direct_collect.md)。

查看参数（在 `lerobot_piper` 根目录执行）：

```bash
script/one_click_collect.sh --help
script/direct_collect.sh --help
script/ros_lerobot_capture.py --help
script/hdf5_to_lerobot_v2.py --help
```
