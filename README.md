# lerobot_piper：PiPER 双臂机器人实验

本仓库是 LeRobot 在 AgileX PiPER 双臂上的实验版本，包含三路 RealSense
同步采集、LeRobot 数据集转换、Pi0.5 微调、PiPER 直接推理，以及可选的
EEG-RLHF 训练代码。仓库中的训练和部署脚本围绕 **30 FPS、14 维双臂状态/动作、
三路 RGB 图像** 设计。

## 1. 路径约定

当前训练机使用 `/root/wxwu`，机器人机端脚本示例使用 `/home/agilex/wxwu`。
这两个路径代表同一套目录结构在不同主机上的挂载点；换机器后需要修改 YAML、
Shell 脚本中的绝对路径，或通过脚本支持的环境变量覆盖。

```bash
export REPO_ROOT=/root/wxwu/lerobot_piper
export DATA_ROOT=/root/wxwu/dataset/EEG
export MODEL_ROOT=/root/wxwu/model
export OUTPUT_ROOT=$REPO_ROOT/outputs
cd "$REPO_ROOT"
```

| 内容 | 当前路径 | 说明 |
| --- | --- | --- |
| 代码 | `/root/wxwu/lerobot_piper` | 本仓库 |
| LeRobot 数据 | `/root/wxwu/dataset/EEG` | v2.1/v3 数据集、原始采集和 EEG 资料 |
| Pi0.5 基座 | `/root/wxwu/model/pi05_base` | 微调的 `pretrained_path` |
| PaliGemma tokenizer | `/root/wxwu/model/paligemma-3b-pt-224` | Pi0.5 文本/视觉 tokenizer |
| 训练输出 | `/root/wxwu/lerobot_piper/outputs/train` | checkpoints、`train_config.json`、日志 |
| EEG-RLHF 输出 | `/root/wxwu/lerobot_piper/outputs/eeg_rlhf` | scorer、stage1 特征、stage2 policy |
| 运行时日志 | `/tmp/lerobot-piper-*` | ROS、相机、policy server、client 日志 |

数据和模型文件很大，不提交到 Git。提交前只检查代码、配置和文档；数据集根目录
必须包含 `meta/info.json`，视频在 `videos/`，状态和动作在 `data/`。

## 2. 已使用的数据集

下表是当前机器上可以直接检查的路径。训练配置中的 `dataset.repo_id` 是本地逻辑
名称，实际读取位置由 `dataset.root` 决定。

| 实验 | 数据集根目录 | 规模/格式 | 训练入口 |
| --- | --- | --- | --- |
| 组装笔帽 | `$DATA_ROOT/0907/ATTACH_CAP_TO_PEN_1_1` | 152 episodes，约 168.7k 帧，30 FPS，14D，v2.1 | `scripts/train_piper/train_pi05_full.sh` |
| 放置试管（清洗集） | `$DATA_ROOT/PLACE_THE_TEST_TUBE_NEW/PLACE_THE_TEST_TUBE_FIX` | 151 episodes，74,357 帧，30 FPS，14D，v3.0 | `scripts/train_piper/train_pi05_place_test_tube.sh` |
| 放置试管（成功轨迹合并） | `$DATA_ROOT/PLACE_THE_TEST_TUBE_NEW/PLACE_THE_TEST_TUBE_MERGED` | 173 episodes，158,863 帧，30 FPS，14D，v2.1 | `scripts/train_piper/train_pi05_place_test_tube_full_bs8_ga6_ep4_20260922.yaml` |
| Galbot 从微波炉取面包 | `$DATA_ROOT/Galbot_G1_take_bread_from_microwave_0909_2720_pi05_23d_3cam` | 100 episodes，73,826 帧，30 FPS，23D 投影，v2.1 | `scripts/train_piper/train_pi05_galbot_g1_bread.sh` |
| EEG/RLHF 原始资料 | `$DATA_ROOT/RLHF_EEG_new_paradigm` | EEG、winner/loser 记录和预处理结果 | `lerobot-train-eeg-rlhf` |

原始 Galbot 数据在 `$DATA_ROOT/Galbot_G1_take_bread_from_microwave_0909_2720`；
`train_pi05_galbot_g1_bread.sh` 会调用
`scripts/train_piper/project_galbot_g1_for_pi05.py`，从 38D 源数据删除底盘位置、
速度和里程计字段，生成 23D 的三相机训练集。Pi0.5 内部仍将动作/状态补齐到
32D，以保持 checkpoint 接口一致。

### 数据目录结构

```text
<dataset-root>/
├── meta/info.json
├── meta/episodes.jsonl                 # 每个 episode 的长度和任务
├── meta/stats.json                     # 状态/动作统计（若数据集提供）
├── data/chunk-*/episode_*.parquet     # state、action、timestamp
└── videos/chunk-*/observation.images.*/episode_*.mp4
```

PiPER 物理接口始终是左臂 7D + 右臂 7D，共 14D。Pi0.5 的 `max_state_dim` 和
`max_action_dim` 为 32；这只是模型输入输出的 padding，不改变发送给 PiPER 的
14D 关节目标。三路相机的常用别名为 `cam_high`、`cam_left_wrist`、
`cam_right_wrist`；不同数据集的重命名规则写在对应 YAML 的 `rename_map`。

## 3. 环境安装

```bash
cd /root/wxwu/lerobot_piper
uv sync --locked --extra pi --extra async --extra dataset
```

机器人采集还需要主机上的 ROS Noetic、PiPER ROS 工作空间、RealSense 驱动、
CAN 配置和 `aloha` conda 环境。训练只需要 CUDA、PyTorch 和仓库的 `.venv`；
8 GPU 训练默认使用 FSDP2、bf16、每卡 batch size 4、gradient accumulation 6。

先确认 CLI 和环境可用：

```bash
uv run lerobot-info
uv run lerobot-train --help
script/one_click_collect.sh --help
script/direct_collect.sh --help
```

## 4. 数据采集流程

### 4.1 采集前检查

1. 支撑两条机械臂，确认急停可触达，清空工作空间。
2. 配置 `can_left`、`can_right`，并确认均为 1 Mbit/s：

   ```bash
   ip -details link show can_left
   ip -details link show can_right
   ```

3. 启动 ROS master 和三路 RGB 话题，确认以下话题持续有帧：

   ```text
   /camera_f/color/image_raw
   /camera_l/color/image_raw
   /camera_r/color/image_raw
   /master/joint_left    /master/joint_right
   /puppet/joint_left    /puppet/joint_right
   ```

4. 直接 SDK 推理时不能同时运行 PiPER ROS 双臂控制节点；ROS 只保留相机话题。

### 4.2 兼容采集：先写 HDF5，再转 LeRobot v2.1

`one_click_collect.sh` 会把每个 episode 临时写入 HDF5，结束采集会话后按索引
串行转换并写出 MP4、Parquet 和 metadata。episode 必须从 0 连续递增：

```bash
cd /home/agilex/wxwu/lerobot_piper
./script/one_click_collect.sh \
  --dataset-path /home/agilex/wxwu/data/piper_lerobot_v2 \
  --repo-id local/piper_dual_arm \
  --episode-idx 0 \
  --timesteps 3000 \
  --fps 30 \
  --task "dual-arm manipulation"
```

异常退出时，脚本会保留 `.piper_capture.*` 临时目录和转换日志，先修复失败的
episode，再从下一个连续索引继续。转换前可单独检查目标目录：

```bash
uv run python script/hdf5_to_lerobot_v2.py \
  --validate-target \
  --dataset-path /home/agilex/wxwu/data/piper_lerobot_v2 \
  --repo-id local/piper_dual_arm \
  --episode-idx 0 \
  --fps 30
```

### 4.3 直接采集：边采集边写 v2.1

`direct_collect.sh` 通过 Unix socket 将同步帧交给
`examples/piper/lerobot_v21_stream_writer.py`，实时写出三路 H.264 MP4 和
Parquet，默认相机分辨率为 `960x540`：

```bash
cd /home/agilex/wxwu/lerobot_piper
./script/direct_collect.sh \
  --dataset-path /home/agilex/wxwu/data/piper_lerobot_direct_v2 \
  --repo-id local/piper_dual_arm \
  --episode-idx 0 \
  --timesteps 3000 \
  --fps 30 \
  --camera-resolution 960x540 \
  --continuous
```

网页控制台入口为 `script/collect_web.py`，默认地址 `http://127.0.0.1:8765`：

```bash
cd /home/agilex/wxwu/lerobot_piper
UV_CACHE_DIR=/tmp/wxwu-uv-cache uv run --frozen --extra dataset \
  python script/collect_web.py
```

## 5. Pi0.5 训练流程

每次训练按以下顺序执行：

1. 确认 `meta/info.json` 的 `fps`、episode 数、总帧数、相机 key 和 action/state 维度。
2. 在 YAML 中确认 `dataset.root`、`pretrained_path`、输出目录和重命名规则。
3. 先用 `DRY_RUN=1` 做路径、metadata、GPU 和梯度累积检查。
4. 使用 8 GPU 启动全参数 Pi0.5 微调，保存 checkpoint 和训练日志。
5. 从 `outputs/train/<run-id>/checkpoints/` 选择 checkpoint，先离线验证输入输出
   shape，再进行 `observe`，最后才进入真实机械臂 `execute`。

### 5.1 组装笔帽

`train_pi05_full.sh` 默认使用 8 GPU、每卡 batch 4、GA 6、5 个 epoch，当前
数据约对应 26,370 个 micro-steps。当前机器的数据实际位于
`$DATA_ROOT/0907/ATTACH_CAP_TO_PEN_1_1`，而脚本/YAML 仍写有
`/root/wxwu/dataset/0907/ATTACH_CAP_TO_PEN_1_1`；运行前请将两处 `dataset.root`
和脚本中的 `DATASET_ROOT` 改为实际存在的路径，或建立等价软链接。

```bash
cd /root/wxwu/lerobot_piper
DRY_RUN=1 ./scripts/train_piper/train_pi05_full.sh
./scripts/train_piper/train_pi05_full.sh
```

可用环境变量覆盖本次运行而不改 YAML：
`RUN_ID`、`OUTPUT_DIR`、`STEPS_OVERRIDE`、`MASTER_PORT`、`WANDB_ENABLE`、
`WANDB_MODE`、`CUDA_VISIBLE_DEVICES`。

### 5.2 放置试管

清洗集入口：

```bash
cd /root/wxwu/lerobot_piper
DRY_RUN=1 ./scripts/train_piper/train_pi05_place_test_tube.sh
./scripts/train_piper/train_pi05_place_test_tube.sh
```

成功轨迹合并集使用
`scripts/train_piper/train_pi05_place_test_tube_full_bs8_ga6_ep4_20260922.yaml`。
它对应 `$DATA_ROOT/PLACE_THE_TEST_TUBE_NEW/PLACE_THE_TEST_TUBE_MERGED`、8 GPU、
每卡 batch 8、GA 6、4 个 epoch（9,936 micro-steps）。该 YAML 历史上也出现过
不带 `EEG` 的 `/root/wxwu/dataset/PLACE_THE_TEST_TUBE_NEW/...` 路径，需以本机
实际存在的 `meta/info.json` 为准。

### 5.3 Galbot 面包任务

脚本会先把源数据投影成 23D，再训练三相机 Pi0.5：

```bash
cd /root/wxwu/lerobot_piper
DATASET_SOURCE="$DATA_ROOT/Galbot_G1_take_bread_from_microwave_0909_2720" \
PROJECTED_DATASET_ROOT="$DATA_ROOT/Galbot_G1_take_bread_from_microwave_0909_2720_pi05_23d_3cam" \
DRY_RUN=1 ./scripts/train_piper/train_pi05_galbot_g1_bread.sh

DATASET_SOURCE="$DATA_ROOT/Galbot_G1_take_bread_from_microwave_0909_2720" \
PROJECTED_DATASET_ROOT="$DATA_ROOT/Galbot_G1_take_bread_from_microwave_0909_2720_pi05_23d_3cam" \
./scripts/train_piper/train_pi05_galbot_g1_bread.sh
```

输出目录默认为 `outputs/train/piper_pi05_galbot_g1_bread_bs4_ga6_ep5_3cam_*`；
若续训，设置 `RESUME_CHECKPOINT=/absolute/path/to/checkpoint`。

## 6. PiPER 推理与真实机器人验证

部署 checkpoint 需要包含 `config.json`、policy 权重、预处理/后处理器和训练配置。
机器人机端当前示例默认使用：

```text
/home/agilex/wxwu/model/pretrained_model
/home/agilex/wxwu/data/ATTACH_CAP_TO_PEN_1/meta/info.json
```

这两个路径不在本仓库中，需将训练结果导出或复制到机器人机端。

### 6.1 启动基础服务和 policy server

```bash
cd /home/agilex/wxwu/lerobot_piper
bash examples/piper/start_inference_stack.sh
```

该脚本只启动 CAN 检查、roscore、三路 RealSense 和 policy server，不启动旧的
PiPER ROS 双臂控制节点。也可以分开运行 `run_policy_server.sh` 和客户端。

### 6.2 按安全顺序验证

先做不使能的输入输出验证：

```bash
cd /home/agilex/wxwu/lerobot_piper
bash examples/piper/run_pi05_inference.sh \
  --mode observe \
  --checkpoint /home/agilex/wxwu/model/pretrained_model \
  --dataset-info /home/agilex/wxwu/data/ATTACH_CAP_TO_PEN_1/meta/info.json \
  --task "Pick up the pen cap and pen body, attach the cap to the body, then place the assembled pen into the pen holder." \
  --fps 30 \
  --actions-per-chunk 50
```

确认 server 报告 `type=pi05` 和 `[1, 50, 14]` 后，再运行 `hold` 检查使能和保持
当前位置，最后使用 `execute`。执行模式必须有人支撑两臂并保持急停可触达；可用
`--max-policy-actions N` 或 `PIPER_MAX_POLICY_ACTIONS=N` 限制试验长度。

控制端的四种模式为：

| 模式 | 行为 |
| --- | --- |
| `validate` | 只校验 checkpoint、数据集 schema 和归一化配置，不打开 ROS/CAN |
| `observe` | 读取相机/状态并运行 policy，不使能、不发送机械动作 |
| `hold` | 使能并持续发送实测位姿，不执行 policy |
| `execute` | 先 hold，再以 30 Hz 发送 policy 的 14D 绝对关节目标 |

遇到异常时先 `Ctrl+C` 让控制器保持实测位姿，再关闭 CAN；不要让多个 ROS/SDK
进程同时占用同一条 CAN 总线。更完整的参数和故障处理见
[`examples/piper/README.md`](examples/piper/README.md)。

## 7. EEG-RLHF 实验流程（可选）

EEG-RLHF 分为两个阶段，入口为 `lerobot-train-eeg-rlhf`：

1. **Stage 1：** 用 winner/loser 成对轨迹、行为 hidden features 和 EEG 窗口训练
   Bradley–Terry scorer 及 EEG 对齐分支，保存完整 scorer 和 EEG-free rollout scorer。
2. **Stage 2：** 冻结 EEG-free scorer，使用相同的成功 replay、FlowPRO/RPRO 和
   SFT anchor 更新 Pi0/Pi0.5。部署时只带更新后的 policy，不带 EEG 编码器和 scorer。

相关资料和输出路径：

```text
原始 EEG/manifest：$DATA_ROOT/RLHF_EEG_new_paradigm
BrainMU-H 源码：    /root/wxwu/Brianmu_tokenizer_human_demo/src
BrainMU-H checkpoint：/root/wxwu/model/unitok_tok-0507-16-rope-ft_0030000_final
实验输出：          /root/wxwu/lerobot_piper/outputs/eeg_rlhf
```

示例命令（manifest 需替换成实际 winner/loser 配对文件）：

```bash
cd /root/wxwu/lerobot_piper
uv run lerobot-train-eeg-rlhf \
  --stage all \
  --dataset-repo-id PLACE_THE_TEST_TUBE_MERGED \
  --dataset-root "$DATA_ROOT/PLACE_THE_TEST_TUBE_NEW/PLACE_THE_TEST_TUBE_MERGED" \
  --winner-manifest /absolute/path/to/winner.csv \
  --loser-manifest /absolute/path/to/loser.csv \
  --pretrained-policy "$MODEL_ROOT/pi05_base" \
  --policy pi05 \
  --output-dir "$OUTPUT_ROOT/eeg_rlhf/piper_pi05_place_test_tube" \
  --eeg-tokenizer-src /root/wxwu/Brianmu_tokenizer_human_demo/src \
  --eeg-checkpoint-path /root/wxwu/model/unitok_tok-0507-16-rope-ft_0030000_final \
  --trust-eeg-checkpoint \
  --device cuda
```

不要把 EEG latent 预先缓存后混入训练集；当前实现会在 DataLoader 中按
`timestamp[t]` 读取并编码 EEG。完整接口说明见 [`docs/source/eeg_rlhf.mdx`](docs/source/eeg_rlhf.mdx)。

## 8. 检查和复现

提交前执行：

```bash
git status --short
git diff --check
uv run pytest tests/datasets/test_eeg_rlhf.py tests/policies/common/test_flow_matching.py tests/rewards/test_dense_eeg_reward.py -q
```

训练复现至少记录：commit、数据集 `meta/info.json`、预训练 checkpoint 路径、
GPU 数量、每卡 batch、GA、steps、随机种子和输出目录。训练日志位于
`outputs/train/` 或仓库根目录下的 `piper_pi05_*.log`、`stage2_rpro_*.log`。

## 9. 相关入口

- 采集说明：[`script/README.md`](script/README.md)、[`script/README_direct_collect.md`](script/README_direct_collect.md)
- PiPER 推理：[`examples/piper/README.md`](examples/piper/README.md)
- Pi0.5 配置：[`scripts/train_piper/`](scripts/train_piper/)
- EEG-RLHF 设计：[`docs/source/eeg_rlhf.mdx`](docs/source/eeg_rlhf.mdx)
- LeRobot API 和通用 CLI：[`AGENT_GUIDE.md`](AGENT_GUIDE.md)
