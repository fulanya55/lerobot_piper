# PiPER dual-arm Pi0/Pi0.5 direct deployment

This deployment keeps one owner per hardware interface:

```text
Pi0/Pi0.5 policy server <-> async client <-> piper-sdk <-> can_left / can_right
                                         `-> ROS RealSense RGB topics
```

The PiPER ROS arm-control nodes must not run at the same time. ROS remains in
use only for the three RealSense topics. The direct client checks ROS node names
and refuses to start if it finds the old left/right PiPER nodes.

## Model and control contract

The default checkpoint is:

```text
/home/agilex/wxwu/model/PLATE_THE_TUBE_PI05_bs192/14k/pretrained_model
```

The client reads `config.json` to select the policy and robot interface. When
present, it also cross-checks `train_config.json`; when `--dataset-info` is
provided, it validates dataset FPS/schema before opening CAN:

- policy type is selected from the checkpoint (`pi05` and `pi0` are supported);
- output is exactly 14 absolute joint/gripper positions in a recognized
  left-7D then right-7D schema;
- joints use radians; gripper metres versus 0–1 scale is derived from the
  checkpoint feature names and converted explicitly;
- Pi0/Pi0.5 state may be real 14D or padded 32D, while robot control stays 14D;
- checkpoint normalization and unnormalization processors are present;
- the three checkpoint camera aliases map to the actual front/left/right ROS topics;
- the serialized training policy agrees with the exported policy on type,
  shapes, normalization, action names, camera rename map, and horizons;
- the dataset metadata supplies 30 FPS and independently confirms the real
  14D state/action order and all three 30 FPS camera streams.

Policy type is not hardcoded. The currently supported action interfaces are:

| Checkpoint schema | Joint unit | Gripper unit | Deployment conversion |
| --- | --- | --- | --- |
| `left_joint_1..6, left_gripper, ...` | rad | m | direct |
| `left_arm_joint_1_rad..6, left_gripper_open_scale, ...` | rad | 0–1 open scale | scale to 0–0.08 m |
| `left_arm_joint_1_rad..6, left_gripper_open, ...` | rad | 0–1 open scale | scale to 0–0.08 m |

The selected names are also used for the policy-facing observation state, so
legacy Pi0 receives gripper feedback in the same 0–1 convention in which it was
trained. The physical full-open value defaults to 0.08 m and can be changed
with `--gripper-range-m` after hardware calibration. Unknown 14D names/units
fail closed instead of being guessed.
The checkpoint path defaults to the current Pi0.5 deployment only as a
convenience; changing `--checkpoint` selects the model type dynamically and
requires an explicit dataset-appropriate `--task`.

The current default Pi0.5 training dataset is 30 FPS. That checkpoint predicts a 50-action chunk and
stores `n_action_steps=10`. Deployment defaults to 30 Hz and requests all 50
predicted actions for the strict chunk-horizon baseline. `--actions-per-chunk
10` is an explicit trial override: the model still predicts 50, while the
server returns only the first 10.

## Direct PiPER implementation

```text
async_policy_client.py           checkpoint validation and session configuration
  -> BiPiper                     14D ordering, ROS cameras, limits, enable watchdog
       -> PiperSDKArm            radians/metres conversion and absolute MOVE J transport
            -> piper-sdk 0.6.1
```

This follows `/home/agilex/wxwu/PCD-LeRobot`: each arm uses
`C_PiperInterface_V2`, `MotionCtrl_2(0x01, 0x01, ...)`, `JointCtrl`, and
`GripperCtrl`. The deployment layer adds bounded enable timeout, measured-pose
hold, per-step slew limits, and stale asynchronous-action invalidation.

## Enable and stop behavior

There are four explicit modes:

- `validate`: validate checkpoint/dataset interfaces only; do not import policy
  weights or open ROS/CAN.
- `observe`: read feedback, cameras, and policy output without enabling or
  commanding the arms. Mechanically support both arms because a disabled PiPER
  is not self-locking.
- `hold`: enable all 12 joint drivers and command the measured pose; do not load
  or execute a policy.
- `execute`: enable/hold first, then send postprocessed 14D policy positions.

In `hold` and `execute`, a 10 Hz watchdog checks every enable bit. Sustained
enable loss causes re-enable at the measured pose and aborts the rollout so old
queued policy actions cannot resume. Hold mode refreshes the measured-pose
command continuously. Normal `Ctrl+C` sends one final hold and deliberately
does not call `DisableArm()`, but this controller may still lose enable when CAN
closes. Physically support both arms before exit. A process or power failure
cannot be handled by software, so the emergency stop must remain reachable.

## 1. Stable CAN names and preflight

The existing Piper ROS script defines USB path `1-1:1.0` as `can_left` and
`1-2:1.0` as `can_right`. On this host those adapters have stable USB serials:

```text
can_left   002100355631511920313857
can_right  002800175746570A20383839
```

For the current boot, the original script remains the reference command:

```bash
sudo bash /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/can_config.sh
```

To make the names independent of enumeration order and bring both interfaces
up at 1 Mbit/s on every boot, install the serial-matched systemd configuration:

```bash
cd /home/agilex/wxwu/lerobot_piper
sudo bash examples/piper/can/install.sh
sudo reboot
```

The installer only writes configuration and enables the boot service; it does
not touch the live CAN links. If applying it without rebooting, first stop all
PiPER ROS/SDK processes, physically support both arms, and then run `sudo
systemctl start lerobot-piper-can.service`.

Stop only the old PiPER arm-control launch; keep `roscore` and the RealSense
launch running:

```bash
rosnode list | grep -E 'piper_(left|right)' || true
rostopic hz /camera_f/color/image_raw
rostopic hz /camera_l/color/image_raw
rostopic hz /camera_r/color/image_raw
ip -details link show can_left
ip -details link show can_right
```

Both CAN interfaces must be `UP` at 1 Mbit/s. Do not allow any second SDK or ROS
process to own either CAN interface.

For the current host, the complete CAN + roscore + RealSense + policy-server
preflight can be started in one terminal and kept alive with:

```bash
cd /home/agilex/wxwu/lerobot_piper
bash examples/piper/start_inference_stack.sh
```

Then start the current Pi0.5 checkpoint in read-only inference mode from a
second terminal:

```bash
cd /home/agilex/wxwu/lerobot_piper
bash examples/piper/run_pi05_inference.sh
```

## 2. Start the policy server

```bash
cd /home/agilex/wxwu/lerobot_piper
uv run python examples/piper/policy_server.py \
  --host 127.0.0.1 \
  --port 18080 \
  --fps 30 \
  --no-compile-model
```

The checkpoint was trained with `compile_model=true`, but compilation does not
change model weights or preprocessing. This server explicitly overrides it to
`false` by default, avoiding the previously measured roughly 302-second first
`max-autotune` compile. A real eager smoke test of this checkpoint loaded in
163.9 seconds and produced a finite `[1, 50, 14]` chunk in 0.955 seconds. Use
`--compile-model` only after the eager path is validated and repeated-inference
throughput is worth the warmup cost. An identical later client reuses the
already loaded policy in the same server.

## 3. Read-only inference at the training horizon

Support both arms mechanically; this mode intentionally does not enable them:

```bash
cd /home/agilex/wxwu/lerobot_piper
uv run python examples/piper/async_policy_client.py \
  --mode observe \
  --checkpoint /home/agilex/wxwu/model/PLATE_THE_TUBE_PI05_bs192/14k/pretrained_model \
  --dataset-info /home/agilex/wxwu/data/PLACE_THE_TEST_TUBE/meta/info.json \
  --task "Place the test tube on the test tube rack on the desk with the gripper." \
  --fps 30 \
  --actions-per-chunk 50
```

Confirm that the server reports `type=pi05` and action shape `[1, 50, 14]`.
This proves preprocessing/inference/postprocessing but does not validate motion.

## 4. Enable and hold without policy motion

With an operator supporting each arm and the emergency stop reachable:

```bash
cd /home/agilex/wxwu/lerobot_piper
uv run python examples/piper/async_policy_client.py \
  --mode hold \
  --confirm-enable \
  --checkpoint /home/agilex/wxwu/model/PLATE_THE_TUBE_PI05_bs192/14k/pretrained_model \
  --fps 30
```

Leave this running long enough to verify that all 12 drivers remain enabled and
that the measured pose does not drift. Before `Ctrl+C`, physically support both
arms: the client does not send `DisableArm()`, but closing CAN may still cause
controller-side enable loss.

## 5. Bounded first policy-motion trial

This is a real-motion step and must not be run until the read-only and hold
checks pass. Keep the full 50-step deployment horizon so the roughly 0.3-second
inference fits inside the prefetched overlap. `--max-policy-actions 3` still
executes only three commands total, then sends a hold and leaves both arms
enabled:

```bash
cd /home/agilex/wxwu/lerobot_piper
uv run python examples/piper/async_policy_client.py \
  --mode execute \
  --confirm-enable \
  --confirm-live \
  --checkpoint /home/agilex/wxwu/model/PLATE_THE_TUBE_PI05_bs192/14k/pretrained_model \
  --task "Place the test tube on the test tube rack on the desk with the gripper." \
  --fps 30 \
  --actions-per-chunk 50 \
  --max-policy-actions 3 \
  --velocity 20
```

After reviewing feedback versus requested/sent targets, increase
`--max-policy-actions` gradually. Omit it only when a continuous rollout has
been separately approved and the limited trials are already safe.

Every observe/execute run writes JSONL telemetry to
`logs/piper_actions_<timestamp>.jsonl`. Each action row includes the raw policy
target, the target actually returned by the robot safety layer, measured joint
state, action timestep, and remaining queue size. A `queue_starved` row is a
control-path fault and should be investigated before continuing a live rollout.

## Save each inference as a LeRobot v2.1 episode

Recording is enabled only for `execute` mode so a dry-run policy target cannot
be mistaken for an action that reached the robot. One rollout appends one
episode; the next `episode_idx` is read from `meta/info.json` automatically:

```bash
cd /home/agilex/wxwu/lerobot_piper
uv run --frozen python examples/piper/async_policy_client.py \
  --mode execute \
  --confirm-enable \
  --confirm-live \
  --checkpoint /home/agilex/wxwu/model/PLATE_THE_TUBE_PI05_bs192/14k/pretrained_model \
  --dataset-info /home/agilex/wxwu/data/PLACE_THE_TEST_TUBE/meta/info.json \
  --task "Place the test tube on the test tube rack on the desk with the gripper." \
  --fps 30 \
  --actions-per-chunk 50 \
  --record-dataset-path /home/agilex/wxwu/data/PI05_INFERENCE_RECORDS
```

Recording starts with the first action actually returned by the safety layer.
Press Enter to stop before the next action. The client first sends its normal
final measured-pose hold and closes CAN/cameras, then converts the episode to
the same LeRobot v2.1 contract used by `one_click_collect.sh`: three H.264 MP4
streams, Parquet, metadata, 14D state/action, velocity, effort, and source
timestamps. The direct SDK supplies no velocity/effort in this deployment, so
velocity is a measured-state finite difference and effort is zero-filled with
that provenance written into the staging HDF5. The saved `action` is the 14D
post-limit command actually sent, not the unclipped raw model target.

The raw file is staged next to the dataset under
`.PI05_INFERENCE_RECORDS_inference_staging/episode_N.hdf5`. It is removed only
after successful conversion; on an inference or conversion error it remains
there for recovery. To assert an index instead of auto-selecting it, add
`--record-episode-idx N`.

`run_pi05_inference.sh` is a thin argparse entrypoint. Configure speed and data
with command-line arguments:

```bash
bash examples/piper/run_pi05_inference.sh \
  --mode execute \
  --confirm-enable \
  --confirm-live \
  --velocity 45 \
  --dataset-info /home/agilex/wxwu/data/PLACE_THE_TEST_TUBE/meta/info.json \
  --record-dataset-path /home/agilex/wxwu/data/MY_INFERENCE_DATA \
  --record-repo-id local/my_inference
```

Argparse keeps the original execution behavior by default: SDK
`velocity=30`, joint step `0.05 rad`, gripper step `0.005 m`, and trajectory
smoothing disabled:

| Argument | Default | Purpose |
| --- | --- | --- |
| `--velocity` | `30` | PiPER SDK MOVE J velocity, valid range 1–100 |
| `--max-joint-step-rad` | `0.05` | Final measured-position joint step envelope |
| `--max-gripper-step-m` | `0.005` | Final measured-position gripper step envelope |
| `--trajectory-smoothing` | disabled | Opt in to velocity/acceleration trajectory limiting |
| `--dataset-info` | `PLACE_THE_TEST_TUBE/meta/info.json` | FPS and 14D schema validation source |
| `--record-dataset-path` | unset | LeRobot v2.1 inference dataset destination |
| `--record-repo-id` | `local/piper_inference` | Logical repo id stored by the converter |
| `--record-episode-idx` | auto | Optional assertion for the next episode index |
| `--record-staging-dir` | next to destination | Raw HDF5 recovery directory |

## Smooth chunk execution

The direct-CAN client now uses three model-independent smoothing layers:

1. At 50% queue depth it sends exactly one forced prefetch request. The server's
   generic joint-state similarity filter cannot discard that request, and no
   second request is sent while inference is in flight.
2. When the new chunk arrives, the next 5 overlapping actions stay committed
   and the following 10 actions use a smoothstep cross-fade from the old chunk
   to the new chunk.
3. When `--trajectory-smoothing` is explicitly enabled, the dual-PiPER adapter
   applies stateful joint/gripper velocity and acceleration limits before
   `JointCtrl`. The default original-speed path leaves it disabled. Mechanical
   limits and the measured-pose per-frame safety envelope remain active in both
   modes as a final guard.

Defaults can be changed with:

```text
--prefetch-ratio 0.5
--inference-request-timeout-s 3
--action-commit-steps 5
--action-blend-steps 10
--trajectory-smoothing / --no-trajectory-smoothing  # default: disabled
--max-joint-velocity-rad-s 1.0
--max-joint-acceleration-rad-s2 4.0
--max-gripper-velocity-m-s 0.08
--max-gripper-acceleration-m-s2 0.4
--telemetry-path PATH
```

Do not use `--actions-per-chunk 10` for a continuous 30 Hz rollout while
inference takes about 0.3 seconds: ten actions cover only 0.33 seconds and leave
almost no latency margin. Use the full 50-step chunk and limit a first trial
with `--max-policy-actions` instead.

## RTC status and configuration

Pi0.5 in this LeRobot revision declares RTC support, but this direct-CAN gRPC
deployment does **not** currently enable RTC. The prefetch, committed prefix,
cross-fade, and trajectory limiter above improve continuity for Pi0, Pi0.5, and
other compatible policies, but they do not feed the unexecuted prefix and
measured inference delay back into Pi0.5's denoising process.
`train_config.json` also records `rtc_config=null`.

The repository's complete RTC implementation is the `lerobot-rollout` backend.
Its relevant options are:

```text
--inference.type=rtc
--inference.rtc.enabled=true
--inference.rtc.execution_horizon=10
--inference.rtc.max_guidance_weight=10
--inference.rtc.prefix_attention_schedule=linear
--inference.queue_threshold=30
--use-torch-compile=false
```

That backend keeps the original and postprocessed action queues together,
tracks actual inference latency, and passes `prev_chunk_left_over` plus
`inference_delay` to Pi0.5. It should be integrated as a separate second stage,
after the non-RTC read-only/hold/bounded-motion baseline is validated. The
initial RTC trial should keep `execution_horizon=10`, disable compilation, and
use the same 30 Hz, 14D absolute-position safety limits; do not interpret RTC
as a replacement for per-frame joint/gripper slew limiting.
