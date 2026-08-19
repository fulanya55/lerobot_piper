# PiPER dual-arm Pi0 direct deployment

The LeRobot client now controls both PiPER arms directly through
`piper-sdk==0.6.1`:

```text
Pi0 policy server <-> async client <-> piper-sdk <-> can_left / can_right
                                 `-> ROS RealSense RGB topics
```

The old PiPER ROS control nodes must not run at the same time. They would be a
second owner of the same CAN interfaces. The direct client checks ROS node names
and refuses to start when it finds the old left/right PiPER nodes.

## Implementation layers and reuse

The deployment code separates policy-framework concerns from the motor
transport:

```text
async_pi0_client.py              policy/session configuration
  -> BiPiper                     LeRobot feature names, ROS cameras, limits, watchdog
       -> PiperSDKArm            framework-independent CAN state/enable/MOVE J transport
            -> piper-sdk 0.6.1
```

`PiperSDKArm` lives in
`src/lerobot/robots/bi_piper/piper_sdk_arm.py` and imports neither ROS nor
LeRobot. A StarVLA or another VLA integration should reuse this class and add a
small adapter that maps its observation/action tensors to the seven-element
`[joint_1 ... joint_6, gripper]` arrays. The safety policy (limits, watchdog,
and aborting an old asynchronous action queue) should stay in an orchestration
layer above the single-arm transport.

The policy server caches the loaded policy for an identical checkpoint,
device, policy type, and rename map. The first load of this 16 GB Pi0
checkpoint can take several minutes; later client reconnects reuse it.

## Control implementation comparison

| Implementation | CAN owner and command path | Enable/disconnect behavior | Deployment trade-off |
| --- | --- | --- | --- |
| PCD-LeRobot | One `Piper` object per arm; direct `MotionCtrl_2` + `JointCtrl` + `GripperCtrl` | Repeats `EnablePiper` at startup; no bounded timeout or enable watchdog | Small and useful as the verified direct-control reference, but safety and failure handling are minimal |
| Evo-RL | Reusable single-arm `PiperFollower`, composed by `BiPiperFollower`; direct SDK and optional USB-CAN serial resolution | Bounded startup enable; optional disable on disconnect | Best reference for reusable LeRobot composition and device discovery, but it does not abort an already-buffered async rollout after enable loss |
| This project | Framework-independent `PiperSDKArm`, composed by `BiPiper`; direct SDK for arms and ROS only for cameras | All 12 drivers required; measured-pose hold; 10 Hz watchdog; re-enable then abort stale action queue; keep enabled on normal exit by default | More deployment safeguards and a portable hardware boundary; the camera adapter is currently ROS1-specific |

## Safety behavior

- Dry-run connects only for feedback and never enables or commands the arms.
- Live mode waits until all 12 joint drivers report enabled.
- Immediately after enabling, each arm receives its measured joint position as
  a hold target before policy actions are accepted.
- Enabling and the initial hold happen immediately after CAN feedback becomes
  available, before waiting for camera startup or loading the policy.
- Every action is clipped to the SDK joint limits and to a per-step delta from
  measured feedback.
- A 10 Hz watchdog monitors all joint-driver enable bits. A short three-sample
  confirmation window filters one-frame SDK status glitches.
- If enable is lost, the client re-enables at the measured pose and aborts the
  current rollout. Old queued policy actions are not resumed.
- Normal `Ctrl+C` sends one final hold target and leaves the motors enabled. It
  closes the SDK connection but does not call `DisableArm()`.

Keep both arms mechanically supported during the first deployment test and keep
the physical emergency stop reachable.

## 1. Stop the old PiPER ROS control path

Stop the terminal running:

```bash
roslaunch piper start_ms_piper.launch ...
```

Verify that no old PiPER node remains:

```bash
rosnode list | grep -E 'piper_(left|right)' || true
```

Do not stop `roscore` or the RealSense launch. The cameras still use these ROS
topics:

```text
/camera_f/color/image_raw
/camera_l/color/image_raw
/camera_r/color/image_raw
```

## 2. Confirm CAN interfaces

```bash
ip -details link show can_left
ip -details link show can_right
```

Both interfaces must be `UP` at 1 Mbit/s. Run the existing `can_config.sh` only
if they are not configured.

## 3. Start the policy server

```bash
cd /home/agilex/wxwu/lerobot_piper
uv run python examples/piper/policy_server.py \
  --host 127.0.0.1 \
  --port 18080 \
  --fps 30
```

## 4. Observation-only check

The arms must be physically supported because dry-run deliberately does not
enable them:

```bash
cd /home/agilex/wxwu/lerobot_piper
uv run python examples/piper/async_pi0_client.py \
  --server-address 127.0.0.1:18080 \
  --checkpoint /home/agilex/wxwu/model/piper_10000_single/pretrained_model \
  --task "Pick up the cube block from the table with the gripper and place it on the plate with the gripper." \
  --fps 30 \
  --actions-per-chunk 10
```

## 5. Live direct-CAN inference

Use two operators for the first run: one at the terminal and one supporting the
arms with access to the emergency stop.

```bash
cd /home/agilex/wxwu/lerobot_piper
uv run python examples/piper/async_pi0_client.py \
  --server-address 127.0.0.1:18080 \
  --checkpoint /home/agilex/wxwu/model/piper_10000_single/pretrained_model \
  --task "Pick up the cube block from the table with the gripper and place it on the plate with the gripper." \
  --fps 30 \
  --actions-per-chunk 10 \
  --can-left can_left \
  --can-right can_right \
  --velocity 30 \
  --live \
  --confirm-live
```

If any motor fails to enable within 10 seconds, the client exits without
starting policy actions. Do not bypass this check.
