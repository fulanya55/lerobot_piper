#!/usr/bin/env python
"""Run the local dual-arm PiPER direct-CAN client against a LeRobot async policy server.

The three cameras remain ROS topics, while piper-sdk owns can_left/can_right.
The default is deliberately read-only until ``--live`` is supplied explicitly.
"""

import argparse
import threading


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/home/agilex/wxwu/model/piper_10000_single/pretrained_model",
    )
    parser.add_argument("--server-address", default="127.0.0.1:18080")
    parser.add_argument("--task", default="place the cube block")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--actions-per-chunk", type=int, default=10)
    parser.add_argument("--can-left", default="can_left")
    parser.add_argument("--can-right", default="can_right")
    parser.add_argument("--velocity", type=int, default=30)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable both arms and send actions directly through piper-sdk. Omit for dry-run.",
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required together with --live as an explicit actuation acknowledgement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.live and not args.confirm_live:
        raise SystemExit("Refusing live actuation without --confirm-live")

    print("[PiPER client] Loading LeRobot client modules...", flush=True)
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.robots.bi_piper import BiPiperConfig

    robot_config = BiPiperConfig(
        id="piper_dual",
        can_left=args.can_left,
        can_right=args.can_right,
        velocity=args.velocity,
        dry_run=not args.live,
        keep_enabled_on_disconnect=True,
    )
    client_config = RobotClientConfig(
        robot=robot_config,
        server_address=args.server_address,
        policy_device="cuda",
        client_device="cpu",
        policy_type="pi0",
        pretrained_name_or_path=args.checkpoint,
        actions_per_chunk=args.actions_per_chunk,
        task=args.task,
        fps=args.fps,
        chunk_size_threshold=0.5,
        aggregate_fn_name="latest_only",
    )
    print("[PiPER client] Connecting to ROS cameras and direct PiPER CAN feedback...", flush=True)
    client = RobotClient(client_config)
    mode = "LIVE" if args.live else "DRY-RUN"
    print(f"[PiPER client] Cameras and PiPER SDK ready. Mode: {mode}", flush=True)
    print(
        "[PiPER client] Requesting checkpoint load from policy server "
        "(the first 16 GB checkpoint load can take several minutes)...",
        flush=True,
    )
    if not client.start():
        client.stop()
        raise RuntimeError(f"Unable to connect to policy server at {args.server_address}")
    print("[PiPER client] Policy ready. Starting inference loop.", flush=True)

    action_receiver = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver.start()
    try:
        client.control_loop(task=args.task)
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        action_receiver.join()


if __name__ == "__main__":
    main()
