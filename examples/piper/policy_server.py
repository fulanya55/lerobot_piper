#!/usr/bin/env python
"""Start the LeRobot async policy server for the local PiPER deployment."""

import argparse

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    serve(PolicyServerConfig(host=args.host, port=args.port, fps=args.fps))


if __name__ == "__main__":
    main()
