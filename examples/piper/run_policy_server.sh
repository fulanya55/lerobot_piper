#!/usr/bin/env bash
set -euo pipefail

# Policy-only process. It never opens CAN or ROS; the client owns both arms.
readonly REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
exec uv run --frozen --extra pi --extra async python examples/piper/policy_server.py \
  --host "${PIPER_SERVER_HOST:-127.0.0.1}" \
  --port "${PIPER_SERVER_PORT:-18080}" \
  --fps "${PIPER_FPS:-30}" \
  --no-compile-model "$@"
