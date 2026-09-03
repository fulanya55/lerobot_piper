#!/usr/bin/env bash
set -euo pipefail

# Thin argparse entrypoint. All model, speed, and dataset settings are supplied
# as command-line options to async_policy_client.py.

readonly REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$REPO_DIR"
exec uv run --frozen python examples/piper/async_policy_client.py "$@"
