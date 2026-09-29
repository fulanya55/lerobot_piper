#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../"
exec uv run --frozen --extra pi --extra async python examples/piper/direct_inference.py "$@"
