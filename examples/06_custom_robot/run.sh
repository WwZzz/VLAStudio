#!/usr/bin/env bash
# Example 06 launcher (Linux/macOS).
#   ./run.sh          -> keyboard teleop + live camera (default)
#   ./run.sh camera   -> live camera only
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
./.venv/bin/python examples/06_custom_robot/run.py "${1:-teleop}"
