#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${VLASTUDIO_PYTHON:-/home/jetson/miniforge3/envs/vlastudio/bin/python}" "$ROOT/scripts/rebot_b601rs_vr_tele.py" "$@"
