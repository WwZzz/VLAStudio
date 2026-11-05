#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Keep using the previously established per-body fault records on this Jetson.
export VLASTUDIO_ALICIA_FAULT_DIR="${VLASTUDIO_ALICIA_FAULT_DIR:-$ROOT/local/alicia_serial/thermal_faults}"
exec "${VLASTUDIO_PYTHON:-/home/jetson/miniforge3/envs/vlastudio/bin/python}" "$ROOT/scripts/bi_alicia_d_vr_tele.py" "$@"
