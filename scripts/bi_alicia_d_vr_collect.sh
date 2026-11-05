#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export VLASTUDIO_ALICIA_FAULT_DIR="${VLASTUDIO_ALICIA_FAULT_DIR:-$ROOT/local/alicia_serial/thermal_faults}"
mkdir -p "$HOME/.cache"
exec 9>"$HOME/.cache/vlastudio-bi-alicia-teleop.lock"
flock -n 9 || { echo 'Another Alicia-D teleop/capture session is running.' >&2; exit 1; }
exec "${VLASTUDIO_PYTHON:-/home/jetson/miniforge3/envs/vlastudio/bin/python}" -m vlastudio collect \
  --runtime current \
  --robot "$ROOT/src/vlastudio/configs/robot/bi_alicia_d_rel_ee.yaml" \
  --teleop "$ROOT/src/vlastudio/configs/teleop/quest3_bi_alicia_rel_ee.yaml" \
  --no-record-teleop --frequency 25 --dataset-format hdf5 "$@"
