#!/usr/bin/env bash
# Record Rebot feedback and the head camera without enabling any motor.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HOME/.cache"
exec 8>"$HOME/.cache/vlastudio-rebot-session.lock"
exec 9>"$HOME/.cache/vlastudio-bi-alicia-teleop.lock"
flock -n 8 || { echo "Rebot session already running" >&2; exit 1; }
flock -n 9 || { echo "Alicia camera session already running" >&2; exit 1; }
exec "${VLASTUDIO_PYTHON:-/home/jetson/miniforge3/envs/vlastudio/bin/python}" -m vlastudio.entrypoints.collect_data \
  --robot "$root/src/vlastudio/configs/robot/rebot_b601rs_head_readonly.yaml" \
  --frequency 30 --dataset-format hdf5 --hdf5-compression lzf --no-record-teleop \
  --output-dir "${1:-$root/data/rebot_head}"
