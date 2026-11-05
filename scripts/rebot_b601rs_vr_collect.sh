#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HOME/.cache"
exec 8>"$HOME/.cache/vlastudio-rebot-session.lock"
exec 9>"$HOME/.cache/vlastudio-bi-alicia-teleop.lock"
flock -n 8 || { echo "Rebot session already running" >&2; exit 1; }
flock -n 9 || { echo "Camera or Quest session already running" >&2; exit 1; }
exec "${VLASTUDIO_PYTHON:-/home/jetson/miniforge3/envs/vlastudio/bin/python}" -m vlastudio.entrypoints.collect_data \
  --robot "$root/src/vlastudio/configs/robot/rebot_b601rs_head_rel_ee.yaml" \
  --teleop "$root/src/vlastudio/configs/teleop/quest3_rebot_b601rs_rel_ee.yaml" \
  --sample-quest-actions --frequency 30 --dataset-format hdf5 --hdf5-compression lzf \
  --output-dir "$root/data/rebot_vr" "$@"
