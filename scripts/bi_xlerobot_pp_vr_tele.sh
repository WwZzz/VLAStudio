#!/usr/bin/env bash
# XLeRobot++ dual SO101++ arms + Quest3 VR teleop.
# Head and wheels are initialized safely but are not controlled by VR.

set -euo pipefail
cd "$(dirname "$0")/.."

# Older local setup used the reversed name. LeRobot reads HF_LEROBOT_HOME.
if [[ -z "${HF_LEROBOT_HOME:-}" && -n "${HF_HOME_LEROBOT:-}" ]]; then
  export HF_LEROBOT_HOME="$HF_HOME_LEROBOT"
fi

# Default to the local cache so the robot can run without the external data
# drive. Interactive shells may still override this via HF_LEROBOT_HOME.
if [[ -z "${HF_LEROBOT_HOME:-}" ]]; then
  export HF_LEROBOT_HOME="${HOME}/.cache/huggingface/lerobot"
fi

for calibration_id in so101_pp_left so101_pp_right; do
  calibration_file="$HF_LEROBOT_HOME/calibration/robots/so101_pp/${calibration_id}.json"
  if [[ ! -f "$calibration_file" ]]; then
    echo "Missing PP calibration file: $calibration_file" >&2
    exit 1
  fi
done

_CMEEL_LIB="$(pwd)/.venv/lib/python3.10/site-packages/cmeel.prefix/lib"
if [[ -d "$_CMEEL_LIB" ]]; then
  export LD_LIBRARY_PATH="${_CMEEL_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

PYTHON="$(pwd)/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing ILStudio virtual environment: $PYTHON" >&2
  exit 1
fi

"$PYTHON" collect_data.py \
  -r configs/robot/xlerobot_pp_rel_ee.yaml \
  -t configs/teleop/quest3_bi_xlerobot_pp_rel_ee.yaml \
  -o data/bi_xlerobot_pp_vr_teleop \
  -f 30 \
  --visualize \
  --no-record-teleop \
  "$@"
