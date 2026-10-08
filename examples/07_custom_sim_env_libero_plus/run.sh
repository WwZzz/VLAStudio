#!/usr/bin/env bash
# Run inside the prepared environment after sourcing libero-plus/env.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
checkpoint="${CHECKPOINT:-checkpoints/example07_smolvla}"
python examples/07_custom_sim_env_libero_plus/train.py --output-dir "$checkpoint" \
  --steps "${STEPS:-5000}" --batch-size "${BATCH_SIZE:-16}"
python examples/07_custom_sim_env_libero_plus/evaluate.py --checkpoint "$checkpoint"
