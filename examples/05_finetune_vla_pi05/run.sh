#!/usr/bin/env bash
# Run from the repository root.
vlastudio env create --policy pi05 -n pi05
vlastudio env create --env aloha_sim -n sim
eval "$(vlastudio env activate -n pi05 --shell bash)"
python examples/05_finetune_vla_pi05/train.py
python examples/05_finetune_vla_pi05/serve.py &
eval "$(vlastudio env activate -n sim --shell bash)"
export MUJOCO_GL=egl
python examples/05_finetune_vla_pi05/evaluate.py
