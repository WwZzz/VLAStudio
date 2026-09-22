#!/usr/bin/env bash
# Run from the repository root.
vlastudio env create --policy smolvla -n smolvla
eval "$(vlastudio env activate -n smolvla --shell bash)"
python examples/04_multitask_policy/train_and_evaluate.py
