#!/usr/bin/env bash
# Run from the repository root.
vlastudio env create --policy act --env aloha_sim -n act
eval "$(vlastudio env activate -n act --shell bash)"
python examples/03_customize_policy/train_and_evaluate.py
