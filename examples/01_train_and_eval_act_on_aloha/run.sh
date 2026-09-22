#!/usr/bin/env bash
# Run from the repository root.
vlastudio env create --policy act --env aloha_sim -n act
eval "$(vlastudio env activate -n act --shell bash)"
python examples/01_train_and_eval_act_on_aloha/train_and_evaluate.py
