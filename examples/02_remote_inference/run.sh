#!/usr/bin/env bash
# Run from the repository root.
vlastudio env create --policy act --env aloha_sim -n act
eval "$(vlastudio env activate -n act --shell bash)"
python examples/02_remote_inference/serve.py &
python examples/02_remote_inference/evaluate.py
