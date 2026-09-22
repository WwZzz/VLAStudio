# LoRA-finetune π0.5 on Tabletop-Sim

Fine-tune π0.5 with LoRA on Tabletop-Sim dish-drainer in the OpenPI environment,
serve the checkpoint, and evaluate in a separate simulator environment over TCP.

From the repository root:

```bash
examples/05_finetune_vla_pi05/run.sh
```

- Train: `tabletop_sim.dish_drainer` + this example's `pi05.yaml` (OpenPI env)
- Serve: `checkpoints/pi05_tabletop` on `0.0.0.0:5000` (OpenPI env)
- Evaluate: `tabletop_sim.dish_drainer` connects to `127.0.0.1:5000` (simulator env)

Policy and simulator do not share an interpreter. To run the two sides yourself:

```bash
eval "$(vlastudio env activate -n pi05 --shell bash)"
python examples/05_finetune_vla_pi05/serve.py
```

```bash
eval "$(vlastudio env activate -n sim --shell bash)"
export MUJOCO_GL=egl
python examples/05_finetune_vla_pi05/evaluate.py
```

Install Tabletop-Sim into the simulator environment before evaluating; see
`src/vlastudio/benchmark/tabletop_sim/README.md`.

## Troubleshooting

- `vlastudio: command not found`: from the repository root, `python -m pip install -e .`
- `python: can't open file 'examples/...'`: run `run.sh` from the repository root, not from this directory.
- `No module named 'vlastudio'` after activate: `vlastudio env install --policy pi05 -n pi05` (or `--env aloha_sim -n sim`), then activate again.
- Connection refused: start `serve.py` in the OpenPI env and wait until it is listening, then run `evaluate.py`.
- Missing `checkpoints/pi05_tabletop`: run `train.py` in the OpenPI env first.
- Missing `pi05_base_torch`: download and convert the JAX checkpoint as in `src/vlastudio/policy/openpi/README.md`, and write the PyTorch weights to the path in `pi05.yaml`.
- HuggingFace timeout / missing `jellyho/aloha_dish_drainer`: download the LeRobot v3.0 dataset locally and point the task YAML `dataset_path_list` at it.
- OpenPI environment create is slow or fails: this profile needs Python 3.11, Torch 2.7.1, glibc >= 2.31. Pass `-i <index-url>` on `vlastudio env create` if the default index is unreachable.
- Evaluation cannot import `tabletop`: install Tabletop-Sim into `-n sim`.
- EGL / offscreen rendering: `run.sh` sets `MUJOCO_GL=egl` on the simulator side. Use `osmesa` if the host has no EGL.
- Stale `No module named 'examples/...'` imports: `unset PYTHONPATH` and rerun.
