# Train and evaluate ACT on ALOHA

From the repository root:

```bash
examples/01_train_and_eval_act_on_aloha/run.sh
```

## Troubleshooting

- `vlastudio: command not found`: from the repository root, `python -m pip install -e .`
- `python: can't open file 'examples/...'`: run `run.sh` from the repository root, not from this directory.
- `Unknown environment: act`: `vlastudio env create` failed; check its output. Pass `-i <index-url>` if package downloads fail.
- HuggingFace timeout / missing ALOHA hdf5: the built-in task config downloads `cadene/aloha_sim_transfer_cube_scripted_raw`. Use a local copy and point the task YAML `root` at it.
- CUDA / torch import errors: the host needs a matching GPU driver. ACT uses the base profile (Torch 2.4).
- Stale `No module named 'examples/...'` imports: `unset PYTHONPATH` and rerun.
