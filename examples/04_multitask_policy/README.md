# Multitask SmolVLA on LIBERO

Train SmolVLA on the LIBERO-Object suite and evaluate the checkpoint on every
object task in the same process. No policy server is required.

From the repository root:

```bash
examples/04_multitask_policy/run.sh
```

- Dataset: `libero.libero_object`
- Policy: this example's `smolvla.yaml` (LIBERO action_dim 7, state_dim 8)
- Training: built-in `smolvla` config
- Evaluation: `libero.object` (ten `LiberoEnv` tasks)

To split policy and simulator across environments, use the remote-inference
example: serve the checkpoint from the SmolVLA env and evaluate from a LIBERO env.

## Troubleshooting

- `vlastudio: command not found`: from the repository root, `python -m pip install -e .`
- `python: can't open file 'examples/...'`: run `run.sh` from the repository root, not from this directory.
- `No module named 'vlastudio'` after activate: `vlastudio env install --policy smolvla -n smolvla`, then activate again.
- `OSError: Error no file named pytorch_model.bin...` / only `policy_metadata.json` in `checkpoints/smolvla_libero`: training wrote metadata first and did not finish `save_model`. Wait for training to complete, or remove that directory and rerun.
- `No module named 'libero'`: LIBERO's setup.py installs an empty wheel. Re-run `vlastudio env install --policy smolvla -n smolvla` so VLAStudio can put the git checkout on `sys.path`, or set `LIBERO_ROOT` to a LIBERO clone. `robosuite` is a separate package in the same profile.
- SmolVLA environment create is slow or fails: first run downloads Torch 2.7.1 / LeRobot. Pass `-i <index-url>` on `vlastudio env create` if the default index is unreachable.
- VLM download errors: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` must be reachable, or pre-cache it.
- Missing LIBERO hdf5: the task YAML `root` defaults to `/inspire/hdd/project/robot-action/public/data/libero/`. Point it at your local copy.
- Stale `No module named 'examples/...'` imports: `unset PYTHONPATH` and rerun.
