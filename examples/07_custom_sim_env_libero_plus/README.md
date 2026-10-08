# 07 — Add a custom simulation benchmark

This example connects an external LIBERO-Plus adapter to VLAStudio through a
YAML `type:` path. Run all commands from the repository root on a Linux GPU machine.

## Install once

```bash
sudo apt-get install libmagickwand-6.q16-6 libgl1 libegl1 libopengl0
pip install -e .
vlastudio env create -n example07 --runtime-manifest examples/07_custom_sim_env_libero_plus/runtime.yaml
vlastudio env run -n example07 -- python examples/07_custom_sim_env_libero_plus/setup.py --download-assets
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/libero-plus/env.sh"
```

Setup downloads a pinned simulator and approximately 6.4 GB of assets, using
`~/.cache/vlastudio` by default. Set `VLASTUDIO_CACHE` to use another cache.
To reuse existing files, replace `--download-assets` with
`--root /path/to/LIBERO-plus --assets /path/to/assets`.
Package indexes follow your system configuration; environment creation accepts
`-i URL`. If a native dependency needs compilation, install a C compiler and
matching Python headers (for Python 3.10: `python3.10-dev build-essential`).

## Train and evaluate

```bash
vlastudio env run -n example07 -- python examples/07_custom_sim_env_libero_plus/train_and_evaluate.py
```

The five-line script loads the dataset and policy, trains for 5,000 steps with
batch size 16, then evaluates seven Plus variants. Edit that script to change
the training budget, checkpoint directory or results directory. Use a fresh
results directory for each evaluation.

Training uses the original ten-task Object demonstrations. They download to
`$VLASTUDIO_CACHE/data/libero`; set `LIBERO_DATASET_ROOT` to an existing directory
containing `libero_object/*.hdf5`. Source the generated `env.sh` again in each new shell.

To evaluate an existing checkpoint without training:

```bash
vlastudio env run -n example07 -- python -m vlastudio evaluate --runtime current \
  -m checkpoints/example07_smolvla \
  -e examples/07_custom_sim_env_libero_plus/env_object.yaml \
  -am examples/07_custom_sim_env_libero_plus/action_manager.yaml \
  -o results/example07_existing -n 1 -bs 0
```

## Add your own benchmark

Start with `libero_plus_env.py`: implement `reset`, `obs2meta`, `meta2act` and
`step`, then point `env_object.yaml` at your class:

```yaml
type: examples/07_custom_sim_env_libero_plus/libero_plus_env.py:LiberoPlusEnv
```

`obs2meta` supplies KCHW images, state and language. Actions stay in native units;
VLAStudio handles normalization. `step` must report task success, not a timeout.
The adapter restores the selected initial state after reset and uses
`config.rollout_index` to select distinct states across rollouts.

`dataset.py` and `task_object.yaml` define the training data. `smolvla.yaml`
selects the policy. Training and evaluation both use main + wrist images rotated
180 degrees, 8D end-effector state and 7D actions. `action_manager.yaml` executes
16 of each 50 predicted actions. Keep these settings aligned with your checkpoint.
`runtime.yaml` and `setup.py` prepare the dependencies and simulator assets.

The seven variants form a teaching subset, not the full LIBERO-Plus benchmark.
The recorded 5,000-step run scored 2/7 on this subset; it does not promise a 90%+
success rate. The detailed validation history is preserved in Git commit `6b455b9`.
