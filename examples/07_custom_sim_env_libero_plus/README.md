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

## Verified run (2026-10-08)

The five-line example at commit [3d549ed](https://github.com/WwZzz/VLAStudio/commit/3d549ed47f7b2e829697f2fda4fe818f3fcb1f97)
completed 5,000 training steps and automatic evaluation on one RTX 4090,
exiting with code 0 and saving the policy checkpoint and normalization statistics.
Training used batch size 16, seed 0 and the original ten-task Object demonstrations;
it took 28.8 minutes with mean training loss 0.1596. The seven default teaching
scenes scored **2/7 (28.6%)**.

The run used Python 3.10, PyTorch 2.7.1/CUDA 12.6 and LeRobot 0.3.3.
Dependencies, simulator assets and the pretrained backbone were reused from cache;
training started with a fresh output directory without resuming an action-policy
checkpoint. A fresh dependency installation was not part of this validation.

A separate evaluation of the same saved checkpoint covered all ten Object tasks,
with five trials per task in each condition (50 rollouts per row). Cases were
selected before evaluation, using initial-state offsets and seeds 0 through 4.

| Condition | Successful rollouts | Success rate |
| --- | ---: | ---: |
| Standard LIBERO-Object | 19/50 | 38% |
| Background textures | 7/50 | 14% |
| Camera viewpoints | 3/50 | 6% |
| Language instructions | 0/50 | 0% |
| Light conditions | 15/50 | 30% |
| Object layouts | 16/50 | 32% |
| Robot initial states | 3/50 | 6% |
| Sensor noise | 8/50 | 16% |

In a separate language control, restoring the canonical training instruction in
the same 50 language-variant scenes and initial states raised success from 0/50
to **17/50 (34%)**. This indicates sensitivity to the tested paraphrases; the
control is diagnostic and is excluded from the LIBERO-Plus scores above.

The completed runs produced 457 rollout videos: 7 from the example, 400 from the
condition comparison and 50 from the language control. The checkpoint hash stayed
unchanged throughout evaluation. Sensor-noise evaluation used a test-only
acceleration of glass blur, verified to preserve exact pixels and NumPy RNG state
in 30 comparisons; the complete 50-case rerun supplies the reported score.

These results demonstrate successful training, checkpoint loading and evaluation.
The condition comparison is a sampled robustness test, not a full LIBERO-Plus
benchmark result. Checkpoints, videos and raw evaluation outputs are kept outside Git.
