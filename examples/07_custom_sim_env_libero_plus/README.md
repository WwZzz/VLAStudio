# 07 — Add a custom simulation benchmark

This example connects an external LIBERO-Plus adapter to VLAStudio through a
YAML `type:` path. Run all commands from the repository root on a Linux GPU machine.

## Install and run

```bash
bash examples/07_custom_sim_env_libero_plus/setup.sh
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/envs/example07/activate.sh"
python examples/07_custom_sim_env_libero_plus/train_and_evaluate.py
```

Setup creates a Python 3.10 virtual environment, installs VLAStudio and the pinned
policy/simulator dependencies, then downloads the simulator and approximately
6.4 GB of assets. It installs native libraries through `apt-get` (using `sudo`
when needed). A Linux machine with Python 3.10 and an NVIDIA GPU is required.
Package indexes follow your pip configuration; pass `-i URL` to override the index.

Everything is cached under `~/.cache/vlastudio` by default. Set `VLASTUDIO_CACHE`
before setup to change that location, `PYTHON` to choose a Python 3.10 interpreter,
or `VENV_DIR` to choose the virtual environment directory. Setup prints the exact
activation command; run it again in each new shell. Existing environments are reused.
To reuse simulator files:

```bash
bash examples/07_custom_sim_env_libero_plus/setup.sh --root /path/to/LIBERO-plus --assets /path/to/assets
```

Use `--skip-system` when native dependencies are already installed. On systems
without `apt-get`, install a C compiler, matching Python development headers and
venv support, Git, MagickWand, GL, EGL and OpenGL libraries before using that option.

The five-line Python script trains for 5,000 steps with batch size 16, then evaluates
seven Plus variants in the activated environment. Edit it to change the training
budget, checkpoint directory or results directory. Use a fresh results directory
for each evaluation. Original ten-task Object demonstrations download to
`${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/data/libero`; set `LIBERO_DATASET_ROOT`
to an existing directory containing `libero_object/*.hdf5` to reuse them.

To evaluate an existing checkpoint without training, run this Python code from
the repository root in the same activated environment:

```python
import vlastudio as vla
policy = vla.load_policy("examples/07_custom_sim_env_libero_plus/smolvla.yaml", checkpoint="checkpoints/example07_smolvla")
vla.load_env("examples/07_custom_sim_env_libero_plus/env_object.yaml").evaluate(policy, output_dir="results/example07_existing", num_rollout=1, action_manager="examples/07_custom_sim_env_libero_plus/action_manager.yaml")
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
`setup.sh` installs dependencies from `runtime.yaml` and calls `setup.py` to prepare
the simulator assets and configuration.

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
