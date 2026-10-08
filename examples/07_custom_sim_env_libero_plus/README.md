# 07 — Add a custom simulation benchmark

This example implements an external `MetaEnv` adapter for the official
[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) simulator. Its Python code
stays in this directory; a YAML `type:` reference connects it to VLAStudio.
You can replace the adapter with your own simulator without writing a trainer.

**Standard LIBERO-Object and LIBERO-Plus are different evaluations.**
The pinned Plus revision contains 2,518 Object variants. The default example
selects seven named variants of the alphabet-soup task, one per perturbation
category; it is a teaching
subset, **not the full Plus benchmark or a reproduction of a 97% Object score**.
Training uses the original ten-task Object demonstrations.

A matched-checkpoint audit obtained 18/20 in both legacy and packaged VLAStudio.
See [the validation record](VALIDATION.md#matched-checkpoint-audit--2026-09-26)
for the exact protocol and the separate 5,000-step example results.
Use the camera orientation, camera count and execution horizon associated with
your checkpoint; this example's defaults are not universal SmolVLA settings.

## Files

| File | Purpose |
| --- | --- |
| `libero_plus_env.py` | Simulator construction, reset, observation/action conversion, success |
| `env_object.yaml` | Seven named Plus variants, with checked perturbation labels |
| `env_standard_object.yaml` | Ten vanilla Object tasks for a separate control evaluation |
| `dataset.py`, `task_object.yaml` | Original HDF5 data with explicit image orientation and cache paths |
| `smolvla.yaml` | Two-camera SmolVLA, 50 predicted actions per chunk |
| `action_manager.yaml` | Execute 16 of those 50 actions before observing again |
| `make_env_config.py` | Generate a full suite or one perturbation category |
| `train.py`, `evaluate.py` | Python API training and local/remote evaluation |
| `setup.py`, `runtime.yaml` | Pinned simulator source, assets, isolated configuration, dependencies |

Run the commands below from the repository root.

## Install

Linux is required for this simulator example. On a headless GPU machine use
EGL. LIBERO-Plus additionally needs ImageMagick's native library; on Debian/Ubuntu:

```bash
sudo apt-get install libmagickwand-6.q16-6 libgl1 libegl1 libopengl0
pip install -e .
vlastudio env create -n example07 --runtime-manifest examples/07_custom_sim_env_libero_plus/runtime.yaml
```

If pip needs to build a native dependency, install your interpreter's matching
Python development headers and a C compiler (for Ubuntu 22.04 / Python 3.10:
`sudo apt-get install python3.10-dev build-essential`).

No shell initialization hook is needed. Run Python in that environment with:

```bash
vlastudio env run -n example07 -- python examples/07_custom_sim_env_libero_plus/setup.py --download-assets
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/libero-plus/env.sh"
```

The official assets archive is about 6.4 GB
(roughly 9 GB extracted, with many small files). To reuse your existing checkout
and extracted assets, pass `--root /path/to/LIBERO-plus --assets /path/to/assets`
instead of `--download-assets`. The default checkout is pinned to
`4976dc30028e805ff8094b55501d532c48fec182`; an explicitly supplied checkout
is preserved and its revision is printed.

The setup creates an isolated LIBERO config under the VLAStudio cache. It does
not rewrite `~/.libero/config.yaml`. The adapter checks the imported source and
the configuration, and refuses to label vanilla LIBERO as Plus.

For an already activated Python environment, use
`vlastudio env install --runtime-manifest examples/07_custom_sim_env_libero_plus/runtime.yaml`,
then run the Python scripts directly.

Package indexes follow your system configuration. You can supply `-i URL` to
the environment installation command.

## Train through Python

```bash
vlastudio env run -n example07 -- python examples/07_custom_sim_env_libero_plus/train.py \
  --steps 5000 --batch-size 16 \
  --output-dir checkpoints/example07_smolvla
```

Optionally add `--data-root /path/to/datasets`, where that directory contains
`libero_object/*.hdf5`. You can also set `LIBERO_DATASET_ROOT`.
Without either, the Object data is downloaded under
`$VLASTUDIO_CACHE/data/libero` (default `~/.cache/vlastudio/data/libero`).

The API usage is:

```python
import vlastudio as vla

dataset = vla.load_dataset("examples/07_custom_sim_env_libero_plus/task_object.yaml")
policy = vla.load_policy("examples/07_custom_sim_env_libero_plus/smolvla.yaml")
vla.train(policy, dataset, "smolvla", output_dir="checkpoints/my_policy",
          overrides={"training.max_steps": 5000})
bench = vla.load_env("examples/07_custom_sim_env_libero_plus/env_object.yaml")
bench.evaluate(policy, output_dir="results/my_policy_plus", num_rollout=1,
               action_manager="examples/07_custom_sim_env_libero_plus/action_manager.yaml")
```

5,000 steps is a functional training run, not a guaranteed convergence target.
A custom training budget, policy, dataset, output directory and checkpoint can
all be supplied without changing VLAStudio.

## Evaluate the Plus subset

```bash
vlastudio env run -n example07 -- python examples/07_custom_sim_env_libero_plus/evaluate.py \
  --checkpoint checkpoints/example07_smolvla \
  --output-dir results/example07_plus --num-rollout 1
```

This performs one rollout for each of the seven explicit variants.
Use a fresh output directory for each experiment. Per-task JSON and videos are
written there; `protocol.json` records the chosen configuration and source revision.

To generate all 2,518 Object variants from the pinned official catalog:

```bash
vlastudio env run -n example07 -- python examples/07_custom_sim_env_libero_plus/make_env_config.py \
  --suite libero_object --output results/libero_plus_object_all.yaml
vlastudio env run -n example07 -- python examples/07_custom_sim_env_libero_plus/evaluate.py \
  --checkpoint checkpoints/example07_smolvla --env results/libero_plus_object_all.yaml \
  --output-dir results/example07_plus_full --num-rollout 1
```

Use `--category "Camera Viewpoints"` to select a category, or `--limit N` for a
smaller diagnostic run. The generator reports an invalid category with the
available names. Generated configurations contain an absolute adapter path;
regenerate them when moving the checkout.

The equivalent CLI for the default subset is:

```bash
vlastudio evaluate --runtime current \
  -m checkpoints/example07_smolvla \
  -e examples/07_custom_sim_env_libero_plus/env_object.yaml \
  -am examples/07_custom_sim_env_libero_plus/action_manager.yaml \
  -o results/example07_plus_cli -n 1 -bs 0
```

That CLI command assumes the current shell's Python has the required packages.
Use `vlastudio env run -n example07 -- python -m vlastudio ...` to select the named
environment explicitly.

For separate policy and simulation environments, start a server using the
checkpoint's policy environment:

```python
import vlastudio as vla
policy = vla.load_policy("smolvla", checkpoint="checkpoints/example07_smolvla")
vla.serve(policy, address="0.0.0.0", port=5000)
```

Then run `evaluate.py --server 127.0.0.1:5000 --output-dir results/example07_remote`
in the simulation environment. HTTP(S) and `shm://` addresses are also accepted
when supported by the corresponding server transport.

## Run the standard Object control

Set up vanilla LIBERO in a separate source/configuration directory:

```bash
vlastudio env run -n example07 -- python examples/07_custom_sim_env_libero_plus/setup.py --backend standard
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/libero-standard/env.sh"
vlastudio env run -n example07 -- python examples/07_custom_sim_env_libero_plus/evaluate.py \
  --checkpoint checkpoints/example07_smolvla \
  --env examples/07_custom_sim_env_libero_plus/env_standard_object.yaml \
  --output-dir results/example07_standard --num-rollout 10
```

This is 10 tasks × 10 initial states = 100 rollouts. Switch back by sourcing
`libero-plus/env.sh` and starting a new evaluation process. A standard config
refuses the Plus backend, so the two results cannot silently mix.

## Implement your own benchmark

The YAML selects a class outside the installed package:

```yaml
type: examples/07_custom_sim_env_libero_plus/libero_plus_env.py:LiberoPlusEnv
name: libero_object_0
args:
  task: libero_object_0
  task_name: pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_15
  require_plus: true
  perturbation: Background Textures
  max_timesteps: 280
  use_wrist: true
  image_size: [256, 256]
  image_transform: rotate180
```

A minimal environment implements:

- `__init__(config)`: create the simulator.
- `reset() -> MetaObs`: initialize an episode.
- `obs2meta(raw_obs) -> MetaObs`: provide images in KCHW format, state and language.
- `meta2act(MetaAction)`: map native-unit policy actions to simulator controls.
- `step(action)`: override when the simulator's termination flag is not equivalent
  to task success. The default VLAStudio evaluator counts `done` as success.

`MetaEnv` provides the ordinary step conversion and close method. This example
overrides step to use `check_success()`, so timeout is never counted as success.

The evaluator supplies `config.rollout_index` (also unique within a parallel
batch). This adapter restores state `(init_state_offset + rollout_index) % N`
**after** the simulator reset, then executes ten settling actions. This avoids
random resets overwriting the selected state or repeating initial state zero
when a fresh environment is created for each episode.

For the **Robot Initial States** category, the shared state file also contains
standard robot joints. After restoring object placements, this adapter reapplies
the named variant's `init_qpos` and zero joint velocity before settling. Otherwise
the full-state restore silently removes that perturbation. This is an explicit
adapter correction beyond the upstream generic reset recipe; record it when
comparing with published results.

## Interpreting low success

Check these before changing training:

1. **Backend/task identity:** a Plus Object index is not a vanilla Object index.
   Select Plus tasks by name and verify their category. Strip perturbation suffixes
   before constructing the standard task instruction; otherwise tokens such as
   `table 15` or `view ... initstate ...` enter the model prompt. Only deliberate
   language variants use the rewritten BDDL instruction. Some base BDDL wording
   differs from the HDF5 training metadata, so blindly switching all prompts to
   BDDL also changes the standard evaluation contract.
2. **Image contract:** this example rotates both HDF5 images and live images by
   180 degrees and uses main + wrist cameras. Match the convention and camera
   order of your checkpoint; do not add a second rotation in a processor.
3. **State/action contract:** state is position (3), axis-angle (3), gripper qpos
   (2). Actions are native 7D end-effector deltas including the native gripper.
   Normalization belongs to the policy pipeline, not this adapter.
4. **Execution horizon:** predicting 50 actions does not mean executing all 50.
   The example explicitly executes 16. Change the manager when the checkpoint
   uses another execution horizon.
5. **Evaluation protocol:** record checkpoint, task names, initial state indices,
   simulator revision, image convention, action horizon and number of rollouts.

To diagnose an existing checkpoint, first run it on the standard control with
the same input convention, then change one item at a time. Report the Plus subset
separately; it cannot establish a full-suite robustness score.

## Verified run

See [the validation record](VALIDATION.md) for the tested environment, 5,000-step
training run, measured success rates, and the separate execution-horizon control.
