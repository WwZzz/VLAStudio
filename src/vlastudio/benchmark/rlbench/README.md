# RLBench Environment

## Installation
```shell
# Download and install CoppeliaSim
# set env variables
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

uv pip install git+https://github.com/stepjam/RLBench.git
```



## Usage

⚠️ **IMPORTANT**: RLBench does **NOT** support parallel execution because CoppeliaSim cannot run in multiple processes.

Always use `--batch_size 0` to force sequential execution:

```bash
# Evaluate on ReachTarget task
python eval_sim.py -e rlbench_reach -m <model_path> --batch_size 0

# Evaluate on PickAndLift task  
python eval_sim.py -e rlbench_pick -m <model_path> --batch_size 0

# Multiple rollouts in sequence
python eval_sim.py -e rlbench_reach -m <model_path> --batch_size 0 --num_rollout 10
```

## Dataset Generation

Generate HDF5 datasets for training using the built-in demo collection:

```bash
# Generate 50 demos for ReachTarget task
python benchmark/rlbench/generate_dataset.py \
    --env_config configs/env/rlbench_reach.yaml \
    --output_dir data/rlbench/reach_target \
    --num_demos 50

# Generate demos for other tasks (create corresponding env config first)
python benchmark/rlbench/generate_dataset.py \
    --env_config configs/env/rlbench_pick.yaml \
    --output_dir data/rlbench/pick_and_lift \
    --num_demos 100
```

### Batch Generation

Generate datasets for all RLBench tasks automatically:

```bash
# Generate datasets for all tasks (50 demos per task, default)
bash benchmark/rlbench/generate_all_datasets.sh

# Generate with custom number of demos
bash benchmark/rlbench/generate_all_datasets.sh --num_demos 100

# Custom output directory
bash benchmark/rlbench/generate_all_datasets.sh --base_dir /path/to/output --num_demos 50

# Disable headless mode (for debugging)
bash benchmark/rlbench/generate_all_datasets.sh --no-headless --num_demos 10
```

**Note**: The script automatically:
- Finds all `rlbench_*.yaml` config files (excluding `*_ee.yaml` versions)
- Skips tasks if dataset already exists (with confirmation prompt)
- Generates one dataset per task (joint and ee versions share the same dataset)
- Provides progress summary at the end

### Generated Data Format

Each episode is saved as an HDF5 file with the following structure:
- `/action`: Action sequence (T, 8) - depends on ctrl_space
- `/state`ations/joint_velocities`: (T, 7)
- `/observations/gripper_pose`: (T, 7)
- `/observations/gripper_open`: (T, 1)
- `/observations/images/front`: (T, H, W, 3)
- `/observations/images/wrist`: (T, H, W, 3)
- `/language_instruction`: Task descriptions
- `/episode_len`: Episode length: State sequence (T, 8) - depends on ctrl_space
- `/observations/joint_positions`: (T, 7)
- `/observ

### Training with Generated Data

```bash
# Train ACT policy
python train.py --policy act --task rlbench_reach --output_dir ckpt/act_rlbench_reach

# Train Diffusion Policy
python train.py --policy diffusion_policy --task rlbench_reach --output_dir ckpt/dp_rlbench_reach
```

## Available Tasks

See full list: https://github.com/stepjam/RLBench/tree/master/rlbench/tasks

# TroubleShooting 

## Headless Server 
You can use the following command to test whether the rlbench was succesfully installed.
```shell
cd benchmark/rlbench
python test.py
```

Since the RLBench needs Xorg environment, you can use the following commands to add a virtual screen.
```shell
nohup X :99 & disown
export DISPLAY=:99
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

If X was not Found, here is a possible solution
```shell
apt-get install -y xvfb
nohup Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

If it raises errors like 
```shell
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/root/CoppeliaSim" even though it was found.

This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application
```
you can further try
```shell
apt-get install -y \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxcb-randr0 \
    libxi6 \
    libxrender1 \
    libxkbcommon-x11-0 \
    libfontconfig1 \
    libdbus-1-3
```

