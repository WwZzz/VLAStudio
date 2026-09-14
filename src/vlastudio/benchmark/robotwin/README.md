# RoboTwin Environment for ILStudio

https://github.com/TianxingChen/RoboTwin

RoboTwin is a dual-arm manipulation benchmark with 50 diverse tasks, supporting domain randomization and various observation modalities.

## Installation


```bash
cd benchmark/robotwin
uv venv
source .venv/bin/activate
uv sync

# adjust code
SAPIEN_LOCATION=$(uv pip show sapien | grep 'Location' | awk '{print $2}')/sapien
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $URDF_LOADER
MPLIB_LOCATION=$(uv pip show mplib | grep 'Location' | awk '{print $2}')/mplib
PLANNER=$MPLIB_LOCATION/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER

uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
cd RoboTwin/envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
uv pip install -e . --no-build-isolation

# Download assets
cd ../..
bash script/_download_assets.sh
cd ../../..
```

## Data Collection & Dataset Usage

To collect or use RoboTwin demonstration data with ILStudio, you need to use the `RoboTwinDataset`.

### Data Download & Processing

The `RoboTwinDataset` automatically handles downloading data from Hugging Face or loading from a local path.

**Hugging Face Example (downloads and extracts)**:
```yaml
# configs/task/robotwin_example.yaml
datasets:
  - type: data_utils.datasets.robotwin_dataset.RoboTwinDataset
    name: robotwin_pick_bottles_aloha
    args:
      dataset_name: adjust_bottle/aloha-agilex_clean_50  # Hugging Face path
      image_size: [480, 640]
      chunk_size: 16
      camera_names: ['head_camera', 'left_camera', 'right_camera']
```

**Local Path Example (uses local .zip or extracted directory)**:
```yaml
# configs/task/robotwin_example.yaml
datasets:
  - type: data_utils.datasets.robotwin_dataset.RoboTwinDataset
    name: robotwin_pick_bottles_aloha
    args:
      dataset_path: /path/to/local/aloha-agilex_clean_50.zip
      # Or if already extracted:
      # dataset_path: /path/to/local/aloha-agilex_clean_50/data
      image_size: [480, 640]
      chunk_size: 16
      camera_names: ['head_camera', 'left_camera', 'right_camera']
```

**Cache Location**:
Downloaded data will be cached in `~/.cache/ilstudio/robotwin`. Original data is NOT copied or converted - only metadata is cached for efficiency.

### Generate data locally



## Usage with ILStudio

RoboTwin must run with its own Python environment:

```bash
# Use RoboTwin's Python
python eval_sim.py -e robotwin_pick_bottles -m <model> --batch_size 0
```

⚠️ **Sequential mode only** (`--batch_size 0`) due to Sapien limitations.

## Configuration

Create task configs in `configs/env/robotwin_<task>.yaml`:

**Dual-Arm Robot (e.g., ALOHA)**
```yaml
type: benchmark.robotwin.RoboTwinEnv
name: robotwin_<task>
args:
  task: pick_diverse_bottles
  task_config: demo_clean
  max_timesteps: 1000
  ctrl_space: qpos
  ctrl_type: abs
  seed: 0
  
  # Camera Configuration - Use RoboTwin's original names
  camera_names:
    - head_camera             # Fixed overhead camera (required)
    - left_camera             # Left arm wrist camera
    - right_camera            # Right arm wrist camera
  image_size: [480, 640]
  
  # Robot Embodiment
  embodiment: aloha-agilex    # Dual-arm robot
  
  # Motion Planner
  use_planner: false
  
  # robotwin_root: /path/to/RoboTwin
```

### `max_timesteps` alignment (important)

RoboTwin tasks have **different** evaluation step limits. The authoritative mapping lives in
`benchmark/robotwin/RoboTwin/task_config/_eval_step_limit.yml`.

In ILStudio, `max_timesteps` comes from your env YAML and directly controls evaluation horizon
(`benchmark/utils.py:evaluate` loops `range(args.max_timesteps)`), so keeping it aligned matters.

To (re)sync all RobotWin env configs under `configs/env/robotwin/` (including `ee/`):

```bash
python scripts/sync_robotwin_max_timesteps.py --repo_root .
```

**Single-Arm Robot (e.g., Franka Panda)**
```yaml
type: benchmark.robotwin.RoboTwinEnv
name: robotwin_<task>
args:
  task: pick_diverse_bottles
  task_config: demo_clean
  max_timesteps: 1000
  ctrl_space: qpos
  ctrl_type: abs
  seed: 0
  
  # Camera Configuration - Use RoboTwin's original names
  camera_names:
    - head_camera             # Fixed overhead camera (required)
    - left_camera             # Wrist camera (single-arm uses left_camera)
  image_size: [480, 640]
  
  # Robot Embodiment
  embodiment: franka-panda    # Single-arm robot
  
  # Motion Planner
  use_planner: false
  
  # robotwin_root: /path/to/RoboTwin
```

### Flexible Path Configuration

If your RoboTwin installation is in a custom location, specify it in the config:

```yaml
args:
  robotwin_root: /custom/path/to/RoboTwin
  # ... other args
```

This resolves relative paths (like `./assets/objects/objaverse/list.json`) correctly regardless of where you run ILStudio from.

### Camera Configuration

Specify which cameras to collect in `camera_names` using **RoboTwin's original camera names**:

- **`head_camera`**: Fixed overhead camera (always required)
- **`left_camera`**: Left arm wrist camera
- **`right_camera`**: Right arm wrist camera

**Example: Dual-arm robot (e.g., aloha-agilex)**
```yaml
camera_names:
  - head_camera
  - left_camera
  - right_camera
```

**Example: Single-arm robot (e.g., franka-panda)**
```yaml
camera_names:
  - head_camera
  - left_camera    # Single-arm robots also use left_camera
```

**Camera Configuration Table**:

| Robot Type | Camera Names | Notes |
|-----------|-------------|-------|
| Dual-arm (aloha-agilex) | `head_camera`<br>`left_camera`<br>`right_camera` | Use left/right for each arm |
| Single-arm (franka-panda) | `head_camera`<br>`left_camera` | Single-arm uses `left_camera` |

**Note**: 
- Use RoboTwin's original camera names directly - no mapping needed.
- Enabling wrist cameras increases observation dimensionality and may slow down inference.
- For single-arm robots, RoboTwin internally uses the same URDF `camera` link for both `left_camera` and `right_camera`, so you can use `left_camera` to access the wrist camera.

### Robot Embodiment Configuration

Override the robot type used in the task. If not specified, uses the embodiment from `task_config/<config>.yml`.

**Available Robots**:
- `aloha-agilex`: ALOHA dual-arm mobile manipulator (default)
- `piper`: Piper robot
- `franka-panda`: Franka Emika Panda
- `ARX-X5`: ARX-X5 robot
- `ur5-wsg`: UR5 with WSG gripper

**Single Robot (Dual-Arm)**
```yaml
embodiment: aloha-agilex
```

**Heterogeneous Dual-Arm Setup**
```yaml
embodiment: [ur5-wsg, franka-panda, 0.5]
# Format: [left_robot, right_robot, distance_between_bases]
```

**Note**: Changing embodiment requires ensuring the task is compatible with the robot's workspace and DOF.

### Planner Control

RoboTwin's original `eval_policy.py` uses `expert_check` to control whether to validate seeds with expert demonstrations before policy evaluation:
- Expert demos (`play_once`) **require** motion planners (Curobo + TOPP)
- Policy evaluation (`take_action`) has **fallback logic** that works without planners

ILStudio's `use_planner` parameter controls planner initialization:

**`use_planner: false` (Default, Recommended)**
- ✅ **Fast**: ~1-2 steps/second
- ✅ **No GPU required**: Skips Curobo/TOPP initialization entirely
- ✅ **Fallback execution**: Actions executed via 50-step linear interpolation (RoboTwin's original fallback)
- 📝 **Equivalent to**: `expert_check=False` in RoboTwin's eval_policy.py

**`use_planner: true` (Optional, GPU required)**
- ⚠️ **Slower**: ~0.1-0.2 steps/second
- ⚠️ **Requires GPU**: Initializes Curobo motion planner
- ✅ **Trajectory smoothing**: TOPP for time-optimal parameterization
- ✅ **Collision avoidance**: Curobo for collision-free paths
- ✅ **Bug fixed**: Fixed Curobo autograd issue (missing `requires_grad` on output buffers in `bound_cost.py` and `dist_cost.py`)
- 📝 **Equivalent to**: `expert_check=True` in RoboTwin's eval_policy.py (but ILStudio doesn't run expert demos)

## Available Tasks

50 tasks available in `RoboTwin/envs/`:
- **Pick & Place**: `pick_diverse_bottles`, `place_bread_basket`, etc.
- **Stacking**: `stack_blocks_two`, `stack_bowls_three`, etc.
- **Opening**: `open_laptop`, `open_microwave`, etc.
- **Manipulation**: `press_stapler`, `shake_bottle`, `rotate_qrcode`, etc.

## Quick Comparison

| Feature | No Planner (Default) | With Planner |
|---------|---------------------|--------------|
| Speed | ~1-2 steps/sec | ~0.1-0.2 steps/sec |
| GPU Required | ❌ No | ✅ Yes |
| Smooth Trajectories | ❌ | ✅ TOPP |
| Collision Avoidance | ❌ | ✅ Curobo |
| Best For | Dense actions, fast eval | Sparse waypoints, smooth motion |
| Config | `use_planner: false` | `use_planner: true` |

## Testing

```bash
# Test without planner (fast, recommended)
benchmark/robotwin/.venv/bin/python eval_sim.py \
  -m __dummy-16random \
  -e robotwin_pick_bottles \
  --batch_size 0 \
  --num_rollout 1

# Test with planner (slow, smooth trajectories)
benchmark/robotwin/.venv/bin/python eval_sim.py \
  -m __dummy-16random \
  -e robotwin_pick_bottles_with_planner \
  --batch_size 0 \
  --num_rollout 1
```

## Motion Planner Configuration

RoboTwin's original design **attempts** to use motion planners (TOPP/Curobo) for trajectory smoothing, but **gracefully falls back** to simple interpolation if they fail.

### Mode 1: Skip Planner (Default, Recommended for ILStudio)
```yaml
args:
  use_planner: false
```

**What it does:**
- Skips planner initialization entirely
- Uses direct action execution (no TOPP smoothing attempt)

**Benefits:**
- ✅ Fast startup (~5sec vs ~15sec)
- ✅ No GPU required
- ✅ Predictable performance (~1-2 steps/sec)
- ✅ No risk of Curobo initialization failure

**Use when:**
- Running in CPU-only environments
- Policy outputs dense actions (e.g., ACT with chunk_size=50)
- Fast evaluation is priority

### Mode 2: Try Planner (Original RoboTwin Behavior)
```yaml
args:
  use_planner: true
```

**What it does:**
- Initializes Curobo + TOPP planners
- **Attempts** TOPP trajectory smoothing on each action
- **Automatically falls back** to interpolation if TOPP fails

**Benefits:**
- ✅ Smoother trajectories when TOPP succeeds
- ✅ Same as official ACT evaluation behavior
- ✅ Has fallback mechanism (won't crash if planning fails)

**Drawbacks:**
- ⚠️ Requires GPU for Curobo warmup
- ⚠️ Slower initialization and inference
- ⚠️ May fail in some environments

**Use when:**
- Have GPU available
- Want trajectory smoothing
- Testing with official RoboTwin policy implementations

## Notes

- Uses Sapien physics engine (similar to ManiSkill2)
- Some episodes may fail initialization (unstable object placement)
- Success detection: Environment returns `done=True` only when task succeeds

## Return Values (ILStudio Convention)

- `done`: Task success status (`True` only when task succeeds)
- `info['success']`: Explicit task success flag
- `info['terminated']`: Episode ended (success or timeout)
- `info['truncated']`: Episode timed out without success
