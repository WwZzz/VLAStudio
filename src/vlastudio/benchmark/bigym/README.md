# BiGym Environment for ILStudio

BiGym is a bimanual manipulation benchmark featuring an H1 humanoid robot with two 7-DoF arms.

## Installation

```bash
# Navigate to bigym directory
cd benchmark/bigym

# Create and activate virtual environment
uv venv
source .venv/bin/activate
uv sync

# Install bigym from local directory
pip install -e bigym/
```

## Quick Start - One-Click Dataset Generation

```bash
# Generate datasets for all recommended tasks (one command!)
python generate_dataset.py

# Or generate specific tasks
python generate_dataset.py --tasks ReachTarget ReachTargetDual

# List all available tasks
python generate_dataset.py --list_tasks
```

## Available Tasks

| Task | Description |
|------|-------------|
| `ReachTarget` | Reach the red target sphere with either hand |
| `ReachTargetSingle` | Reach the red target sphere with the left hand |
| `ReachTargetDual` | Reach both target spheres, one with each hand |
| `StackBlocks` | Stack the blocks on top of each other |
| `MovePlate` | Move the plate to the target location |
| `DishwasherOpen` | Open the dishwasher door |
| `DishwasherClose` | Close the dishwasher door |
| `GroceriesStoreLower` | Store groceries on the lower shelf |
| ... | And many more |

## Dataset Generation

BiGym provides pre-recorded demonstrations that can be downloaded and converted to LeRobot format.

### One-Click Generation (Recommended)

```bash
# Generate all recommended tasks at once
python generate_dataset.py

# Generate specific tasks
python generate_dataset.py --tasks ReachTarget ReachTargetDual

# Custom settings
python generate_dataset.py \
    --tasks ReachTarget \
    --output_root ./my_datasets \
    --num_demos 100 \
    --resolution 256 256

# List all available tasks
python generate_dataset.py --list_tasks
```

Output datasets will be saved to `datasets/` directory by default:
```
datasets/
├── bigym_reachtarget/
│   ├── data/chunk-000/file-000.parquet
│   └── meta/
│       ├── info.json
│       ├── stats.json
│       ├── tasks.jsonl
│       └── episodes.jsonl
├── bigym_reachtargetsingle/
└── bigym_reachtargetdual/
```

## Observation & Action Structure

### Action (15 dims) - with `floating_base=True`

| Index | Name | Description |
|-------|------|-------------|
| 0 | `floating_base_x` | Pelvis X position (delta) |
| 1 | `floating_base_y` | Pelvis Y position (delta) |
| 2 | `floating_base_rz` | Pelvis rotation around Z (delta) |
| 3 | `left_shoulder_pitch` | Left arm shoulder pitch |
| 4 | `left_shoulder_roll` | Left arm shoulder roll |
| 5 | `left_shoulder_yaw` | Left arm shoulder yaw |
| 6 | `left_elbow` | Left arm elbow |
| 7 | `left_wrist` | Left arm wrist |
| 8 | `right_shoulder_pitch` | Right arm shoulder pitch |
| 9 | `right_shoulder_roll` | Right arm shoulder roll |
| 10 | `right_shoulder_yaw` | Right arm shoulder yaw |
| 11 | `right_elbow` | Right arm elbow |
| 12 | `right_wrist` | Right arm wrist |
| 13 | `left_gripper` | Left gripper position |
| 14 | `right_gripper` | Right gripper position |

**Structure**: `[floating_base(3), left_arm(5), right_arm(5), grippers(2)]`

### observation.state (15 dims) - matches action structure

Same structure as action, representing the current robot state that corresponds to each action dimension.

| Field | Dims | Description |
|-------|------|-------------|
| `observation.state` | 15 | `[floating_base(3), left_arm(5), right_arm(5), grippers(2)]` |

### observation.state_arm (12 dims) - arm-only state

For policies that don't need floating base:

| Field | Dims | Description |
|-------|------|-------------|
| `observation.state_arm` | 12 | `[left_arm(5), right_arm(5), grippers(2)]` |

### Full Proprioception

| Field | Dims | Description |
|-------|------|-------------|
| `observation.qpos` | 29 | Full joint positions (all 29 joints) |
| `observation.qvel` | 29 | Full joint velocities (all 29 joints) |
| `observation.gripper` | 2 | Gripper states `[left, right]` |
| `observation.floating_base` | 3 | Floating base position `[x, y, rz]` |
| `observation.floating_base_actions` | 3 | Accumulated floating base actions since reset |

### Action Components

| Field | Dims | Description |
|-------|------|-------------|
| `action` | 15 | Full action `[floating_base(3), arms(10), grippers(2)]` |
| `action.floating_base` | 3 | Floating base action `[x, y, rz]` |
| `action.arms` | 12 | Arms + grippers action `[left_arm(5), right_arm(5), grippers(2)]` |

### Images

| Field | Shape | Description |
|-------|-------|-------------|
| `observation.images.head` | [256, 256, 3] | Head camera RGB |
| `observation.images.left_wrist` | [256, 256, 3] | Left wrist camera RGB |
| `observation.images.right_wrist` | [256, 256, 3] | Right wrist camera RGB |

## Manual Generation (Advanced)

For more control, use `convert_to_lerobot.py`:

```bash
python convert_to_lerobot.py \
    --task ReachTarget \
    --output_dir /path/to/output/dataset \
    --num_demos 100 \
    --control_frequency 50 \
    --cameras head left_wrist right_wrist \
    --resolution 256 256
```

## Quick Test

Test the environment with a random policy:

```python
from benchmark.bigym import create_env
from easydict import EasyDict

config = EasyDict({
    'task': 'ReachTarget',
    'control_frequency': 50,
    'cameras': ['head', 'left_wrist', 'right_wrist'],
    'camera_resolution': (256, 256),
    'max_timesteps': 200,
    'action_mode': 'joint_position',
    'absolute': True,
    'floating_base': True,
})

env = create_env(config)
obs = env.reset()

print(f"State shape: {obs.state.shape}")  # (15,)
print(f"Action dim: {env.action_dim}")    # 15

for t in range(100):
    # Action: 15 dims [floating_base(3), left_arm(5), right_arm(5), grippers(2)]
    action = env.env.action_space.sample()
    obs, reward, done, info = env.step({'action': action})
    if done:
        break

env.close()
print("Test passed!")
```

## Configuration Example

Create a YAML config file `configs/env/bigym/reach_target.yaml`:

```yaml
type: benchmark.bigym.BiGymEnv
name: bigym_reach_target
args:
  task: ReachTarget
  max_timesteps: 500
  control_frequency: 50
  action_mode: joint_position
  absolute: true
  floating_base: true
  use_state_arm: false          # Set true for 12-dim arm-only state
  cameras:
    - head
    - left_wrist
    - right_wrist
  camera_resolution: [256, 256]
```

## Evaluation

Run evaluation with eval_sim.py:

```bash
python eval_sim.py \
    -e bigym.reach_target \
    -m /path/to/checkpoint \
    --max_timesteps 500
```

## Troubleshooting

1. **MuJoCo rendering issues**: Ensure you have proper OpenGL drivers installed.

2. **Demo download fails**: Check network connectivity. Demos are hosted on GitHub releases.

3. **Import errors**: Make sure bigym is installed correctly:
   ```bash
   pip install -e bigym/
   ```

4. **EGL cleanup errors on env.close()**: These are benign MuJoCo rendering cleanup errors and are automatically suppressed.
