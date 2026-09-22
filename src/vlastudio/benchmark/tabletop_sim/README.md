# Tabletop-Sim Environment for VLAStudio

[Tabletop-Sim](https://github.com/jellyho/Tabletop-Sim) is a `dm_control`-based
bimanual tabletop manipulation simulator built on ALOHA (TwinVLA, ICLR 2026).
This module adapts it to VLAStudio's `MetaEnv` interface.

## Installation

The original integration used `mujoco==3.2.0` and `dm_control==1.0.21` in the
repository virtual environment. That combination was tested with the earlier
ALOHA integration. Current base profiles may pin different versions; prepare
a compatible environment before replacing existing dependencies.

From the repository root, with a checkout of Tabletop-Sim available:

```bash
source .venv/bin/activate
uv pip install "mujoco==3.2.0" "dm_control==1.0.21"
cd Tabletop-Sim && git submodule update --init --recursive && cd ..
uv pip install -e ./Tabletop-Sim --no-deps
```

## Datasets

Tabletop-Sim provides LeRobot v3.0 datasets, as well as RLDS and HDF5 formats.
This integration uses LeRobot v3.0:

```bash
hf download --repo-type dataset jellyho/aloha_dish_drainer
hf download --repo-type dataset jellyho/aloha_handover_box
hf download --repo-type dataset jellyho/aloha_shoes_table
hf download --repo-type dataset jellyho/aloha_lift_box
hf download --repo-type dataset jellyho/aloha_box_into_pot_easy
```

Set `HF_ENDPOINT` explicitly if you need an alternative download endpoint.

## Observations and actions

- State: 14-dimensional `qpos` with `ctrl_space=joint` (six joints and one gripper
  per arm); 20 dimensions with `ctrl_space=ee` and `action_space=ee_6d_pos`.
- Action: selected by `action_space`; the default `joint_pos` uses 14 absolute joint targets.
- Cameras: `back`, `wrist_left`, and `wrist_right`. The dataset calls the back
  camera `observation.images.agentview`; the configuration accepts `agentview`
  or `back`, and the module resolves the alias.

## Dummy-policy test

Run from the repository root with the simulator dependencies installed:

```bash
MUJOCO_GL=egl vlastudio evaluate --runtime current \
  -m __dummy-14random \
  -e tabletop_sim.dish_drainer \
  --batch_size 0 \
  --num_rollout 1 \
  -o results/tabletop_sim_dummy
```

## ACT training and evaluation

```bash
MUJOCO_GL=egl vlastudio train --runtime current -p act_tabletop_sim -t tabletop_sim.dish_drainer -c act -o ckpt/act_tabletop_dish_drainer
MUJOCO_GL=egl vlastudio evaluate --runtime current -m ckpt/act_tabletop_dish_drainer/checkpoint-10000 -e tabletop_sim.dish_drainer -o results/test_ --batch_size 0 --num_rollout 20
```

`--batch_size 0` uses `SequentialVectorEnv` to avoid multiprocessing context
issues with MuJoCo/EGL. On machines supporting parallel EGL rendering,
try `--batch_size 2` or larger.

## Tasks

The task list follows `tabletop.aloha_env.ALOHA_TASK_CONFIGS`:

| Task | Episode length (seconds) | Maximum timesteps (`DT=0.04`) |
| --- | --- | --- |
| `aloha_dish_drainer` | 10 | 250 |
| `aloha_handover_box` | 15 | 375 |
| `aloha_shoes_table` | 15 | 375 |
| `aloha_lift_box` | 15 | 375 |
| `aloha_box_into_pot` | 10 | 250 |
| `aloha_box_into_pot_easy` | 10 | 250 |
| `aloha_dish_drainer_new` | 10 | 250 |
| `aloha_handover_box_new` | 15 | 375 |
| `aloha_shoes_table_new` | 15 | 375 |
| `aloha_lift_box_new` | 15 | 375 |

Example environment and task configurations are under
`src/vlastudio/configs/env/tabletop_sim/` and
`src/vlastudio/configs/task/tabletop_sim/`. Use the dish-drainer configuration
as a template for other tasks.
