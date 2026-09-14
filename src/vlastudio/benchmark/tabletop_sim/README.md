# Tabletop-Sim Environment for ILStudio

[Tabletop-Sim](https://github.com/jellyho/Tabletop-Sim) is a `dm_control` based
bimanual tabletop manipulation simulator built on the ALOHA platform (TwinVLA,
ICLR 2026).  This module wraps it into ILStudio's `MetaEnv` interface.

## 1. 环境配置 (Installation)

Tabletop-Sim 依赖 `mujoco==3.2.0` / `dm_control==1.0.21`，经实测与 ILStudio
主仓自带的 `.venv` 完全兼容（主仓原来声明的 `mujoco==2.3.7` / `dm_control==1.0.14`
可以直接升级，原有 `benchmark/aloha` 等环境升级后仍可正常工作）。因此 **推荐
直接复用主仓 `.venv`，无需再建立独立 uv 环境**。

```bash
source .venv/bin/activate
uv pip install "mujoco==3.2.0" "dm_control==1.0.21"
cd Tabletop-Sim && git submodule update --init --recursive && cd ..
uv pip install -e ./Tabletop-Sim --no-deps
```

## 2. 数据集

Tabletop-Sim 官方提供 **LeRobot v3.0** 格式数据集（也提供 RLDS / HDF5，但本
集成只使用 LeRobot v3.0）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download --repo-type dataset jellyho/aloha_dish_drainer
hf download --repo-type dataset jellyho/aloha_handover_box
hf download --repo-type dataset jellyho/aloha_shoes_table
hf download --repo-type dataset jellyho/aloha_lift_box
hf download --repo-type dataset jellyho/aloha_box_into_pot_easy
```

## 3. 观测 / 动作格式

- **State**：`ctrl_space=joint` 时为 14-dim `qpos`（左臂6 + 左夹爪1 + 右臂6 + 右夹爪1），
  `ctrl_space=ee` + `action_space=ee_6d_pos` 时为 20-dim。
- **Action**：由 `action_space` 决定，默认 `joint_pos`（14-dim，绝对关节目标）。
- **Camera**：仿真中视角名为 `back` / `wrist_left` / `wrist_right`；在 LeRobot
  数据集里 `back` 对应 `observation.images.agentview`。在 config 里写
  `agentview`（或 `back`）皆可，module 内部会做 alias。

## 4. Quick test with dummy policy

```bash
cd /home/xudawei/ILStudio
source .venv/bin/activate
MUJOCO_GL=egl python eval_sim.py \
  -m __dummy-14random \
  -e tabletop_sim.dish_drainer \
  --batch_size 0 \
  --num_rollout 1 \
  -o results/tabletop_sim_dummy
```

## 5. Train / Eval with ACT

```bash
# ==== Train ====
MUJOCO_GL=egl python train.py -p act_tabletop_sim -t tabletop_sim.dish_drainer -c act -o ckpt/act_tabletop_dish_drainer

# ==== Evaluate ====
MUJOCO_GL=egl python eval_sim.py -m ckpt/act_tabletop_dish_drainer/checkpoint-10000 -e tabletop_sim.dish_drainer -o results/test_ --batch_size 0 --num_rollout 20
```

`--batch_size 0` 使用 `SequentialVectorEnv`，避开 MuJoCo / EGL 下多进程可能出现的上下文问题，与 `benchmark/robotwin` 做法一致。如果机器对并行 EGL 友好，可尝试 `--batch_size 2`（或更大）。

## 6. 任务列表

与 `tabletop.aloha_env.ALOHA_TASK_CONFIGS` 保持一致：

| Task | episode_len (s) | max_timesteps (DT=0.04) |
|------|-----------------|-------------------------|
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

本仓库提供的示例 env / task config 见 `configs/env/tabletop_sim/`、
`configs/task/tabletop_sim/`（以 `aloha_dish_drainer` 为主，其余任务可按此
模板复制修改）。
