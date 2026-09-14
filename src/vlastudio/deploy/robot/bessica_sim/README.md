# Bessica-D Simulation Robot (`bessica_sim`)

ILStudio **自带**双臂 Bessica-D v1.0 仿真：URDF、MJCF、场景均在 `deploy/robot/bessica_sim/` 下，**不依赖**任何外部 Bessica / Synria Python 包或仓库路径。

- **运动学 / IK**：本仓库内 `kinematics.py` + Pinocchio，读包内 `assets/Bessica-D_Covered.urdf`
- **可视化 / 步进**：MuJoCo，读包内 `mujoco_model/scene.xml` → `bessica_d.xml`

## 网格文件（一次性放入本目录）

STL 体积大，通常不强制进 Git；由你在本地**放进 ILStudio 树内**这两处（路径固定，与 MJCF / URDF 一致）：

| 用途 | 目录 |
|------|------|
| MuJoCo | `mujoco_model/meshes/Bessica-D_v1_0/` |
| URDF（Pinocchio 可选加载 mesh） | `assets/meshes/Bessica-D_v1_0/` |

两处应包含**同一套** `*.STL` / `*.stl`（与 MJCF 里列出的 mesh 名一致，如 `base_link.STL` 等）。

**推荐**：用脚本把 STL **复制进** `bessica_sim`（之后仿真只读包内文件，不依赖 SDK 的 Python 包；`SRC` 任意）：

```bash
cd deploy/robot/bessica_sim
./vendor_meshes.sh /path/to/folder/containing/stl/files
```

若 ILStudio 仓库根目录下自带厂商资源树 **`Bessica-D-SDK/bessica_d_sdk/robocore_main/assets/robot/meshes/Bessica-D_v1_0/`**（与 `urdf/`、`mjcf/` 并列），可直接：

```bash
cd deploy/robot/bessica_sim
./vendor_meshes.sh ../../Bessica-D-SDK/bessica_d_sdk/robocore_main/assets/robot/meshes/Bessica-D_v1_0
```

（这与此前只检查 `.local/Bessica-D-SDK` 时「看不到」mesh 不同：根目录 `Bessica-D-SDK/` 里可以包含完整 `meshes/Bessica-D_v1_0/`。）

若你希望克隆即可跑仿真，将 `bessica_sim` 下两份 `meshes/Bessica-D_v1_0/` **提交进 Git**（大文件可用 Git LFS）；**不要**指望运行时去读 `Bessica-D-SDK/` 目录，否则换机器会丢路径。

## 控制模式

- **`delta_ee`**（默认）：14D — 右臂 7 + 左臂 7：`dx,dy,dz, dr,dp,dy, gripper`。夹爪通道控制 slide 关节（负=闭合，0=不动）。
- **`qpos`**：16D — `right_arm_joint1..7, right_gripper_width, left_arm_joint1..7, left_gripper_width`。臂关节为弧度，夹爪值为两指间宽度（米）`[0, 0.101]`（0=完全闭合，0.101=全开）。

每只手的两根手指通过 MuJoCo `equality` 约束镜像联动，只需控制一个值。

## 观测

- `qpos`: (16,) — `[R_joint1..7, R_gripper, L_joint1..7, L_gripper]`
- `gpos`: (12,) — 右 `link7` 再左 `link7` 的 `[x,y,z,roll,pitch,yaw]`（Pinocchio FK）

## 配置

- `configs/robot/bessica_sim.yaml`
- `configs/robot/bessica_sim_qpos.yaml`

一般**不要**在 YAML 里写 `xml_path` / `urdf_path`，除非你要临时换模型；默认始终用本包内路径。

## 依赖

与 ILStudio 一致：**mujoco**、**numpy**、**pin**（Pinocchio：`import pinocchio as pin`）。

## 烟测

```bash
# 已放好 STL 且环境已安装 pinocchio
python deploy/robot/bessica_sim/robot.py --mode qpos --visualize
```
