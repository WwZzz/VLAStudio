# Bessica-D Simulation Robot (`bessica_sim`)

VLAStudio includes a dual-arm Bessica-D v1.0 simulation. Its URDF, MJCF, and
scene files live inside `src/vlastudio/deploy/robot/bessica_sim/`; simulation
does not depend on an external Bessica/Synria Python package or SDK path.

- Kinematics and IK: the included `kinematics.py` and Pinocchio use
  `assets/Bessica-D_Covered.urdf`.
- Rendering and stepping: MuJoCo loads `mujoco_model/scene.xml` and `bessica_d.xml`.

## Mesh files

STL files are large and may be omitted from Git. Place the meshes inside this
module at both locations expected by its MJCF and URDF:

| Usage | Directory relative to this module |
| --- | --- |
| MuJoCo | `mujoco_model/meshes/Bessica-D_v1_0/` |
| URDF and optional Pinocchio mesh loading | `assets/meshes/Bessica-D_v1_0/` |

Both directories must contain the same `*.STL`/`*.stl` files, with names matching
the model declarations, such as `base_link.STL`.

Use the helper to copy meshes into the module:

```bash
cd src/vlastudio/deploy/robot/bessica_sim
./vendor_meshes.sh /path/to/folder/containing/stl/files
```

If a vendor checkout is available at the repository root, the source meshes may
be under `Bessica-D-SDK/bessica_d_sdk/robocore_main/assets/robot/meshes/Bessica-D_v1_0/`.
Pass its absolute path to the helper. After copying, simulation reads only the
module's own files and does not need the vendor SDK at runtime.

To distribute a checkout or wheel with meshes included, retain both mesh copies
under this module. Large assets may be tracked with Git LFS. Do not rely on an
external SDK directory being present on another machine.

## Control modes

- `delta_ee` (default): 14 dimensions, right arm then left arm, each containing
  `dx, dy, dz, dr, dp, dy, gripper`. Gripper channels control slide joints;
  negative closes and zero keeps the current position.
- `qpos`: 16 dimensions, ordered as `right_arm_joint1..7, right_gripper_width,
  left_arm_joint1..7, left_gripper_width`. Arm joints use radians; grippers use
  the distance between fingers in meters, from `0` (closed) to `0.101` (open).

MuJoCo equality constraints mirror each pair of fingers, so one value controls
both fingers on each hand.

## Observations

- `qpos`, shape `(16,)`: `[R_joint1..7, R_gripper, L_joint1..7, L_gripper]`.
- `gpos`, shape `(12,)`: `[x, y, z, roll, pitch, yaw]` for right `link7`, then
  left `link7`, computed by Pinocchio forward kinematics.

## Configuration

- `src/vlastudio/configs/robot/bessica_sim.yaml`
- `src/vlastudio/configs/robot/bessica_sim_qpos.yaml`

Leave `xml_path` and `urdf_path` unset to use packaged model paths. Override them
only when intentionally testing another model.

## Dependencies

Install `mujoco`, `numpy`, and `pin` (Pinocchio, imported as `pinocchio`).

## Smoke test

With meshes copied and dependencies installed, run from the repository root:

```bash
python -m vlastudio.deploy.robot.bessica_sim.robot --mode qpos --visualize
```
