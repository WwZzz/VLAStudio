# SO101 Plus simulation robot

MuJoCo viewer + Pinocchio/Piper IK on the bundled `so101_plus.urdf` (same flange
frame `gripper_body` as the real `So101Plus`).

## Files

| Path | Role |
|------|------|
| `mujoco_model/so101_plus.xml` | MJCF from URDF (`convert_urdf_to_mjcf.py`) |
| `mujoco_model/scene.xml` | Floor + lighting |
| `mujoco_model/assets` | Symlink → `../so101_plus/so101_plus_model/assets` |
| `robot.py` | `So101PlusSimRobot` (`qpos` / `delta_ee` / `rel_ee`) |

## VR teleop (Quest)

```bash
bash scripts/so101_plus_sim_vr_tele.sh
# or
python collect_data.py \
  -r configs/robot/so101_plus_sim_rel_ee.yaml \
  -t configs/teleop/quest3_so101_plus_rel_ee.yaml \
  -o data/so101_plus_sim_vr \
  --no-record-teleop
```

Uses the **same** Quest teleop YAML as the real arm. Viewer opens in-process.

## Smoke test (no VR)

```bash
python deploy/robot/so101_plus_sim/robot.py --mode rel_ee
```

## Regenerate MJCF after URDF edits

```bash
python deploy/robot/so101_plus_sim/convert_urdf_to_mjcf.py
```
