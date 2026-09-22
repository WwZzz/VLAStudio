# 06. Add a New Robot Externally + Simulated SO101 with Keyboard Teleop

This example shows the full flow for adding a **new robot** to VLAStudio
**without modifying its source**, then driving it in simulation with a keyboard
teleoperator:

1. Write a robot class (your file).
2. Reference it from a YAML config by a local file path.
3. Wire it to a teleoperator via `vlastudio collect`.
4. Watch the MuJoCo viewer follow your keyboard.

It reuses the package's MuJoCo SO101 simulation (`So101SimRobot`), so it runs on
any desktop with `mujoco` installed. For real hardware you implement the same
interface and connect to your vendor SDK / ROS inside the class.

## 1. The robot interface

Every robot is a class. The contract is `AbstractRobotInterface`
(`vlastudio/deploy/robot/base.py`). The abstract members you must provide:

| Method | Responsibility | Notes |
|---|---|---|
| `connect()` | connect hardware/SDK | return bool |
| `get_observation() -> dict` | one synchronized observation | **blocking** until fresh data |
| `publish_action(action: ndarray)` | send command to the robot | **non-blocking** |
| `is_running() -> bool` | is the robot alive | |
| `meta2act(mact) -> action` | policy `MetaAction` -> executable command | default returns `mact['action']` |
| `obs2meta(device_data) -> dict` | SHM device data -> partial obs (`state`, `image`) | merged across devices |
| `shutdown()` | safe disconnect | |
| `get_action_dim()` | action dimension | |

For a simulation robot you can instead subclass `MujocoDeviceBase` /
`So101SimRobot` to reuse physics + IK (this example does exactly that).
`examples/06_custom_robot/my_robot.py` shows both: a `MySimSo101(So101SimRobot)`
subclass with custom `args` (e.g. `custom_serial`) and overridden extension
points.

## 2. Config that references your external class

```yaml
# examples/06_custom_robot/my_robot.yaml
type: examples/06_custom_robot/my_robot.py:MySimSo101   # <-- external file reference
name: my_so101_sim
args:
  name: my_so101_sim
  fps: 50.0
  scene_name: empty
  enable_viewer: true
  control_mode: delta_ee        # 7D [dx,dy,dz,d_roll,d_pitch,d_yaw,d_gripper]
  custom_serial: "SO101-CUSTOM-06"
```

The `type` field supports several external references (see `extensions.py`):

- **Local file**: `/path/my_robot.py:MyRobotInterface` or `/path/my_robot.py`
- **Local package dir**: `/path/my_robot_pkg/` (loads `__init__.py`)
- **Installed plugin**: `@robot/my_robot` (entry point group `vlastudio.robot`)
- **Dotted / module**: `my_package.devices:MyRobot`

`control_shm_name` is **auto-filled** from the first teleop device by
`vlastudio collect`, so you do not set it manually.

## 3. Keyboard teleop config

```yaml
# examples/06_custom_robot/keyboard_so101.yaml
type: vlastudio.deploy.teleoperator.keyboard.Keyboard
name: keyboard
args:
  name: keyboard
  fps: 50.0
  reset_key: "0"
  key_mappings:
    - {name: "X forward/back", key_positive: "w", key_negative: "s", scale: 0.005}
    - {name: "Y left/right",   key_positive: "a", key_negative: "d", scale: 0.005}
    - {name: "Z up/down",      key_positive: "r", key_negative: "f", scale: 0.005}
    - {name: "Roll",           key_positive: "u", key_negative: "j", scale: 0.05}
    - {name: "Pitch",          key_positive: "i", key_negative: "k", scale: 0.05}
    - {name: "Yaw",            key_positive: "o", key_negative: "l", scale: 0.05}
    - {name: "Gripper open/close", key_positive: "g", key_negative: "h", scale: 0.05}
```

One `key_mappings` entry = one action dimension; positive/negative keys map to
`+/- scale`.

## 4. Run it (pure Python)

Install the dependencies, then launch with the Python script (no `vlastudio`
console command needed):

```bash
# Windows
uv venv --python 3.11 .venv
uv pip install -e . mujoco opencv-python scipy pillow
.\examples\06_custom_robot\run.bat            # or: python examples/06_custom_robot/run.py

# Linux / macOS
uv venv --python 3.11 .venv
uv pip install -e . mujoco opencv-python scipy pillow
./examples/06_custom_robot/run.sh              # or: python examples/06_custom_robot/run.py
```

`run.py` is written to be read — it shows the whole flow with the public
building blocks, in order:

```python
# 1. Load the robot config and the teleop config (also auto-fills the
#    robot's control_shm_name from the teleop device)
all_configs, all_shm_names, teleop_configs, visualizer_configs = \
    load_device_configs(deploy_args, None)

# 2. Drop orphaned shared memory from a previous run
cleanup_all_shm(all_shm_names)

# 3. Start the robot and the teleop, each in its own process
device_procs = start_devices(all_configs)

# 4. Start the camera visualizer (reads the robot's observation SHM)
viz_procs = start_all_visualizers(all_configs, get_shm_name,
                                 is_camera_config, visualizer_configs)

# 5. Run until Ctrl+C, then stop
stop_all_visualizers(viz_procs)
for proc in device_procs:
    proc.terminate()
```

This opens two windows:

- a **keyboard teleop** window — click it, then:
  - `W/S` `A/D` `R/F` — move in X/Y/Z
  - `U/J` `I/K` `O/L` — roll/pitch/yaw
  - `G/H` — gripper open/close
  - `0` — reset to zero action
- a **live camera** window showing the robot's `frontview` camera

While keys are held, the robot follows on the camera feed. Press `Ctrl+C` to stop.

The camera window comes from the second row of `my_robot.yaml`:

```yaml
- type: vlastudio.deploy.visualizer.camera_view.CameraViewVisualizer
  name: my_so101_sim_view
  args:
    shm_name: my_so101_sim   # robot's observation shared memory
    camera_names: [frontview]
    fps: 30
```

Any `BaseVisualizer` subclass listed in the robot YAML is started by
`start_all_visualizers` and lives as long as the run.

## 5. Real-time camera feed only (cross-platform, headless-aware)

Show the live camera without teleop, purely through the public API:

```python
import vlastudio as vla

vla.show_camera(
    "examples/06_custom_robot/my_robot.yaml",
    camera_names=["frontview"],
    fps=30,
)
```

```bash
python examples/06_custom_robot/run.py camera
```

`vla.show_camera` starts the robot device in a subprocess, and the viewer
(`vlastudio.deploy.camera_viewer.LiveCameraViewer`) reads the observation frames
from the robot's shared memory and shows them with OpenCV.

Display policy (shared by the visualizer and `show_camera`):

- **Windows**: a window opens (desktop always available).
- **Linux/macOS**: a window opens only when `DISPLAY` is set. In a **headless**
  session without `DISPLAY`, frames are still read and logged but no window is
  shown (`show=False` forces this even with a display).
- **Xvfb / virtual display**: run under `xvfb-run -a python run.py camera` (or
  set `DISPLAY=:99` with an Xvfb server) — the feed renders into the virtual
  screen exactly as with a real display.

## Note on a from-scratch robot

If you are not reusing `So101SimRobot`, implement `AbstractRobotInterface`'s
abstract members in your own file, keep the same YAML shape (`type` pointing at
your file), and everything else (collect / infer / data collection) works the
same. The `meta2act`/`obs2meta` pair must stay consistent with the
`ActionManager` and the policy `state` layout you configure.
