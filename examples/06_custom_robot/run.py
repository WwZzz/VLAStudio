"""Example 06 launcher (pure Python, explicit device wiring).

    python examples/06_custom_robot/run.py           -> keyboard (delta_ee) teleop + live camera
    python examples/06_custom_robot/run.py slider    -> 6D joint slider (qpos) + live camera
    python examples/06_custom_robot/run.py camera    -> live camera only

This script is written to be read: it shows, step by step, how to load the robot
config, load the teleop config, start the devices, and start the camera
visualizer -- all with the public VLAStudio building blocks, without the
``vlastudio`` console command.

Two teleop modes exercise the custom SO101 robot through two control paths:
  * ``teleop``: delta end-effector control via the keyboard (7D, uses IK).
  * ``slider``: direct joint position control via a 6D slider (no IK), useful
    for comparing against the delta_ee path.

Each opens the teleop window plus a live camera window. Press Ctrl+C to stop.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from vlastudio.deploy.base import is_camera_config
from vlastudio.deploy.shm_utils import cleanup_all_shm
from vlastudio.deploy.utils import get_shm_name, load_device_configs, start_devices
from vlastudio.deploy.visualizer.base import start_all_visualizers, stop_all_visualizers

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_YAML = REPO_ROOT / "examples" / "06_custom_robot" / "my_robot.yaml"
TELEOP_YAML = REPO_ROOT / "examples" / "06_custom_robot" / "keyboard_so101.yaml"
ROBOT_QPOS_YAML = REPO_ROOT / "examples" / "06_custom_robot" / "my_robot_qpos.yaml"
SLIDER_YAML = REPO_ROOT / "examples" / "06_custom_robot" / "slider_so101.yaml"
OUTPUT_DIR = REPO_ROOT / "data" / "teleop_recordings"

# Which robot/teleop each teleop mode wires up.
MODES = {
    "teleop": (ROBOT_YAML, TELEOP_YAML),
    "slider": (ROBOT_QPOS_YAML, SLIDER_YAML),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", default="teleop", choices=["teleop", "slider", "camera"],
        help="teleop/slider: teleop + live camera; camera: live camera only",
    )
    options = parser.parse_args()

    os.chdir(REPO_ROOT)          # YAML paths and the robot ``type`` are repo-root relative
    sys.path.insert(0, str(REPO_ROOT))

    if options.mode == "camera":
        import vlastudio as vla
        frames = vla.show_camera(
            str(ROBOT_YAML), camera_names=["frontview"], fps=30,
            title="SO101 Sim - frontview",
        )
        print(f"viewed {frames} frames")
        return 0

    robot_yaml, teleop_yaml = MODES[options.mode]

    # 1. Load the robot config and the teleop config.
    #    ``load_device_configs`` returns:
    #      all_configs       -- device rows (robot + teleop), each with its SHM name
    #      all_shm_names     -- their shared-memory names
    #      teleop_configs    -- the teleop device rows
    #      visualizer_configs-- the visualizer rows from the YAML
    #    It also auto-fills the robot's ``control_shm_name`` from the teleop
    #    device, which is how the teleop drives the robot.
    deploy_args = SimpleNamespace(robot=str(robot_yaml), teleop=str(teleop_yaml))
    all_configs, all_shm_names, teleop_configs, visualizer_configs = (
        load_device_configs(deploy_args, None)
    )
    print(f"devices: {[get_shm_name(c) for c in all_configs]}")
    print(f"visualizers: {[c.get('name') for c in visualizer_configs]}")

    # 2. Drop orphaned shared-memory segments from a previous run.
    cleanup_all_shm(all_shm_names)

    # 3. Start the robot and the teleop, each in its own process.
    device_procs = start_devices(all_configs)
    time.sleep(1.0)

    # 4. Start the camera visualizer. It reads the robot's observation SHM and
    #    shows the ``frontview`` camera frames in a window.
    viz_procs = start_all_visualizers(
        device_configs=all_configs,
        get_shm_name_func=get_shm_name,
        is_camera_config_func=is_camera_config,
        visualizer_configs=visualizer_configs,
    )

    # 5. Run until Ctrl+C.
    print("Running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # 6. Stop the visualizer and the device processes.
        stop_all_visualizers(viz_procs)
        for proc in device_procs:
            proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
