"""
CLI helpers for SO101++ (parallel gripper) setup / calibrate.

Examples:
  python -m deploy.robot.so101_pp --port /dev/ttyACM0 --id so101_pp_leader --calibrate
  python -m deploy.robot.so101_pp --port /dev/ttyACM1 --id so101_pp_follower --calibrate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import SO101PPConfig
from .so101_pp import SO101PP, JOINT_NAMES, ARM_JOINTS


def main():
    parser = argparse.ArgumentParser(description="SO101++ (so101_pp) setup / calibrate")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--id", default="so101_pp_follower", help="Robot id (calibration JSON name)")
    parser.add_argument("--calibration-dir", default=None, help="Optional calibration directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--setup-motors", action="store_true", help="Program motor IDs 1-7 one-by-one")
    group.add_argument("--calibrate", action="store_true", help="Two-phase calibration (arm + gripper slider)")
    args = parser.parse_args()

    config = SO101PPConfig(port=args.port, id=args.id, cameras={})
    if args.calibration_dir:
        config.calibration_dir = Path(args.calibration_dir).expanduser()

    robot = SO101PP(config)
    print(f"Robot type: {robot.name}")
    print(f"Joints ({len(JOINT_NAMES)}): {list(JOINT_NAMES)}")
    print(f"Phase-1 arm joints: {list(ARM_JOINTS)}")
    print("ID map: shoulder_pan=1 ... wrist_roll=5, gripper=6, wrist_yaw=7")

    if args.setup_motors:
        robot.bus.connect()
        try:
            robot.setup_motors()
        finally:
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=True)
        return

    # Connect bus only — do NOT call robot.connect()/configure(), which re-enables
    # torque via torque_disabled()'s finally block and locks the arm before calib.
    robot.bus.connect()
    try:
        robot.calibrate()
    finally:
        if robot.bus.is_connected:
            try:
                robot.bus.disconnect(disable_torque=True)
            except Exception as exc:
                print(f"Warning: failed to disconnect cleanly: {exc}")


if __name__ == "__main__":
    main()
