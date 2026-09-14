"""
CLI helpers for SO101 Plus follower motor setup and calibration.

Examples:
  # Assign servo IDs (connect ONE motor at a time; includes wrist_yaw = ID 7)
  python -m deploy.robot.so101_plus --port /dev/so101_left_arm --id so101_plus_left --setup-motors

  # Run joint range calibration (all 7 joints, including wrist_yaw)
  python -m deploy.robot.so101_plus --port /dev/so101_left_arm --id so101_plus_left --calibrate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import SO101PlusConfig
from .so101_plus import SO101Plus, JOINT_NAMES


def main():
    parser = argparse.ArgumentParser(description="SO101 Plus follower setup / calibrate")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/so101_left_arm")
    parser.add_argument("--id", default="so101_plus_left", help="Robot id (selects calibration JSON name)")
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help="Optional calibration directory (default: ~/.cache/huggingface/lerobot/calibration/robots/so101_plus)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--setup-motors", action="store_true", help="Program motor IDs 1-7 one-by-one")
    group.add_argument("--calibrate", action="store_true", help="Run homing + range-of-motion calibration")
    args = parser.parse_args()

    config = SO101PlusConfig(port=args.port, id=args.id, cameras={})
    if args.calibration_dir:
        config.calibration_dir = Path(args.calibration_dir)

    robot = SO101Plus(config)
    print(f"Robot type: {robot.name}")
    print(f"Joints ({len(JOINT_NAMES)}): {list(JOINT_NAMES)}")
    print("ID map: shoulder_pan=1, shoulder_lift=2, elbow_flex=3, wrist_flex=4,")
    print("        wrist_roll=5, gripper=6, wrist_yaw=7")

    if args.setup_motors:
        robot.setup_motors()
        return

    # calibrate
    robot.connect(calibrate=False)
    try:
        robot.calibrate()
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
