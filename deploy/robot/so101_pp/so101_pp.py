#!/usr/bin/env python

"""
SO-101++ (so101_pp) Follower — so101_plus hardware with parallel gripper.

Motor ID map (same as so101_plus):
  1 shoulder_pan, 2 shoulder_lift, 3 elbow_flex, 4 wrist_flex,
  5 wrist_roll, 6 gripper, 7 wrist_yaw

Calibration is two-phase:
  1) Manual ROM for all joints except gripper
  2) Tkinter slider ROM for parallel gripper (cannot be moved by hand)
"""

import copy
import logging
import sys
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position

try:
    from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
except ImportError:
    from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import SO101PPConfig
from .gripper_norm import disarm_bus_gripper_drive_mode, flip_gripper_norm
from .gripper_slider_calib import calibrate_gripper_with_slider

logger = logging.getLogger(__name__)

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "wrist_yaw",
    "gripper",
)
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "wrist_yaw",
)
GRIPPER_JOINT = "gripper"


class SO101PP(Robot):
    """SO-101++ follower with parallel-gripper-aware calibration."""

    config_class = SO101PPConfig
    name = "so101_pp"

    def __init__(self, config: SO101PPConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        # Deepcopy so disarming bus drive_mode does not wipe the JSON invert flag.
        bus_calibration = copy.deepcopy(self.calibration) if self.calibration else {}
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "wrist_yaw": Motor(7, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=bus_calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        # drive_mode bit in JSON = invert flag for PP gripper (closed_raw > open_raw)
        self._gripper_invert_norm = disarm_bus_gripper_drive_mode(self.bus)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file "
                "or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        # Always unlock first — connect()/configure() may have re-enabled torque.
        self.bus.disable_torque()

        if self.calibration:
            # Subprocesses (collect_data) have no usable stdin — auto-apply the file.
            use_file = True
            try:
                if sys.stdin is not None and sys.stdin.isatty():
                    user_input = input(
                        f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                        f"or type 'c' and press ENTER to run calibration: "
                    )
                    use_file = user_input.strip().lower() != "c"
                else:
                    print(f"[so101_pp] Non-interactive: applying calibration file id={self.id}")
            except EOFError:
                print(f"[so101_pp] No stdin: applying calibration file id={self.id}")
                use_file = True
            if use_file:
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self._apply_calibration_to_bus()
                return

        print(
            f"\n========== SO101-PP calibration of {self} ==========\n"
            f"Joints: {', '.join(JOINT_NAMES)}\n"
            "Phase 1: arm joints (hand-move ROM, excludes gripper)\n"
            "Phase 2: parallel gripper via GUI slider\n"
        )

        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # ----- Phase 1: arm joints -----
        print("\n----- Phase 1/2: arm joints (no gripper) -----")
        print(f"  Calibrating: {', '.join(ARM_JOINTS)}")
        input("Move ARM joints (not gripper) to the middle of their ranges and press ENTER....")
        arm_homing = self.bus.set_half_turn_homings(list(ARM_JOINTS))

        print(
            "Move ALL arm joints sequentially through their entire ranges of motion "
            "(including wrist_yaw / ID 7; skip gripper).\n"
            "Recording positions. Press ENTER to stop..."
        )
        arm_mins, arm_maxes = self.bus.record_ranges_of_motion(list(ARM_JOINTS))

        # ----- Phase 2: parallel gripper via slider -----
        print("\n----- Phase 2/2: parallel gripper (GUI slider) -----")
        print("  A slider window will open: first close (min), then open (max).")
        g_min, g_max, g_homing, g_drive = calibrate_gripper_with_slider(self.bus, motor=GRIPPER_JOINT)
        print(f"  Gripper range recorded: min={g_min}, max={g_max}, drive_mode={g_drive}")

        self.calibration = {}
        for motor in ARM_JOINTS:
            m = self.bus.motors[motor]
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=arm_homing[motor],
                range_min=arm_mins[motor],
                range_max=arm_maxes[motor],
            )
        self.calibration[GRIPPER_JOINT] = MotorCalibration(
            id=self.bus.motors[GRIPPER_JOINT].id,
            drive_mode=g_drive,
            homing_offset=g_homing,
            range_min=g_min,
            range_max=g_max,
        )

        self._save_calibration()
        self._apply_calibration_to_bus()
        print("Calibration saved to", self.calibration_fpath)
        print(
            "Calibrated motors:",
            ", ".join(f"{name}(id={m.id})" for name, m in self.bus.motors.items()),
        )
        print(f"Gripper norm invert: {self._gripper_invert_norm} (API 0=closed, 100=open)")

    def _apply_calibration_to_bus(self) -> None:
        """Write calibration to motors; keep JSON invert flag; disarm bus drive_mode."""
        self.bus.write_calibration(copy.deepcopy(self.calibration), cache=True)
        self._gripper_invert_norm = disarm_bus_gripper_drive_mode(self.bus)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == GRIPPER_JOINT:
                    self.bus.write("Max_Torque_Limit", motor, 500)
                    self.bus.write("Protection_Current", motor, 250)
                    self.bus.write("Overload_Torque", motor, 25)

    def setup_motors(self) -> None:
        print(
            "\nSO101-PP motor setup\n"
            "Connect ONE motor at a time when prompted.\n"
            "ID map: 1 shoulder_pan, 2 shoulder_lift, 3 elbow_flex, 4 wrist_flex,\n"
            "        5 wrist_roll, 6 gripper, 7 wrist_yaw.\n"
        )
        for motor in reversed(self.bus.motors):
            motor_id = self.bus.motors[motor].id
            input(f"Connect the controller board to the '{motor}' motor only (will set ID={motor_id}) and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {motor_id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return flip_gripper_norm(obs_dict, self._gripper_invert_norm)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Flip API norm (0=closed) back to bus norm before write when needed.
        action = flip_gripper_norm(dict(action), self._gripper_invert_norm)
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        self.bus.sync_write("Goal_Position", goal_pos)
        sent = {f"{motor}.pos": val for motor, val in goal_pos.items()}
        return flip_gripper_norm(sent, self._gripper_invert_norm)

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
