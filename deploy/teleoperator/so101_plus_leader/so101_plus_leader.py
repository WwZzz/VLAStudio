#!/usr/bin/env python

"""
SO101 Plus Leader teleoperator (7 motors, wrist_yaw on ID 7).

Motor map matches deploy.robot.so101_plus:
  1 shoulder_pan, 2 shoulder_lift, 3 elbow_flex, 4 wrist_flex,
  5 wrist_roll, 6 gripper, 7 wrist_yaw
"""

import logging
import sys
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.teleoperators.teleoperator import Teleoperator

try:
    from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
except ImportError:
    from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import SO101PlusLeaderConfig

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


class SO101PlusLeader(Teleoperator):
    """SO-101 Plus Leader Arm with wrist_yaw (servo ID 7)."""

    config_class = SO101PlusLeaderConfig
    name = "so101_plus_leader"

    def __init__(self, config: SO101PlusLeaderConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
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
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

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

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
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
                    logger.debug(
                        "[so101_plus_leader] Non-interactive: applying calibration file id=%s",
                        self.id,
                    )
            except EOFError:
                logger.debug(
                    "[so101_plus_leader] No stdin: applying calibration file id=%s",
                    self.id,
                )
                use_file = True
            if use_file:
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        print(
            f"\nRunning calibration of {self} (7 joints):\n"
            f"  {', '.join(JOINT_NAMES)}\n"
        )
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move ALL 7 joints sequentially through their entire ranges of motion "
            "(including wrist_yaw / ID 7).\n"
            "Recording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            motor_id = self.bus.motors[motor].id
            input(f"Connect the controller board to the '{motor}' motor only (will set ID={motor_id}) and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {motor_id}")

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
