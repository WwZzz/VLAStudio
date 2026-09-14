#!/usr/bin/env python

"""SO101++ Leader teleoperator (parallel-gripper calibration)."""

import copy
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

from deploy.robot.so101_pp.gripper_norm import disarm_bus_gripper_drive_mode, flip_gripper_norm
from deploy.robot.so101_pp.gripper_slider_calib import calibrate_gripper_with_slider
from deploy.robot.so101_pp.so101_pp import ARM_JOINTS, GRIPPER_JOINT, JOINT_NAMES

from .config import SO101PPLeaderConfig

logger = logging.getLogger(__name__)


class SO101PPLeader(Teleoperator):
    config_class = SO101PPLeaderConfig
    name = "so101_pp_leader"

    def __init__(self, config: SO101PPLeaderConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
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
        self._gripper_invert_norm = disarm_bus_gripper_drive_mode(self.bus)

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
                    print(f"[so101_pp_leader] Non-interactive: applying calibration file id={self.id}")
            except EOFError:
                print(f"[so101_pp_leader] No stdin: applying calibration file id={self.id}")
                use_file = True
            if use_file:
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self._apply_calibration_to_bus()
                return

        print(
            f"\n========== SO101-PP Leader calibration of {self} ==========\n"
            f"Joints: {', '.join(JOINT_NAMES)}\n"
            "Phase 1: arm joints (hand-move ROM, excludes gripper)\n"
            "Phase 2: parallel gripper via GUI slider\n"
        )

        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

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
        print(f"Calibration saved to {self.calibration_fpath}")
        print(f"Gripper norm invert: {self._gripper_invert_norm} (API 0=closed, 100=open)")

    def _apply_calibration_to_bus(self) -> None:
        self.bus.write_calibration(copy.deepcopy(self.calibration), cache=True)
        self._gripper_invert_norm = disarm_bus_gripper_drive_mode(self.bus)

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
        action = flip_gripper_norm(action, self._gripper_invert_norm)
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
