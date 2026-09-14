"""XLeRobot++: BiSo101PP arms with head and omni-wheel motors."""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
from loguru import logger

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import OperatingMode

from deploy.robot.bi_so101_pp.robot import BiSo101PP
from deploy.robot.so101_pp.gripper_norm import flip_gripper_norm
from deploy.robot.so101_pp.robot import So101PP
from deploy.robot.so101_pp.so101_pp import JOINT_NAMES, SO101PP


class _XLerobotPPArmBackend(SO101PP):
    """Keep arm observations at 7D after auxiliary motors join the bus."""

    def get_observation(self) -> dict[str, Any]:
        obs = self.bus.sync_read("Present_Position", list(JOINT_NAMES))
        obs_dict = {f"{motor}.pos": value for motor, value in obs.items()}
        for camera_name, camera in self.cameras.items():
            obs_dict[camera_name] = camera.async_read()
        return flip_gripper_norm(obs_dict, self._gripper_invert_norm)


class _XLerobotPPArm(So101PP):
    ROBOT_BACKEND_CLS = _XLerobotPPArmBackend


class XLerobotPP(BiSo101PP):
    """XLeRobot body using the existing, calibrated BiSo101PP as its arms.

    The left serial bus carries the left arm (IDs 1-7) and head (IDs 8-9).
    The right serial bus carries the right arm (IDs 1-7) and base (IDs 8-10).
    VR actions remain the same 14D dual-arm actions as BiSo101PP.
    """

    ARM_CLS = _XLerobotPPArm

    HEAD_MOTOR_IDS = {
        "head_motor_1": 8,
        "head_motor_2": 9,
    }
    BASE_MOTOR_IDS = {
        "base_left_wheel": 8,
        "base_back_wheel": 9,
        "base_right_wheel": 10,
    }

    def __init__(
        self,
        *args,
        enable_head: bool = True,
        enable_base: bool = True,
        head_home_raw: list[int] | None = None,
        head_pan_range_ticks: int = 300,
        head_tilt_range_ticks: int = 250,
        head_pan_sign: int = 1,
        head_tilt_sign: int = 1,
        base_max_linear_m_s: float = 0.15,
        base_max_angular_deg_s: float = 45.0,
        base_max_raw: int = 2000,
        base_forward_sign: int = 1,
        base_turn_sign: int = 1,
        thumbstick_deadzone: float = 0.12,
        base_watchdog_s: float = 0.3,
        **kwargs,
    ):
        self.enable_head = bool(enable_head)
        self.enable_base = bool(enable_base)
        self._head_ready = False
        self._base_ready = False
        self._head_home_raw = None if head_home_raw is None else np.asarray(head_home_raw, dtype=int)
        if self._head_home_raw is not None and self._head_home_raw.shape != (2,):
            raise ValueError("head_home_raw must contain [pan, tilt]")
        self.head_pan_range_ticks = max(0, int(head_pan_range_ticks))
        self.head_tilt_range_ticks = max(0, int(head_tilt_range_ticks))
        self.head_pan_sign = 1 if int(head_pan_sign) >= 0 else -1
        self.head_tilt_sign = 1 if int(head_tilt_sign) >= 0 else -1
        self.base_max_linear_m_s = max(0.0, float(base_max_linear_m_s))
        self.base_max_angular_deg_s = max(0.0, float(base_max_angular_deg_s))
        self.base_max_raw = max(1, int(base_max_raw))
        self.base_forward_sign = 1 if int(base_forward_sign) >= 0 else -1
        self.base_turn_sign = 1 if int(base_turn_sign) >= 0 else -1
        self.thumbstick_deadzone = float(np.clip(thumbstick_deadzone, 0.0, 0.95))
        self.base_watchdog_s = max(0.1, float(base_watchdog_s))
        self._last_base_stick_at = 0.0
        self._base_is_moving = False

        # This constructs and connects the two PP arms with their original
        # calibration IDs before auxiliary, uncalibrated motors are registered.
        super().__init__(*args, **kwargs)

        try:
            if self.enable_head:
                self._attach_head()
            if self.enable_base:
                self._attach_base()
        except Exception:
            logger.exception("[XLerobotPP] failed to initialize head/base motors")
            self.shutdown()
            raise

        logger.info(
            "[XLerobotPP] arms=BiSo101PP, head_ids={}, base_ids={}, "
            "VR control=arms + head/base thumbsticks",
            list(self.HEAD_MOTOR_IDS.values()),
            list(self.BASE_MOTOR_IDS.values()),
        )

    @property
    def head_bus(self):
        return self.left._robot.bus

    @property
    def base_bus(self):
        return self.right._robot.bus

    @staticmethod
    def _register_aux_motors(bus, motors: dict[str, Motor]) -> None:
        """Register motors on an already-connected LeRobot bus.

        MotorsBus builds ID lookup tables and cached ID/model lists in its
        constructor. Updating only ``bus.motors`` leaves those structures stale
        and causes a KeyError on the first access to a newly added motor ID.
        """
        existing_ids = {motor.id for motor in bus.motors.values()}
        new_ids = [motor.id for motor in motors.values()]
        duplicate_ids = existing_ids.intersection(new_ids)
        if duplicate_ids or len(new_ids) != len(set(new_ids)):
            raise ValueError(f"Duplicate motor IDs on {bus.port}: {sorted(duplicate_ids)}")

        bus.motors.update(motors)
        bus._id_to_model_dict.update({motor.id: motor.model for motor in motors.values()})
        bus._id_to_name_dict.update({motor.id: name for name, motor in motors.items()})

        # These are functools.cached_property values populated while the arm
        # connected with IDs 1-7. Force them to reflect the extended bus.
        for cached_name in ("ids", "models", "_has_different_ctrl_tables"):
            bus.__dict__.pop(cached_name, None)

    def _attach_head(self) -> None:
        motors = {
            name: Motor(motor_id, "sts3215", MotorNormMode.RANGE_M100_100)
            for name, motor_id in self.HEAD_MOTOR_IDS.items()
        }
        self._register_aux_motors(self.head_bus, motors)
        names = list(motors)

        self.head_bus.disable_torque(names)
        for name in names:
            self.head_bus.write("Operating_Mode", name, OperatingMode.POSITION.value, normalize=False)
            self.head_bus.write("P_Coefficient", name, 16, normalize=False)
            self.head_bus.write("I_Coefficient", name, 0, normalize=False)
            self.head_bus.write("D_Coefficient", name, 43, normalize=False)

        # Enabling position torque with a stale goal can make the head jump.
        current = self.head_bus.sync_read("Present_Position", names, normalize=False, num_retry=3)
        self.head_bus.sync_write("Goal_Position", current, normalize=False, num_retry=3)
        self.head_bus.enable_torque(names)
        if self._head_home_raw is None:
            self._head_home_raw = np.asarray([current[name] for name in names], dtype=int)
        self._head_ready = True
        logger.info(
            "[XLerobotPP] head attached; current={} home_raw={}",
            current,
            self._head_home_raw.tolist(),
        )

    def _attach_base(self) -> None:
        motors = {
            name: Motor(motor_id, "sts3215", MotorNormMode.RANGE_M100_100)
            for name, motor_id in self.BASE_MOTOR_IDS.items()
        }
        self._register_aux_motors(self.base_bus, motors)
        names = list(motors)

        self.base_bus.disable_torque(names)
        for name in names:
            self.base_bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value, normalize=False)
        self.base_bus.sync_write("Goal_Velocity", dict.fromkeys(names, 0), normalize=False, num_retry=3)
        self.base_bus.enable_torque(names)
        self._base_ready = True
        logger.info("[XLerobotPP] base attached on right bus with zero wheel velocity")

    def _apply_deadzone(self, value: float) -> float:
        value = float(np.clip(value, -1.0, 1.0))
        magnitude = abs(value)
        if magnitude <= self.thumbstick_deadzone:
            return 0.0
        return float(np.sign(value) * (magnitude - self.thumbstick_deadzone) / (1.0 - self.thumbstick_deadzone))

    def _body_to_wheel_raw(self, forward: float, theta_deg_s: float) -> dict[str, int]:
        theta_rad_s = np.deg2rad(theta_deg_s)
        velocity = np.array([forward, 0.0, theta_rad_s], dtype=np.float64)
        angles = np.radians(np.array([240.0, 0.0, 120.0]) - 90.0)
        matrix = np.array([[np.cos(angle), np.sin(angle), 0.125] for angle in angles])
        wheel_rad_s = matrix.dot(velocity) / 0.05
        wheel_raw = wheel_rad_s * (180.0 / np.pi) * (4096.0 / 360.0)
        peak = float(np.max(np.abs(wheel_raw)))
        if peak > self.base_max_raw:
            wheel_raw *= self.base_max_raw / peak
        values = np.rint(wheel_raw).astype(int)
        return dict(zip(self.BASE_MOTOR_IDS, values.tolist()))

    @staticmethod
    def _stick(action_dict: dict, key: str) -> tuple[float, float] | None:
        raw = action_dict.get(key)
        if raw is None:
            return None
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if values.size < 2 or not np.all(np.isfinite(values[:2])):
            return None
        return float(values[0]), float(values[1])

    def _command_head_from_stick(self, stick: tuple[float, float]) -> None:
        if not self._head_ready or self._head_home_raw is None:
            return
        x = self._apply_deadzone(stick[0])
        y = self._apply_deadzone(stick[1])
        pan = int(self._head_home_raw[0] + self.head_pan_sign * x * self.head_pan_range_ticks)
        # WebXR forward is negative Y; forward should raise the head.
        tilt = int(self._head_home_raw[1] - self.head_tilt_sign * y * self.head_tilt_range_ticks)
        target = {
            "head_motor_1": int(np.clip(pan, 0, 4095)),
            "head_motor_2": int(np.clip(tilt, 0, 4095)),
        }
        self.head_bus.sync_write("Goal_Position", target, normalize=False, num_retry=3)

    def _command_base_from_stick(self, stick: tuple[float, float]) -> None:
        if not self._base_ready:
            return
        x = self._apply_deadzone(stick[0])
        y = self._apply_deadzone(stick[1])
        forward = -y * self.base_forward_sign * self.base_max_linear_m_s
        # WebXR left is negative X; positive theta is counterclockwise.
        theta = -x * self.base_turn_sign * self.base_max_angular_deg_s
        command = self._body_to_wheel_raw(forward, theta)
        self.base_bus.sync_write("Goal_Velocity", command, normalize=False, num_retry=3)
        self._last_base_stick_at = time.monotonic()
        self._base_is_moving = any(command.values())

    def process_action(self, action_dict: dict) -> dict:
        left_stick = self._stick(action_dict, "left_thumbstick")
        right_stick = self._stick(action_dict, "right_thumbstick")
        try:
            if left_stick is not None:
                self._command_head_from_stick(left_stick)
            if right_stick is not None:
                self._command_base_from_stick(right_stick)
        except Exception as exc:
            logger.warning("[XLerobotPP] thumbstick command failed: {}", exc)
        return super().process_action(action_dict)

    def _stop_base(self) -> None:
        if not self._base_ready:
            return
        if not self.base_bus.is_connected:
            self._base_ready = False
            return
        try:
            # SIGINT/SIGTERM can interrupt an SDK read after it sets is_using
            # but before it clears it. The interrupted call will never resume,
            # so release that stale flag before the emergency zero write.
            if getattr(self.base_bus.port_handler, "is_using", False):
                self.base_bus.port_handler.is_using = False
            self.base_bus.sync_write(
                "Goal_Velocity",
                dict.fromkeys(self.BASE_MOTOR_IDS, 0),
                normalize=False,
                num_retry=5,
            )
        except Exception as exc:
            logger.warning("[XLerobotPP] failed to stop base cleanly: {}", exc)
        self._base_is_moving = False

    def get_observation(self) -> Dict[str, Any] | None:
        if (
            self._base_ready
            and self._base_is_moving
            and time.monotonic() - self._last_base_stick_at > self.base_watchdog_s
        ):
            self._stop_base()
        observation = super().get_observation()
        if observation is None:
            return None
        try:
            if self._head_ready:
                observation["head_raw_qpos"] = self.head_bus.sync_read(
                    "Present_Position", list(self.HEAD_MOTOR_IDS), normalize=False
                )
            if self._base_ready:
                observation["base_raw_velocity"] = self.base_bus.sync_read(
                    "Present_Velocity", list(self.BASE_MOTOR_IDS), normalize=False
                )
        except Exception as exc:
            logger.warning("[XLerobotPP] auxiliary observation failed: {}", exc)
        return observation

    def set_home(self):
        self._stop_base()
        super().set_home()
        if self._head_ready and self._head_home_raw is not None:
            target = dict(zip(self.HEAD_MOTOR_IDS, self._head_home_raw.astype(int).tolist()))
            self.head_bus.sync_write("Goal_Position", target, normalize=False, num_retry=3)

    def shutdown(self):
        self._stop_base()
        # Give the zero-velocity command a moment to reach every wheel before
        # the inherited disconnect disables torque and closes both buses.
        if self._base_ready:
            time.sleep(0.05)
        super().shutdown()
        self._head_ready = False
        self._base_ready = False
