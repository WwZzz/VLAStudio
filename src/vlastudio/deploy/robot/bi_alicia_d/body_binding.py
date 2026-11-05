"""Jetson VLAStudio adapter: select the arm by the serial number in its firmware."""
import fcntl
import inspect
import os
from pathlib import Path
import sys

import numpy as np
import serial
from serial.tools import list_ports
from loguru import logger
from vlastudio.deploy.robot.alicia_d import AliciaD
from .thermal_policy import ThermalPolicy

# Use the very same bundled driver as the installed VLAStudio adapter.
_sdk = str(Path(inspect.getfile(AliciaD)).parent)
if _sdk not in sys.path:
    sys.path.insert(0, _sdk)
from alicia_d_sdk.hardware import SerialComm, ServoDriver


class _CheckedSerial(serial.Serial):
    def write(self, data):
        self.check_packet(bytes(data))
        return super().write(data)


class _ExactPort(SerialComm):
    """Open precisely once, exclusively. A new connection must recheck identity."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thermal_policy = ThermalPolicy()

    def connect(self):
        if getattr(self, "_attempted", False):
            return False
        self._attempted = True
        self.serial_port = _CheckedSerial(
            self.port_name, self.baudrate, timeout=self.timeout,
            write_timeout=self.timeout, exclusive=True,
        )
        self.serial_port.check_packet = self._check_packet
        self._initialize_serial_port()
        return True

    def bind_fault_record(self, body_serial):
        if not body_serial.isalnum():
            raise ValueError("Invalid verified body serial")
        self.fault_file = Path(os.environ.get("VLASTUDIO_ALICIA_FAULT_DIR", str(Path.home() / ".local/state/vlastudio/alicia/thermal_faults"))) / body_serial
        self.thermal_policy.bind(self.fault_file)

    def read_frame(self):
        frame = super().read_frame()  # SDK has already checked framing and CRC
        if (isinstance(frame, (list, bytes, bytearray)) and len(frame) >= 21
                and bytes(frame[:3]) == b"\xaa\x06\x00" and frame[3] >= 15
                ):
            self.thermal_policy.observe(frame[18])
            # The SDK invokes read_frame while holding its communication lock.
            # Never write console logs here: a slow sink stalls both parsing and
            # motion writes. Count transient E1/E2 silently; confirmed faults are
            # still surfaced by _check_packet when a command is inhibited.
        return frame

    def _check_packet(self, data):
        if os.environ.get("VLASTUDIO_ALICIA_READ_ONLY") == "1":
            allowed = {bytes(ServoDriver.INFO_COMMAND_MAP[k]) for k in ("version", "joint_gripper")}
            if bytes(data) not in allowed:
                raise RuntimeError("Read-only startup smoke blocked a non-query packet")
        if data[:3] == b"\xaa\x06\x03":
            reason = self.thermal_policy.reason()
            if reason:
                raise RuntimeError("Joint/gripper command inhibited: " + reason)

    def acknowledge_warning_recovery(self, temperatures):
        self.thermal_policy.acknowledge_recovery(temperatures)


class BodySerialAliciaD(AliciaD):
    def refresh_rel_ee_anchor(self, use_actual_joints=True):
        # This pair has measured encoder feedback: anchor to fresh hardware,
        # rather than the upstream adapter's last commanded IK target.
        with self._lock:
            if not self._driver.acquire_info("joint_gripper", wait=True, timeout=.5):
                return False
            obs = self.get_observation()
            if obs is None:
                return False
            q = np.asarray(obs["joint_angles"], dtype=float).copy()
            self._target_qpos = q.copy()
            self._smooth_qpos = q.copy()
            self._rel_ee_anchor_pose = self.ik_solver.compute_fk(q).copy()
            self.ik_solver.reset(q)
            self.ik_solver.set_reference(q, self._rel_ee_anchor_pose)
            self._rel_ee_anchor_gripper = float(obs["gripper"])
            self._rel_ee_anchor_initialized = True
            return True

    def __init__(self, body_serial, **kwargs):
        if not body_serial or kwargs.get("port"):
            raise ValueError("Specify body_serial, not a fixed port")
        super().__init__(**kwargs)
        self.body_serial = body_serial
        self.verified_version = None

    def get_observation(self):
        obs = super().get_observation()
        if obs is not None:
            # In this serial-bound adapter qpos means measured configuration.
            # Never substitute the last commanded target for actual telemetry.
            obs["qpos"] = np.r_[obs["joint_angles"], float(obs["gripper"]) / 1000.0].astype(np.float32)
        return obs

    def connect(self):
        if self._connected:
            return self._driver.serial_comm.is_connected()
        lockdir = Path(os.environ.get("XDG_RUNTIME_DIR") or Path.home() / ".cache")
        lockdir.mkdir(exist_ok=True)
        # Serialize discovery so two VLAStudio device processes can start together.
        # Each verified port remains exclusively locked by pyserial during use.
        with (lockdir / "vlastudio-alicia-discovery.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ports = sorted(p.device for p in list_ports.comports()
                           if (p.vid, p.pid) == (0x1a86, 0x55d3))
            for port in ports:
                driver = ServoDriver(port=port, debug_mode=self.debug)
                driver.serial_comm = _ExactPort(port=port, lock=driver._lock)
                try:
                    if not driver.connect() or not driver.acquire_info("version", wait=True, timeout=2):
                        continue
                    version = driver.data_parser.get_info("version") or {}
                    if version.get("serial_number") != self.body_serial:
                        continue
                    driver.serial_comm.bind_fault_record(self.body_serial)
                    if not driver.acquire_info("joint_gripper", wait=True, timeout=2):
                        continue
                    state = driver.data_parser.get_joint_state()
                    if state is None or state.gripper is None:
                        continue
                    joints = np.asarray(state.angles, dtype=float)
                    if joints.shape != (6,) or not np.all(np.isfinite(joints)):
                        continue
                    # Seed from measured state; connect never sends a home/target command.
                    self._current_joint_angles = joints.copy()
                    self._target_qpos = joints.copy()
                    self._smooth_qpos = joints.copy()
                    self._last_sent_qpos = joints.copy()
                    self._current_gripper = self._target_gripper = state.gripper
                    self._last_sent_gripper = int(state.gripper)
                    self.port, self.verified_version = port, dict(version)
                    self._driver, self._connected = driver, True
                    logger.info("Bound {} to {} (firmware {})", self.body_serial, port,
                                version.get("firmware_version"))
                    return True
                except (OSError, serial.SerialException) as exc:
                    logger.debug("Skipping {}: {}", port, exc)
                finally:
                    if self._driver is not driver:
                        driver.disconnect()
        logger.error("No accessible arm returned body serial {} on {}", self.body_serial, ports)
        return False
