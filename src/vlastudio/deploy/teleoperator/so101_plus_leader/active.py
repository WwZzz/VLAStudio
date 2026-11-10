"""A hand-guided joint source. Only this worker owns the leader servo bus."""
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
from loguru import logger

from vlastudio.deploy.teleoperator.base import BaseTeleopDevice
from . import SO101PlusLeader, SO101PlusLeaderConfig
from .controller import _default_robots_calib_dir
from .mapping import JointMap, ORDER
from .calibration import load_reference, lock_port


class ActivePlusLeader(BaseTeleopDevice):
    @staticmethod
    def action_source_settings(params):
        directory = Path(params["calibration_dir"]).expanduser() if params.get("calibration_dir") else _default_robots_calib_dir()
        ref = load_reference(directory, params.get("robot_id", "so101_leader_plus"))
        if not Path(params.get("com") or ref["port"]).exists():
            raise ValueError("Calibrated leader USB port is unavailable")
        return dict(control_mode="qpos", absolute_gripper=True, arms=1,
                    qpos_contract="rebot_b601rs", align_leader=True)

    def __init__(self, name, joints, com=None, fps=50, robot_id="so101_leader_plus",
                 calibration_dir=None, **kwargs):
        super().__init__(name, fps=fps)
        directory = Path(calibration_dir).expanduser() if calibration_dir else _default_robots_calib_dir()
        ref = load_reference(directory, robot_id)
        self.port = com or ref["port"]
        self.port_lock = None
        self.device = SO101PlusLeader(SO101PlusLeaderConfig(port=self.port, id=robot_id,
            calibration_dir=directory))
        self.bus = self.device.bus
        self.socket = None
        self.token = None
        self.goal = None
        self.last = np.zeros(7)
        self.mapping = JointMap.from_reference(joints, ref)
        self.previous_limit = {}

    def _read(self):
        raw = self.bus.sync_read("Present_Position")
        pos = np.array([raw[k] for k in ORDER])
        if not np.isfinite(pos).all():
            raise RuntimeError("Invalid leader feedback")
        return pos

    def _release(self):
        self.bus.disable_torque(num_retry=2)
        if any(self.bus.sync_read("Torque_Enable").values()):
            raise RuntimeError("Leader torque-off was not confirmed")
        self.goal = None
        if self.previous_limit:
            self.bus.sync_write("Torque_Limit", self.previous_limit)
            self.previous_limit = {}

    def _align(self, target, token):
        goal = self.mapping.leader(target)
        pos = self._read()
        self.token = None
        self.last = np.array(target)
        self.goal = goal
        self.pending = token
        self.command = pos.copy()
        self.settled = 0
        self.started = self.tick = time.monotonic()
        self.previous_limit = self.bus.sync_read("Torque_Limit")
        limits = {k: min(int(v), 350) for k, v in self.previous_limit.items()}
        self.bus.sync_write("Torque_Limit", limits)
        # Seed the exact encoder positions before enabling. A normalized read
        # could conceal a position already outside its calibrated interval.
        raw = self.bus.sync_read("Present_Position", normalize=False)
        for name, value in raw.items():
            cal = self.device.calibration[name]
            if not cal.range_min - 3 <= value <= cal.range_max + 3:
                raise ValueError(f"{name} is outside its calibrated range")
        self.bus.sync_write("Goal_Position", raw, normalize=False)
        goals = self.bus.sync_read("Goal_Position", normalize=False)
        if self.bus.sync_read("Torque_Limit") != limits or any(
            abs(goals[name] - value) > 3 for name, value in raw.items()
        ):
            raise RuntimeError("Leader did not confirm the initial position and torque limits")
        self.bus.enable_torque()

    def _step(self, pos):
        now = time.monotonic()
        if now-self.started > 22 or np.max(abs(self.command-pos)) > 12:
            raise RuntimeError("Leader alignment timed out or tracking error exceeded allowed travel")
        step = 16 * min(now-self.tick, .05)  # normalized travel units per second
        self.tick = now
        self.command += np.clip(self.goal-self.command,-step,step)
        self.bus.sync_write("Goal_Position", dict(zip(ORDER,self.command)))
        self.settled = self.settled+1 if np.max(abs(self.goal-pos))<=1.5 else 0
        if self.settled >= 4:
            self._release()
            self.token = self.pending
            logger.info("Leader aligned and released for hand guidance")

    def start(self):
        endpoint = os.environ.get("VLASTUDIO_ACTION_CONTRACT")
        if not endpoint or os.environ.get("VLASTUDIO_WEB_GATE") != "collect":
            raise ValueError("Active leader requires a managed WebUI collection session")
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.bind(str(Path(endpoint).with_name("leader.sock")))
        self.socket.setblocking(False)
        self.port_lock = lock_port(self.port)
        self.bus.connect()
        if not self.device.is_calibrated or any(self.bus.sync_read("Operating_Mode").values()):
            raise ValueError("Leader calibration or position mode does not match; run calibration")
        self._release()
        self._align(np.zeros(7), None)
        while self.goal is not None:
            self._step(self._read())
            time.sleep(.02)
        self.shm = self.create_shm(is_writer=True)
        self.is_running = True
        while self.is_running:
            started = time.monotonic()
            data = self.get_data()
            self.write_data_to_shm(data)
            time.sleep(max(0,1/self.fps-(time.monotonic()-started)))

    def get_data(self):
        try:
            req = json.loads(self.socket.recv(2048))
        except BlockingIOError:
            req = None
        if req is not None:
            if self.goal is not None:
                self._release()
            self.token = None
            if "target" in req:
                self._align(req["target"], req["token"])
        pos = self._read()
        if self.goal is not None:
            self._step(pos)
            return dict(align_token="", action=self.last.copy())
        self.last = self.mapping.target(pos, self.last)
        return dict(align_token=self.token or "", action=self.last.copy())

    def convert_data_to_action(self, data):
        return data["action"]

    def close(self):
        try:
            if self.bus.is_connected:
                try:
                    self._release()
                finally:
                    self.bus.disconnect()
        finally:
            if self.socket:
                self.socket.close()
            if self.port_lock:
                self.port_lock.close()
            super().close()
