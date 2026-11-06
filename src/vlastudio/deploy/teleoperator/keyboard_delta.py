"""Browser keyboard input producing device-independent Cartesian delta messages."""
import json
import os
import socket
import time

import numpy as np

from vlastudio.deploy.teleoperator.base import BaseTeleopDevice


class KeyboardDelta(BaseTeleopDevice):
    browser_keyboard = True

    def __init__(self, name="keyboard_delta", arms=1, fps=20,
                 translation_speed=0.06, rotation_speed=0.4, gripper_speed=1.5,
                 timeout=0.25, **kwargs):
        super().__init__(name, fps=fps)
        self.arms = int(arms)
        self.timeout = float(timeout)
        self.scales = np.array([translation_speed] * 3 + [rotation_speed] * 3 + [gripper_speed]) / fps
        self.keys = set()
        self.arm = 0
        self.received = 0
        self.socket = None

    def get_data(self):
        while True:
            try:
                packet = json.loads(self.socket.recv(2048))
            except BlockingIOError:
                break
            self.keys = set(packet["keys"])
            self.arm = int(packet["arm"])
            self.received = time.monotonic()
        if time.monotonic() - self.received > self.timeout:
            self.keys.clear()
        return {"keys": self.keys, "arm": self.arm}

    def convert_data_to_action(self, data):
        actions = np.zeros((self.arms, 7), dtype=np.float32)
        pairs = [('w', 's'), ('a', 'd'), ('r', 'f'), ('q', 'e'), ('t', 'g'), ('z', 'c'), ('v', 'b')]
        if 0 <= data['arm'] < self.arms:
            actions[data['arm']] = np.array([int(p in data['keys']) - int(n in data['keys'])
                                             for p, n in pairs]) * self.scales
        return actions.reshape(-1)

    def start(self):
        path = os.environ.get("VLASTUDIO_KEYBOARD_SOCKET")
        if not path:
            raise RuntimeError("KeyboardDelta requires a WebUI keyboard session")
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.bind(path)
        self.socket.setblocking(False)
        self.shm = self.create_shm()
        self.is_running = True
        try:
            while self.is_running:
                start = time.monotonic()
                action = self.convert_data_to_action(self.get_data())
                self.write_data_to_shm({"action": action, "ctrl_space": "ee", "ctrl_type": "delta"})
                time.sleep(max(0, 1 / self.fps - (time.monotonic() - start)))
        finally:
            self.close()

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        super().close()
