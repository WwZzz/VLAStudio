"""Browser keyboard input producing device-independent Cartesian delta messages."""
import json
import os
import socket
import time

import numpy as np

from vlastudio.deploy.teleoperator.base import BaseTeleopDevice


class KeyboardDelta(BaseTeleopDevice):
    browser_keyboard = True

    @staticmethod
    def action_source_settings(config):
        return dict(arms=int(config.get("arms", 1)), control_mode="delta_ee", absolute_gripper=False)

    @staticmethod
    def bindings_for_arms(arms, key_bindings=None):
        bindings = key_bindings or [
            [['w', 's'], ['a', 'd'], ['r', 'f'], ['q', 'e'], ['t', 'g'], ['z', 'c'], ['v', 'b']],
            [['i', 'k'], ['j', 'l'], ['u', 'o'], ['y', 'h'], ['p', ';'], ['n', 'm'], [',', '.']],
        ][:arms]
        if len(bindings) != arms or any(len(arm) != 7 or any(len(pair) != 2 for pair in arm) for arm in bindings):
            raise ValueError('Provide seven positive/negative key pairs per arm')
        keys = [key for arm in bindings for pair in arm for key in pair]
        if any(not isinstance(key, str) or not key or key != key.lower() for key in keys) or len(set(keys)) != len(keys):
            raise ValueError('Keyboard bindings must contain unique lowercase key names')
        return bindings

    @staticmethod
    def bindings_for_base(base_bindings=None):
        if base_bindings is None:
            return []
        if len(base_bindings) != 3 or any(len(pair) != 2 for pair in base_bindings):
            raise ValueError('Provide forward, lateral and yaw positive/negative key pairs')
        keys = [key for pair in base_bindings for key in pair]
        if any(not isinstance(key, str) or not key or key != key.lower() for key in keys) or len(set(keys)) != len(keys):
            raise ValueError('Base bindings must contain unique lowercase key names')
        return base_bindings

    def __init__(self, name="keyboard_delta", arms=1, fps=20,
                 translation_speed=0.06, rotation_speed=0.4, gripper_speed=1.5,
                 timeout=0.25, key_bindings=None, base_bindings=None,
                 base_translation_speed=0.25, base_rotation_speed=0.5, **kwargs):
        super().__init__(name, fps=fps)
        self.arms = int(arms)
        self.timeout = float(timeout)
        self.scales = np.array([translation_speed] * 3 + [rotation_speed] * 3 + [gripper_speed]) / fps
        self.keys = set()
        self.bindings = self.bindings_for_arms(self.arms, key_bindings)
        self.base_bindings = self.bindings_for_base(base_bindings)
        arm_keys = {key for arm in self.bindings for pair in arm for key in pair}
        if any(key in arm_keys for pair in self.base_bindings for key in pair):
            raise ValueError('Base and arm key bindings must not overlap')
        self.base_scales = np.array([base_translation_speed, base_translation_speed, base_rotation_speed], dtype=np.float32)
        self.received = 0
        self.socket = None

    def get_data(self):
        while True:
            try:
                packet = json.loads(self.socket.recv(2048))
            except BlockingIOError:
                break
            self.keys = set(packet["keys"])
            self.received = time.monotonic()
        if time.monotonic() - self.received > self.timeout:
            self.keys.clear()
        return {"keys": self.keys}

    def convert_data_to_action(self, data):
        actions = np.zeros((self.arms, 7), dtype=np.float32)
        for arm, pairs in enumerate(self.bindings):
            actions[arm] = np.array([int(p in data['keys']) - int(n in data['keys'])
                                     for p, n in pairs]) * self.scales
        return actions.reshape(-1)

    def message_from_data(self, data):
        message = dict(action=self.convert_data_to_action(data), ctrl_space='ee', ctrl_type='delta')
        if self.base_bindings:
            # Base input is a body-frame velocity, independent of arm delta units.
            message['base_velocity'] = np.array([int(p in data['keys']) - int(n in data['keys'])
                                                for p, n in self.base_bindings], dtype=np.float32) * self.base_scales
            message['base_frame'] = 'robot'
        return message

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
                self.write_data_to_shm(self.message_from_data(self.get_data()))
                time.sleep(max(0, 1 / self.fps - (time.monotonic() - start)))
        finally:
            self.close()

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        super().close()
