"""Shared Cartesian and native-joint action conversion.

Adapters declare environment_class or provide a custom create_environment(config)
factory. An explicit environment.type in the robot configuration takes precedence.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from vlastudio.benchmark.base import MetaAction


class CartesianAdapter:
    def __init__(self, config):
        self.config = dict(config)
        self.arms = int(config.get('arms', 1))
        self.native_type = config.get('type', 'delta')
        self.rotation = config.get('rotation', 'rotvec')
        self.space = config.get('space', 'ee')
        self.width = 8 if self.rotation in ('quat_xyzw', 'quat_wxyz') else 7
        self.target = None
        if self.space != 'ee' or self.native_type not in ('abs', 'delta') or self.arms < 1:
            raise ValueError('CartesianAdapter requires EE abs/delta control')

    def reset(self, env, observation):
        if self.native_type == 'abs':
            self.target = np.asarray(observation[self.config['initial_target']], dtype=float).copy()
            if self.target.size != self.arms * self.width:
                raise ValueError('Initial target shape does not match configured arms/rotation')
        else:
            self.target = np.zeros(self.arms * self.width)
            self.target[self.width-1::self.width] = self.config.get('initial_gripper', 1)

    def idle_action(self):
        action = self.target.copy()
        if self.native_type == 'delta':
            action.reshape(self.arms, self.width)[:, :-1] = 0
        return action

    def environment_action(self, action):
        native = action.copy().reshape(self.arms, self.width)
        native[:, -1] = (native[:, -1] * self.config.get('gripper_scale', 1)
                          + self.config.get('gripper_offset', 0))
        return MetaAction(action=native.reshape(-1), ctrl_space=self.space, ctrl_type=self.native_type)

    def _rotation(self, values):
        if self.rotation == "quat_wxyz":
            return Rotation.from_quat(np.roll(values, -1))
        if self.rotation == "quat_xyzw":
            return Rotation.from_quat(values)
        if self.rotation == "euler_xyz":
            return Rotation.from_euler("xyz", values)
        return Rotation.from_rotvec(values)

    def _encode_rotation(self, rotation):
        if self.rotation == "quat_wxyz":
            return np.roll(rotation.as_quat(), 1)
        if self.rotation == "quat_xyzw":
            return rotation.as_quat()
        if self.rotation == "euler_xyz":
            return rotation.as_euler("xyz")
        return rotation.as_rotvec()

    def teleop_action(self, message, observation):
        if message.get("ctrl_space", "ee") != "ee" or message.get("ctrl_type", "delta") != "delta":
            raise ValueError("Teleoperation input must declare ee/delta")
        delta = np.asarray(message["action"], dtype=float).reshape(self.arms, 7)
        if not np.all(np.isfinite(delta)):
            raise ValueError("Non-finite teleoperation input")
        result = self.target.copy().reshape(self.arms, self.width)
        scale = np.broadcast_to(self.config.get("delta_scale", [1] * 6), (6,))
        for i, increment in enumerate(delta):
            if self.native_type == "abs":
                current = self._rotation(result[i, 3:-1])
                translation = increment[:3] * scale[:3]
                if self.config.get("translation_frame", "world") == "local":
                    translation = current.apply(translation)
                result[i, :3] += translation
                rotation = Rotation.from_rotvec(increment[3:6] * scale[3:])
                if self.config.get("rotation_frame", "world") == "local":
                    rotation = current * rotation
                else:
                    rotation = rotation * current
                result[i, 3:-1] = self._encode_rotation(rotation)
            else:
                result[i, :6] = increment[:6] * scale
            # Inputs always express gripper opening delta in normalized units.
            result[i, -1] = np.clip(result[i, -1] + increment[6], 0, 1)
        self.target = result.reshape(-1)
        return {"action": self.target.copy()}


class NativeJointAdapter:
    """Hold joint targets between policy messages; no Cartesian teleoperation."""
    def __init__(self, config):
        self.arms = int(config['arms'])
        self.width = int(config['width'])
        self.space, self.native_type = 'joint', 'abs'
        self.target = None

    def reset(self, env, observation):
        self.target = np.asarray(observation['state_joint'], dtype=float).copy()
        if self.target.size != self.arms * self.width:
            raise ValueError('Joint state dimension does not match native action dimension')

    def idle_action(self):
        return self.target.copy()

    def teleop_action(self, message, observation):
        raise ValueError('Joint inference profile does not support Cartesian teleoperation')

    def environment_action(self, action):
        return MetaAction(action=action.copy(), ctrl_space=self.space, ctrl_type=self.native_type)
