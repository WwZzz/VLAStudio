"""Configured adapters isolate simulator control semantics from device lifecycle.

Custom adapters implement reset(env, observation), teleop_action(message, observation),
idle_action(), and environment_action(action), and expose target/arms/width/space/native_type.
The environment remains responsible for emitting MetaObs and its step result.
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
                result[i, :3] += increment[:3] * scale[:3]
                rotation = Rotation.from_rotvec(increment[3:6] * scale[3:])
                current = self._rotation(result[i, 3:-1])
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


class AlohaAdapter(CartesianAdapter):
    """Preserve the model's calibrated mocap-to-gripper weld transform."""
    def reset(self, env, observation):
        super().reset(env, observation)
        if self.arms != 2 or self.rotation != 'quat_wxyz' or self.native_type != 'abs':
            raise ValueError('ALOHA EE uses two absolute xyz + wxyz + gripper targets')
        physics = env.env.physics
        for i in range(self.arms):
            self.target.reshape(self.arms, self.width)[i, :7] = np.concatenate([
                physics.data.mocap_pos[i], physics.data.mocap_quat[i]])
        expected_hz = 1.0 / env.env.control_timestep()
        self.control_frequency = expected_hz

    def preview_data(self, env, observation):
        physics = env.env.physics
        cameras = []
        for index in env.camera_ids:
            name = ['top', 'left_wrist', 'right_wrist'][int(index)]
            cameras.append(dict(position=physics.named.data.cam_xpos[name].tolist(),
                                rotation=physics.named.data.cam_xmat[name].reshape(3, 3).tolist(),
                                fovy=float(physics.named.model.cam_fovy[name])))
        effectors = [dict(label=side[0].upper(), position=physics.named.data.xpos[f'vx300s_{side}/gripper_link'].tolist())
                     for side in ('left', 'right')]
        return dict(cameras=cameras, effectors=effectors, frame='world')


class LiberoAdapter(CartesianAdapter):
    """Convert physical Cartesian increments using the active OSC controller scale."""
    def reset(self, env, observation):
        super().reset(env, observation)
        controller = env.env.env.robots[0].controller
        low = np.asarray(controller.output_min)
        high = np.asarray(controller.output_max)
        in_low = np.asarray(controller.input_min)
        in_high = np.asarray(controller.input_max)
        if low.size != 6 or not np.allclose(high, -low):
            raise ValueError('LIBERO teleoperation requires a symmetric 6D OSC_POSE controller')
        self.config['delta_scale'] = (in_high - in_low) / (high - low)
        self.control_frequency = float(env.env.env.control_freq)

    def environment_action(self, action):
        result = super().environment_action(action)
        result.action[:6] = np.clip(result.action[:6], -1, 1)
        return result

    def preview_data(self, env, observation):
        from robosuite import macros
        sim = env.env.env.sim
        names = ['agentview'] + (['robot0_eye_in_hand'] if env.use_wrist else [])
        cameras = []
        for name in names:
            index = sim.model.camera_name2id(name)
            cameras.append(dict(position=sim.data.cam_xpos[index].tolist(),
                                rotation=sim.data.cam_xmat[index].reshape(3, 3).tolist(),
                                fovy=float(sim.model.cam_fovy[index]), flip_x=True,
                                flip_y=macros.IMAGE_CONVENTION == 'opencv'))
        return dict(cameras=cameras, effectors=[dict(label='EE', position=observation['state_ee'][:3].tolist())], frame='world')
