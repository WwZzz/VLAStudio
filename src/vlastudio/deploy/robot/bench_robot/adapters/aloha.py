"""Interactive ALOHA semantics, separate from the benchmark evaluation contract."""
from copy import copy
import numpy as np
from vlastudio.benchmark.aloha import AlohaSimEnv
from .base import CartesianAdapter, NativeJointAdapter


class AlohaEnvironment(AlohaSimEnv):
    def __init__(self, config):
        config = copy(config)
        if getattr(config, 'continue_after_success', False):
            config.time_limit = float('inf')
        super().__init__(config)

    def create_env(self):
        env = super().create_env()
        if self.ctrl_space == 'ee':
            # Helpers belong to the preview overlay, never to recorded sensors.
            for index in range(env.physics.model.nsite):
                name = env.physics.model.id2name(index, 'site') or ''
                if name.startswith('mocap_'):
                    env.physics.model.site_rgba[index, 3] = 0
        return env

    def obs2meta(self, obs):
        result = super().obs2meta(obs)
        result.state_joint = result.state
        physics = self.env.physics
        result.state_ee = np.concatenate([
            np.concatenate([physics.named.data.xpos[f'vx300s_{side}/gripper_link'],
                            physics.named.data.xquat[f'vx300s_{side}/gripper_link'],
                            [result.state[index]]])
            for side, index in [('left', 6), ('right', 13)]
        ]).astype(np.float32)
        return result

    def step(self, action):
        obs, reward, success, timestep = super().step(action)
        if not getattr(self.config, 'separate_termination', True):
            return obs, reward, success, timestep
        return obs, reward, False, bool(timestep.last()), {'success': bool(success)}


class AlohaAdapter(CartesianAdapter):
    """Preserve the model's calibrated mocap-to-gripper weld transform."""
    @staticmethod
    def create_environment(config):
        return AlohaEnvironment(config)

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


class AlohaJointAdapter(NativeJointAdapter):
    """Native joint control using ALOHA's deployment lifecycle."""
    @staticmethod
    def create_environment(config):
        return AlohaEnvironment(config)
