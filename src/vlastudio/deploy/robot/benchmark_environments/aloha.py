"""Interactive ALOHA semantics, separate from the benchmark evaluation contract."""
from copy import copy
import numpy as np
from vlastudio.benchmark.aloha import AlohaSimEnv


class AlohaDeploymentEnv(AlohaSimEnv):
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
