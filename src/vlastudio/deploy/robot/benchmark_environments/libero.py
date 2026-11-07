"""Interactive LIBERO lifecycle, isolated from the evaluation environment."""
from copy import copy
from dataclasses import asdict
import numpy as np
from vlastudio.benchmark.libero import LiberoEnv


class LiberoDeploymentEnv(LiberoEnv):
    def __init__(self, config):
        config = copy(config)
        if getattr(config, 'continue_after_success', False):
            config.ignore_done = True
        super().__init__(config)

    def obs2meta(self, obs):
        result = super().obs2meta(obs)
        result.state_ee = result.state
        result.state_joint = np.concatenate([
            obs['robot0_joint_pos'], obs['robot0_gripper_qpos']
        ]).astype(np.float32)
        return result

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if not getattr(self.config, 'separate_termination', True):
            return obs, reward, done, info
        info = dict(info or {})
        success = bool(self.env.check_success())
        info['success'] = success
        return obs, reward, bool(done and not success), False, info

    def reset(self):
        if not getattr(self.config, 'restore_initial_state', False):
            return super().reset()
        self.env.reset()
        self.env.set_init_state(self.initial_state)
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self.env.step(self.get_libero_dummy_action())
        self.prev_obs = self.obs2meta(obs)
        return self.prev_obs
