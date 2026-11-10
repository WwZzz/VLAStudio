"""Interactive MetaWorld lifecycle; the evaluation wrapper stays unchanged."""
import numpy as np
import gymnasium as gym
from vlastudio.benchmark.metaworld import MetaWorldEnv
from .base import CartesianAdapter
from vlastudio.benchmark.base import MetaAction


def _without_time_limits(env):
    """Remove horizon wrappers while retaining task selection and success logic."""
    if isinstance(env, gym.wrappers.TimeLimit):
        return _without_time_limits(env.env)
    if isinstance(env, gym.Wrapper):
        env.env = _without_time_limits(env.env)
    return env


class MetaWorldEnvironment(MetaWorldEnv):
    def __init__(self, config):
        super().__init__(config)
        self.ctrl_space, self.ctrl_type = 'ee', 'delta'

    def create_env(self):
        width, height = getattr(self.config, 'image_size', [320, 240])
        env = gym.make('Meta-World/MT1', env_name=self.config.task,
                       render_mode='rgb_array', camera_name=self.camera_name or 'corner',
                       width=width, height=height)
        if getattr(self.config, 'continuous', True):
            env = _without_time_limits(env)
            # MetaWorld itself also rejects step() after max_path_length.
            # Keep its public horizon consistent with the removed wrappers.
            env.unwrapped.max_path_length = float('inf')
        self.min_action, self.max_action = env.action_space.low, env.action_space.high
        return env

    def obs2meta(self, obs):
        result = super().obs2meta(obs)
        quaternion = np.roll(self.env.unwrapped.data.body('hand').xquat, -1)
        result.state_ee = np.r_[obs[:3], quaternion, obs[3]].astype(np.float32)
        result.state_joint = np.r_[self.env.unwrapped.data.qpos[:7], obs[3]].astype(np.float32)
        return result

    def reset(self):
        obs, _ = self.env.reset(seed=int(getattr(self.config, 'seed', 7)))
        for _ in range(self.num_steps_wait):
            obs, _, terminated, truncated, _ = self.env.step(np.zeros(4))
            if terminated or truncated:
                break
        self.prev_obs = self.obs2meta(obs)
        return self.prev_obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action['action'])
        info = dict(info)
        success = bool(info.get('success', False))
        info['success'] = success
        if success and getattr(self.config, 'continue_after_success', True):
            terminated = False
        self.prev_obs = self.obs2meta(obs)
        return self.prev_obs, reward, bool(terminated), bool(truncated), info


class MetaWorldAdapter(CartesianAdapter):
    """Map physical XYZ increments to MetaWorld's four native controls.

    MetaWorld has fixed orientation. Its fourth control is signed gripper
    effort (positive closes), not a normalized gripper position.
    """
    @staticmethod
    def create_environment(config):
        return MetaWorldEnvironment(config)

    def __init__(self, config):
        super().__init__(config)
        if self.arms != 1 or self.native_type != 'delta':
            raise ValueError('MetaWorld uses one arm with delta control')
        self.width = 4

    def reset(self, env, observation):
        self.target = np.array([0., 0., 0., -1.])
        self.action_scale = float(env.env.unwrapped.action_scale)
        if self.action_scale <= 0:
            raise ValueError('MetaWorld action_scale must be positive')

    def teleop_action(self, message, observation):
        if message.get('ctrl_space', 'ee') != 'ee' or message.get('ctrl_type', 'delta') != 'delta':
            raise ValueError('Teleoperation input must declare ee/delta')
        delta = np.asarray(message['action'], dtype=float).reshape(7)
        if not np.all(np.isfinite(delta)) or np.any(delta[3:6]):
            raise ValueError('MetaWorld supports XYZ translation and gripper only')
        self.target[:3] = np.clip(delta[:3] / self.action_scale, -1, 1)
        self.target[-1] = np.clip(self.target[-1] - 2 * delta[6], -1, 1)
        return {'action': self.target.copy()}

    def environment_action(self, action):
        return MetaAction(action=np.clip(action, -1, 1), ctrl_space='ee', ctrl_type='delta')
