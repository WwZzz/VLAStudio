"""Interactive RoboTwin lifecycle using the installed native task/control APIs."""
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path

import numpy as np

from vlastudio.benchmark.robotwin import RoboTwinEnv


class RoboTwinDeploymentEnv(RoboTwinEnv):
    def __init__(self, config):
        root = Path(getattr(config, 'robotwin_root', os.environ.get('VLASTUDIO_RUNTIME_ROBOTWIN_ROOT', '')))
        if not (root / 'envs').is_dir():
            raise ValueError('Configure VLASTUDIO_RUNTIME_ROBOTWIN_ROOT for the installed RoboTwin runtime')
        self.robotwin_root = str(root.resolve())
        self.continuous = bool(getattr(config, 'continuous', True))
        if not (root / 'env_cfg').is_dir():
            super().__init__(config)
            return
        # Current RoboTwin uses env_cfg and its own public XPolicyLab loader.
        self.config = config
        self.task_name = config.task
        self.task_config_name = getattr(config, 'task_config', 'demo_clean')
        self.ctrl_space = getattr(config, 'ctrl_space', 'ee')
        self.ctrl_space = 'qpos' if self.ctrl_space == 'joint' else self.ctrl_space
        self.ctrl_type = 'abs'
        self.seed = int(getattr(config, 'seed', 7))
        self.max_timesteps = int(getattr(config, 'max_timesteps', 1000000000))
        self.camera_names = list(getattr(config, 'camera_names', ['head_camera', 'left_camera', 'right_camera']))
        self.image_size = list(getattr(config, 'image_size', [240, 320]))
        self._env_initialized = False
        self.prev_obs = None
        with self._context():
            from scripts.eval_policy_xpolicylab import class_decorator, load_task_args
            self.task_config, _ = load_task_args(dict(task_name=self.task_name,
                task_config=self.task_config_name, policy_name='vlastudio_webui'))
            self.task_config.update(eval_mode=True, eval_video_log=False, save_data=False, render_freq=0)
            self.task_env = class_decorator(self.task_name)

    @contextmanager
    def _context(self):
        previous = os.getcwd()
        os.chdir(self.robotwin_root)
        try:
            yield
        finally:
            os.chdir(previous)

    def obs2meta(self, observation):
        result = super().obs2meta(observation)
        endpose = observation['endpose']
        result.state_ee = np.r_[endpose['left_endpose'], endpose['left_gripper'],
                                endpose['right_endpose'], endpose['right_gripper']].astype(np.float32)
        robot = self.task_env.robot
        result.state_joint = np.asarray(robot.get_left_arm_real_jointState() +
                                        robot.get_right_arm_real_jointState(), dtype=np.float32)
        if not result.raw_lang:
            result.raw_lang = self.task_name.replace('_', ' ')
        return result

    def get_tcp_poses(self):
        """Measured gripper centers, distinct from native EE action origins."""
        robot = self.task_env.robot
        return np.asarray([robot.get_left_tcp_pose(), robot.get_right_tcp_pose()], dtype=float)

    def reset(self):
        with self._context():
            if self._env_initialized:
                self.task_env.close_env(clear_cache=False)
            settings = deepcopy(self.task_config)
            settings.update(seed=self.seed, now_ep_num=0, eval_mode=True)
            self.task_env.setup_demo(is_test=True, **settings)
            self._env_initialized = True
            self.task_env.step_lim = float('inf') if self.continuous else self.max_timesteps
            self.task_env.take_action_cnt = 0
            self.prev_obs = self.obs2meta(self.task_env.get_obs())
        return self.prev_obs

    def step(self, action):
        with self._context():
            if self.continuous:
                # Native evaluation returns immediately after success; collection can continue.
                self.task_env.eval_success = False
            self.task_env.take_action(self.meta2act(action), action_type=self.ctrl_space)
            success = bool(self.task_env.eval_success)
            self.prev_obs = self.obs2meta(self.task_env.get_obs())
            truncated = self.task_env.take_action_cnt >= self.task_env.step_lim
        return self.prev_obs, float(success), False, bool(truncated), {'success': success}

    def close(self):
        if self._env_initialized:
            with self._context():
                self.task_env.close_env(clear_cache=True)
            self._env_initialized = False
