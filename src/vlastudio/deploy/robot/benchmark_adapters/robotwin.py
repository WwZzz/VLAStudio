"""Interactive RoboTwin lifecycle using the installed native task/control APIs."""
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from vlastudio.benchmark.robotwin import RoboTwinEnv


class RoboTwinEnvironment(RoboTwinEnv):
    def __init__(self, config):
        root = Path(getattr(config, 'robotwin_root', os.environ.get('VLASTUDIO_RUNTIME_ROBOTWIN_ROOT', '')))
        if not (root / 'envs').is_dir():
            raise ValueError('Configure VLASTUDIO_RUNTIME_ROBOTWIN_ROOT for the installed RoboTwin runtime')
        self.robotwin_root = str(root.resolve())
        self._control_frames = None
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

    @staticmethod
    def _pose_matrix(pose):
        pose = np.asarray(pose, dtype=float).reshape(7)
        result = np.eye(4)
        result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        result[:3, 3] = pose[:3]
        return result

    @staticmethod
    def _encode_poses(matrices):
        matrices = np.asarray(matrices)
        return np.c_[matrices[:, :3, 3],
                     Rotation.from_matrix(matrices[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]]

    def _bind_control_frames(self):
        robot = self.task_env.robot
        frames = []
        for side in ('left', 'right'):
            native = self._pose_matrix(self.task_env.get_arm_pose(side))
            tcp = self._pose_matrix(getattr(robot, f'get_{side}_tcp_pose')())
            # Ask the loaded robot for its action-to-planner-link transform.
            # Its EE joint frame alone omits rotation of the final wrist joint.
            endlink = robot._trans_from_gripper_to_endlink([0., 0., 0., 1., 0., 0., 0.], arm_tag=side)
            link_to_control = np.linalg.inv(endlink.to_transformation_matrix())
            frames.append((side, link_to_control, np.linalg.inv(native) @ tcp))
        self._control_frames = frames

    def _measured_tool_matrices(self, tcp):
        if self._control_frames is None:
            self._bind_control_frames()
        robot = self.task_env.robot
        result = []
        for side, link_to_control, control_to_tcp in self._control_frames:
            link = getattr(robot, side + '_ee').child_link
            control = link.get_entity_pose().to_transformation_matrix() @ link_to_control
            result.append(control @ control_to_tcp if tcp else control)
        return result

    def get_ee_control_poses(self):
        """Measured action-frame poses, including the final wrist joint's motion."""
        return self._encode_poses(self._measured_tool_matrices(tcp=False))

    def get_tcp_poses(self):
        """Measured gripper TCP poses from native link FK and tool calibration."""
        return self._encode_poses(self._measured_tool_matrices(tcp=True))

    def reset(self):
        with self._context():
            if self._env_initialized:
                self.task_env.close_env(clear_cache=False)
            settings = deepcopy(self.task_config)
            settings.update(seed=self.seed, now_ep_num=0, eval_mode=True)
            self.task_env.setup_demo(is_test=True, **settings)
            self._control_frames = None
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
