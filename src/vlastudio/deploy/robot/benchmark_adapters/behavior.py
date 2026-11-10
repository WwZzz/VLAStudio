"""Interactive OmniGibson configuration; benchmark training classes stay unchanged."""
from copy import copy
import os
from pathlib import Path
import sys

import numpy as np
import yaml

from vlastudio.benchmark.behavior1k import Behavior1kEnv, BEHAVIOR_1K_PATH


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


class BehaviorEnvironment(Behavior1kEnv):
    def __init__(self, config):
        config = copy(config)
        if getattr(config, 'task', None) == 'empty':
            config.task = None
        self.control_frequency = int(getattr(config, 'control_freq', 20))
        self.continuous = bool(getattr(config, 'continuous', True))
        self.camera_names = ['head', 'left_wrist', 'right_wrist']
        super().__init__(config)

    def _create_og_env(self):
        from vlastudio.benchmark.behavior1k import _setup_omnigibson_macros
        sys.argv = [sys.argv[0]]
        _setup_omnigibson_macros(headless=self.headless)
        import omnigibson as og
        cfg = self._build_og_config()
        if hasattr(os, 'geteuid') and os.geteuid() == 0:
            sys.argv.append('--allow-root')
        threads = int(getattr(self.config, 'cpu_threads', 8))
        if threads < 1:
            raise ValueError('cpu_threads must be positive')
        sys.argv += [f'--/plugins/carb.tasking.plugin/threadCount={threads}',
                     f'--/plugins/omni.tbb.globalcontrol/maxThreadCount={threads}',
                     f'--/persistent/physics/numThreads={threads}']
        if og.sim is not None:
            og.sim.stop()
        return og.Environment(configs=cfg)

    def _build_og_config(self):
        from omnigibson.learning.utils.eval_utils import generate_basic_environment_config

        cfg = self._build_fallback_config()
        robot = cfg['robots'][0]
        if self.task_name is None:
            cfg['scene'] = dict(type='Scene')
        else:
            source = Path(BEHAVIOR_1K_PATH) / 'joylo/sampled_task/available_tasks.yaml'
            tasks = yaml.safe_load(source.read_text())
            if self.task_name not in tasks or self.instance_id not in tasks[self.task_name]:
                raise ValueError('Task/instance is not in the installed BEHAVIOR task catalog')
            task = tasks[self.task_name][self.instance_id]
            cfg = generate_basic_environment_config(self.task_name, task)
            robot.update(position=task['robot_start_position'], orientation=task['robot_start_orientation'])
            cfg['robots'] = [robot]
            cfg['task']['activity_instance_id'] = self.instance_id
            cfg['task']['termination_config']['max_steps'] = self.max_timesteps
        cfg['env'].update(action_frequency=self.control_frequency,
                          rendering_frequency=self.control_frequency, physics_frequency=120)
        # Physical arm deltas (m/rad), body-frame base velocity (m/s, rad/s).
        controllers = dict(base=dict(name='HolonomicBaseJointController', motor_type='velocity',
                                    command_input_limits=None, command_output_limits=None),
                           trunk=dict(name='JointController', motor_type='position', use_delta_commands=False,
                                      command_input_limits=None, command_output_limits=None))
        for arm in ('left', 'right'):
            controllers['arm_' + arm] = dict(name='InverseKinematicsController', mode='pose_delta_ori',
                command_input_limits=None, command_output_limits=None, smoothing_filter_size=None)
            controllers['gripper_' + arm] = dict(name='MultiFingerGripperController', mode='smooth',
                motor_type='position', command_input_limits='default', command_output_limits='default')
        robot.update(controller_config=controllers, action_normalize=False)
        return cfg

    def no_op_action(self):
        result = np.zeros(self.robot.action_dim, dtype=np.float32)
        control = self.robot.get_control_dict()
        for name, controller in self.robot.controllers.items():
            result[array(self.robot.controller_action_idx[name])] = array(controller.compute_no_op_action(control))
        return result

    def obs2meta(self, observation):
        from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES

        result = super().obs2meta(observation)
        flat = self._flatten_obs_dict(observation)
        cameras = ROBOT_CAMERA_NAMES[self.robot.model_name]
        result.image = np.stack([array(flat[cameras[name] + '::rgb'])[..., :3].transpose(2, 0, 1)
                                 for name in self.camera_names])
        joints = array(self.robot.get_joint_positions()).astype(np.float32)
        result.state_joint = joints
        result.state = joints.copy()
        result.state_ee = np.concatenate([
            np.r_[array(self.robot.get_eef_position(arm)), array(self.robot.get_eef_orientation(arm)),
                  joints[array(self.robot.gripper_control_idx[arm])].mean()]
            for arm in self.robot.arm_names]).astype(np.float32)
        return result

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(self.meta2act(action), n_render_iterations=1)
        self.prev_obs = self.obs2meta(observation)
        info['success'] = bool(info.get('done', {}).get('success', False))
        if self.continuous and info['success']:
            terminated = False
        return self.prev_obs, reward, terminated, truncated, info
