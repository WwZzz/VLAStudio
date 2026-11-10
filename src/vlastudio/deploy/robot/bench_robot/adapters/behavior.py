"""Interactive OmniGibson configuration; benchmark training classes stay unchanged."""
from copy import copy
import os
import signal
from pathlib import Path
import sys
import numpy as np
import yaml

from ..robot import BenchmarkRobot
from vlastudio.benchmark.behavior1k import Behavior1kEnv, BEHAVIOR_1K_PATH
from scipy.spatial.transform import Rotation
from vlastudio.benchmark.base import MetaAction


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


class BehaviorAdapter:
    arms, width, space, native_type = 2, 7, 'ee', 'delta'

    @staticmethod
    def create_environment(config):
        return BehaviorEnvironment(config)

    def __init__(self, config):
        self.config = dict(config)
        self.target = None

    def reset(self, env, observation):
        self.env, self.robot = env, env.robot
        self.control_frequency = env.control_frequency
        self.action_dim = self.robot.action_dim
        self.indices = {name: array(idx) for name, idx in self.robot.controller_action_idx.items()}
        self.target = env.no_op_action()
        self.opening = np.array([(self.target[self.indices['gripper_' + arm]][0] + 1) / 2
                                 for arm in self.robot.arm_names])
        if len(self.indices['base']) != 3 or len(self.robot.arm_names) != self.arms:
            raise ValueError('BEHAVIOR teleoperation requires a holonomic base and two arms')
        if any(len(self.indices['arm_' + arm]) != 6 for arm in self.robot.arm_names):
            raise ValueError('BEHAVIOR teleoperation requires six-dimensional IK controllers')

    def idle_action(self):
        result = self.target.copy()
        result[self.indices['base']] = 0
        for arm in self.robot.arm_names:
            result[self.indices['arm_' + arm]] = 0
        return result

    def teleop_action(self, message, observation):
        if message.get('ctrl_space', 'ee') != 'ee' or message.get('ctrl_type', 'delta') != 'delta':
            raise ValueError('BEHAVIOR teleoperation expects ee/delta increments')
        delta = np.asarray(message['action'], dtype=float).reshape(self.arms, 7)
        base = np.asarray(message.get('base_velocity', [0, 0, 0]), dtype=float).reshape(3)
        if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(base)):
            raise ValueError('Non-finite teleoperation command')
        if message.get('base_frame', 'robot') != 'robot':
            raise ValueError('Base velocity must use the robot frame')
        result = self.idle_action()
        limit = np.asarray(self.config.get('base_velocity_limit', [.25, .25, .5]))
        result[self.indices['base']] = np.clip(base, -limit, limit)
        world_to_base = Rotation.from_quat(array(self.robot.get_position_orientation()[1])).inv()
        for i, arm in enumerate(self.robot.arm_names):
            # The UI axes are world-aligned; OG IK commands are relative to the robot base.
            result[self.indices['arm_' + arm]] = np.r_[world_to_base.apply(delta[i, :3]),
                                                        world_to_base.apply(delta[i, 3:6])]
            self.opening[i] = np.clip(self.opening[i] + delta[i, 6], 0, 1)
            result[self.indices['gripper_' + arm]] = 2 * self.opening[i] - 1
        self.target = result.copy()
        return dict(action=result)

    def environment_action(self, action):
        return MetaAction(action=action.copy(), ctrl_space=self.space, ctrl_type=self.native_type)


class BehaviorRobot(BenchmarkRobot):
    process_shutdown_timeout_s = 30.0

    def _request_stop(self, signum, frame):
        # Closing Kit from a signal callback can interrupt its own native update.
        self._stop_requested = True
        self.is_running = False

    def _install_stop_handlers(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, self._request_stop)

    def connect(self):
        connected = super().connect()
        # OmniGibson installs its own SIGINT handler during import.
        self._install_stop_handlers()
        if getattr(self, '_stop_requested', False):
            raise KeyboardInterrupt
        return connected

    def start(self):
        self._stop_requested = False
        self._install_stop_handlers()
        return super().start()

    def close(self):
        if getattr(self, '_native_closed', False):
            return
        self._native_closed = True
        try:
            super().close()
        finally:
            # This device owns its dedicated process, so Kit can close here.
            og = sys.modules.get('omnigibson')
            if og is not None and og.app is not None:
                og.shutdown()
