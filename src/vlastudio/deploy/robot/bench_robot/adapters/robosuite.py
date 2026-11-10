"""Interactive robosuite lifecycle, preserving the robomimic policy contract."""
from copy import deepcopy
import random
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from vlastudio.benchmark.base import MetaEnv
from vlastudio.benchmark.robomimic import RobomimicEnv
from vlastudio.benchmark.robomimic.constant import ALL_ENV_CONFIGS, ALL_ENV_LANGUAGES
from vlastudio.data_utils.pose_utils import quat2axisangle
from .base import CartesianAdapter


class RobosuiteEnvironment(RobomimicEnv):
    def create_env(self):
        env_name, robot_name = self.config.task.split('_')
        metadata = deepcopy(ALL_ENV_CONFIGS[env_name][robot_name])
        self.raw_lang = ALL_ENV_LANGUAGES.get(env_name, '')
        self.state_keys = self.get_state_keys(env_name)
        self.img_keys = self.get_img_keys(env_name)
        width, height = getattr(self.config, 'image_size', [256, 256])
        self.seed = int(getattr(self.config, 'seed', 7))
        np.random.seed(self.seed)
        random.seed(self.seed)
        metadata['env_kwargs'].update(
            camera_names=[key.removesuffix('_image') for key in self.img_keys],
            camera_widths=width, camera_heights=height,
            control_freq=int(getattr(self.config, 'control_freq', 20)),
            ignore_done=bool(getattr(self.config, 'continuous', True)),
        )
        env = EnvUtils.create_env_from_metadata(
            env_meta=metadata, env_name=env_name, render=False,
            render_offscreen=True, use_image_obs=True, use_depth_obs=False,
        )
        # EnvRobosuite forces ignore_done=True while constructing the environment.
        env.base_env.ignore_done = bool(getattr(self.config, 'continuous', True))
        names = env.base_env.observation_names
        ObsUtils.initialize_obs_utils_with_obs_specs({'obs': {
            'low_dim': [key for key in names if 'image' not in key and 'depth' not in key],
            'rgb': self.img_keys, 'depth': [], 'scan': [],
        }})
        self.min_action, self.max_action = env.base_env.action_spec
        return env

    def obs2meta(self, obs):
        result = super().obs2meta(obs)
        # Keep state identical to the original benchmark (including both fingers).
        result.state_ee = np.concatenate([
            obs[key] if 'quat' not in key else quat2axisangle(obs[key])
            for key in self.state_keys if key != 'object'
        ]).astype(np.float32)
        result.state_joint = np.concatenate([
            obs[f'robot{i}_{key}']
            for i in range(len(self.env.base_env.robots))
            for key in ('joint_pos', 'gripper_qpos')
        ]).astype(np.float32)
        return result

    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.env.base_env.rng.bit_generator.state = np.random.default_rng(self.seed).bit_generator.state
        return MetaEnv.reset(self)

    def step(self, action):
        obs, reward, _, info = MetaEnv.step(self, action)
        info = dict(info or {})
        info['success'] = bool(self.env.is_success().get('task', False))
        # robomimic suppresses done; read the underlying robosuite horizon.
        return obs, reward, False, bool(self.env.base_env.done), info

    def close(self):
        self.env.base_env.close()


class RobosuiteAdapter(CartesianAdapter):
    @staticmethod
    def create_environment(config):
        return RobosuiteEnvironment(config)

    def reset(self, env, observation):
        super().reset(env, observation)
        base = env.env.base_env
        if self.arms != len(base.robots) or self.width != 7 or self.native_type != 'delta':
            raise ValueError('robosuite profile requires one 7D OSC action per robot')
        scales = []
        for robot in base.robots:
            controllers = getattr(robot, 'part_controllers', {})
            controller = controllers.get('right')
            if (controller is None or controller.name != 'OSC_POSE'
                    or controller.input_type != 'delta' or controller.input_ref_frame != 'world'
                    or list(robot._action_split_indexes.items()) != [
                        ('right', (0, 6)), ('right_gripper', (6, 7))]):
                raise ValueError('Use robosuite 1.5 BASIC with world-frame delta OSC_POSE and a 1D gripper')
            low, high = np.asarray(controller.output_min), np.asarray(controller.output_max)
            in_low, in_high = np.asarray(controller.input_min), np.asarray(controller.input_max)
            if (low.size != 6 or np.any(high <= 0) or not np.allclose(high, -low)
                    or not np.allclose(in_high, -in_low)):
                raise ValueError('Keyboard control requires symmetric 6D OSC input/output bounds')
            scales.append((in_high - in_low) / (high - low))
        self.scales = np.stack(scales)
        self.low, self.high = base.action_spec
        self.control_frequency = float(base.control_freq)
        if getattr(env.config, 'input_mode', 'ee_delta') == 'native':
            # Native idle must hold the open gripper in native units too.
            self.target = self.environment_action(self.target).action

    def teleop_action(self, message, observation):
        result = super().teleop_action(message, observation)
        self.target.reshape(self.arms, 7)[:, :6] *= self.scales
        result['action'] = self.target.copy()
        return result

    def environment_action(self, action):
        result = super().environment_action(action)
        result.action = np.clip(result.action, self.low, self.high)
        return result
