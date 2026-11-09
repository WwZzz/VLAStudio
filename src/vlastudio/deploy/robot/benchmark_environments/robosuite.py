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


class RobosuiteDeploymentEnv(RobomimicEnv):
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

    def get_preview_metadata(self, observation):
        """Read camera/EE geometry without adding simulator markers or changing RGB."""
        from robosuite import macros
        sim = self.env.base_env.sim
        cameras = []
        for key in self.img_keys:
            camera = None
            try:
                index = sim.model.camera_name2id(key.removesuffix('_image'))
                if index >= 0:
                    camera = dict(position=sim.data.cam_xpos[index].tolist(),
                                  rotation=sim.data.cam_xmat[index].reshape(3, 3).tolist(),
                                  fovy=float(sim.model.cam_fovy[index]),
                                  # robomimic flips RGB vertically after robosuite's convention.
                                  flip_y=macros.IMAGE_CONVENTION == 'opencv')
            except (AttributeError, KeyError, ValueError, IndexError):
                pass
            cameras.append(camera)
        if not any(cameras):
            return None
        poses = np.asarray(observation['state_ee']).reshape(-1, 8)
        effectors = [dict(label='EE' if len(poses) == 1 else f'R{i}', position=pose[:3].tolist())
                     for i, pose in enumerate(poses)]
        return dict(cameras=cameras, effectors=effectors, frame='world')

    def close(self):
        self.env.base_env.close()
