import robomimic.utils.env_utils as EnvUtils
import sys
from tqdm import tqdm
import time
import torch
# from eval.utils import run_rollout, setup_seed
# from eval.config import ALL_ENV_CONFIGS
import argparse
import numpy as np
from collections import defaultdict
import json
import robomimic.utils.obs_utils as ObsUtils
from vlastudio.benchmark.robomimic.constant import ALL_ENV_CONFIGS, ALL_ENV_LANGUAGES
from vlastudio.data_utils.pose_utils import quat2axisangle
from ..base import *
from .constant import ALL_ENV_CONFIGS
from multiprocessing import current_process
import robomimic.utils.tensor_utils as TensorUtils

ALL_TASKS = ['Lift_Panda', "PickPlaceCan_Panda", "NutAssemblySquare_Panda", "ToolHang_Panda", "TwoArmTransport_Panda"]

def create_env(config):
    return RobomimicEnv(config)

class RobomimicEnv(MetaEnv):
    def __init__(self, config, *args):
        # 初始化env，仅从 config 读取参数
        self.config = config
        self.ctrl_space = getattr(self.config, 'ctrl_space', 'ee')
        self.ctrl_type = getattr(self.config, 'ctrl_type', 'delta')
        self.use_low_dim = getattr(self.config, 'use_low_dim', False)
        self.use_img = getattr(self.config, 'use_img', True)
        self.use_wrist_img = getattr(self.config, 'use_wrist_img', False)
        env = self.create_env()
        super().__init__(env)
    
    def get_state_keys(self, env_name):
        state_keys = [
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ]
        if env_name=='TwoArmTransport':
            state_keys += ['robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos']
        if self.use_low_dim:
            state_keys = ['object'] + state_keys
        return state_keys
    
    def get_img_keys(self, env_name):
        img_keys = []
        if env_name in ['Lift', 'PickPlaceCan', 'NutAssemblySquare',]:
            if self.use_img:
                img_keys.append('agentview_image')
            if self.use_wrist_img:
                img_keys.append('robot0_eye_in_hand_image')
        elif env_name in ['ToolHang',]:
            if self.use_img:
                img_keys.append('sideview_image')
            if self.use_wrist_img:
                img_keys.append('robot0_eye_in_hand_image') 
        elif env_name=='TwoArmTransport':
            if self.use_img:
                img_keys.extend(['shouldercamera0_image', 'shouldercamera1_image'])
            if self.use_wrist_img:
                img_keys.extend(['robot0_eye_in_hand_image', 'robot1_eye_in_hand_image'])
        return img_keys
    
    def create_env(self):
        task_info = self.config.task.split('_')
        env_name, robot_name = task_info[0], task_info[1]
        env_meta = ALL_ENV_CONFIGS[env_name][robot_name]
        self.raw_lang = ALL_ENV_LANGUAGES.get(env_name, '')
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_name,
            render=False,
            render_offscreen=True,
            use_image_obs=True,
            use_depth_obs=False,
        )
        self.state_keys = self.get_state_keys(env_name)
        self.img_keys = self.get_img_keys(env_name)
        modalities = {
            'obs':{
                "low_dim": [x for x in env.base_env.observation_names if 'image' not in x and 'depth' not in x],
                "rgb": [x for x in env.base_env.observation_names if 'image' in x],
                "depth": [x for x in env.base_env.observation_names if 'depth' in x],
                "scan": [],
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(modalities)
        
        # Get action bounds from action_space
        # Robomimic uses robosuite environments which have action_spec
        if hasattr(env, 'action_spec'):
            action_spec = env.action_spec
            if isinstance(action_spec, tuple) and len(action_spec) == 2:
                self.min_action = action_spec[0]  # low
                self.max_action = action_spec[1]  # high
            else:
                self.min_action = None
                self.max_action = None
        elif hasattr(env, 'action_space') and hasattr(env.action_space, 'low'):
            self.min_action = env.action_space.low
            self.max_action = env.action_space.high
        else:
            # Fallback for robosuite delta control (typical range)
            self.min_action = np.array([-1.0] * 7, dtype=np.float32)  # Typical delta EE control
            self.max_action = np.array([1.0] * 7, dtype=np.float32)
        
        return env
        
    def meta2act(self, maction: MetaAction):
        # MetaAct to action of libero
        assert maction['ctrl_space']==self.ctrl_space, f"The ctrl_space of MetaAction {maction['ctrl_space']} doesn't match the action space of environment {self.ctrl_space}"
        assert maction['ctrl_type']==self.ctrl_type, "Action must be delta action for LIBERO"
        actions = maction['action'] # (action_dim, )
        return actions
        
    def obs2meta(self, obs):
        # gripper state
        all_states = [obs[k] if 'quat' not in k else quat2axisangle(obs[k]) for k in self.state_keys]
        state_ee = np.concatenate(all_states, axis=0).astype(np.float32)
        # image - apply camera selection based on camera_ids
        all_imgs = [obs[k] for k in self.img_keys]
        image = np.stack(all_imgs)
        if np.max(image)<=1.0:
            image = (image*255.0).astype(np.uint8)
        return MetaObs(state=state_ee, image=image, raw_lang=self.raw_lang)

    def step(self, *args, **kwargs):
        obs, r, done, info = super().step(*args, **kwargs)
        done = self.env.is_success().get('task', False)
        return obs, r, done, info
    
    def reset(self):
        pid = current_process().pid  # 获取当前进程 ID
        seed = (pid * 1000 + time.time_ns()) % (2**32)  # 基于时间戳生成种子
        np.random.seed(seed)
        return super().reset()