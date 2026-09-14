import time
import numpy as np
from ..base import *
import gymnasium as gym
from multiprocessing import current_process
import metaworld
from .task_desc import TASK_DESC
import warnings

ALL_CAMERA_NAMES = ['corner', 'corner2', 'corner3', 'corner4', 'topview', 'behindGripper', 'gripperPOV']
ALL_TASKS = metaworld.ALL_V3_ENVIRONMENTS

def create_env(config):
    return MetaWorldEnv(config)

class MetaWorldEnv(MetaEnv):
    def __init__(self, config, *args):
        # 初始化env，仅从 config 读取参数
        self.config = config
        self.ctrl_space = getattr(self.config, 'ctrl_space', 'joint')
        self.ctrl_type = 'abs'
        self.render_mode = getattr(self.config, 'render_mode', 'rgb_array')
        self.camera_name = getattr(self.config, 'camera_name', None)
        self.use_camera = getattr(self.config, 'use_camera', True)
        self.robot_state_only = getattr(self.config, 'robot_state_only', True)
        self.num_steps_wait = getattr(self.config, 'num_steps_wait', 10)
        assert self.camera_name is None or self.camera_name in ALL_CAMERA_NAMES
        self.raw_lang = TASK_DESC[self.config.task][1]
        env = self.create_env()
        super().__init__(env)
    
    def create_env(self):
        task = self.config.task
        # 过滤 gymnasium 的 passive_env_checker 警告
        warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium.utils.passive_env_checker')
        if self.camera_name is not None:
            env = gym.make('Meta-World/MT1', env_name=task, render_mode=self.render_mode, camera_name=self.camera_name)
        else:
            env = gym.make('Meta-World/MT1', env_name=task, render_mode=self.render_mode)
        
        # Get action bounds from action_space
        if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
            self.min_action = env.action_space.low
            self.max_action = env.action_space.high
        else:
            # Fallback
            self.min_action = None
            self.max_action = None
        
        return env
        
    def meta2act(self, maction: MetaAction):
        # MetaAct to action of libero
        # assert maction['ctrl_space']==self.ctrl_space, f"The ctrl_space of MetaAction {maction['ctrl_space']} doesn't match the action space of environment {self.ctrl_space}"
        # assert maction['ctrl_type']==self.ctrl_type, "Action must be abs action for PandaGym"
        actions = maction['action'] # (action_dim, )
        return actions
        
    def obs2meta(self, obs):
        state = obs.astype(np.float32)    
        if self.robot_state_only: state = state[:4]
        if self.use_camera:
            # PandaGym only has one camera view from env.render()
            image = self.env.render()
            image = image[::-1, ::-1]
            if image is not None:
                # Convert to (N, C, H, W) format for consistency with other environments
                if len(image.shape) == 3:  # (H, W, C)
                    image = image[np.newaxis, ...]  # Add batch dimension -> (1, H, W, C)
                if image.shape[-1] == 3:  # (N, H, W, C) -> (N, C, H, W)
                    image = image.transpose(0, 3, 1, 2)
        else:
            image = None
        return MetaObs(state=state, image=image, raw_lang=self.raw_lang)

    def step(self, *args, **kwargs):
        action = args[0]['action']
        observation, reward, terminated, truncated, info = self.env.step(action)
        obs = self.obs2meta(observation)
        done = info['success']>0.
        info['terminated'] = terminated
        info['truncated'] = truncated
        return obs, reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.num_steps_wait):
            obs = self.env.step([0,0,0,0])
        return self.obs2meta(obs[0])