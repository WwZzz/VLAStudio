import sys
import os
import threading
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'third_party', 'libero'))
from vlastudio.benchmark.base import MetaAction, MetaEnv, MetaObs
from libero.libero import benchmark as libero_bench
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from dataclasses import dataclass, field, fields, asdict
from vlastudio.data_utils.pose_utils import quat2axisangle
import numpy as np
from torchvision import transforms
import pickle
import time
import copy
import json
from PIL import Image, ImageDraw, ImageFont
from typing import List
from pathlib import Path
import argparse
from collections import deque
import imageio
from robosuite.controllers import load_controller_config
from loguru import logger 

benchmark_dict = libero_bench.get_benchmark_dict()

def create_env(config):
    return LiberoEnv(config)

class LiberoEnv(MetaEnv):
    def __init__(self, config, *args):
        # 初始化env，仅从 config 读取参数
        self.config = config
        self.ctrl_space = getattr(self.config, 'ctrl_space', 'ee')
        self.ctrl_type = getattr(self.config, 'ctrl_type', 'delta')
        self.camera_ids = getattr(self.config, 'camera_ids', [0,])
        self.use_openvla_gripper = getattr(self.config, 'use_openvla_gripper', False)
        self.use_wrist = getattr(self.config, 'use_wrist', False)
        self.num_steps_wait = getattr(self.config, 'num_steps_wait', 10)
        env = self.create_env()
        super().__init__(env)
        
    def create_env(self):
        task_info = self.config.task.split('_')
        task_name = task_info[0] + '_' + task_info[1] # libero_{object, goal, spatial, 10, 90}
        task_id = int(task_info[-1])
        ctrl_space = "OSC_POSE" if self.ctrl_space=='ee' else "JOINT_POSITION"  # ee or joint
        task_suite = benchmark_dict[task_name]()
        init_states = task_suite.get_task_init_states(task_id)
        task = task_suite.get_task(task_id)
        self.raw_lang =  task.language
        self.task_name = task.name
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        # step over the environment
        image_size = getattr(self.config, 'image_size', None)
        if image_size is not None:
            if isinstance(image_size, (list, tuple)):
                height, width = image_size
            elif isinstance(image_size, int):
                height, width = image_size, image_size
            else:
                raise ValueError("image_size should be list [height, width] or int")
            self.image_size = (height, width)
        else:
            self.image_size = None
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256,
        }
        env = OffScreenRenderEnv(**env_args)
        np.random.seed(None)
        state_index = np.random.choice(len(init_states))
        state = init_states[state_index]
        logger.info(f"Setting initial state {state_index} for task {self.task_name}")
        env.set_init_state(state)
        
        # Get action bounds from action_space
        # LIBERO uses robosuite environments with Box action space
        if hasattr(env, 'action_spec'):
            action_spec = env.action_spec
            self.min_action = action_spec[0]  # low
            self.max_action = action_spec[1]  # high
        elif hasattr(env, 'action_space') and hasattr(env.action_space, 'low'):
            self.min_action = env.action_space.low
            self.max_action = env.action_space.high
        else:
            # Fallback: LIBERO typically uses delta control with small bounds
            # [dx, dy, dz, droll, dpitch, dyaw, gripper]
            self.min_action = np.array([-1.0] * 7, dtype=np.float32)
            self.max_action = np.array([1.0] * 7, dtype=np.float32)
        
        return env
        
    def meta2act(self, maction: MetaAction):
        assert maction['ctrl_space']==self.ctrl_space, f"The ctrl_space of MetaAction {maction['ctrl_space']} doesn't match the action space of environment {self.ctrl_space}"
        assert maction['ctrl_type']==self.ctrl_type, "Action must be delta action for LIBERO"
        actions = maction['action'] # (action_dim, )
        # actions[:6] = actions[:6]*np.array([0.5, 0.5, 0.5, 0.05, 0.05, 0.05, ])
        if self.use_openvla_gripper:
            actions[6] = 1.-2.*actions[6]
        return actions
    
    def get_libero_dummy_action(self):
        """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
        return [0, 0, 0, 0, 0, 0, -1]

    def obs2meta(self, obs):
        gripper_state = obs['robot0_gripper_qpos'] # (2,) 
        xyz = obs['robot0_eef_pos'] # (3,)
        euler = quat2axisangle(obs['robot0_eef_quat']) # (3,)
        state_ee = np.concatenate([xyz, euler, gripper_state], axis=0).astype(np.float32)
        # joint state
        state_joint = np.concatenate([obs["robot0_joint_pos"], gripper_state], axis=0).astype(np.float32)
        # image - apply camera selection based on camera_ids
        img_primary = obs["agentview_image"][::-1, ::-1]
        all_imgs = [img_primary]
        if self.use_wrist:
            img_second = obs['robot0_eye_in_hand_image'][::-1, ::-1]
            all_imgs.append(img_second)
        image = np.stack(all_imgs)
        image = image.transpose(0, 3, 1, 2)
        # depth
        # depth_primary = obs["agentview_depth"][::-1, ::-1]
        # depth_second = obs['robot0_eye_in_hand_depth']
        # depth = np.stack([depth_primary, depth_second])
        return MetaObs(state=state_ee, image=image, raw_lang=self.raw_lang)
    
    def reset(self):
        self.env.reset()
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self.env.step(self.get_libero_dummy_action())
        self.prev_obs = self.obs2meta(obs)
        return self.prev_obs

    
    
