"""
Observation Format
===============================================================================================================
sim_transfer_cube_scripted / sim_insertion_scripted
---------------------------------------------------------------------------------------------------------------
obs [dict] = {
    'qpos': ndarray:(14,),
    'qvel': ndarray:(14,),
    'env_state': ndarray:(7,),
    'images': {
        'top': ndarray:(480, 460, 3),
        'angle': ndarray:(480, 460, 3),
        'vis': ndarray:(480, 460, 3),
    }
}
---------------------------------------------------------------------------------------------------------------
act [ndarray(14,)]
---------------------------------------------------------------------------------------------------------------
===============================================================================================================
Environment for simulated robot bi-manual manipulation, with joint position control
Action space:      [left_arm_qpos (6),             # absolute joint position
                    left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                    right_arm_qpos (6),            # absolute joint position
                    right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                    left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                    right_arm_qpos (6),         # absolute joint position
                                    right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                    "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                    left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                    right_arm_qvel (6),         # absolute joint velocity (rad)
                                    right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                    "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
"""
import os
import sys

# Configure offscreen rendering before dm_control or MuJoCo is imported.
if (sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")):
    os.environ.setdefault("MUJOCO_GL", "egl")

# Removed problematic numpy import that causes compatibility issues
from vlastudio.benchmark.base import MetaEnv, MetaAction, MetaObs, MetaPolicy
from .constants import SIM_TASK_CONFIGS
from .ee_sim_env import make_ee_sim_env
from .sim_env import make_sim_env, BOX_POSE
from .utils import sample_box_pose, sample_insertion_pose
import numpy as np
from dataclasses import asdict
import numpy as np
import time
import cv2
from multiprocessing import current_process
from loguru import logger

TASK_PROMPT = {
    'sim_transfer_cube_scripted': 'Transfer the red cube from the right arm to the left arm.',
    'sim_insertion_scripted': 'Insert the red peg into the blue socket.'
}

def create_env(config):
    return AlohaSimEnv(config)

class AlohaSimEnv(MetaEnv):
    def __init__(self, config, *args):
        # 初始化env，仅从 config 读取参数
        self.config = config
        self.ctrl_space = getattr(self.config, 'ctrl_space', 'joint')
        self.ctrl_type = getattr(self.config, 'ctrl_type', 'abs')
        self.camera_ids = getattr(self.config, 'camera_ids', [0])
        # 统一使用 image_size；同时用于 primary 与 wrist
        image_size = getattr(self.config, 'image_size', [640, 480])
        if isinstance(image_size, int):
            width, height = image_size, image_size
        elif isinstance(image_size, (list, tuple)):
            # assume [w, h] or (w, h)
            width, height = image_size
        else:
            raise ValueError("image_size should be list [width, height] or int")
        self.image_size_primary = (width, height)
        self.image_size_wrist = (width, height)
        env = self.create_env()
        super().__init__(env)
        
    def create_env(self):
        self.task_name = 'transfer_cube' if 'transfer' in self.config.task else 'insertion'
        self.raw_lang = TASK_PROMPT['sim_transfer_cube_scripted'] if 'transfer' in self.task_name else TASK_PROMPT['sim_insertion_scripted']
        # step over the environment
        env = make_sim_env('sim_'+self.task_name)
        # Get action bounds from action_spec
        action_spec = env.action_spec()
        self.min_action = action_spec.minimum
        self.max_action = action_spec.maximum
        return env
        
    def meta2act(self, maction: MetaAction):
        # assert maction['ctrl_space']==self.ctrl_space, f"The ctrl_space of MetaAction {maction['ctrl_space']} doesn't match the action space of environment {self.ctrl_space}"
        # assert maction['ctrl_type']==self.ctrl_type, "Action must be relative action for LIBERO"
        actions = maction['action'] # (action_dim, )
        return actions
        
    def obs2meta(self, obs):
        state_joint = np.concatenate([obs["qpos"], ], axis=0).astype(np.float32)
        # image
        img_primary = cv2.resize(obs['images']['top'], self.image_size_primary)
        img_wrist_left = cv2.resize(obs['images']['left_wrist'], self.image_size_wrist)
        img_wrist_right = cv2.resize(obs['images']['right_wrist'], self.image_size_wrist)
        all_imgs = [img_primary]
        if '1' in self.camera_ids or 1 in self.camera_ids:
            all_imgs = all_imgs + [img_wrist_left, ]
        if '2' in self.camera_ids or 2 in self.camera_ids:
            all_imgs = all_imgs + [img_wrist_right, ]
        image = np.stack(all_imgs)
        image = image.transpose(0, 3, 1, 2)
        return MetaObs(state=state_joint, image=image, raw_lang=self.raw_lang)
    
    def step(self, *args, **kwargs):
        act = self.meta2act(*args, **kwargs)
        ts = self.env.step(act)
        results = [ts.observation, ts.reward, self.env.task.max_reward==ts.reward, ts]
        self.prev_obs = results[0] = self.obs2meta(results[0])
        if isinstance(results[0], MetaObs): results[0] = asdict(results[0])
        return tuple(results)

    def reset(self):
        pid = current_process().pid  # 获取当前进程 ID
        seed = (pid * 1000 + time.time_ns()) % (2**32)  # 基于时间戳生成种子
        np.random.seed(seed)
        # global BOX_POSE
        if 'transfer' in self.task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif 'insertion' in self.task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset
        ts = self.env.reset()
        self.prev_obs = self.obs2meta(ts.observation)
        return self.prev_obs
    
    def close(self):
        self.env.close()
    
