import os
os.environ['MS2_REAL2SIM_ASSET_DIR'] = os.path.join(os.path.dirname(__file__), 'ManiSkill2_real2sim', 'data')
import simpler_env
import numpy as np
from ..base import *
from scipy.spatial.transform import Rotation as R


def create_env(config):
    return SimplerEnv(config)

class SimplerEnv(MetaEnv):
    def __init__(self, config, *args):
        self.config = config
        self.ctrl_space = getattr(self.config, 'ctrl_space', 'joint')
        self.state_type = getattr(self.config, 'obs_type', 'pose_rpy') # qpos, pose, or pose_rpy
        self.ctrl_type = 'abs'
        self.max_timesteps = getattr(self.config, 'max_timesteps', 200)
        self.max_timesteps = getattr(self.config, 'max_timesteps', 200)
        env = self.create_env()
        if "google_robot" in env.robot_uid:
            self.camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            self.camera_name = "3rd_view_camera"
        if "google_robot" in env.robot_uid:
            self.camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            self.camera_name = "3rd_view_camera"
        self.raw_lang = env.get_language_instruction()
        super().__init__(env)

    
    def create_env(self):
        task = self.config.task
        env = simpler_env.make(task)
        
        # Get action bounds from action_space
        # SimplerEnv uses ManiSkill2 which has a gymnasium-style action_space
        if hasattr(env, 'action_space'):
            if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
                self.min_action = env.action_space.low
                self.max_action = env.action_space.high
            elif hasattr(env.action_space, 'minimum') and hasattr(env.action_space, 'maximum'):
                # dm_control style BoundedArray
                self.min_action = env.action_space.minimum
                self.max_action = env.action_space.maximum
            else:
                self.min_action = None
                self.max_action = None
        else:
            self.min_action = None
            self.max_action = None
        
        return env
        
    def meta2act(self, maction: MetaAction):
        actions = maction['action'] # (action_dim, )
        return actions
        
    def obs2meta(self, obs):
        state = obs['agent']['qpos']
        image = np.stack([obs['image'][self.camera_name]['rgb']])
        return MetaObs(state=state, image=image, raw_lang=self.raw_lang)


    def pose_to_rpy(self, tcp_pose_7d):
        """
        Args:
            tcp_pose_7d: [x, y, z, qx, qy, qz, qw] (from SimplerEnv)
        Returns:
            target_pose: [x, y, z, roll, pitch, yaw] (for Bridge V2 alignment)
        """
        # 1. 提取位置
        xyz = tcp_pose_7d[:3]
        
        # 2. 提取旋转 (Scalar-Last)
        quat_xyzw = tcp_pose_7d[3:]
        
        # 3. 转换为 Euler XYZ (Bridge Data 标准)
        r = R.from_quat(quat_xyzw)
        rpy = r.as_euler('xyz', degrees=False) # 返回弧度
    
        return np.concatenate([xyz, rpy])

    def step(self, *args, **kwargs):
        action = args[0]['action']
        observation, reward, terminated, truncated, info = self.env.step(action)
        obs = self.obs2meta(observation)
        done = info['success']
        info['terminated'] = terminated
        info['truncated'] = truncated
        return obs, reward, done, info
    
    def reset(self):
        obs, reset_info = self.env.reset()
        return self.obs2meta(obs)


