"""
Gymnasium Environment Adapter

Provides a MetaEnv wrapper for standard Gymnasium environments,
including MuJoCo environments like HalfCheetah, Ant, Humanoid, etc.

Usage:
    # Via config:
    python train_rl.py -a td3 -e gymnasium/halfcheetah
    
    # Programmatically:
    from benchmark.gymnasium import create_env
    env = create_env(config)
"""

import numpy as np
import gymnasium as gym
from ..base import MetaEnv, MetaObs, MetaAction
import torch



def create_env(config):
    """Factory function for creating GymnasiumEnv."""
    return GymnasiumEnv(config)


class GymnasiumEnv(MetaEnv):
    """
    MetaEnv wrapper for standard Gymnasium environments.
    
    Supports continuous control environments like:
    - MuJoCo: HalfCheetah-v5, Ant-v5, Humanoid-v5, Walker2d-v5, Hopper-v5, etc.
    - Classic Control: Pendulum-v1, MountainCarContinuous-v0
    - Box2D: LunarLanderContinuous-v3, BipedalWalker-v3
    
    Config attributes:
        task (str): Gymnasium environment ID (e.g., 'HalfCheetah-v5')
        render_mode (str): Render mode ('human', 'rgb_array', None). Default: None
        use_camera (bool): Whether to capture images. Default: False
        max_episode_steps (int): Override max steps. Default: use env default
        raw_lang (str): Language instruction. Default: task name
    """
    
    def __init__(self, config, *args):
        self.config = config
        self.task = config.task
        self.render_mode = getattr(config, 'render_mode', None)
        self.use_camera = getattr(config, 'use_camera', False)
        self.max_episode_steps = getattr(config, 'max_episode_steps', None)
        self.raw_lang = getattr(config, 'raw_lang', self.task)
        
        # Control space info (most MuJoCo envs use normalized actions)
        self.ctrl_space = getattr(config, 'ctrl_space', 'joint')
        self.ctrl_type = getattr(config, 'ctrl_type', 'abs')
        
        env = self._create_env()
        super().__init__(env)
        
        # Store action space bounds for reference
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.debug_step=0
    
    def _create_env(self):
        """Create the underlying Gymnasium environment."""
        kwargs = {}
        if self.render_mode:
            kwargs['render_mode'] = self.render_mode
        if self.max_episode_steps:
            kwargs['max_episode_steps'] = self.max_episode_steps
            
        env = gym.make(self.task, **kwargs)
        return env
    
    @property
    def action_space(self):
        """Return the action space of the environment."""
        return self.env.action_space
    
    @property
    def observation_space(self):
        """Return the observation space of the environment."""
        return self.env.observation_space
    
    def meta2act(self, maction: MetaAction):
        """Convert MetaAction to Gymnasium action."""
        actions = maction['action']  # (action_dim, )
        return actions
    
    def obs2meta(self, obs) -> MetaObs:
        """Convert Gymnasium observation to MetaObs."""
        # Handle different observation types
        if isinstance(obs, dict):
            # For goal-conditioned environments
            state = obs.get('observation', obs.get('state', None))
            if state is None:
                state = np.concatenate([v.flatten() for v in obs.values()])
        else:
            state = obs
        
        state = np.asarray(state, dtype=np.float32)
        
        # Capture image if requested
        image = None
        if self.use_camera and self.render_mode == 'rgb_array':
            image = self.env.render()
            if image is not None:
                # Convert to (N, C, H, W) format
                if len(image.shape) == 3:  # (H, W, C)
                    image = image[np.newaxis, ...]  # -> (1, H, W, C)
                if image.shape[-1] == 3:  # (N, H, W, C) -> (N, C, H, W)
                    image = image.transpose(0, 3, 1, 2)
        
        return MetaObs(state=state, image=image, raw_lang=self.raw_lang)
    
    def step(self, maction):
        """Execute action and return (obs, reward, done, info)."""
        action = self.meta2act(maction)
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        obs = self.obs2meta(observation)
        
        # Combine terminated and truncated into done
        # Store both for proper handling in RL algorithms
        done = terminated or truncated
        info['terminated'] = terminated
        info['truncated'] = truncated
        info['TimeLimit.truncated'] = truncated and not terminated
        self.debug_step += 1
        
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        """Reset environment and return initial observation."""
        obs, info = self.env.reset(**kwargs)
        self.debug_step=0
        return self.obs2meta(obs)
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def ensure_action_reasonable(self, action):
        """Ensure action is reasonable."""
        if self.action_low is not None or self.action_high is not None:
            # Convert Tensor to numpy array if needed
            if torch is not None and torch.is_tensor(action):
                action = action.detach().cpu().numpy()
            action = np.asarray(action)
            
            if np.any(action <= self.action_high) and np.any(action >= self.action_low):
                return True
            
        return False


# Common MuJoCo environment configurations
MUJOCO_ENVS = {
    'halfcheetah': 'HalfCheetah-v5',
    'ant': 'Ant-v5',
    'humanoid': 'Humanoid-v5',
    'walker2d': 'Walker2d-v5',
    'hopper': 'Hopper-v5',
    'swimmer': 'Swimmer-v5',
    'reacher': 'Reacher-v5',
    'pusher': 'Pusher-v5',
    'inverted_pendulum': 'InvertedPendulum-v5',
    'inverted_double_pendulum': 'InvertedDoublePendulum-v5',
}


def get_env_id(name: str) -> str:
    """Get full environment ID from short name."""
    return MUJOCO_ENVS.get(name.lower(), name)

