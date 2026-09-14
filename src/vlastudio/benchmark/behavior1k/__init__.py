"""
BEHAVIOR-1K Benchmark Environment for ILStudio

This module provides a wrapper for OmniGibson environments to integrate with ILStudio.
Aligned with official OmniGibson eval.py settings.

IMPORTANT: OmniGibson/Isaac Sim parses sys.argv and crashes on unknown arguments.
This module clears sys.argv before importing omnigibson to prevent this issue.
"""

import sys
import os

# Add BEHAVIOR-1K packages to path BEFORE any imports
BEHAVIOR_1K_PATH = os.path.join(os.path.dirname(__file__), 'BEHAVIOR-1K')
sys.path.insert(0, os.path.join(BEHAVIOR_1K_PATH, 'bddl3'))
sys.path.insert(0, os.path.join(BEHAVIOR_1K_PATH, 'OmniGibson'))
sys.path.insert(0, os.path.join(BEHAVIOR_1K_PATH, 'joylo'))

# Set OmniGibson environment variables BEFORE importing omnigibson
os.environ.setdefault("OMNIGIBSON_HEADLESS", "True")
os.environ.setdefault("OMNIGIBSON_REMOTE_STREAMING", "")

# CRITICAL: Save and clear sys.argv to prevent OmniGibson/Isaac Sim from crashing
_ORIGINAL_ARGV = sys.argv.copy()
sys.argv = [sys.argv[0]] if sys.argv else ['']

import numpy as np
import torch as th
from dataclasses import asdict
from loguru import logger

from vlastudio.benchmark.base import MetaAction, MetaEnv, MetaObs

# Global flag to track if OmniGibson has been initialized
_OG_INITIALIZED = False


def _setup_omnigibson_macros(headless=True):
    """
    Setup OmniGibson macros before creating environment.
    Aligned with official eval.py settings.
    """
    global _OG_INITIALIZED
    
    from omnigibson.macros import gm, macros
    
    # Set headless mode
    gm.HEADLESS = headless
    gm.RENDER_VIEWER_CAMERA = not headless
    
    # Official eval.py settings
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False  # Note: False in eval.py for stability
    gm.ENABLE_TRANSITION_RULES = True
    gm.ENABLE_OBJECT_STATES = True
    
    # Set grasp window to larger value (from eval.py)
    with macros.unlocked():
        macros.robots.manipulation_robot.GRASP_WINDOW = 0.75
    
    _OG_INITIALIZED = True


def create_env(config):
    """
    Create a BEHAVIOR-1K environment instance.
    
    Args:
        config: Environment configuration object with:
            - task (str): Task name (e.g., "assembling_gift_baskets")
            - instance_id (int, optional): Task instance ID (default: 0)
            - max_timesteps (int, optional): Max steps per episode
            - headless (bool, optional): Run in headless mode (default: True)
    
    Returns:
        Behavior1kEnv: Wrapped environment instance
    """
    global _ORIGINAL_ARGV
    if len(sys.argv) > 1:
        _ORIGINAL_ARGV = sys.argv.copy()
        sys.argv = [sys.argv[0]] if sys.argv else ['']
        logger.debug(f"Cleared sys.argv before creating env")
    
    return Behavior1kEnv(config)


class Behavior1kEnv(MetaEnv):
    """
    BEHAVIOR-1K environment wrapper for ILStudio.
    Aligned with official OmniGibson eval.py.
    """
    
    def __init__(self, config):
        """
        Initialize the BEHAVIOR-1K environment.
        
        Args:
            config: Environment configuration object
        """
        self.config = config
        
        # Extract configuration parameters
        self.task_name = getattr(config, 'task', None)
        self.instance_id = getattr(config, 'instance_id', 0)
        self.max_timesteps = getattr(config, 'max_timesteps', None)
        self.headless = getattr(config, 'headless', True)
        self.ctrl_space = getattr(config, 'ctrl_space', 'joint')
        self.ctrl_type = getattr(config, 'ctrl_type', 'delta')
        
        # Image settings (for obs conversion)
        self.image_size = getattr(config, 'image_size', [256, 256])
        
        # Create OmniGibson environment
        env = self._create_og_env()
        super().__init__(env)
        
        # Store robot reference
        self.robot = self.env.robots[0]
        self.raw_lang = self._get_task_language()
        
        # Load task instance if specified
        if self.task_name is not None and self.instance_id >= 0:
            self._load_task_instance(self.instance_id)
        
        logger.info(f"Initialized BEHAVIOR-1K environment: {self.task_name}")
        logger.info(f"  Robot: {self.robot.model_name}, Instance: {self.instance_id}")
        logger.info(f"  Control space: {self.ctrl_space}, Control type: {self.ctrl_type}")
    
    def _create_og_env(self):
        """Create the OmniGibson environment using official config generators."""
        global _OG_INITIALIZED, _ORIGINAL_ARGV
        
        # Clear sys.argv before importing omnigibson
        if len(sys.argv) > 1:
            _ORIGINAL_ARGV = sys.argv.copy()
            sys.argv = [sys.argv[0]] if sys.argv else ['']
        
        # Setup macros
        if not _OG_INITIALIZED:
            _setup_omnigibson_macros(headless=self.headless)
        
        import omnigibson as og
        from omnigibson.macros import gm
        
        # Build configuration
        cfg = self._build_og_config()
        
        # Stop existing sim if running
        if og.sim is not None:
            og.sim.stop()
        
        logger.info("Creating OmniGibson environment...")
        env = og.Environment(configs=cfg)
        
        return env
    
    def _build_og_config(self):
        """Build OmniGibson environment configuration aligned with eval.py."""
        # Try to use official config generators
        try:
            from gello.robots.sim_robot.og_teleop_utils import (
                load_available_tasks,
                generate_robot_config,
            )
            from omnigibson.learning.utils.eval_utils import (
                generate_basic_environment_config,
                PROPRIOCEPTION_INDICES,
            )
            
            if self.task_name is not None:
                available_tasks = load_available_tasks()
                if self.task_name in available_tasks:
                    task_cfg = available_tasks[self.task_name][0]
                    
                    # Generate configs using official functions
                    cfg = generate_basic_environment_config(
                        task_name=self.task_name,
                        task_cfg=task_cfg
                    )
                    
                    # Generate robot config
                    robot_config = generate_robot_config(
                        task_name=self.task_name,
                        task_cfg=task_cfg
                    )
                    # Update observation modalities
                    robot_config["obs_modalities"] = ["proprio", "rgb"]
                    robot_config["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
                    
                    cfg["robots"] = [robot_config]
                    
                    # Set max steps
                    if self.max_timesteps is not None:
                        cfg["task"]["termination_config"]["max_steps"] = self.max_timesteps
                    
                    cfg["task"]["include_obs"] = False
                    
                    logger.info(f"Using official config for task: {self.task_name}")
                    return cfg
                else:
                    logger.warning(f"Task {self.task_name} not in available_tasks, using fallback config")
        except Exception as e:
            logger.warning(f"Failed to use official config generators: {e}")
            logger.info("Falling back to manual config")
        
        # Fallback: manual configuration
        return self._build_fallback_config()
    
    def _build_fallback_config(self):
        """Build fallback configuration when official generators are not available."""
        # Robot config
        robot_config = {
            "type": "R1Pro",
            "name": "robot_r1",
            "obs_modalities": ["proprio", "rgb"],
            "action_normalize": False,
            "self_collisions": True,
            "grasping_mode": "assisted",
            "sensor_config": {
                "VisionSensor": {
                    "sensor_kwargs": {
                        "image_height": self.image_size[0],
                        "image_width": self.image_size[1],
                    }
                }
            },
        }
        
        # Scene config - default to Rs_int if no task specified
        scene_model = getattr(self.config, 'scene_model', 'Rs_int')
        scene_config = {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "include_robots": False,
        }
        
        # Task config
        if self.task_name is not None:
            task_config = {
                "type": "BehaviorTask",
                "activity_name": self.task_name,
                "activity_definition_id": 0,
                "activity_instance_id": 0,
                "online_object_sampling": False,
                "highlight_task_relevant_objects": False,
                "termination_config": {
                    "max_steps": self.max_timesteps or 5000,
                },
                "include_obs": False,
            }
        else:
            task_config = {
                "type": "DummyTask",
            }
        
        # Environment config
        env_config = {
            "action_frequency": 30,
            "physics_frequency": 120,
            "rendering_frequency": 30,
        }
        
        return {
            "scene": scene_config,
            "robots": [robot_config],
            "task": task_config,
            "env": env_config,
        }
    
    def _load_task_instance(self, instance_id: int):
        """
        Load a specific task instance configuration.
        Aligned with Evaluator.load_task_instance() from eval.py.
        
        Args:
            instance_id: The ID of the task instance to load
        """
        import omnigibson as og
        from omnigibson.macros import gm
        from omnigibson.utils.asset_utils import get_task_instance_path
        from omnigibson.utils.python_utils import recursively_convert_to_torch
        import json
        
        try:
            scene_model = self.env.task.scene_name
            tro_filename = self.env.task.get_cached_activity_scene_filename(
                scene_model=scene_model,
                activity_name=self.env.task.activity_name,
                activity_definition_id=self.env.task.activity_definition_id,
                activity_instance_id=instance_id,
            )
            
            tro_file_path = os.path.join(
                get_task_instance_path(scene_model),
                f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
            )
            
            if not os.path.exists(tro_file_path):
                logger.warning(f"Task instance file not found: {tro_file_path}")
                return
            
            with open(tro_file_path, "r") as f:
                tro_state = recursively_convert_to_torch(json.load(f))
            
            for tro_key, state in tro_state.items():
                if tro_key == "robot_poses":
                    presampled_robot_poses = state
                    robot_pos = presampled_robot_poses[self.robot.model_name][0]["position"]
                    robot_quat = presampled_robot_poses[self.robot.model_name][0]["orientation"]
                    self.robot.set_position_orientation(robot_pos, robot_quat)
                    self.env.scene.write_task_metadata(key=tro_key, data=state)
                else:
                    if tro_key in self.env.task.object_scope:
                        self.env.task.object_scope[tro_key].load_state(state, serialized=False)
            
            # Stabilize objects
            for _ in range(25):
                og.sim.step_physics()
                for entity in self.env.task.object_scope.values():
                    if not entity.is_system and entity.exists:
                        entity.keep_still()
            
            self.env.scene.update_initial_file()
            self.env.scene.reset()
            
            logger.info(f"Loaded task instance {instance_id}")
            
        except Exception as e:
            logger.warning(f"Failed to load task instance {instance_id}: {e}")
    
    def _get_task_language(self):
        """Get the natural language description of the current task."""
        if hasattr(self.env, 'task') and hasattr(self.env.task, 'natural_language_goal_conditions'):
            conditions = self.env.task.natural_language_goal_conditions
            if conditions:
                return " ".join(conditions)
        if self.task_name:
            return self.task_name.replace("_", " ")
        return ""
    
    def _flatten_obs_dict(self, obs: dict, parent_key: str = "") -> dict:
        """
        Flatten observation dictionary (aligned with eval_utils.flatten_obs_dict).
        """
        processed_obs = {}
        for key, value in obs.items():
            new_key = f"{parent_key}::{key}" if parent_key else key
            if isinstance(value, dict):
                processed_obs.update(self._flatten_obs_dict(value, parent_key=new_key))
            else:
                processed_obs[new_key] = value
        return processed_obs
    
    def obs2meta(self, obs):
        """
        Convert OmniGibson observations to MetaObs format.
        """
        # Flatten observation dict
        flat_obs = self._flatten_obs_dict(obs)
        
        robot_name = self.robot.name
        
        # Extract proprioceptive state
        proprio_key = f"{robot_name}::proprio"
        state_joint = None
        state_ee = None
        
        if proprio_key in flat_obs:
            proprio = flat_obs[proprio_key]
            if isinstance(proprio, th.Tensor):
                proprio = proprio.cpu().numpy()
            state_joint = proprio.astype(np.float32)
        
        # Try to get EEF state
        try:
            eef_pos = self.robot.get_eef_position()
            eef_quat = self.robot.get_eef_orientation()
            
            if isinstance(eef_pos, th.Tensor):
                eef_pos = eef_pos.cpu().numpy()
            if isinstance(eef_quat, th.Tensor):
                eef_quat = eef_quat.cpu().numpy()
            
            from scipy.spatial.transform import Rotation
            euler = Rotation.from_quat(eef_quat).as_euler('xyz')
            
            state_ee = np.concatenate([eef_pos, euler]).astype(np.float32)
        except Exception:
            state_ee = state_joint
        
        # Determine state based on control space
        if self.ctrl_space == 'ee':
            state = state_ee if state_ee is not None else state_joint
        else:
            state = state_joint if state_joint is not None else state_ee
        
        # Extract images
        images = []
        for key, value in flat_obs.items():
            if "::rgb" in key:
                rgb = value
                if isinstance(rgb, th.Tensor):
                    rgb = rgb.cpu().numpy()
                if rgb.ndim == 3 and rgb.shape[-1] in [3, 4]:
                    rgb = rgb[..., :3]  # Remove alpha if present
                    rgb = rgb.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
                images.append(rgb)
        
        image = np.stack(images) if images else None
        
        return MetaObs(
            state=state,
            state_ee=state_ee,
            state_joint=state_joint,
            image=image,
            raw_lang=self.raw_lang,
        )
    
    def meta2act(self, maction: MetaAction):
        """Convert MetaAction to OmniGibson action format.
        
        OmniGibson expects numpy array, not torch tensor.
        """
        action = maction['action']
        
        if isinstance(action, th.Tensor):
            action = action.cpu().numpy()
        
        # Ensure numpy array with correct dtype
        action = np.asarray(action, dtype=np.float32)
        
        return action
    
    def step(self, maction: MetaAction):
        """Execute one environment step.
        
        Returns:
            tuple: (obs, reward, done, info) - 4 values to match ILStudio convention
        """
        action = self.meta2act(maction)
        
        # Use n_render_iterations=1 for speed (from eval.py)
        obs, reward, terminated, truncated, info = self.env.step(action, n_render_iterations=1)
        
        meta_obs = self.obs2meta(obs)
        self.prev_obs = meta_obs
        
        # Combine terminated and truncated into done
        done = terminated or truncated
        
        # Extract success from nested info
        if 'done' in info and 'success' in info['done']:
            info['success'] = info['done']['success']
        
        # Store truncated in info for reference
        info['truncated'] = truncated
        
        # Return 4 values to match ILStudio's SequentialVectorEnv expectation
        return asdict(meta_obs), reward, done, info
    
    def reset(self):
        """Reset the environment."""
        obs, info = self.env.reset()
        self.prev_obs = self.obs2meta(obs)
        self.raw_lang = self._get_task_language()
        return self.prev_obs
    
    def close(self):
        """Close the environment and clean up resources."""
        if self.env is None:
            return
        
        # Don't call og.shutdown() here - it terminates the entire process
        # Just close the environment properly
        try:
            if self.env is not None:
                self.env.close()
        except Exception as e:
            logger.warning(f"Error during OmniGibson env close: {e}")
        finally:
            self.env = None
    
    def render(self):
        """Render the environment."""
        import omnigibson as og
        og.sim.render()
    
    @property
    def action_space(self):
        """Return the action space of the environment."""
        return self.env.action_space
    
    @property
    def observation_space(self):
        """Return the observation space of the environment."""
        return self.env.observation_space


def evaluate(args, policy, env, video_writer=None, save_example_dir=None):
    """
    Custom evaluation function for BEHAVIOR-1K environment.
    """
    from vlastudio.benchmark.utils import evaluate as default_evaluate
    return default_evaluate(args, policy, env, video_writer=video_writer, save_example_dir=save_example_dir)
