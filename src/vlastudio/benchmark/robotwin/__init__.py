"""
RoboTwin Environment Integration for ILStudio

RoboTwin is a dual-arm manipulation environment with 50 tasks.
"""

import sys
import os
import gc
import numpy as np
import yaml
import importlib
import cv2
from pathlib import Path
from loguru import logger
# Add RoboTwin to path
ROBOTWIN_ROOT = os.path.join(os.path.dirname(__file__), 'RoboTwin')
ROBOTWIN_ROOT = os.path.abspath(ROBOTWIN_ROOT)
GLOBAL_EPISODE_NUM = 0
# Change to RoboTwin directory for relative path resolution
_original_cwd = None

def _enter_robotwin_context():
    """Change to RoboTwin directory to resolve relative paths."""
    global _original_cwd
    _original_cwd = os.getcwd()
    os.chdir(ROBOTWIN_ROOT)

def _exit_robotwin_context():
    """Restore original working directory."""
    global _original_cwd
    if _original_cwd is not None:
        os.chdir(_original_cwd)

sys.path.insert(0, ROBOTWIN_ROOT)
sys.path.insert(0, os.path.join(ROBOTWIN_ROOT, 'policy'))
sys.path.insert(0, os.path.join(ROBOTWIN_ROOT, 'description/utils'))

# RoboTwin imports are done lazily in _enter_robotwin_context to avoid path issues
# These will be set when first needed
_UnStableError = None
_generate_episode_descriptions = None


def _is_cuda_oom(err: BaseException) -> bool:
    s = str(err)
    return (
        "CUDA out of memory" in s
        or "torch.OutOfMemoryError" in s
        or err.__class__.__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"}
    )


def _cleanup_torch_cuda(reason: str = "") -> None:
    """
    Best-effort cleanup for exception paths.
    Notes:
    - PyTorch uses a caching allocator; memory may look "not freed" in nvidia-smi even without leaks.
    - This helps when we repeatedly create/destroy heavy GPU objects (e.g., curobo warmup).
    """
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch  # local import to avoid hard dependency at module import time

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # helps free IPC handles in some multi-process cases
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        # If torch isn't available, nothing to do.
        pass
    if reason:
        logger.warning(f"[RoboTwinEnv] CUDA cleanup: {reason}")

def _ensure_robotwin_imports():
    """Lazily import RoboTwin utilities that require being in RoboTwin directory."""
    global _UnStableError, _generate_episode_descriptions
    if _UnStableError is None:
        from envs.utils.create_actor import UnStableError
        _UnStableError = UnStableError
    if _generate_episode_descriptions is None:
        from generate_episode_instructions import generate_episode_descriptions
        _generate_episode_descriptions = generate_episode_descriptions
    return _UnStableError, _generate_episode_descriptions

from ..base import MetaEnv, MetaObs, MetaAction


def create_env(config):
    """
    Create RoboTwin environment instance.
    
    Args:
        config: Configuration object with task and settings
        
    Returns:
        RoboTwinEnv instance
    """
    return RoboTwinEnv(config)


class RoboTwinEnv(MetaEnv):
    """
    RoboTwin Environment wrapper for ILStudio.
    
    RoboTwin features:
    - Dual-arm manipulation tasks
    - Rich visual observations (RGB, depth, point cloud)
    - 50 different manipulation tasks
    - Domain randomization support
    """
    
    def __init__(self, config):
        """
        Initialize RoboTwin environment.
        
        Args:
            config: Configuration with attributes:
                - task: Task name (e.g., 'pick_diverse_bottles')
                - task_config: Config file name (e.g., 'demo_clean')
                - max_timesteps: Maximum episode steps
                - ctrl_space: 'qpos' or 'ee' (default: 'qpos')
                - ctrl_type: 'abs' (RoboTwin only supports absolute control)
                - seed: Random seed
                - image_size: [H, W] for cameras (default: [480, 640])
                - camera_names: List of cameras to use (default: ['head_camera'])
                - robotwin_root: Custom RoboTwin root path (optional)
        """
        self.config = config
        
        # Allow custom RoboTwin root path
        global ROBOTWIN_ROOT
        custom_root = getattr(config, 'robotwin_root', None)
        if custom_root:
            ROBOTWIN_ROOT = os.path.abspath(custom_root)
        
        self.robotwin_root = ROBOTWIN_ROOT
        self.task_name = config.task
        self.task_config_name = getattr(config, 'task_config', 'demo_clean')
        self.max_timesteps = config.max_timesteps
        self.ctrl_space = getattr(config, 'ctrl_space', 'qpos')
        if self.ctrl_space=='joint': self.ctrl_space = 'qpos'
        self.ctrl_type = 'abs'  # RoboTwin only supports absolute control
        self.seed = getattr(config, 'seed', 0)
        self.image_size = getattr(config, 'image_size', [480, 640])
        self.camera_names = getattr(config, 'camera_names', ['head_camera'])
        
        # Robot embodiment configuration (optional override)
        # If not specified, will use embodiment from task_config YAML
        self.embodiment = getattr(config, 'embodiment', None)
        
        # Expert check: run play_once() to verify seed produces a solvable episode
        # If expert fails, skip to next seed (same as RoboTwin's eval_policy.py)
        self.expert_check = getattr(config, 'expert_check', True)
        
        # Instruction type for language instructions ('seen' or 'unseen')
        self.instruction_type = getattr(config, 'instruction_type', 'seen')
        
        # Planner configuration
        # use_planner=False: Skip planner initialization (faster, no GPU required)
        # use_planner=True: Try to use TOPP/Curobo for trajectory smoothing (with automatic fallback)
        # Note: Original RoboTwin tries to use planner but gracefully falls back if it fails
        self.use_planner = getattr(config, 'use_planner', False)
        
        # Change to RoboTwin directory for initialization
        _enter_robotwin_context()
        
        try:
            # Load task config
            task_config_path = os.path.join(self.robotwin_root, 'task_config', f'{self.task_config_name}.yml')
            with open(task_config_path, 'r') as f:
                self.task_config = yaml.safe_load(f)
            
            # Override with ILStudio settings
            self.task_config['task_name'] = self.task_name
            self.task_config['eval_mode'] = True
            self.task_config['save_data'] = False
            self.task_config['render_freq'] = 0  # No rendering by default
            self.task_config['seed'] = self.seed
            self.task_config['use_planner'] = self.use_planner  # Motion planner (Curobo + TOPP)
            self.task_config['step_lim'] = self.max_timesteps  # Override RoboTwin's default step limit
            
            # Load camera config and set dimensions
            self._load_camera_configs()
            
            # Load embodiment config to get action dimensions
            self._load_embodiment_configs()
            
            # Import and create task environment
            self.task_env = self._create_task_env()
            
        finally:
            # Restore original directory
            _exit_robotwin_context()
        
        # Initialize episode counter
        self.episode_num = GLOBAL_EPISODE_NUM
        self._env_initialized = False
        self._first_episode_done = False
        
        # Pre-initialize the first episode's environment BEFORE torch.inference_mode()
        # This ensures Curobo planner initialization happens with gradients enabled
        # Subsequent episodes will reuse this task_env to avoid the inference_mode issue
        _enter_robotwin_context()
        try:
            # Find a valid seed for initialization
            max_init_attempts = 50
            oom_fallback_done = False
            for init_attempt in range(max_init_attempts):
                try:
                    self.task_config['seed'] = self.seed + init_attempt
                    self.task_config['now_ep_num'] = 0
                    self.task_env.setup_demo(is_test=True, **self.task_config)
                    self.task_env.step_lim = self.max_timesteps
                    self._env_initialized = True
                    self._first_episode_done = True
                    self.episode_num = init_attempt
                    break
                except Exception as e:
                    # If planner warmup OOMs, do a one-time fallback to no-planner.
                    # Otherwise we may repeatedly warm up curobo and accumulate reserved VRAM.
                    if _is_cuda_oom(e) and self.use_planner and not oom_fallback_done:
                        oom_fallback_done = True
                        self.use_planner = False
                        self.task_config['use_planner'] = False
                        logger.warning(
                            f"[RoboTwinEnv] Planner warmup OOM during pre-init (seed_offset={init_attempt}). "
                            f"Falling back to use_planner=False and retrying."
                        )
                    # Try to close and recreate for next attempt
                    try:
                        self.task_env.close_env(clear_cache=True)
                    except:
                        pass
                    # Drop references and force CUDA cleanup on exception paths
                    try:
                        del self.task_env
                    except Exception:
                        pass
                    _cleanup_torch_cuda(reason=f"pre-init attempt failed: {type(e).__name__}: {e}")
                    # Recreate task_env for next attempt
                    self.task_env = self._create_task_env()
                    if init_attempt == max_init_attempts - 1:
                        raise RuntimeError(f"Failed to pre-initialize environment after {max_init_attempts} attempts: {e}")
        finally:
            _exit_robotwin_context()
        
        # Calculate action dimension: (left_arm + left_gripper + right_arm + right_gripper)
        self.action_dim = self.left_arm_dim + 1 + self.right_arm_dim + 1
        
        # Create a dummy action_space for compatibility (gymnasium.spaces.Box)
        import gymnasium as gym
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
        # Get action bounds from action_space
        self.min_action = self.action_space.low
        self.max_action = self.action_space.high
        
        # Don't call parent init as we handle env differently
        self.env = None
        self.prev_obs = None
        
    def _load_camera_configs(self):
        """Load camera configurations and set dimensions."""
        # Use absolute path to RoboTwin's config directory
        configs_path = os.path.join(self.robotwin_root, 'task_config')
        
        # Load camera config
        camera_config_path = os.path.join(configs_path, "_camera_config.yml")
        with open(camera_config_path, 'r') as f:
            camera_configs = yaml.safe_load(f)
        
        # Get head camera type from task config
        if 'camera' not in self.task_config:
            self.task_config['camera'] = {}
        
        head_camera_type = self.task_config['camera'].get('head_camera_type', '480p')
        
        # Set camera dimensions based on camera type
        if head_camera_type in camera_configs:
            self.task_config['head_camera_h'] = camera_configs[head_camera_type]['h']
            self.task_config['head_camera_w'] = camera_configs[head_camera_type]['w']
        else:
            # Fallback to default
            self.task_config['head_camera_h'] = self.image_size[0]
            self.task_config['head_camera_w'] = self.image_size[1]
        
        # Override camera collection based on camera_names
        # Use RoboTwin's original camera names directly
        self.task_config['camera']['collect_head_camera'] = 'head_camera' in self.camera_names
        self.task_config['camera']['collect_wrist_camera'] = ('left_camera' in self.camera_names or 
                                                               'right_camera' in self.camera_names)
    
    def _load_embodiment_configs(self):
        """Load embodiment configurations to determine action dimensions."""
        # Use absolute path to RoboTwin's config directory
        configs_path = os.path.join(self.robotwin_root, 'task_config')
        
        # Load embodiment config
        embodiment_config_path = os.path.join(configs_path, "_embodiment_config.yml")
        with open(embodiment_config_path, 'r') as f:
            embodiment_types = yaml.safe_load(f)
        
        # Get embodiment type from ILStudio config (if specified) or task config
        if self.embodiment is not None:
            embodiment_type = self.embodiment if isinstance(self.embodiment, list) else [self.embodiment]
        else:
            embodiment_type = self.task_config.get('embodiment', ['aloha-agilex'])
        
        def get_embodiment_file(embodiment_name):
            """Get the robot file path for an embodiment."""
            robot_file = embodiment_types[embodiment_name]['file_path']
            if robot_file is None:
                raise ValueError(f"No file path for embodiment '{embodiment_name}'")
            return robot_file
        
        def get_embodiment_config(robot_file):
            """Load embodiment config from robot file."""
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Handle single or dual embodiment
        if len(embodiment_type) == 1:
            # Same robot for both arms (dual-arm embodied)
            left_robot_file = get_embodiment_file(embodiment_type[0])
            right_robot_file = left_robot_file
            self.task_config['dual_arm_embodied'] = True
        elif len(embodiment_type) == 3:
            # Different robots for each arm
            left_robot_file = get_embodiment_file(embodiment_type[0])
            right_robot_file = get_embodiment_file(embodiment_type[1])
            self.task_config['embodiment_dis'] = embodiment_type[2]
            self.task_config['dual_arm_embodied'] = False
        else:
            raise ValueError("embodiment_type must have 1 or 3 elements")
        
        # Set robot files in task config
        self.task_config['left_robot_file'] = left_robot_file
        self.task_config['right_robot_file'] = right_robot_file
        
        # Load embodiment configs
        left_config = get_embodiment_config(left_robot_file)
        right_config = get_embodiment_config(right_robot_file)
        
        self.task_config['left_embodiment_config'] = left_config
        self.task_config['right_embodiment_config'] = right_config
        
        # Get action dimensions
        self.left_arm_dim = len(left_config.get('arm_joints_name', [[]])[0])
        self.right_arm_dim = len(right_config.get('arm_joints_name', [[]])[1])
    
    def _create_task_env(self):
        """Dynamically load the task environment class."""
        try:
            envs_module = importlib.import_module(f"envs.{self.task_name}")
            env_class = getattr(envs_module, self.task_name)
            return env_class()
        except Exception as e:
            raise ValueError(f"Failed to load task '{self.task_name}': {e}")
    
    def _init_env_for_episode(self, seed_offset=0):
        """
        Initialize environment for a new episode.
        
        Args:
            seed_offset: Additional offset to add to seed (used when retrying with expert_check)
        """
        # Update seed for this episode
        global GLOBAL_EPISODE_NUM
        self.task_config['seed'] = self.seed + GLOBAL_EPISODE_NUM + seed_offset
        self.task_config['now_ep_num'] = GLOBAL_EPISODE_NUM

        # Change to RoboTwin directory for episode setup
        _enter_robotwin_context()
        
        try:
            # For first reset() call, the environment is already initialized in __init__
            # We need to close and re-setup with a fresh seed
            if self._env_initialized:
                # Close the previous episode's environment state
                try:
                    self.task_env.close_env(clear_cache=False)
                except:
                    pass
            
            # Note: We do NOT create a new task_env here because Curobo planner
            # initialization requires gradients, which aren't available in torch.inference_mode()
            # The task_env was pre-initialized in __init__ with the planner already warmed up
            # We just call setup_demo to reset the scene with a new seed
            
            # Setup the demo/episode with new seed
            # print(self.task_config)
            self.task_config['eval_mode'] = True
            self.task_env.setup_demo(is_test=True, **self.task_config)
            
            # Force override step_lim after setup to ensure our max_timesteps is used
            self.task_env.step_lim = self.max_timesteps
            
            self._env_initialized = True
        except Exception as e:
            # Some episodes may fail due to unstable object placement
            # This is normal in RoboTwin
            self._env_initialized = False
            raise RuntimeError(f"Failed to initialize episode {self.episode_num}: {e}")
        finally:
            # Restore original directory
            _exit_robotwin_context()
    
    def _run_expert_check(self, seed_offset=0):
        """
        Run expert check to verify if the current seed produces a solvable episode.
        
        This mimics RoboTwin's expert_check logic in eval_policy.py:
        1. Setup environment with current seed (no rendering)
        2. Run play_once() to execute expert trajectory
        3. Check if plan_success and check_success() both pass
        4. Return True if solvable, False otherwise (with episode_info for instruction)
        
        Args:
            seed_offset: Additional offset to add to seed
            
        Returns:
            tuple: (success: bool, episode_info: dict or None)
        """
        _enter_robotwin_context()
        
        try:
            # Ensure imports are available
            UnStableError, _ = _ensure_robotwin_imports()
            
            # Close previous environment if initialized
            if self._env_initialized:
                try:
                    self.task_env.close_env(clear_cache=False)
                except:
                    pass
                self._env_initialized = False
            
            # Setup with no rendering for expert check (faster)
            global GLOBAL_EPISODE_NUM
            self.task_config['seed'] = self.seed + GLOBAL_EPISODE_NUM + seed_offset
            self.task_config['now_ep_num'] = GLOBAL_EPISODE_NUM
            original_render_freq = self.task_config.get('render_freq', 0)
            self.task_config['render_freq'] = 0  # Disable rendering for expert check
            self.task_config['eval_mode'] = True
            
            try:
                self.task_env.setup_demo(is_test=True, **self.task_config)
                episode_info = self.task_env.play_once()
                self.task_env.close_env(clear_cache=False)
            except UnStableError as e:
                # Unstable object placement - seed is invalid
                try:
                    self.task_env.close_env(clear_cache=False)
                except:
                    pass
                self.task_config['render_freq'] = original_render_freq
                return False, None
            except Exception as e:
                # Other errors - seed is invalid
                try:
                    self.task_env.close_env(clear_cache=False)
                except:
                    pass
                self.task_config['render_freq'] = original_render_freq
                print(f"Expert check error at seed offset {seed_offset}: {e}")
                return False, None
            
            # Restore render_freq
            self.task_config['render_freq'] = original_render_freq
            
            # Check if expert succeeded
            if self.task_env.plan_success and self.task_env.check_success():
                return True, episode_info
            else:
                return False, None
                
        finally:
            _exit_robotwin_context()
    
    def _generate_instruction(self, episode_info):
        """
        Generate language instruction for the episode using RoboTwin's instruction generator.
        
        Args:
            episode_info: Episode info dict from play_once() containing placeholders
            
        Returns:
            str: Generated instruction text
        """
        if episode_info is None:
            return None
        
        _enter_robotwin_context()
        try:
            # Ensure imports are available
            _, generate_episode_descriptions = _ensure_robotwin_imports()
            
            episode_info_list = [episode_info.get("info", {})]
            results = generate_episode_descriptions(
                self.task_config['task_name'], 
                episode_info_list, 
                max_descriptions=100
            )
            
            if results and len(results) > 0 and self.instruction_type in results[0]:
                instructions = results[0][self.instruction_type]
                if instructions:
                    return np.random.choice(instructions)
            return None
        except Exception as e:
            print(f"Failed to generate instruction: {e}")
            return None
        finally:
            _exit_robotwin_context()
    
    def reset(self):
        """
        Reset environment and return initial observation.
        
        If expert_check is enabled (default), this method will:
        1. Try seeds until finding one where expert trajectory succeeds
        2. Generate language instruction from the successful expert's episode info
        3. Setup the environment with that seed for policy evaluation
        
        Returns:
            MetaObs: Initial observation in ILStudio format
        """
        global GLOBAL_EPISODE_NUM
        
        # Close previous environment if exists
        if self._env_initialized:
            _enter_robotwin_context()
            try:
                self.task_env.close_env()
            except:
                pass
            finally:
                _exit_robotwin_context()
            self._env_initialized = False
        
        max_attempts = 50  # Maximum seeds to try
        seed_offset = 0
        episode_info = None
        
        if self.expert_check:
            # Expert check mode: find a seed where expert succeeds
            while seed_offset < max_attempts:
                success, episode_info = self._run_expert_check(seed_offset)
                if success:
                    # Found a valid seed, now setup for actual evaluation
                    break
                seed_offset += 1
            
            if seed_offset >= max_attempts:
                raise RuntimeError(f"Failed to find valid seed after {max_attempts} attempts with expert_check")
        
        # Initialize environment for the selected seed
        last_error = None
        for attempt in range(10):
            try:
                self._init_env_for_episode(seed_offset)
                break
            except Exception as e:
                last_error = e
                seed_offset += 1  # Try next seed
                if attempt == 9:
                    raise RuntimeError(f"Failed to initialize environment after 10 attempts. Last error: {last_error}")
        
        # Update global episode number now that we have a valid episode
        GLOBAL_EPISODE_NUM += 1 + seed_offset
        
        # Generate and set instruction if episode_info available
        if episode_info is not None:
            instruction = self._generate_instruction(episode_info)
            if instruction is not None:
                _enter_robotwin_context()
                try:
                    self.task_env.set_instruction(instruction=instruction)
                    print(f"[RoboTwin] Instruction: {instruction}")
                finally:
                    _exit_robotwin_context()
        
        # Get initial observation
        obs_dict = self.task_env.get_obs()
        
        # Convert to MetaObs
        meta_obs = self.obs2meta(obs_dict)
        self.prev_obs = meta_obs
        
        # Reset step counter
        self.task_env.take_action_cnt = 0
        
        return meta_obs
    
    def _get_joint_state_from_obs(self, obs_dict):
        """
        Get joint state from observation dict (uses drive_target, same as training data).
        
        This method extracts joint state from obs_dict['joint_action'], which is 
        populated by get_left_arm_jointState() / get_right_arm_jointState() that
        read from drive_target. This ensures consistency with training data.
        
        Args:
            obs_dict: Observation dictionary from get_obs()
            
        Returns:
            np.ndarray: Joint state vector [left_arm, left_gripper, right_arm, right_gripper]
        """
        joint_action = obs_dict.get('joint_action', {})
        
        # Use 'vector' if available (pre-concatenated)
        if 'vector' in joint_action and joint_action['vector'] is not None:
            return np.array(joint_action['vector'], dtype=np.float32)
        
        # Otherwise construct from individual components
        left_arm = joint_action.get('left_arm', [])
        left_gripper = joint_action.get('left_gripper', 0.0)
        right_arm = joint_action.get('right_arm', [])
        right_gripper = joint_action.get('right_gripper', 0.0)
        
        state = np.array(
            list(left_arm) + [left_gripper] + list(right_arm) + [right_gripper],
            dtype=np.float32
        )
        return state
    
    def obs2meta(self, obs_dict):
        """
        Convert RoboTwin observation to MetaObs format.
        
        RoboTwin obs structure:
        {
            'observation': {
                'head_camera': {'rgb': (H, W, 3)},
                'left_camera': {'rgb': (H, W, 3)},
                'right_camera': {'rgb': (H, W, 3)},
            },
            'joint_action': {
                'left_arm': (N,),
                'left_gripper': float,
                'right_arm': (N,),
                'right_gripper': float,
                'vector': (2N+2,)
            },
            'endpose': {
                'left_endpose': (7,) [x,y,z,qx,qy,qz,qw],
                'left_gripper': float,
                'right_endpose': (7,),
                'right_gripper': float
            }
        }
        
        IMPORTANT: For qpos control, we use joint_action from obs_dict which uses
        get_*_arm_jointState() (drive_target). This ensures consistency with training
        data collected by RoboTwin's data collection pipeline.
        
        Returns:
            MetaObs with state and images
        """
        # Extract images
        images = []
        observation = obs_dict.get('observation', {})
        
        for cam_name in self.camera_names:
            # Directly use camera name from config
            # RoboTwin's camera keys: 'head_camera', 'left_camera', 'right_camera'
            if cam_name in observation and 'rgb' in observation[cam_name]:
                img = observation[cam_name]['rgb']  # (H, W, 3) RGB
                
                # Resize to target image_size if needed (for consistency with training)
                target_h, target_w = self.image_size
                if img.shape[0] != target_h or img.shape[1] != target_w:
                    img = cv2.resize(img, (target_w, target_h))
                
                # Convert to (C, H, W)
                img = np.transpose(img, (2, 0, 1))
                images.append(img)
        
        if len(images) > 0:
            image = np.stack(images)  # (N, C, H, W)
        else:
            image = None
        
        # Extract state based on control space
        if self.ctrl_space == 'qpos':
            # Use joint_action from obs_dict for consistency with training data
            # Training data uses get_*_arm_jointState() which reads from drive_target
            state = self._get_joint_state_from_obs(obs_dict)
                
        elif self.ctrl_space == 'ee':
            # Use end-effector poses as state
            endpose = obs_dict.get('endpose', {})
            left_endpose = endpose.get('left_endpose', np.zeros(7))
            left_gripper = np.array([endpose.get('left_gripper', 0.0)])
            right_endpose = endpose.get('right_endpose', np.zeros(7))
            right_gripper = np.array([endpose.get('right_gripper', 0.0)])
            state = np.concatenate([left_endpose, left_gripper, right_endpose, right_gripper]).astype(np.float32)
        else:
            raise ValueError(f"Unsupported ctrl_space: {self.ctrl_space}")
        
        # Also get real joint state for state_joint field (for debugging/compatibility)
        
        return MetaObs(
            state=state,
            image=image,
            raw_lang=getattr(self.task_env, 'instruction', '')
        )
    
    def meta2act(self, maction: MetaAction):
        """
        Convert MetaAction to RoboTwin action format.
        
        RoboTwin expects action as numpy array:
        - For qpos: [left_arm_joints, left_gripper, right_arm_joints, right_gripper]
        - For ee: [left_ee_pose(7), left_gripper, right_ee_pose(7), right_gripper]
        
        Args:
            maction: MetaAction with action array
            
        Returns:
            action: numpy array for RoboTwin
        """
        action = maction['action']
        
        # Ensure action is float32
        action = np.array(action, dtype=np.float32)
        
        return action
    
    def step(self, *args, **kwargs):
        """
        Execute one step in the environment.
        
        Args:
            maction: MetaAction or dict with 'action' key
            
        Returns:
            tuple: (obs, reward, done, info)
        """
        # Extract action
        if isinstance(args[0], MetaAction):
            action = args[0]['action']
        elif isinstance(args[0], dict):
            action = args[0]['action']
        else:
            action = args[0]
        
        # Convert to RoboTwin format
        robotwin_action = self.meta2act(MetaAction(action=action))
        
        # Change to RoboTwin directory for step execution
        _enter_robotwin_context()
        
        try:
            # Execute action
            action_type = self.ctrl_space  # 'qpos' or 'ee'
            self.task_env.take_action(robotwin_action, action_type=action_type)
            
            # Get new observation
            obs_dict = self.task_env.get_obs()
        finally:
            # Restore original directory
            _exit_robotwin_context()
        
        meta_obs = self.obs2meta(obs_dict)
        self.prev_obs = meta_obs
        
        # Check success
        success = self.task_env.eval_success
        
        # In ILStudio, done indicates task success, not episode termination
        # This matches the convention used in other environments (simplerenv, gymnasium_robotics, etc.)
        done = success
        
        # Check termination conditions
        terminated = (self.task_env.take_action_cnt >= self.max_timesteps) or success
        truncated = self.task_env.take_action_cnt >= self.max_timesteps and not success
        
        # Reward (0 or 1 based on success)
        reward = float(success)
        
        # Info
        info = {
            'success': success,
            'terminated': terminated,
            'truncated': truncated,
            'step_count': self.task_env.take_action_cnt
        }
        
        return meta_obs, reward, done, info
    
    def close(self):
        """Close the environment and cleanup."""
        if self._env_initialized:
            _enter_robotwin_context()
            try:
                # clear_cache=True helps release sapien caches; CUDA cleanup is handled below
                self.task_env.close_env(clear_cache=True)
            except:
                pass
            finally:
                _exit_robotwin_context()
        self._env_initialized = False
        _cleanup_torch_cuda(reason="env.close()")
    
    def get_action_dim(self):
        """Return action dimension based on control space."""
        # Action dim depends on the robot configuration
        # For dual-arm ALOHA-Agilex: 7+1 joints per arm = 16 total for qpos
        # For ee control: 7+1 per arm = 16 total
        
        if self.ctrl_space == 'qpos':
            # Joint space: number of joints + gripper per arm
            return self.left_arm_dim + 1 + self.right_arm_dim + 1
        elif self.ctrl_space == 'ee':
            # End-effector: 7-DoF pose + gripper per arm
            return 7 + 1 + 7 + 1  # Always 16 for ee mode
        else:
            raise ValueError(f"Unsupported ctrl_space: {self.ctrl_space}")

