"""
CALVIN Environment Integration for ILStudio

This module integrates the CALVIN benchmark environment into ILStudio.
CALVIN is unique in that it evaluates long-horizon task completion over sequences of subtasks.
"""

import sys
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import os
import hydra
import multiprocessing as mp
import contextlib


@contextlib.contextmanager
def _suppress_pybullet_output():
    """
    Suppress all output from pybullet and OpenGL/EGL.
    This is CALVIN-specific as it uses pybullet which produces verbose output.
    """
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    
    # Save copies of original stdout/stderr
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    
    # Open devnull
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    
    try:
        # Redirect stdout and stderr to devnull at the file descriptor level
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        
        yield
        
    finally:
        # Restore original stdout/stderr
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        
        # Close temporary file descriptors
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)

# Add calvin_env_10 to the path
CALVIN_ROOT = Path(__file__).parent / "calvin_env_10" / "src"
if str(CALVIN_ROOT) not in sys.path:
    sys.path.insert(0, str(CALVIN_ROOT))

from calvin_env_10.envs.play_table_env import get_env
from calvin_env_10.evaluation.sequences import get_sequences
from calvin_env_10.evaluation.utils import get_env_state_for_initial_condition

from vlastudio.benchmark.base import MetaEnv, MetaObs, MetaAction

# Export evaluation functions
# The 'evaluate' function will be used by eval_sim.py if env_module has it
from .evaluate import evaluate, count_success

# Global counter for sequence allocation (shared across processes via Manager)
_sequence_counter = None
_sequence_lock = None

def _init_sequence_counter():
    """Initialize shared sequence counter for parallel environments."""
    global _sequence_counter, _sequence_lock
    if _sequence_counter is None:
        manager = mp.Manager()
        _sequence_counter = manager.Value('i', 0)
        _sequence_lock = manager.Lock()


def create_env(config):
    """
    Create a CALVIN environment instance.
    
    Args:
        config: Configuration object containing:
            - task: Task name (e.g., 'task_D', 'task_A', 'task_B', 'task_C')
            - show_gui: Whether to show GUI (default: False)
            - num_sequences: Number of evaluation sequences (default: 1000)
            - sequence_idx: Which sequence to use (-1 for next available, default: -1)
    """
    return CalvinEnv(config)


class CalvinEnv(MetaEnv):
    """
    CALVIN Environment wrapper that adapts calvin_env_10 to ILStudio's MetaEnv interface.
    
    Key differences from other environments:
    1. Each env instance represents ONE sequence of 5 subtasks
    2. The language instruction changes as subtasks progress
    3. Evaluation focuses on the number of successfully completed subtasks in sequence
    4. Different task suites (task_A, task_B, task_C, task_D) provide different challenges
    """
    
    def __init__(self, config):
        self.config = config
        self.task_name = getattr(config, 'task', 'task_D')
        self.show_gui = getattr(config, 'show_gui', False)
        self.num_sequences = getattr(config, 'num_sequences', 1)
        sequence_idx_config = getattr(config, 'sequence_idx', -1)
        self.use_wrist = getattr(config, 'use_wrist', False)
        self._closed = False  # Track whether environment has been closed
        
        # Action and observation space configuration
        self.ctrl_space = 'ee'  # CALVIN uses end-effector control
        self.ctrl_type = 'delta'  # CALVIN uses delta/relative actions
        self.action_dim = 7  # [dx, dy, dz, droll, dpitch, dyaw, gripper]
        
        # Camera configuration
        self.camera_names = ['static', 'gripper']  # CALVIN has two cameras
        
        # Load evaluation sequences
        self._load_sequences()
        
        # Allocate sequence index
        # If sequence_idx == -1, auto-allocate from global counter
        # This ensures each parallel environment gets a different sequence
        if sequence_idx_config == -1:
            # Auto-allocate using shared counter
            global _sequence_counter, _sequence_lock
            if _sequence_counter is None:
                _init_sequence_counter()
            
            with _sequence_lock:
                self.sequence_idx = _sequence_counter.value % len(self.sequences)
                _sequence_counter.value += 1
        else:
            self.sequence_idx = sequence_idx_config % len(self.sequences)
        
        # Create the underlying CALVIN environment
        # Suppress OpenGL/EGL/pybullet output messages
        with _suppress_pybullet_output():
            env = get_env(self.task_name, show_gui=self.show_gui)
        
        # CALVIN uses delta control with fixed bounds
        # Action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        # gripper: 1 = open, -1 = close
        self.min_action = np.array([-0.02, -0.02, -0.02, -0.05, -0.05, -0.05, -1.0], dtype=np.float32)
        self.max_action = np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 1.0], dtype=np.float32)
        
        super().__init__(env)
        
        # Initialize sequence state
        self.current_subtask_idx = 0
        self.subtasks_completed = 0
        self.max_subtasks = 5
        
        # Get current sequence info
        self.initial_state, self.eval_sequence = self.sequences[self.sequence_idx]
        
        # Load task checker for success detection
        from calvin_env_10.evaluation.multi_step_evaluation import BASE_DIR
        task_cfg = OmegaConf.load(os.path.join(BASE_DIR, "conf/tasks/new_playtable_tasks.yaml"))
        self.task_oracle = hydra.utils.instantiate(task_cfg)
        
        # Load language annotations
        self.val_annotations = OmegaConf.load(
            os.path.join(BASE_DIR, "conf/annotations/new_playtable_validation.yaml")
        )
        
        # Episode length per subtask (same as original CALVIN)
        self.ep_len = 360
        self.current_step = 0
        self.start_info = None
        
    def _load_sequences(self):
        """Load evaluation sequences for the task."""
        # Sequences are cached by get_sequences
        # Use num_workers=1 to avoid multiprocessing issues in SubprocVectorEnv
        # (daemon processes cannot spawn child processes)
        self.sequences = get_sequences(self.num_sequences, num_workers=1)
        
    def get_current_language(self):
        """Get the language instruction for the current subtask."""
        if self.current_subtask_idx >= len(self.eval_sequence):
            return ""
        subtask = self.eval_sequence[self.current_subtask_idx]
        # Get the first annotation for this subtask
        return self.val_annotations[subtask][0]
    
    def meta2act(self, maction: MetaAction):
        """
        Convert MetaAction to CALVIN action format.
        
        CALVIN expects: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        where gripper: 1 = open, -1 = close
        """
        action = maction['action']  # (7,)
        
        # Convert continuous gripper value to discrete CALVIN format
        # CALVIN uses: 1=open, -1=close
        calvin_action = action.copy()
        
        # Threshold: > 0 means open (1), <= 0 means close (-1)
        # This handles both [0,1] normalized values and arbitrary continuous values
        if action[-1] > 0:
            calvin_action[-1] = 1.0
        else:
            calvin_action[-1] = -1.0
        
        return calvin_action.astype(np.float32)
        
    def obs2meta(self, obs):
        """
        Convert CALVIN observation to MetaObs format.
        
        CALVIN obs structure:
        {
            'rgb_obs': {
                'rgb_static': (H, W, 3),
                'rgb_gripper': (H, W, 3)
            },
            'depth_obs': {...},
            'robot_obs': {
                'robot_state_full': [tcp_pos, tcp_orn, gripper_width],
                ...
            },
            'scene_obs': {...}
        }
        """
        # Extract robot state
        # In CALVIN, robot_obs is directly a numpy array, not a dict
        robot_state = obs['robot_obs']  # (15,)
        state = robot_state.astype(np.float32)
        
        # Extract images
        # CALVIN has: static (200x200) and gripper (84x84)
        # Primary view: static camera
        img_primary = obs['rgb_obs']['rgb_static']  # (200, 200, 3)
        all_imgs = [img_primary]
        
        # Add wrist/gripper camera if requested
        if self.use_wrist:
            import cv2
            img_wrist = obs['rgb_obs']['rgb_gripper']  # (84, 84, 3)
            # Resize wrist camera to match primary camera size
            img_wrist = cv2.resize(img_wrist, (200, 200), interpolation=cv2.INTER_LINEAR)
            all_imgs.append(img_wrist)
        
        # Stack and transpose to (N, C, H, W) format
        image = np.stack(all_imgs)  # (N, H, W, C)
        image = image.transpose(0, 3, 1, 2)  # (N, C, H, W)
        
        # Get current language instruction
        raw_lang = self.get_current_language()
        
        return MetaObs(state=state, image=image, raw_lang=raw_lang)
    
    def reset(self):
        """
        Reset the environment for the current sequence.
        Initializes to the first subtask of the sequence.
        """
        # Reset subtask progress
        self.current_subtask_idx = 0
        self.subtasks_completed = 0
        self.current_step = 0
        
        # Get initial state for this sequence
        robot_obs, scene_obs = get_env_state_for_initial_condition(self.initial_state)
        
        # Reset the environment with the initial state
        raw_obs = self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        
        # Store start info for task checking
        self.start_info = self.env.get_info()
        
        # Convert to MetaObs
        meta_obs = self.obs2meta(raw_obs)
        self.prev_obs = meta_obs
        
        return self.prev_obs
    
    def step(self, *args, **kwargs):
        """
        Execute one step in the environment.
        
        This method:
        1. Converts MetaAction to CALVIN action
        2. Steps the environment
        3. Checks if current subtask is completed
        4. If completed, advances to next subtask or marks sequence as done
        5. Returns (obs, reward, done, info)
        
        For eval_sim.py compatibility:
        - done=True when sequence ends (all subtasks done OR failed)
        - info['success'] = True if at least 1 subtask completed (for standard eval)
        - info['subtasks_completed'] = actual number (for CALVIN metrics)
        
        Returns:
            obs: MetaObs dict
            reward: Always 0 (CALVIN doesn't use rewards)
            done: True if sequence is complete
            info: Dict with 'success', 'subtasks_completed', 'sequence_done', etc.
        """
        # Convert action
        action_dict = args[0] if args else kwargs
        calvin_action = self.meta2act(action_dict)
        
        # Step the environment
        raw_obs, _, _, env_info = self.env.step(calvin_action)
        self.current_step += 1
        
        # Convert observation
        obs = self.obs2meta(raw_obs)
        self.prev_obs = obs
        
        # Check if current subtask is completed
        current_info = self.env.get_info()
        subtask = self.eval_sequence[self.current_subtask_idx]
        current_task_info = self.task_oracle.get_task_info_for_set(
            self.start_info, current_info, {subtask}
        )
        
        subtask_success = len(current_task_info) > 0
        sequence_done = False
        
        if subtask_success:
            # Subtask completed successfully
            self.subtasks_completed += 1
            self.current_subtask_idx += 1
            self.current_step = 0  # Reset step counter for next subtask
            
            # Update start_info for next subtask
            self.start_info = current_info
            
            # Check if sequence is complete
            if self.current_subtask_idx >= self.max_subtasks:
                sequence_done = True
        elif self.current_step >= self.ep_len:
            # Subtask failed (timeout)
            sequence_done = True
        
        # Prepare info dict
        # For eval_sim.py: 'success' means at least 1 subtask completed
        info = {
            'success': self.subtasks_completed > 0,  # True if any subtask completed
            'subtasks_completed': self.subtasks_completed,  # CALVIN metric: 0-5
            'current_subtask_idx': self.current_subtask_idx,
            'sequence_done': sequence_done,
            'current_step': self.current_step,
            'current_subtask': subtask if self.current_subtask_idx < len(self.eval_sequence) else None,
            'calvin_success_rate_1': 1.0 if self.subtasks_completed >= 1 else 0.0,
            'calvin_success_rate_2': 1.0 if self.subtasks_completed >= 2 else 0.0,
            'calvin_success_rate_3': 1.0 if self.subtasks_completed >= 3 else 0.0,
            'calvin_success_rate_4': 1.0 if self.subtasks_completed >= 4 else 0.0,
            'calvin_success_rate_5': 1.0 if self.subtasks_completed >= 5 else 0.0,
        }
        info.update(env_info)
        
        # Return as dict for compatibility
        if isinstance(obs, MetaObs):
            from dataclasses import asdict
            obs = asdict(obs)
        
        # done = True means the sequence evaluation is complete
        done = sequence_done
        
        return obs, 0, done, info
    
    def get_sequence_info(self):
        """Get information about the current sequence."""
        return {
            'sequence_idx': self.sequence_idx,
            'initial_state': self.initial_state,
            'eval_sequence': self.eval_sequence,
            'subtasks_completed': self.subtasks_completed,
            'current_subtask_idx': self.current_subtask_idx,
        }
    
    def load_next_sequence(self):
        """
        Load the next evaluation sequence.
        Call this after completing one sequence to move to the next.
        """
        self.sequence_idx = (self.sequence_idx + 1) % len(self.sequences)
        self.initial_state, self.eval_sequence = self.sequences[self.sequence_idx]
        self.current_subtask_idx = 0
        self.subtasks_completed = 0
        self.current_step = 0
        return self.reset()
    
    def close(self):
        """
        Close the environment properly.
        Prevents double-closing which causes pybullet connection issues.
        """
        if hasattr(self, '_closed'):
            return  # Already closed
        
        self._closed = True
        
        if hasattr(self, 'env') and self.env is not None:
            # Suppress OpenGL/EGL/pybullet disconnect messages
            try:
                with _suppress_pybullet_output():
                    self.env.close()
            except Exception as e:
                # Ignore errors during close (e.g., "Not connected to physics server")
                pass
    
    def __del__(self):
        """
        Ensure clean shutdown when object is garbage collected.
        Suppress all pybullet/OpenGL messages during cleanup.
        """
        try:
            with _suppress_pybullet_output():
                self.close()
        except:
            pass  # Ignore all errors during garbage collection

