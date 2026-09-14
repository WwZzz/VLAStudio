#!/usr/bin/env python3
"""
RLBench Dataset Generation Script

This script generates HDF5 datasets from RLBench environments by collecting
demonstrations using the built-in get_demos() API.

Usage:
    python benchmark/rlbench/generate_dataset.py \
        --env_config configs/env/rlbench_reach.yaml \
        --output_dir data/rlbench/reach_target \
        --num_demos 50

The generated dataset follows ILStudio's standard HDF5 format and can be
loaded using data_utils/datasets/rlbench_dataset.py
"""

import os
import sys
import argparse
import numpy as np
import h5py
import pickle
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
import importlib


def parse_args():
    parser = argparse.ArgumentParser(description='Generate RLBench dataset')
    parser.add_argument('--env_config', '-e', type=str, required=True,
                        help='Path to environment config YAML file')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Output directory for HDF5 files')
    parser.add_argument('--num_demos', '-n', type=int, default=50,
                        help='Number of successful demonstrations to collect')
    parser.add_argument('--max_attempts', type=int, default=None,
                        help='Maximum attempts per demo (default: num_demos * 2)')
    parser.add_argument('--variation', '-v', type=int, default=0,
                        help='Task variation index')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run in headless mode')
    parser.add_argument('--image_size', type=int, nargs=2, default=None,
                        help='Image size [H, W]. If not specified, uses config or default 128x128')
    return parser.parse_args()


def load_env_config(config_path):
    """Load environment configuration from YAML file."""
    import yaml
    
    # Handle both absolute and relative paths
    if not os.path.isabs(config_path):
        # Try relative to project root
        if not os.path.exists(config_path):
            config_path = os.path.join(project_root, config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_task_class(task_name):
    """Dynamically load task class by name."""
    task_module = importlib.import_module('rlbench.tasks')
    task_class = getattr(task_module, task_name)
    return task_class


# All RLBench camera names (native names used in observation attributes)
RLBENCH_ALL_CAMERAS = ['front', 'wrist', 'left_shoulder', 'right_shoulder', 'overhead']


def create_rlbench_env(config, headless=True, image_size=None):
    """Create RLBench environment from config."""
    args = config.get('args', config)
    
    # Get image size
    if image_size is None:
        image_size = args.get('image_size', [128, 128])
    
    # Configure observation with proper image sizes
    obs_config = ObservationConfig()
    
    # Configure cameras with specified image size
    camera_config = CameraConfig(
        rgb=True,
        depth=False,
        point_cloud=False,
        mask=False,
        image_size=tuple(image_size)
    )
    
    obs_config.front_camera = camera_config
    obs_config.wrist_camera = camera_config
    obs_config.left_shoulder_camera = camera_config
    obs_config.right_shoulder_camera = camera_config
    obs_config.overhead_camera = camera_config
    
    # Enable low-dim observations
    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = True
    obs_config.gripper_joint_positions = True
    obs_config.task_low_dim_state = True
    
    # Configure action mode based on ctrl_space
    ctrl_space = args.get('ctrl_space', 'joint')
    if ctrl_space == 'joint' or ctrl_space == 'joint_vel':
        arm_action_mode = JointVelocity()
    elif ctrl_space == 'ee':
        arm_action_mode = EndEffectorPoseViaPlanning()
    else:
        arm_action_mode = JointVelocity()
    
    gripper_action_mode = Discrete()
    action_mode = MoveArmThenGripper(
        arm_action_mode=arm_action_mode,
        gripper_action_mode=gripper_action_mode
    )
    
    # Create environment
    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=headless,
        shaped_rewards=False
    )
    env.launch()
    
    return env, args


def save_demo_to_hdf5(demo, descriptions, output_path, camera_names, config_args):
    """
    Save a single demo to HDF5 file in ILStudio standard format.
    
    Args:
        demo: RLBench Demo object
        descriptions: List of task descriptions (language instructions)
        output_path: Path to save HDF5 file
        camera_names: List of camera names to save
        config_args: Configuration arguments
    """
    num_frames = len(demo)
    
    # Pre-allocate arrays
    joint_positions_list = []
    joint_velocities_list = []
    gripper_pose_list = []
    gripper_open_list = []
    
    # Camera name mapping: config name -> observation attribute
    camera_mapping = {
        'front': 'front_rgb',
        'wrist': 'wrist_rgb',
        'left_shoulder': 'left_shoulder_rgb',
        'right_shoulder': 'right_shoulder_rgb',
        'overhead': 'overhead_rgb',
    }
    
    image_data = {cam: [] for cam in camera_names}
    
    # Extract data from each frame
    for i in range(num_frames):
        obs = demo[i]
        
        # Low-dim state
        joint_positions_list.append(obs.joint_positions)
        joint_velocities_list.append(obs.joint_velocities)
        gripper_pose_list.append(obs.gripper_pose)
        gripper_open_list.append(obs.gripper_open)
        
        # Images
        for cam_name in camera_names:
            attr_name = camera_mapping.get(cam_name, f'{cam_name}_rgb')
            img = getattr(obs, attr_name, None)
            if img is not None:
                image_data[cam_name].append(img)
    
    # Convert to numpy arrays
    joint_positions = np.array(joint_positions_list, dtype=np.float32)  # (T, 7)
    joint_velocities = np.array(joint_velocities_list, dtype=np.float32)  # (T, 7)
    gripper_pose = np.array(gripper_pose_list, dtype=np.float32)  # (T, 7)
    gripper_open = np.array(gripper_open_list, dtype=np.float32).reshape(-1, 1)  # (T, 1)
    
    # Create actions based on ctrl_space
    ctrl_space = config_args.get('ctrl_space', 'joint')
    if ctrl_space == 'ee':
        # End-effector pose + gripper
        action = np.concatenate([gripper_pose, gripper_open], axis=-1)  # (T, 8)
        state = action.copy()
    elif ctrl_space == 'joint_vel':
        # Joint velocities + gripper
        action = np.concatenate([joint_velocities, gripper_open], axis=-1)  # (T, 8)
        state = np.concatenate([joint_positions, gripper_open], axis=-1)  # (T, 8)
    else:  # 'joint'
        # Joint positions + gripper
        action = np.concatenate([joint_positions, gripper_open], axis=-1)  # (T, 8)
        state = action.copy()
    
    # Save to HDF5
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Episode metadata
        f.create_dataset('episode_len', data=np.array([num_frames], dtype=np.int32))
        
        # Language instructions - save all descriptions
        # Create a variable-length string dtype
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('language_instruction', data=descriptions, dtype=dt)
        
        # Primary language (first description)
        f.attrs['language'] = descriptions[0] if descriptions else ''
        
        # Random seed (for reproducibility)
        if hasattr(demo, 'random_seed') and demo.random_seed is not None:
            # random_seed is a tuple from np.random.get_state()
            # Save as pickled bytes
            f.create_dataset('random_seed', data=np.void(pickle.dumps(demo.random_seed)))
        
        # Actions and states (ctrl_space dependent)
        f.create_dataset('action', data=action, dtype=np.float32)
        f.create_dataset('state', data=state, dtype=np.float32)
        
        # Raw observations (always saved regardless of ctrl_space)
        obs_group = f.create_group('observations')
        obs_group.create_dataset('joint_positions', data=joint_positions, dtype=np.float32)
        obs_group.create_dataset('joint_velocities', data=joint_velocities, dtype=np.float32)
        obs_group.create_dataset('gripper_pose', data=gripper_pose, dtype=np.float32)
        obs_group.create_dataset('gripper_open', data=gripper_open, dtype=np.float32)
        
        # Images
        img_group = obs_group.create_group('images')
        for cam_name in camera_names:
            if len(image_data[cam_name]) > 0:
                images = np.stack(image_data[cam_name], axis=0)  # (T, H, W, 3)
                img_group.create_dataset(cam_name, data=images, dtype=np.uint8,
                                        compression='gzip', compression_opts=4)
        
        # Config info
        f.attrs['ctrl_space'] = config_args.get('ctrl_space', 'joint')
        f.attrs['ctrl_type'] = config_args.get('ctrl_type', 'abs')
        f.attrs['task'] = config_args.get('task', 'unknown')


def main():
    args = parse_args()
    
    # Load environment config
    print(f"Loading config from: {args.env_config}")
    config = load_env_config(args.env_config)
    config_args = config.get('args', config)
    
    # Get task info
    task_name = config_args.get('task', 'ReachTarget')
    # Always save all RLBench cameras (ignore config camera_names)
    camera_names = RLBENCH_ALL_CAMERAS
    
    print(f"Task: {task_name}")
    print(f"Camera names: {camera_names}")
    print(f"Number of demos: {args.num_demos}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environment
    print("Creating RLBench environment...")
    env, env_args = create_rlbench_env(config, headless=args.headless, image_size=args.image_size)
    
    # Get task
    task_class = get_task_class(task_name)
    task = env.get_task(task_class)
    
    # Set variation
    task.set_variation(args.variation)
    
    # Get language descriptions from task reset (with retry logic)
    max_reset_attempts = 10
    descriptions = None
    for reset_attempt in range(max_reset_attempts):
        try:
            descriptions, _ = task.reset()
            print(f"Task descriptions: {descriptions}")
            break
        except Exception as e:
            print(f"Warning: task.reset() failed (attempt {reset_attempt + 1}/{max_reset_attempts}): {e}")
            if reset_attempt == max_reset_attempts - 1:
                print(f"Error: Failed to reset task after {max_reset_attempts} attempts. Exiting.")
                env.shutdown()
                raise RuntimeError(f"Failed to initialize task {task_name}") from e
            import time
            time.sleep(1)  # Brief pause before retry
    
    # Collect demos - keep trying until we get the required number of successful demos
    print(f"Collecting {args.num_demos} successful demonstrations...")
    
    # Set max attempts (default: 2x num_demos to allow for some failures)
    max_attempts = args.max_attempts if args.max_attempts is not None else args.num_demos * 2
    
    success_count = 0
    attempt_count = 0
    
    with tqdm(total=args.num_demos, desc="Collecting demos") as pbar:
        while success_count < args.num_demos:
            attempt_count += 1
            
            # Check if we've exceeded max attempts
            if attempt_count > max_attempts:
                print(f"\nWarning: Reached maximum attempts ({max_attempts}). Collected {success_count}/{args.num_demos} demos.")
                break
            
            try:
                # Get a single demo
                demos = task.get_demos(1, live_demos=True)
                demo = demos[0]
                
                # Save to HDF5
                output_path = os.path.join(args.output_dir, f'episode_{success_count:04d}.hdf5')
                save_demo_to_hdf5(demo, descriptions, output_path, camera_names, config_args)
                success_count += 1
                pbar.update(1)
                
            except Exception as e:
                # Continue trying on failure
                if attempt_count % 10 == 0:  # Print warning every 10 failures
                    print(f"\nWarning: Failed to collect demo (attempt {attempt_count}): {e}")
                continue
    
    print(f"\nSuccessfully collected {success_count}/{args.num_demos} demonstrations (after {attempt_count} attempts)")
    
    # Save dataset metadata
    metadata_path = os.path.join(args.output_dir, 'metadata.yaml')
    import yaml
    metadata = {
        'task': task_name,
        'variation': args.variation,
        'num_episodes': success_count,
        'camera_names': camera_names,
        'ctrl_space': config_args.get('ctrl_space', 'joint'),
        'ctrl_type': config_args.get('ctrl_type', 'abs'),
        'image_size': config_args.get('image_size', [128, 128]),
        'language_instructions': descriptions,
    }
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"Metadata saved to: {metadata_path}")
    
    # Shutdown environment
    env.shutdown()
    print("Done!")


if __name__ == '__main__':
    main()
