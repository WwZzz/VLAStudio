#!/usr/bin/env python3
"""
RLBench Dataset Generation Script for LeRobot Format

This script generates LeRobot format datasets from RLBench environments by collecting
demonstrations using the built-in get_demos() API.

Usage:
    python benchmark/rlbench/generate_dataset_lerobot.py \
        --env_config configs/env/rlbench_reach.yaml \
        --output_dir data/rlbench/reach_target \
        --num_demos 50

The generated dataset follows LeRobot's standard format and can be
loaded using lerobot.datasets.lerobot_dataset.LeRobotDataset
"""

import os
import sys
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# Disable datasets progress bars (e.g., Map progress) to keep only our collection progress bar
import datasets
datasets.disable_progress_bars()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
import importlib

# Import LeRobot dataset classes
sys.path.insert(0, str(project_root / "third_party" / "lerobot" / "src"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Generate RLBench dataset in LeRobot format')
    parser.add_argument('--env_config', '-e', type=str, required=True,
                        help='Path to environment config YAML file')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Output directory for LeRobot dataset')
    parser.add_argument('--num_demos', '-n', type=int, default=50,
                        help='Number of successful demonstrations to collect')
    parser.add_argument('--max_attempts', type=int, default=None,
                        help='Maximum attempts per demo (default: num_demos * 2)')
    parser.add_argument('--variation', '-v', type=int, default=0,
                        help='Task variation index')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run in headless mode')
    parser.add_argument('--image_size', type=int, nargs=2, default=None,
                        help='Image size [H, W]. If not specified, uses config or default 256x256')
    parser.add_argument('--robot_type', type=str, default='panda',
                        help='Robot type for dataset metadata')
    parser.add_argument('--video_file_size_mb', type=int, default=500,
                        help='Max video file size in MB. Larger = more episodes per file (default: 500MB)')
    parser.add_argument('--batch_encoding_size', type=int, default=1,
                        help='Number of episodes to batch before encoding video (default: 1). '
                             'WARNING: values >1 will accumulate per-frame PNGs on disk until encoding triggers.')
    parser.add_argument('--image_writer_threads', type=int, default=8,
                        help='Number of threads for async image writing (default: 8). Higher is faster but uses more CPU.')
    parser.add_argument('--image_writer_processes', type=int, default=0,
                        help='Number of processes for async image writing (default: 0). Use threads unless you know you need processes.')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force overwrite if output directory already exists')
    return parser.parse_args()


# RLBench/CoppeliaSim default simulation timestep is 0.05s = 20Hz
RLBENCH_DEFAULT_FPS = 20


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
        image_size = args.get('image_size', [256, 256])
    
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


def create_lerobot_features(config_args, camera_names, image_size):
    """Create LeRobot features dictionary from config.
    
    Always saves:
    - action: joint positions (7) + gripper (1) = 8
    - action.ee: end-effector pose (7) + gripper (1) = 8
    - observation.state: joint positions (7) + gripper (1) = 8
    - observation.state.ee: end-effector pose (7) + gripper (1) = 8
    - observation.state.joint_velocities: joint velocities (7)
    - observation.images.*: camera images
    """
    features = {}
    
    # Joint action: joint positions (7) + gripper (1) = 8
    features['action'] = {
        'dtype': 'float32',
        'shape': (8,),
        'names': ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
    }
    
    # EE action: end-effector pose (7) + gripper (1) = 8
    features['action.ee'] = {
        'dtype': 'float32',
        'shape': (8,),
        'names': ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
    }
    
    # Joint state: joint positions (7) + gripper (1) = 8
    features['observation.state'] = {
        'dtype': 'float32',
        'shape': (8,),
        'names': ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
    }
    
    # EE state: end-effector pose (7) + gripper (1) = 8
    features['observation.state.ee'] = {
        'dtype': 'float32',
        'shape': (8,),
        'names': ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
    }
    
    # Joint velocities: (7)
    features['observation.state.joint_velocities'] = {
        'dtype': 'float32',
        'shape': (7,),
        'names': ['vel_0', 'vel_1', 'vel_2', 'vel_3', 'vel_4', 'vel_5', 'vel_6']
    }
    
    # Observation image features (one per camera)
    #
    # IMPORTANT:
    # We store images directly in parquet (HF `datasets.Image`) to avoid training-time video decoding bottlenecks.
    # Therefore we use dtype="image" (NOT "video") and create the dataset with use_videos=False.
    for cam_name in camera_names:
        features[f'observation.images.{cam_name}'] = {
            'dtype': 'image',  # Store image bytes in parquet (fast random access during training)
            'shape': tuple(image_size) + (3,),  # (H, W, C)
            'names': ['height', 'width', 'channels']
        }
    
    return features


def convert_demo_to_lerobot(demo, descriptions, dataset, camera_names, fps):
    """
    Convert a single RLBench demo to LeRobot format and save it.
    
    Always saves:
    - action: joint positions (7) + gripper (1)
    - action.ee: end-effector pose (7) + gripper (1)
    - observation.state: joint positions (7) + gripper (1)
    - observation.state.ee: end-effector pose (7) + gripper (1)
    - observation.state.joint_velocities: joint velocities (7)
    - observation.images.*: camera images
    
    Args:
        demo: RLBench Demo object
        descriptions: List of task descriptions (language instructions)
        dataset: LeRobotDataset instance
        camera_names: List of camera names to save
        fps: Frames per second
    """
    if demo is None:
        raise ValueError("Demo is None")
    
    num_frames = len(demo)
    if num_frames == 0:
        raise ValueError("Demo has no frames")
    
    # Build episode data fully in memory first, then write once at the end of the episode.
    # This avoids per-step disk I/O while the RLBench simulator is running.
    episode_buffer = dataset.create_episode_buffer()
    
    # Camera name mapping: config name -> observation attribute
    camera_mapping = {
        'front': 'front_rgb',
        'wrist': 'wrist_rgb',
        'left_shoulder': 'left_shoulder_rgb',
        'right_shoulder': 'right_shoulder_rgb',
        'overhead': 'overhead_rgb',
    }
    
    # Randomly select one task description from the available descriptions for this episode
    task_description = random.choice(descriptions) if descriptions else ""
    
    # Process each frame (in memory)
    for i in range(num_frames):
        obs = demo[i]
        if obs is None:
            raise ValueError(f"Observation at frame {i} is None")

        # Validate required attributes exist
        if obs.joint_positions is None:
            raise ValueError(f"joint_positions is None at frame {i}")
        if obs.gripper_pose is None:
            raise ValueError(f"gripper_pose is None at frame {i}")
        if obs.joint_velocities is None:
            raise ValueError(f"joint_velocities is None at frame {i}")

        # Add task/timing info (task is outside `features`, but required in episode_buffer)
        episode_buffer["task"].append(task_description)
        episode_buffer["frame_index"].append(i)
        episode_buffer["timestamp"].append(i / fps)

        # Joint action: joint positions (7) + gripper (1)
        episode_buffer["action"].append(
            np.concatenate([obs.joint_positions, np.array([obs.gripper_open], dtype=np.float32)], axis=0).astype(
                np.float32
            )
        )

        # EE action: end-effector pose (7) + gripper (1)
        episode_buffer["action.ee"].append(
            np.concatenate([obs.gripper_pose, np.array([obs.gripper_open], dtype=np.float32)], axis=0).astype(
                np.float32
            )
        )

        # Joint state: joint positions (7) + gripper (1)
        episode_buffer["observation.state"].append(
            np.concatenate([obs.joint_positions, np.array([obs.gripper_open], dtype=np.float32)], axis=0).astype(
                np.float32
            )
        )

        # EE state: end-effector pose (7) + gripper (1)
        episode_buffer["observation.state.ee"].append(
            np.concatenate([obs.gripper_pose, np.array([obs.gripper_open], dtype=np.float32)], axis=0).astype(
                np.float32
            )
        )

        # Joint velocities: (7)
        episode_buffer["observation.state.joint_velocities"].append(obs.joint_velocities.astype(np.float32))

        # Store images directly in the episode buffer (will be embedded into parquet by LeRobot)
        for cam_name in camera_names:
            video_key = f"observation.images.{cam_name}"
            attr_name = camera_mapping.get(cam_name, f"{cam_name}_rgb")
            img = getattr(obs, attr_name, None)
            if img is None:
                continue
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            episode_buffer[video_key].append(img)

    # Mark episode length
    episode_buffer["size"] = num_frames

    # Save episode in one shot
    dataset.save_episode(episode_data=episode_buffer)


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
    
    # Get image size
    if args.image_size is None:
        image_size = config_args.get('image_size', [256, 256])
    else:
        image_size = args.image_size
    
    # Use RLBench's default FPS (20Hz from CoppeliaSim timestep of 0.05s)
    fps = RLBENCH_DEFAULT_FPS
    
    print(f"Task: {task_name}")
    print(f"Camera names: {camera_names}")
    print(f"Image size: {image_size}")
    print(f"Number of demos: {args.num_demos}")
    print(f"Output directory: {args.output_dir}")
    print(f"FPS: {fps} (RLBench default)")
    print(f"Video file size limit: {args.video_file_size_mb}MB")
    print(f"Batch encoding size: {args.batch_encoding_size} episodes")
    print(f"Image writer: {args.image_writer_processes} processes, {args.image_writer_threads} threads")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate repo_id from output path
    # For output_dir like "data/rlbench/reach_target", we want:
    #   root = full path "data/rlbench/reach_target" (LeRobot expects the complete dataset path)
    #   repo_id = "rlbench/reach_target" (identifier, excluding the first "data" component)
    # This makes the dataset identifiable by its relative path under the data root
    path_parts = output_path.parts
    if len(path_parts) >= 2:
        repo_id = str(Path(*path_parts[1:]))  # e.g., "rlbench/reach_target"
    else:
        repo_id = output_path.name
    
    # Check if dataset directory already exists
    if output_path.exists():
        if args.force:
            import shutil
            print(f"Warning: Output directory {output_path} already exists. Removing it (--force specified)...")
            shutil.rmtree(output_path)
        else:
            print(f"Error: Output directory {output_path} already exists.")
            print(f"Use --force or -f to overwrite, or specify a different output directory.")
            sys.exit(1)
    
    # Create RLBench environment FIRST (before LeRobot dataset)
    # This ensures the simulator is ready before we start recording
    print("Creating RLBench environment...")
    env, env_args = create_rlbench_env(config, headless=args.headless, image_size=image_size)
    
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
            time.sleep(1)  # Brief pause before retry
    
    # Now create LeRobot dataset (after RLBench environment is ready)
    print("Creating LeRobot dataset...")
    features = create_lerobot_features(config_args, camera_names, image_size)
    
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=args.robot_type,
            features=features,
            root=str(output_path),  # LeRobot expects the full dataset path as root
            use_videos=False,  # Store images directly in parquet (no mp4 videos)
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
            batch_encoding_size=args.batch_encoding_size,  # Batch encode for efficiency
        )
    except FileExistsError:
        print(f"Error: Output directory {output_path} already exists (possibly created by another process).")
        print(f"Use --force or -f to overwrite, or specify a different output directory.")
        env.shutdown()
        sys.exit(1)
    
    # Update video file size limit to allow merging many episodes into one video
    # Default is 500MB, we increase it to allow more episodes per video file
    dataset.meta.update_chunk_settings(video_files_size_in_mb=args.video_file_size_mb)
    
    # Collect demos - keep trying until we get the required number of successful demos
    print(f"Collecting {args.num_demos} successful demonstrations...")
    
    # Set max attempts (default: 2x num_demos to allow for some failures)
    max_attempts = args.max_attempts if args.max_attempts is not None else args.num_demos * 2
    
    success_count = 0
    attempt_count = 0
    consecutive_failures = 0
    
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
                
                # Convert and save to LeRobot format
                convert_demo_to_lerobot(demo, descriptions, dataset, camera_names, fps)
                success_count += 1
                consecutive_failures = 0  # Reset on success
                pbar.update(1)
                
            except Exception as e:
                consecutive_failures += 1
                
                # Always print failure reason for debugging
                print(f"\nWarning: Failed to collect demo (attempt {attempt_count}): {e}")
                
                # If too many consecutive failures, try to reset the task
                if consecutive_failures >= 5:
                    print(f"\nToo many consecutive failures ({consecutive_failures}). Attempting to reset task...")
                    try:
                        task.reset()
                        consecutive_failures = 0
                        print("Task reset successful.")
                    except Exception as reset_e:
                        print(f"Task reset failed: {reset_e}")
                
                continue
    
    print(f"\nSuccessfully collected {success_count}/{args.num_demos} demonstrations (after {attempt_count} attempts)")
    
    # Finalize dataset (encode any remaining videos if using batch encoding)
    if dataset.batch_encoding_size > 1 and dataset.episodes_since_last_encoding > 0:
        start_ep = dataset.num_episodes - dataset.episodes_since_last_encoding
        end_ep = dataset.num_episodes
        print(f"\nEncoding remaining {dataset.episodes_since_last_encoding} episodes...")
        dataset._batch_save_episode_video(start_ep, end_ep)

    # Ensure parquet writers are properly closed
    dataset.finalize()

    # LeRobot stores temporary per-frame PNGs under `images/` before encoding videos.
    # After successful collection, we prefer to keep only the encoded mp4s in `videos/`.
    images_dir = output_path / "images"
    if images_dir.exists() and images_dir.is_dir():
        import shutil
        try:
            shutil.rmtree(images_dir)
            print(f"Removed temporary images directory: {images_dir}")
        except Exception as e:
            print(f"Warning: failed to remove temporary images directory {images_dir}: {e}")
    
    # Shutdown environment
    env.shutdown()
    print(f"Dataset saved to: {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()

