#!/usr/bin/env python3
"""
BiGym Dataset Generator - One-Click Dataset Generation

This script downloads BiGym demonstrations and converts them to LeRobot format.
Simply run this script to generate datasets for all or selected tasks.

Usage:
    # Generate all available tasks (default)
    python generate_dataset.py
    
    # Generate specific tasks
    python generate_dataset.py --tasks ReachTarget ReachTargetDual
    
    # Custom output directory
    python generate_dataset.py --output_root /path/to/datasets
    
    # Generate with specific settings
    python generate_dataset.py --tasks ReachTarget --num_demos 200 --resolution 256 256
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from io import BytesIO
from typing import List, Dict, Any, Optional

# Add bigym to path
BIGYM_PATH = Path(__file__).parent / "bigym"
if BIGYM_PATH.exists():
    sys.path.insert(0, str(BIGYM_PATH))

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

# Default output directory (relative to this script)
DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "datasets"

# Default settings
DEFAULT_CONTROL_FREQUENCY = 50
DEFAULT_RESOLUTION = [256, 256]  # Updated: 256x256 resolution
DEFAULT_NUM_DEMOS = -1  # -1 means all available
DEFAULT_CAMERAS = ["head", "left_wrist", "right_wrist"]  # Updated: all 3 cameras
DEFAULT_JPEG_QUALITY = 95

# All available tasks in BiGym (approximately 40 tasks)
ALL_TASKS = [
    # ==== Reach Target (3 tasks) ====
    "ReachTarget",
    "ReachTargetSingle",
    "ReachTargetDual",
    
    # ==== Manipulation (4 tasks) ====
    "FlipCup",
    "FlipCutlery",
    "StackBlocks",
    
    # ==== Move Plates (2 tasks) ====
    "MovePlate",
    "MoveTwoPlates",
    
    # ==== Pick and Place (9 tasks) ====
    "PutCups",
    "TakeCups",
    "StoreBox",
    "PickBox",
    "SaucepanToHob",
    "StoreKitchenware",
    "ToastSandwich",
    "FlipSandwich",
    "RemoveSandwich",
    
    # ==== Cupboards (8 tasks) ====
    "DrawerTopOpen",
    "DrawerTopClose",
    "DrawersAllOpen",
    "DrawersAllClose",
    "WallCupboardOpen",
    "WallCupboardClose",
    "CupboardsOpenAll",
    "CupboardsCloseAll",
    
    # ==== Dishwasher Base (4 tasks) ====
    "DishwasherOpen",
    "DishwasherClose",
    "DishwasherCloseTrays",
    "DishwasherOpenTrays",
    
    # ==== Dishwasher Cups (3 tasks) ====
    "DishwasherUnloadCups",
    "DishwasherUnloadCupsLong",
    "DishwasherLoadCups",
    
    # ==== Dishwasher Cutlery (3 tasks) ====
    "DishwasherUnloadCutlery",
    "DishwasherUnloadCutleryLong",
    "DishwasherLoadCutlery",
    
    # ==== Dishwasher Plates (3 tasks) ====
    "DishwasherUnloadPlates",
    "DishwasherUnloadPlatesLong",
    "DishwasherLoadPlates",
    
    # ==== Groceries (2 tasks) ====
    "GroceriesStoreLower",
    "GroceriesStoreUpper",
]

# Tasks with known available demos (based on BiGym demo store)
RECOMMENDED_TASKS = [
    "ReachTarget",
    "ReachTargetSingle",
    "ReachTargetDual",
]

# =============================================================================
# Dimension Constants - H1 Robot with Floating Base
# =============================================================================

# Action dimensions with 4-DOF floating_base (16 total): [X, Y, Z, RZ]
# This matches the downloaded demo format
ACTION_DIM_FLOATING_BASE_4DOF = 4  # [pelvis_x, pelvis_y, pelvis_z, pelvis_rz]
ACTION_DIM_FLOATING_BASE_3DOF = 3  # [pelvis_x, pelvis_y, pelvis_rz] (legacy)
ACTION_DIM_LEFT_ARM = 5            # [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist]
ACTION_DIM_RIGHT_ARM = 5           # [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist]
ACTION_DIM_GRIPPER = 2             # [left_gripper, right_gripper]
ACTION_DIM_ARMS = ACTION_DIM_LEFT_ARM + ACTION_DIM_RIGHT_ARM + ACTION_DIM_GRIPPER  # 12

# Default: 4-DOF floating base (matches downloaded demos)
ACTION_DIM_FLOATING_BASE = ACTION_DIM_FLOATING_BASE_4DOF  # 4
ACTION_DIM_TOTAL = ACTION_DIM_FLOATING_BASE + ACTION_DIM_ARMS  # 16

# Observation dimensions
OBS_DIM_QPOS = 29               # All joint positions (including legs, torso, arms)
OBS_DIM_QVEL = 29               # All joint velocities
OBS_DIM_GRIPPER = 2             # [left_gripper, right_gripper]
OBS_DIM_FLOATING_BASE_4DOF = 4  # [x, y, z, rz] positions (4-DOF)
OBS_DIM_FLOATING_BASE_3DOF = 3  # [x, y, rz] positions (3-DOF legacy)
OBS_DIM_FLOATING_BASE = OBS_DIM_FLOATING_BASE_4DOF  # Default: 4-DOF
OBS_DIM_FLOATING_BASE_ACTIONS = 4  # accumulated [dx, dy, dz, drz]

# State dimensions for policy input
STATE_DIM_TOTAL = 16            # Matches action: [floating_base(4), arms(10), grippers(2)]
STATE_DIM_ARM = 12              # Arms only: [arms(10), grippers(2)]

# =============================================================================
# Dimension Names for Documentation
# =============================================================================

QPOS_NAMES = [
    # Legs (10 joints, not actuated in floating_base mode)
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    # Torso (1 joint, not actuated)
    "torso",
    # Left arm (5 joints, actuated)
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist",
    # Right arm (5 joints, actuated)
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist",
    # Additional joints (3 joints, may vary by robot model)
    "joint_22", "joint_23", "joint_24",
    # Extra joints if present
    "joint_25", "joint_26", "joint_27", "joint_28",
]

ACTION_NAMES_4DOF = [
    # Floating base (4 dims)
    "floating_base_x", "floating_base_y", "floating_base_z", "floating_base_rz",
    # Left arm (5 dims)
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist",
    # Right arm (5 dims)
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist",
    # Grippers (2 dims)
    "left_gripper", "right_gripper",
]

ACTION_NAMES_3DOF = [
    # Floating base (3 dims)
    "floating_base_x", "floating_base_y", "floating_base_rz",
    # Left arm (5 dims)
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist",
    # Right arm (5 dims)
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist",
    # Grippers (2 dims)
    "left_gripper", "right_gripper",
]

ACTION_NAMES = ACTION_NAMES_4DOF  # Default: 4-DOF floating base
STATE_NAMES = ACTION_NAMES  # observation.state matches action structure

STATE_ARM_NAMES = [
    # Left arm (5 dims)
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist",
    # Right arm (5 dims)
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist",
    # Grippers (2 dims)
    "left_gripper", "right_gripper",
]


# =============================================================================
# Task Class Loader
# =============================================================================

def get_task_class(task_name: str):
    """Import and return task class by name."""
    task_map = {}
    
    # Reach Target tasks (3)
    try:
        from bigym.envs.reach_target import ReachTarget, ReachTargetSingle, ReachTargetDual
        task_map.update({
            'ReachTarget': ReachTarget,
            'ReachTargetSingle': ReachTargetSingle,
            'ReachTargetDual': ReachTargetDual,
        })
    except ImportError:
        pass
    
    # Manipulation tasks (3)
    try:
        from bigym.envs.manipulation import FlipCup, FlipCutlery, StackBlocks
        task_map.update({
            'FlipCup': FlipCup,
            'FlipCutlery': FlipCutlery,
            'StackBlocks': StackBlocks,
        })
    except ImportError:
        pass
    
    # Move plates tasks (2)
    try:
        from bigym.envs.move_plates import MovePlate, MoveTwoPlates
        task_map.update({
            'MovePlate': MovePlate,
            'MoveTwoPlates': MoveTwoPlates,
        })
    except ImportError:
        pass
    
    # Pick and place tasks (9)
    try:
        from bigym.envs.pick_and_place import (
            PutCups, TakeCups, StoreBox, PickBox, SaucepanToHob,
            StoreKitchenware, ToastSandwich, FlipSandwich, RemoveSandwich
        )
        task_map.update({
            'PutCups': PutCups,
            'TakeCups': TakeCups,
            'StoreBox': StoreBox,
            'PickBox': PickBox,
            'SaucepanToHob': SaucepanToHob,
            'StoreKitchenware': StoreKitchenware,
            'ToastSandwich': ToastSandwich,
            'FlipSandwich': FlipSandwich,
            'RemoveSandwich': RemoveSandwich,
        })
    except ImportError:
        pass
    
    # Cupboards tasks (8)
    try:
        from bigym.envs.cupboards import (
            DrawerTopOpen, DrawerTopClose, DrawersAllOpen, DrawersAllClose,
            WallCupboardOpen, WallCupboardClose, CupboardsOpenAll, CupboardsCloseAll
        )
        task_map.update({
            'DrawerTopOpen': DrawerTopOpen,
            'DrawerTopClose': DrawerTopClose,
            'DrawersAllOpen': DrawersAllOpen,
            'DrawersAllClose': DrawersAllClose,
            'WallCupboardOpen': WallCupboardOpen,
            'WallCupboardClose': WallCupboardClose,
            'CupboardsOpenAll': CupboardsOpenAll,
            'CupboardsCloseAll': CupboardsCloseAll,
        })
    except ImportError:
        pass
    
    # Dishwasher base tasks (4)
    try:
        from bigym.envs.dishwasher import (
            DishwasherOpen, DishwasherClose,
            DishwasherCloseTrays, DishwasherOpenTrays
        )
        task_map.update({
            'DishwasherOpen': DishwasherOpen,
            'DishwasherClose': DishwasherClose,
            'DishwasherCloseTrays': DishwasherCloseTrays,
            'DishwasherOpenTrays': DishwasherOpenTrays,
        })
    except ImportError:
        pass
    
    # Dishwasher cups tasks (3)
    try:
        from bigym.envs.dishwasher_cups import (
            DishwasherUnloadCups, DishwasherUnloadCupsLong, DishwasherLoadCups
        )
        task_map.update({
            'DishwasherUnloadCups': DishwasherUnloadCups,
            'DishwasherUnloadCupsLong': DishwasherUnloadCupsLong,
            'DishwasherLoadCups': DishwasherLoadCups,
        })
    except ImportError:
        pass
    
    # Dishwasher cutlery tasks (3)
    try:
        from bigym.envs.dishwasher_cutlery import (
            DishwasherUnloadCutlery, DishwasherUnloadCutleryLong, DishwasherLoadCutlery
        )
        task_map.update({
            'DishwasherUnloadCutlery': DishwasherUnloadCutlery,
            'DishwasherUnloadCutleryLong': DishwasherUnloadCutleryLong,
            'DishwasherLoadCutlery': DishwasherLoadCutlery,
        })
    except ImportError:
        pass
    
    # Dishwasher plates tasks (3)
    try:
        from bigym.envs.dishwasher_plates import (
            DishwasherUnloadPlates, DishwasherUnloadPlatesLong, DishwasherLoadPlates
        )
        task_map.update({
            'DishwasherUnloadPlates': DishwasherUnloadPlates,
            'DishwasherUnloadPlatesLong': DishwasherUnloadPlatesLong,
            'DishwasherLoadPlates': DishwasherLoadPlates,
        })
    except ImportError:
        pass
    
    # Groceries tasks (2)
    try:
        from bigym.envs.groceries import GroceriesStoreLower, GroceriesStoreUpper
        task_map.update({
            'GroceriesStoreLower': GroceriesStoreLower,
            'GroceriesStoreUpper': GroceriesStoreUpper,
        })
    except ImportError:
        pass
    
    if task_name not in task_map:
        available = list(task_map.keys())
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {available}")
    
    return task_map[task_name]


def get_available_tasks() -> List[str]:
    """Get list of all available task names (approximately 40 tasks)."""
    # Simply return all tasks from ALL_TASKS that can be imported
    available = []
    for task_name in ALL_TASKS:
        try:
            get_task_class(task_name)
            available.append(task_name)
        except (ImportError, ValueError):
            pass
    return available


# =============================================================================
# Image Encoding
# =============================================================================

def encode_image_to_bytes(image: np.ndarray, quality: int = 95) -> bytes:
    """Encode a numpy image to JPEG bytes."""
    # Handle CHW -> HWC conversion
    if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
        image = np.transpose(image, (1, 2, 0))
    
    # Ensure uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    return buffer.getvalue()


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_dataset_for_task(
    task: str,
    output_dir: Path,
    num_demos: int = -1,
    control_frequency: int = DEFAULT_CONTROL_FREQUENCY,
    cameras: List[str] = DEFAULT_CAMERAS,
    resolution: List[int] = DEFAULT_RESOLUTION,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    skip_existing: bool = True,
) -> str:
    """
    Generate dataset for a single task.
    
    Returns:
        'success': Generated successfully
        'skipped': Already exists (when skip_existing=True)
        'no_demos': No demonstrations available
        'error': Failed with error
    """
    # Check if dataset already exists
    info_file = output_dir / "meta" / "info.json"
    if skip_existing and info_file.exists():
        print(f"\n[SKIP] {task}: Dataset already exists at {output_dir}")
        return 'skipped'
    
    from bigym.action_modes import JointPositionActionMode
    from bigym.utils.observation_config import ObservationConfig, CameraConfig
    from demonstrations.demo_store import DemoStore
    from demonstrations.utils import Metadata
    
    print(f"\n{'='*60}")
    print(f"Generating dataset for task: {task}")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Demos: {num_demos if num_demos > 0 else 'all'}")
    print(f"  Frequency: {control_frequency} Hz")
    print(f"  Cameras: {cameras}")
    print(f"  Resolution: {resolution}")
    
    try:
        # Create environment
        task_cls = get_task_class(task)
        
        # Use 4-DOF floating base to match downloaded demos: [X, Y, Z, RZ]
        from bigym.action_modes import PelvisDof
        floating_dofs = [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
        action_mode = JointPositionActionMode(
            floating_base=True, 
            absolute=True,
            floating_dofs=floating_dofs,
        )
        
        camera_configs = [
            CameraConfig(name=cam, resolution=tuple(resolution), rgb=True, depth=False)
            for cam in cameras
        ]
        
        env = task_cls(
            action_mode=action_mode,
            observation_config=ObservationConfig(cameras=camera_configs, proprioception=True),
            control_frequency=control_frequency,
        )
        
        # Get demos from demo store
        print("\nDownloading demonstrations...")
        metadata = Metadata.from_env(env)
        demo_store = DemoStore()
        
        try:
            demos = demo_store.get_demos(metadata, amount=num_demos, frequency=control_frequency)
        except Exception as e:
            print(f"  Warning: {e}")
            print("  Trying with lightweight observation mode...")
            from demonstrations.utils import ObservationMode
            metadata.observation_mode = ObservationMode.Lightweight
            demos = demo_store.get_demos(metadata, amount=num_demos, frequency=control_frequency)
        
        if not demos:
            print(f"  No demonstrations available for {task}")
            try:
                env.close()
            except Exception:
                pass
            return 'no_demos'
        
        print(f"  Downloaded {len(demos)} demonstrations")
        
        # Setup output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data_dir = output_dir / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        meta_dir = output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        # Process demonstrations
        all_episodes_data = []  # List of lists, one per episode
        episode_metadata = []
        current_idx = 0
        action_dim = env.action_space.shape[0]
        
        # Actual dimensions from environment
        actual_dims = {
            'qpos': None,
            'qvel': None,
            'gripper': None,
            'floating_base': None,
            'floating_base_actions': None,
        }
        
        for ep_idx, demo in enumerate(tqdm(demos, desc=f"Processing {task}")):
            episode_start = current_idx
            episode_length = len(demo.timesteps)
            episode_data = []
            
            for frame_idx, timestep in enumerate(demo.timesteps):
                obs = timestep.observation
                action = timestep.executed_action
                
                # Build row
                row = {
                    'episode_index': ep_idx,
                    'frame_index': frame_idx,
                    'timestamp': float(frame_idx / control_frequency),
                    'index': current_idx,
                    'task_index': 0,
                }
                
                # =============================================
                # Parse Observations
                # =============================================
                
                # 1. Full proprioception: qpos (29) + qvel (29)
                if 'proprioception' in obs:
                    proprio = np.array(obs['proprioception']).astype(np.float32)
                    n_joints = len(proprio) // 2
                    qpos = proprio[:n_joints]
                    qvel = proprio[n_joints:]
                    
                    row['observation.qpos'] = qpos.tolist()
                    row['observation.qvel'] = qvel.tolist()
                    
                    if actual_dims['qpos'] is None:
                        actual_dims['qpos'] = len(qpos)
                        actual_dims['qvel'] = len(qvel)
                else:
                    qpos = np.zeros(OBS_DIM_QPOS, dtype=np.float32)
                    qvel = np.zeros(OBS_DIM_QVEL, dtype=np.float32)
                    row['observation.qpos'] = qpos.tolist()
                    row['observation.qvel'] = qvel.tolist()
                
                # 2. Gripper states (2)
                if 'proprioception_grippers' in obs:
                    gripper_state = np.array(obs['proprioception_grippers']).astype(np.float32)
                    row['observation.gripper'] = gripper_state.tolist()
                    if actual_dims['gripper'] is None:
                        actual_dims['gripper'] = len(gripper_state)
                else:
                    gripper_state = np.zeros(OBS_DIM_GRIPPER, dtype=np.float32)
                    row['observation.gripper'] = gripper_state.tolist()
                
                # 3. Floating base position (4 dims with 4-DOF: [x, y, z, rz])
                if 'proprioception_floating_base' in obs:
                    floating_base = np.array(obs['proprioception_floating_base']).astype(np.float32)
                    row['observation.floating_base'] = floating_base.tolist()
                    if actual_dims['floating_base'] is None:
                        actual_dims['floating_base'] = len(floating_base)
                else:
                    floating_base = np.zeros(OBS_DIM_FLOATING_BASE, dtype=np.float32)
                    row['observation.floating_base'] = floating_base.tolist()
                
                # 4. Floating base accumulated actions (4 dims with 4-DOF: [dx, dy, dz, drz])
                if 'proprioception_floating_base_actions' in obs:
                    floating_base_actions = np.array(obs['proprioception_floating_base_actions']).astype(np.float32)
                    row['observation.floating_base_actions'] = floating_base_actions.tolist()
                    if actual_dims['floating_base_actions'] is None:
                        actual_dims['floating_base_actions'] = len(floating_base_actions)
                else:
                    floating_base_actions = np.zeros(OBS_DIM_FLOATING_BASE_ACTIONS, dtype=np.float32)
                    row['observation.floating_base_actions'] = floating_base_actions.tolist()
                
                # 5. Build observation.state (16 dims with 4-DOF) - matches action structure
                # [floating_base(4), left_arm(5), right_arm(5), grippers(2)]
                # From qpos, arms are at indices 11:16 (left) and 16:21 (right)
                left_arm_qpos = qpos[11:16] if len(qpos) > 16 else np.zeros(5, dtype=np.float32)
                right_arm_qpos = qpos[16:21] if len(qpos) > 21 else np.zeros(5, dtype=np.float32)
                
                state = np.concatenate([
                    floating_base,          # 4 dims: [x, y, z, rz] (4-DOF)
                    left_arm_qpos,          # 5 dims: [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist]
                    right_arm_qpos,         # 5 dims: [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist]
                    gripper_state,          # 2 dims: [left_gripper, right_gripper]
                ]).astype(np.float32)
                row['observation.state'] = state.tolist()
                
                # 6. Build observation.state_arm (12 dims) - arms only, no floating base
                # [left_arm(5), right_arm(5), grippers(2)]
                state_arm = np.concatenate([
                    left_arm_qpos,          # 5 dims
                    right_arm_qpos,         # 5 dims
                    gripper_state,          # 2 dims
                ]).astype(np.float32)
                row['observation.state_arm'] = state_arm.tolist()
                
                # =============================================
                # Parse Actions
                # =============================================
                
                if action is not None:
                    action_arr = np.array(action).astype(np.float32)
                else:
                    action_arr = np.zeros(action_dim, dtype=np.float32)
                
                # Full action (16 dims with 4-DOF floating base)
                row['action'] = action_arr.tolist()
                
                # Determine floating base dimension from action
                fb_dim = len(action_arr) - ACTION_DIM_ARMS  # Total - arms = floating_base dim
                
                # Split action into components
                # action.floating_base (4 dims with 4-DOF): [x, y, z, rz]
                action_floating_base = action_arr[:fb_dim] if len(action_arr) >= fb_dim else np.zeros(fb_dim, dtype=np.float32)
                row['action.floating_base'] = action_floating_base.tolist()
                
                # action.arms (12 dims): [left_arm(5), right_arm(5), grippers(2)]
                action_arms = action_arr[fb_dim:] if len(action_arr) >= fb_dim else np.zeros(ACTION_DIM_ARMS, dtype=np.float32)
                row['action.arms'] = action_arms.tolist()
                
                # =============================================
                # Images
                # =============================================
                
                for cam_name in cameras:
                    rgb_key = f'rgb_{cam_name}'
                    img_col = f'observation.images.{cam_name}'
                    
                    if rgb_key in obs and obs[rgb_key] is not None:
                        img = np.array(obs[rgb_key])
                        img_bytes = encode_image_to_bytes(img, jpeg_quality)
                        row[img_col] = {'bytes': img_bytes, 'path': None}
                    else:
                        placeholder = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
                        img_bytes = encode_image_to_bytes(placeholder, jpeg_quality)
                        row[img_col] = {'bytes': img_bytes, 'path': None}
                
                episode_data.append(row)
                current_idx += 1
            
            all_episodes_data.append(episode_data)
            
            # Episode metadata
            episode_metadata.append({
                'episode_index': ep_idx,
                'episode_data_index_from': episode_start,
                'episode_data_index_to': current_idx,
                'length': episode_length,
                'task': task,
            })
        
        # Save data as parquet
        print("  Saving data parquet files (one per episode)...")
        for ep_idx, episode_data in enumerate(tqdm(all_episodes_data, desc="Saving episodes")):
            ep_parquet_path = data_dir / f"file-{ep_idx:03d}.parquet"
            df = pa.Table.from_pylist(episode_data)
            pq.write_table(df, ep_parquet_path)
        
        # Save episode metadata as jsonlines
        print("  Saving episode metadata...")
        episodes_jsonl_path = meta_dir / "episodes.jsonl"
        with open(episodes_jsonl_path, 'w') as f:
            for ep_meta in episode_metadata:
                f.write(json.dumps(ep_meta) + '\n')
        
        # Compute and save statistics
        print("  Computing statistics...")
        all_data = [row for episode_data in all_episodes_data for row in episode_data]
        
        def compute_stats_for_key(key):
            """Compute min/max/mean/std for a given key."""
            values = [row.get(key) for row in all_data if row.get(key) is not None]
            if not values or (isinstance(values[0], list) and len(values[0]) == 0):
                return None
            arr = np.array(values, dtype=np.float32)
            return {
                'min': arr.min(axis=0).tolist(),
                'max': arr.max(axis=0).tolist(),
                'mean': arr.mean(axis=0).tolist(),
                'std': arr.std(axis=0).tolist(),
            }
        
        stats = {}
        
        # Observation stats
        for key in ['observation.qpos', 'observation.qvel', 'observation.gripper',
                    'observation.floating_base', 'observation.floating_base_actions',
                    'observation.state', 'observation.state_arm']:
            key_stats = compute_stats_for_key(key)
            if key_stats:
                stats[key] = key_stats
        
        # Action stats
        for key in ['action', 'action.floating_base', 'action.arms']:
            key_stats = compute_stats_for_key(key)
            if key_stats:
                stats[key] = key_stats
        
        stats_path = meta_dir / 'stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create info.json with detailed dimension annotations
        features = {
            # ==== Observations ====
            'observation.qpos': {
                'dtype': 'float32',
                'shape': [actual_dims['qpos'] or OBS_DIM_QPOS],
                'description': 'Full joint positions (29 dims)',
                'names': QPOS_NAMES[:actual_dims['qpos'] or OBS_DIM_QPOS],
            },
            'observation.qvel': {
                'dtype': 'float32',
                'shape': [actual_dims['qvel'] or OBS_DIM_QVEL],
                'description': 'Full joint velocities (29 dims)',
                'names': QPOS_NAMES[:actual_dims['qvel'] or OBS_DIM_QVEL],  # Same names as qpos
            },
            'observation.gripper': {
                'dtype': 'float32',
                'shape': [actual_dims['gripper'] or OBS_DIM_GRIPPER],
                'description': 'Gripper states (2 dims)',
                'names': ['left_gripper', 'right_gripper'],
            },
            'observation.floating_base': {
                'dtype': 'float32',
                'shape': [actual_dims['floating_base'] or OBS_DIM_FLOATING_BASE],
                'description': 'Floating base position (4 dims with 4-DOF)',
                'names': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'][:actual_dims['floating_base'] or OBS_DIM_FLOATING_BASE],
            },
            'observation.floating_base_actions': {
                'dtype': 'float32',
                'shape': [actual_dims['floating_base_actions'] or OBS_DIM_FLOATING_BASE_ACTIONS],
                'description': 'Accumulated floating base actions since reset (4 dims with 4-DOF)',
                'names': ['acc_pelvis_x', 'acc_pelvis_y', 'acc_pelvis_z', 'acc_pelvis_rz'][:actual_dims['floating_base_actions'] or OBS_DIM_FLOATING_BASE_ACTIONS],
            },
            'observation.state': {
                'dtype': 'float32',
                'shape': [STATE_DIM_TOTAL],
                'description': f'Policy state input ({STATE_DIM_TOTAL} dims, matches action structure)',
                'names': STATE_NAMES,
                'structure': '[floating_base(4), left_arm(5), right_arm(5), grippers(2)]',
            },
            'observation.state_arm': {
                'dtype': 'float32',
                'shape': [STATE_DIM_ARM],
                'description': 'Arm-only state (12 dims, no floating base)',
                'names': STATE_ARM_NAMES,
                'structure': '[left_arm(5), right_arm(5), grippers(2)]',
            },
            
            # ==== Actions ====
            'action': {
                'dtype': 'float32',
                'shape': [action_dim],  # Use actual action dim from env
                'description': f'Full action ({action_dim} dims)',
                'names': ACTION_NAMES[:action_dim],
                'structure': '[floating_base(4), left_arm(5), right_arm(5), grippers(2)]',
            },
            'action.floating_base': {
                'dtype': 'float32',
                'shape': [action_dim - ACTION_DIM_ARMS],  # Actual floating base dim
                'description': f'Floating base action ({action_dim - ACTION_DIM_ARMS} dims)',
                'names': ['floating_base_x', 'floating_base_y', 'floating_base_z', 'floating_base_rz'][:action_dim - ACTION_DIM_ARMS],
            },
            'action.arms': {
                'dtype': 'float32',
                'shape': [ACTION_DIM_ARMS],
                'description': 'Arms + grippers action (12 dims)',
                'names': [
                    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 'left_wrist',
                    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow', 'right_wrist',
                    'left_gripper', 'right_gripper',
                ],
                'structure': '[left_arm(5), right_arm(5), grippers(2)]',
            },
        }
        
        # Add image features
        for cam_name in cameras:
            features[f'observation.images.{cam_name}'] = {
                'dtype': 'image',
                'shape': [resolution[0], resolution[1], 3],
                'description': f'{cam_name} camera RGB image',
            }
        
        info = {
            'codebase_version': 'v2.1',
            'robot_type': 'h1_humanoid',
            'fps': control_frequency,
            'total_episodes': len(demos),
            'total_frames': current_idx,
            'chunks_size': len(demos),
            'total_chunks': 1,
            'data_path': 'data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet',
            'features': features,
            'task_name': task,
            'splits': {
                'train': f'0:{len(demos)}',
            },
            'dimension_info': {
                'action_total': ACTION_DIM_TOTAL,
                'action_floating_base': ACTION_DIM_FLOATING_BASE,
                'action_arms': ACTION_DIM_ARMS,
                'obs_qpos': actual_dims['qpos'] or OBS_DIM_QPOS,
                'obs_qvel': actual_dims['qvel'] or OBS_DIM_QVEL,
                'obs_gripper': actual_dims['gripper'] or OBS_DIM_GRIPPER,
                'obs_floating_base': actual_dims['floating_base'] or OBS_DIM_FLOATING_BASE,
                'obs_floating_base_actions': actual_dims['floating_base_actions'] or OBS_DIM_FLOATING_BASE_ACTIONS,
                'state_total': STATE_DIM_TOTAL,
                'state_arm': STATE_DIM_ARM,
            },
        }
        
        with open(meta_dir / 'info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        # Create tasks.jsonl
        tasks_jsonl_path = meta_dir / 'tasks.jsonl'
        with open(tasks_jsonl_path, 'w') as f:
            f.write(json.dumps({'task_index': 0, 'task': task}) + '\n')
        
        print(f"\n  ✓ Success! Generated {len(demos)} episodes, {current_idx} frames")
        print(f"    Output: {output_dir}")
        
        # Close environment
        try:
            env.close()
        except Exception:
            pass
        
        return 'success'
        
    except Exception as e:
        print(f"\n  ✗ Failed to generate dataset for {task}: {e}")
        import traceback
        traceback.print_exc()
        return 'error'


def generate_all_datasets(
    tasks: List[str],
    output_root: Path,
    num_demos: int = -1,
    control_frequency: int = DEFAULT_CONTROL_FREQUENCY,
    cameras: List[str] = DEFAULT_CAMERAS,
    resolution: List[int] = DEFAULT_RESOLUTION,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    skip_existing: bool = True,
):
    """Generate datasets for multiple tasks."""
    
    print("=" * 60)
    print("BiGym Dataset Generator")
    print("=" * 60)
    print(f"Tasks to generate: {len(tasks)} tasks")
    print(f"Output root: {output_root}")
    print(f"Demos per task: {num_demos if num_demos > 0 else 'all'}")
    print(f"Control frequency: {control_frequency} Hz")
    print(f"Resolution: {resolution}")
    print(f"Skip existing: {skip_existing}")
    print()
    print("Tasks:")
    for i, t in enumerate(tasks, 1):
        print(f"  {i:2d}. {t}")
    print()
    
    results = {}
    
    for idx, task in enumerate(tasks, 1):
        print(f"\n[{idx}/{len(tasks)}] Processing {task}...")
        output_dir = output_root / f"bigym_{task.lower()}"
        result = generate_dataset_for_task(
            task=task,
            output_dir=output_dir,
            num_demos=num_demos,
            control_frequency=control_frequency,
            cameras=cameras,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
            skip_existing=skip_existing,
        )
        results[task] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("Generation Summary")
    print("=" * 60)
    
    successful = [t for t, s in results.items() if s == 'success']
    skipped = [t for t, s in results.items() if s == 'skipped']
    no_demos = [t for t, s in results.items() if s == 'no_demos']
    errors = [t for t, s in results.items() if s == 'error']
    
    if successful:
        print(f"\n✓ Successfully generated ({len(successful)} tasks):")
        for task in successful:
            print(f"    - {task}")
    
    if skipped:
        print(f"\n⏭ Skipped (already exists) ({len(skipped)} tasks):")
        for task in skipped:
            print(f"    - {task}")
    
    if no_demos:
        print(f"\n⚠ No demos available ({len(no_demos)} tasks):")
        for task in no_demos:
            print(f"    - {task}")
    
    if errors:
        print(f"\n✗ Failed with errors ({len(errors)} tasks):")
        for task in errors:
            print(f"    - {task}")
    
    print(f"\nTotal: {len(successful)} success, {len(skipped)} skipped, {len(no_demos)} no demos, {len(errors)} errors")
    print(f"Output directory: {output_root}")
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BiGym Dataset Generator - One-click dataset generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all recommended tasks
    python generate_dataset.py
    
    # Generate specific tasks
    python generate_dataset.py --tasks ReachTarget ReachTargetDual
    
    # Custom output and settings
    python generate_dataset.py --output_root ./my_datasets --num_demos 100
    
    # List available tasks
    python generate_dataset.py --list_tasks
"""
    )
    
    parser.add_argument(
        "--tasks", "-t",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to generate (default: recommended tasks)"
    )
    parser.add_argument(
        "--output_root", "-o",
        type=str,
        default=None,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT})"
    )
    parser.add_argument(
        "--num_demos", "-n",
        type=int,
        default=DEFAULT_NUM_DEMOS,
        help="Number of demos per task (-1 for all, default: all)"
    )
    parser.add_argument(
        "--control_frequency", "-f",
        type=int,
        default=DEFAULT_CONTROL_FREQUENCY,
        help=f"Control frequency in Hz (default: {DEFAULT_CONTROL_FREQUENCY})"
    )
    parser.add_argument(
        "--cameras", "-c",
        type=str,
        nargs="+",
        default=DEFAULT_CAMERAS,
        help=f"Camera names (default: {DEFAULT_CAMERAS})"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        nargs=2,
        default=DEFAULT_RESOLUTION,
        help=f"Camera resolution H W (default: {DEFAULT_RESOLUTION})"
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=f"JPEG encoding quality (default: {DEFAULT_JPEG_QUALITY})"
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="List all available tasks and exit"
    )
    parser.add_argument(
        "--all_tasks",
        action="store_true",
        help="Generate datasets for ALL available tasks"
    )
    
    args = parser.parse_args()
    
    # List tasks mode
    if args.list_tasks:
        print("Available BiGym tasks:")
        print("-" * 40)
        try:
            available = get_available_tasks()
            for task in available:
                marker = "★" if task in RECOMMENDED_TASKS else " "
                print(f"  {marker} {task}")
            print()
            print("★ = Recommended (known to have demos)")
        except ImportError:
            print("  Error: BigYm not installed. Run:")
            print("    pip install -e bigym/")
        return 0
    
    # Determine tasks to generate
    if args.all_tasks:
        tasks = get_available_tasks()
    elif args.tasks:
        tasks = args.tasks
    else:
        tasks = RECOMMENDED_TASKS
    
    # Determine output root
    output_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
    
    # Generate datasets
    results = generate_all_datasets(
        tasks=tasks,
        output_root=output_root,
        num_demos=args.num_demos,
        control_frequency=args.control_frequency,
        cameras=args.cameras,
        resolution=args.resolution,
        jpeg_quality=args.jpeg_quality,
    )
    
    # Return exit code based on results
    if all(results.values()):
        return 0
    elif any(results.values()):
        return 1  # Partial success
    else:
        return 2  # Complete failure


if __name__ == "__main__":
    exit(main())
