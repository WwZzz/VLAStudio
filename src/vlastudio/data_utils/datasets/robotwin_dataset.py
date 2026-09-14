import os
import sys
import numpy as np
import cv2
import zipfile
import shutil
import json
import re
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import requests
import h5py
from collections import OrderedDict

# -----------------------------------------------------------------------------
# Base Class Fallback
# -----------------------------------------------------------------------------

from vlastudio.data_utils.datasets.base import EpisodicDataset

# -----------------------------------------------------------------------------
# Cache Directory Helper
# -----------------------------------------------------------------------------

def _get_robotwin_cache_dir() -> Path:
    """Get RoboTwin cache directory with priority:
    1. HF_LEROBOT_HOME/robotwin (if HF_LEROBOT_HOME is set)
    2. HF_HOME/lerobot/robotwin (if HF_HOME is set)
    3. Default: ~/.cache/huggingface/lerobot/robotwin
    """
    # Priority 1: HF_LEROBOT_HOME/robotwin
    lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if lerobot_home:
        return Path(lerobot_home) / "robotwin"
    
    # Priority 2: HF_HOME/lerobot/robotwin
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "lerobot" / "robotwin"
    
    # Priority 3: Default
    return Path.home() / ".cache" / "huggingface" / "lerobot" / "robotwin"

# -----------------------------------------------------------------------------
# RoboTwin Dataset Class
# -----------------------------------------------------------------------------
class RoboTwinDataset(EpisodicDataset):
    """
    RoboTwin dataset for ILStudio.
    
    Key Features:
    - ctrl_space: 'joint' (joint_action) or 'ee' (endpose).
    - Dynamic Dimensions: Automatically detects single/dual arm from data.
    - Unified Path: Accepts local path or HF dataset name.
    """

    HF_REPO_ID = "TianxingChen/RoboTwin2.0"
    HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    CACHE_DIR = _get_robotwin_cache_dir()

    def __init__(
        self,
        dataset_path: str,
        image_size: tuple = (480, 640),
        chunk_size: int = 16,
        camera_names: list = None,
        ctrl_space: str = 'joint',  # 'joint' or 'ee'
        ctrl_type: str = 'abs',
        preload_data: bool = False,
    ):
        if camera_names is None:
            camera_names = ['head_camera', 'left_camera', 'right_camera']
        
        if isinstance(image_size, list):
            image_size = tuple(image_size)
            
        # 1. Validate ctrl_space mapping
        if ctrl_space not in ['joint', 'ee']:
            raise ValueError("ctrl_space must be 'joint' or 'ee'")
        
        # Internal mapping to HDF5 group names
        self.hdf5_group_name = 'joint_action' if ctrl_space == 'joint' else 'endpose'
        self.ctrl_space = ctrl_space
        self._preload_data = preload_data
        self._episode_cache = {}
        self._instruction_cache = {}

        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------
        # Path Logic
        # --------------------------------------------------------
        if dataset_path is None:
            raise ValueError("dataset_path must be provided.")

        path_obj = Path(dataset_path).expanduser().resolve()
        
        if path_obj.exists():
            logger.info(f"Detected local path: {path_obj}")
            self.raw_data_dir = self._extract_local_data(path_obj)
        else:
            logger.info(f"Path not found locally, assuming Hugging Face dataset name: {dataset_path}")
            self.raw_data_dir = self._download_and_extract_hf(dataset_path)

        # Recursively find all episode files
        raw_episode_files = sorted(list(self.raw_data_dir.rglob('episode*.hdf5')))
        if not raw_episode_files:
            raise FileNotFoundError(f"No episode HDF5 files found in: {self.raw_data_dir}")

        self.raw_episode_files = raw_episode_files

        # --------------------------------------------------------
        # Dynamic Dimension Inference
        # --------------------------------------------------------
        # Load the first episode to infer state_dim/action_dim automatically
        try:
            sample_data, _ = self._load_hdf5(str(raw_episode_files[0]))
            inferred_dim = sample_data.shape[1]
            self.state_dim = inferred_dim
            self.action_dim = inferred_dim
            logger.info(f"Inferred dimensions from data: {inferred_dim} (Based on '{ctrl_space}' space)")
        except Exception as e:
            logger.error(f"Failed to infer dimensions from {raw_episode_files[0]}: {e}")
            raise RuntimeError("Could not infer dataset dimensions.")

        super().__init__(
            dataset_path_list=[str(p) for p in raw_episode_files],
            camera_names=camera_names,
            chunk_size=chunk_size,
            ctrl_space=ctrl_space,
            ctrl_type=ctrl_type,
            image_size=image_size,
            preload_data=preload_data
        )

        logger.info(f"RoboTwinDataset initialized from {self.raw_data_dir}")
        logger.info(f"  - Control: {ctrl_space} -> {self.hdf5_group_name} ({ctrl_type})")
        logger.info(f"  - Episodes: {len(self.raw_episode_files)}")

    # ============================================================================
    # Core Data Loading Methods
    # ============================================================================

    def _load_hdf5(self, path):
        """
        Internal method to load raw data from HDF5.
        Dynamically handles single-arm or dual-arm data.
        """
        with h5py.File(path, 'r') as root:
            traj_parts = []
            
            # Use the group determined in __init__ (joint_action or endpose)
            if self.hdf5_group_name not in root:
                raise KeyError(f"Group '{self.hdf5_group_name}' not found in {path}")
            
            group = root[self.hdf5_group_name]
            
            # Define keys to look for based on control space
            # Order matters: usually [Left Arm, Left Gripper, Right Arm, Right Gripper]
            # or just [Arm, Gripper] if single arm
            
            sides = ['left', 'right']
            
            for side in sides:
                # Determine key names based on ctrl_space
                if self.ctrl_space == 'joint':
                    arm_key = f'{side}_arm'      # e.g., left_arm
                    gripper_key = f'{side}_gripper' # e.g., left_gripper
                else: # ee
                    arm_key = f'{side}_endpose'  # e.g., left_endpose
                    gripper_key = f'{side}_gripper'

                # Check if this side exists in the group
                if arm_key in group and gripper_key in group:
                    arm_data = group[arm_key][()]
                    gripper_data = group[gripper_key][()]

                    # Ensure gripper is (T, 1)
                    if gripper_data.ndim == 1:
                        gripper_data = gripper_data[:, None]
                    
                    traj_parts.append(arm_data)
                    traj_parts.append(gripper_data)
            
            if not traj_parts:
                raise ValueError(f"No valid arm data found in group '{self.hdf5_group_name}' for {path}")

            # Concatenate all found parts (T, sum_dims)
            traj_data = np.concatenate(traj_parts, axis=1)

            # --- Load Image Bytes ---
            image_dict = {}
            potential_cams = ['head_camera', 'left_camera', 'right_camera', 'front_camera']
            for camera_name in potential_cams:
                rgb_path = f'/observation/{camera_name}/rgb'
                if rgb_path in root:
                    image_dict[camera_name] = root[rgb_path][()]
        
        return traj_data, image_dict

    def _load_instruction(self, dataset_path):
        """Find and load language instruction from JSON."""
        path_obj = Path(dataset_path)
        match = re.search(r'episode(\d+)\.hdf5', path_obj.name)
        episode_id = int(match.group(1)) if match else None
        
        if episode_id is None:
            return ""

        candidates = [
            path_obj.parent.parent / "instructions" / f"episode{episode_id}.json", 
            path_obj.parent / "instructions" / f"episode{episode_id}.json",        
        ]
        
        instruction_path = None
        for p in candidates:
            if p.exists():
                instruction_path = p
                break
        
        if not instruction_path:
            found = list(self.raw_data_dir.rglob(f"instructions/episode{episode_id}.json"))
            if found:
                instruction_path = found[0]

        if instruction_path and instruction_path.exists():
            try:
                with open(instruction_path, "r") as f:
                    data = json.load(f)
                    instr = data.get("seen", "")
                    return instr[0] if isinstance(instr, list) and len(instr) > 0 else str(instr)
            except Exception:
                return ""
        return ""

    # ============================================================================
    # Public Interface
    # ============================================================================

    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """Load single timestep."""
        if dataset_path not in self._episode_cache:
            self._episode_cache[dataset_path] = self._load_hdf5(dataset_path)
        
        if dataset_path not in self._instruction_cache:
            self._instruction_cache[dataset_path] = self._load_instruction(dataset_path)

        traj_data, image_dict = self._episode_cache[dataset_path]
        language_instruction = self._instruction_cache[dataset_path]

        state = traj_data[start_ts].astype(np.float32)
        
        # if start_ts + self.chunk_size < traj_data.shape[0]:
        action = traj_data[start_ts: start_ts + self.chunk_size].astype(np.float32)
        # else:
        #     action = np.zeros_like(state)

        images = OrderedDict()
        for cam_name in self.camera_names:
            if cam_name in image_dict and len(image_dict[cam_name]) > start_ts:
                img = cv2.imdecode(np.frombuffer(image_dict[cam_name][start_ts], np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
                images[cam_name] = img 
            else:
                images[cam_name] = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

        # if not self._preload_data:
        #     del self._episode_cache[dataset_path]

        return {
            'action': action, 
            'image': images, 
            'state': state, 
            'language_instruction': language_instruction,
            'timestamp': start_ts,
        }

    def load_feat_from_episode(self, dataset_path, feats=[]):
        """Load entire episode."""
        data_dict = {}
        if isinstance(feats, str): feats = [feats]
        
        if 'language_instruction' in feats or len(feats) == 0:
            data_dict['language_instruction'] = self._load_instruction(dataset_path)
        
        if len(feats) == 1 and 'language_instruction' in feats:
            return data_dict

        traj_raw, image_bytes_dict = self._load_hdf5(dataset_path)
        traj_raw = traj_raw.astype(np.float32)

        if 'state' in feats or len(feats) == 0:
            data_dict['state'] = traj_raw

        if 'action' in feats or len(feats) == 0:
            data_dict['action'] = traj_raw

        if 'image' in feats or 'image_wrist' in feats or len(feats) == 0:
            loaded_images = {}
            for cam_name in self.camera_names:
                if cam_name not in image_bytes_dict: continue
                is_wrist = 'left' in cam_name or 'right' in cam_name
                should_load = (len(feats)==0) or ('image' in feats) or ('image_wrist' in feats and is_wrist)
                
                if should_load:
                    imgs = []
                    for b in image_bytes_dict[cam_name]:
                        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                        img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
                        imgs.append(img)
                    loaded_images[cam_name] = np.array(imgs)
            data_dict['image'] = loaded_images

        return data_dict

    def get_episode_len(self):
        all_episode_len = []
        for dataset_path in self.dataset_path_list:
            try:
                # Peek length without full load
                with h5py.File(dataset_path, 'r') as f:
                    # Check which group exists (joint or endpose)
                    if 'joint_action' in f:
                        grp = f['joint_action']
                    elif 'endpose' in f:
                        grp = f['endpose']
                    else:
                        all_episode_len.append(0)
                        continue
                    
                    # Find any valid key to get length
                    if 'left_gripper' in grp: length = grp['left_gripper'].shape[0]
                    elif 'right_gripper' in grp: length = grp['right_gripper'].shape[0]
                    else: length = 0
                    
                    all_episode_len.append(max(0, length - 1))
            except Exception:
                all_episode_len.append(0)
        return all_episode_len


    # ============================================================================
    # Download & Extract Helpers
    # ============================================================================
    def _download_and_extract_hf(self, dataset_name):
        # Note: Append /datasets/ to HF_ENDPOINT
        url = f"{self.HF_ENDPOINT}/datasets/{self.HF_REPO_ID}/resolve/main/dataset/{dataset_name}.zip"
        
        download_path = self.CACHE_DIR / f"{dataset_name}.zip"
        extract_dir = self.CACHE_DIR / dataset_name
        
        download_path.parent.mkdir(parents=True, exist_ok=True)

        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"Using cached: {extract_dir}")
            return extract_dir

        if not download_path.exists():
            logger.info(f"Downloading {url}...")
            
            # --- Auth Token handling ---
            hf_token = os.getenv("HF_TOKEN")
            headers = {}
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            # ---------------------

            try:
                # Explicitly allow redirects (allow_redirects=True is default, but explicit for safety)
                resp = requests.get(url, stream=True, headers=headers, allow_redirects=True)
                
                if resp.status_code == 401:
                    logger.error("Error 401: Unauthorized. Please set HF_TOKEN environment variable.")
                
                resp.raise_for_status()
                total = int(resp.headers.get('content-length', 0))
                with open(download_path, 'wb') as f:
                    for chunk in tqdm(resp.iter_content(8192), total=total//8192, unit='KB'):
                        f.write(chunk)
            except Exception as e:
                logger.error(f"Download error for URL: {url}")  # Print URL for debugging
                if download_path.exists(): download_path.unlink()
                raise e

        logger.info(f"Extracting {download_path}...")
        try:
            with zipfile.ZipFile(download_path, 'r') as z:
                z.extractall(extract_dir.parent)
            return extract_dir
        except Exception as e:
            if extract_dir.exists(): shutil.rmtree(extract_dir)
            raise e
        
    def _extract_local_data(self, path):
        path = Path(path)
        if path.is_dir(): return path
        if path.is_file() and path.suffix == '.zip':
            extract_dir = path.parent / path.stem
            if extract_dir.exists() and any(extract_dir.iterdir()): return extract_dir
            logger.info(f"Extracting local {path}...")
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(path.parent)
            return extract_dir
        raise ValueError(f"Invalid path: {path}")

if __name__ == "__main__":
    """
    datasets:
  - type: data_utils.datasets.robotwin_dataset.RoboTwinDataset
    name: robotwin_hammer_3cam_v2
    args:
      dataset_path: /home/wz/Code/ILStudio/benchmark/robotwin/RoboTwin/data/beat_block_hammer/demo_clean
      image_size: [480, 640]  # [H, W] - matching RoboTwin original size
      chunk_size: 50          # Action chunk length (matching RoboTwin)
      camera_names:           # All 3 cameras (matching RoboTwin)
        - head_camera         # Front/overhead camera (cam_high)
        - left_camera         # Left wrist camera (cam_left_wrist)
        - right_camera        # Right wrist camera (cam_right_wrist)
      ctrl_space: ee          # End-effector control
      ctrl_type: abs          # Absolute control
      preload_data: false     # Set to true to load all data into RAM
    """
    dataset = RoboTwinDataset(dataset_path="/home/wz/Code/ILStudio/benchmark/robotwin/RoboTwin/data/beat_block_hammer/demo_clean", ctrl_space="joint", ctrl_type="abs", chunk_size=50)
    d0 = dataset[0]
    d100 = dataset[100]
    # # d121 = dataset[121]
    # # rawd = dataset.load_onestep_from_episode(dataset.dataset_path_list[0], 0)
    print('ok')