"""
Base dataset class for episodic data loading.

This module contains the EpisodicDataset base class that all specific dataset implementations inherit from.
"""

import numpy as np
import torch
import os
import h5py
import fnmatch
from vlastudio.data_utils.utils import ensure_uint8_image
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

class EpisodicDataset(torch.utils.data.Dataset):
    """
    Base class for episodic datasets.
    
    This class provides the core functionality for loading and processing episodic data,
    including memory management, episode indexing, and data normalization.
    """
    
    def __init__(self, dataset_path_list: list, camera_names: list=[], 
                 chunk_size: int = 16, 
                 ctrl_space: str = 'ee', ctrl_type: str = 'delta',
                 image_size: tuple = (480, 640), preload_data: bool = False):
        """
        Initialize the episodic dataset.
        
        Args:
            dataset_path_list: List containing a single dataset directory path, or list of .h5 file paths (for backward compatibility)
            camera_names: List of camera names to use
            chunk_size: Number of timesteps per sample
            ctrl_space: Control space type ('ee', 'joint', 'other')
            ctrl_type: Control type ('abs', 'rel', 'delta')
            image_size: Target size for image resizing (height, width)
            preload_data: Whether to preload all data into memory
        """
        super(EpisodicDataset).__init__()
        
        # Check if dataset_path_list contains a directory or file paths
        if len(dataset_path_list) == 1:
            # New behavior: dataset_path_list contains a single directory path
            self.dataset_dir = dataset_path_list[0]
            data_cache = os.environ.get('VLASTUDIO_DATA_CACHE_DIR')
            if data_cache and not os.path.isabs(self.dataset_dir):
                # Built-in aliases may use friendly relative locations such as
                # data/sim_transfer_cube_scripted. Keep them out of the caller's
                # source tree when an explicit dataset cache was requested.
                dataset_name = os.path.basename(os.path.normpath(self.dataset_dir))
                self.dataset_dir = os.path.join(data_cache, 'datasets', dataset_name)
            os.makedirs(self.dataset_dir, exist_ok=True)
            self.dataset_path_list = self._find_all_hdf5(self.dataset_dir)
            self.multi_dataset = False
        else:
            # Backward compatibility: dataset_path_list contains file paths
            self.dataset_path_list = dataset_path_list
            self.dataset_dir = os.path.dirname(dataset_path_list[0]) if len(dataset_path_list) > 0 else ""
            self.multi_dataset = True
        
        self.episode_ids = np.arange(len(self.dataset_path_list))
        self.chunk_size = chunk_size
        self.camera_names = camera_names
        self.ctrl_space = ctrl_space  # ['ee', 'joint', 'other']
        self.ctrl_type = ctrl_type  # ['abs', 'rel', 'delta']
        self.image_size = image_size
        self.preload_data = preload_data
        self.freq = -1
        self.max_workers = 8
        self.initialize()
    
    def _find_all_hdf5(self, dataset_dir):
        """
        Find all HDF5 files in the dataset directory.
        
        Args:
            dataset_dir: Directory containing HDF5 files
            
        Returns:
            List of HDF5 file paths
        """
        hdf5_files = []
        for root, dirs, files in os.walk(dataset_dir):
            if 'pointcloud' in root: 
                continue
            for filename in fnmatch.filter(files, '*.hdf5'):
                if 'features' in filename: 
                    continue
                hdf5_files.append(os.path.join(root, filename))
        return hdf5_files
    
    def initialize(self):
        """Initialize the dataset by loading data and computing episode lengths."""
        self.loaded_data = self._load_all_episodes_into_memory() if self.preload_data else None
        self.episode_len = self.get_episode_len()  # Get length of each episode
        self.cumulative_len = np.cumsum(self.episode_len)  # Compute cumulative lengths
        self.max_episode_len = max(self.episode_len)  # Get maximum episode length
    
    def _load_file_into_memory(self, dataset_path):
        """
        Load a single HDF5 file and flatten its content.
        
        Args:
            dataset_path: Path to the HDF5 file
            
        Returns:
            Dictionary containing flattened data from the file
        """
        flattened_data = {}
        with h5py.File(dataset_path, 'r') as f:
            def recursive_load(group, current_path=""):
                for key, item in group.items():
                    full_path = f"{current_path}/{key}" if current_path else f"/{key}"
                    if isinstance(item, h5py.Group):
                        recursive_load(item, full_path)
                    elif isinstance(item, h5py.Dataset):
                        flattened_data[full_path] = item[()]
            recursive_load(f)
        return {dataset_path: flattened_data}
    
    def _load_all_episodes_into_memory(self):
        """
        Load all HDF5 files into memory in parallel.
        
        Returns:
            Dictionary containing all loaded data
        """
        logger.info("Pre-Loading all data into memory...")
        memory_data = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit parallel tasks
            results = executor.map(self._load_file_into_memory, self.dataset_path_list)
            # Collect results
            for result in results:
                memory_data.update(result)
        logger.info("Pre-Loading Done")
        return memory_data

    def get_episode_len(self):
        """
        Get the length of each episode in the dataset.
        
        Returns:
            List of episode lengths
        """
        if self.loaded_data is not None:
            tmp = self.loaded_data[list(self.loaded_data.keys())[0]]
            all_ks = ['/action', '/actions', '/action_ee', '/action_joint', '/state']
            key = None
            for k in all_ks:
                if k in tmp:
                    key = k
                    break
            if key is None: 
                raise NotImplementedError("Failed to get length of episodes")
            all_episode_len = [
                self.loaded_data[pi]['/episode_len'][0].astype(int) 
                if '/episode_len' in self.loaded_data[pi]
                else self.loaded_data[pi][key].shape[0] 
                for pi in self.dataset_path_list
            ]
        else:
            all_episode_len = []
            key = None
            for dataset_path in self.dataset_path_list:
                try:
                    with h5py.File(dataset_path, 'r') as root:
                        if key is None:
                            all_ks = ['/action', '/actions', '/action_ee', '/action_joint', '/state']
                            for k in all_ks:
                                if k in root:
                                    key = k
                                    break
                        elen = root['/episode_len'][0].astype(np.int32) if '/episode_len' in root else root[key][()].shape[0]
                except Exception as e:
                    logger.error(f'Error loading {dataset_path} in get_episode_len')
                    quit()
                all_episode_len.append(elen) 
        return all_episode_len
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return sum(self.episode_len)

    def _locate_transition(self, index):
        """
        Convert sample index to episode index and internal timestep.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (episode_id, start_ts)
        """
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)  # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def extract_from_episode(self, episode_idx, keyname=[]):
        episode_path = self.dataset_path_list[episode_idx]
        feat = self.load_feat_from_episode(episode_path, keyname)
        return feat
    
    @property
    def num_episodes(self):
        return len(self.dataset_path_list)
    
    def get_dataset_dir(self):
        """Get the dataset directory path."""
        return self.dataset_dir

    def get_freq(self):
        """Get the dataset frequency."""
        return self.freq

    def get_language_instruction(self):
        """
        Get language instruction for the dataset.
        
        This method should be overridden by subclasses to provide task-specific instructions.
        
        Returns:
            String containing the language instruction
        """
        raise NotImplementedError("Subclasses must implement get_language_instruction")

    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """
        Load one-step data at start_ts from the episode specified by dataset_path.
        
        This method should be overridden by subclasses to provide dataset-specific loading logic.
        
        Args:
            dataset_path: Path to the dataset file
            start_ts: Starting timestep
            
        Returns:
            Dictionary containing the loaded data
        """
        raise NotImplementedError("Subclasses must implement load_onestep_from_episode")

    def load_feat_from_episode(self, dataset_path, feats=[]):
        """
        Load all steps data from the episode specified by dataset_path.
        
        This method should be overridden by subclasses to provide dataset-specific loading logic.
        
        Args:
            dataset_path: Path to the dataset file
            feats: List of features to load
            
        Returns:
            Dictionary containing the loaded data
        """
        raise NotImplementedError("Subclasses must implement load_feat_from_episode")

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing the sample data
        """
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id] 
        episode_len = self.episode_len[episode_id] 
        data_dict = self.load_onestep_from_episode(dataset_path, start_ts) 
        action, image_dict, state, raw_lang = data_dict['action'], data_dict['image'], data_dict['state'], data_dict['language_instruction']
        reasoning = data_dict.get('reasoning', '')
        padded_action = np.zeros((self.chunk_size, action.shape[1]), dtype=np.float32) 
        padded_action[:action.shape[0]] = action
        is_pad = np.zeros(self.chunk_size) 
        is_pad[action.shape[0]:] = 1
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        if len(self.camera_names)>0:
            all_cam_images = np.stack(all_cam_images, axis=0)  # Stack images into array
        else:
            all_cam_images = None
        
        # Construct observations, convert arrays to tensors
        if all_cam_images is not None:
            # Safety check: ensure images are uint8 with values in [0, 255]
            all_cam_images = ensure_uint8_image(all_cam_images)
            
            image_data = torch.from_numpy(all_cam_images)
            image_data = torch.einsum('k h w c -> k c h w', image_data)  # Swap image channels
        else:
            image_data = None
        state_data = torch.from_numpy(state).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        sample = {
            'image': image_data,
            'state': state_data,
            'action': action_data,
            'is_pad': is_pad,
            'raw_lang': raw_lang,
            'reasoning': reasoning,
            'timestamp': start_ts,  
            'episode_id': episode_id,
            '__index__': index
        }  # Construct sample dict
        assert raw_lang is not None, ""
        return sample
