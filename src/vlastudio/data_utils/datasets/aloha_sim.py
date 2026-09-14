"""
AlohaSimDataset implementation.

This module contains the AlohaSimDataset class for loading ALOHA simulation data.
"""

import numpy as np
import cv2
import h5py
import os
import warnings
from .base import EpisodicDataset


class AlohaSimDataset(EpisodicDataset):
    """
    Dataset class for ALOHA simulation data.
    
    This class handles loading and processing of ALOHA simulation datasets,
    including transfer and insertion tasks.
    """ 
    def _download_dataset_if_needed(self):
        """
        Download dataset from HuggingFace if dataset_dir is empty or doesn't exist.
        
        Returns:
            Updated dataset_dir path
        """
        num_episodes = len(self._find_all_hdf5(self.dataset_dir))
        if num_episodes >= 50:
            return self.dataset_dir
        task_name = self.dataset_dir
        num_cameras = len(self.camera_names)
        # Determine which HuggingFace repo to download from based on task name
        repo_id = None
        if 'transfer' in task_name.lower() and 'scripted' in task_name.lower():
            if num_cameras==1:
                repo_id = "cadene/aloha_sim_transfer_cube_scripted_raw"
            else:
                repo_id = "WWZzz/sim_transfer_cube_scripted"
        elif 'transfer' in task_name.lower() and 'human' in task_name.lower():
            repo_id = "cadene/aloha_sim_transfer_cube_human_raw"
        elif 'insertion' in task_name.lower() and 'scripted' in task_name.lower():
            if num_cameras==1:
                repo_id = "cadene/aloha_sim_insertion_scripted_raw"
            else:
                repo_id = "WWZzz/sim_insertion_scripted"
        elif 'insertion' in task_name.lower() and 'human' in task_name.lower():
            repo_id = "cadene/aloha_sim_insertion_human_raw"
        else:
            return self.dataset_dir
        
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download datasets. "
                "Please install it with: pip install huggingface_hub"
            )
        if not self.dataset_dir:
            self.dataset_dir = os.environ.get('ILSTD_CACHE', os.path.expanduser('~/.cache/ilstd'))
            os.makedirs(self.dataset_dir, exist_ok=True)
        # Download the dataset using HuggingFace's default cache
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=self.dataset_dir,
            local_dir_use_symlinks=False
        )
        # Update dataset_dir with the downloaded path
        if not self.dataset_dir:
            self.dataset_dir = downloaded_path
        return self.dataset_dir
    
    def initialize(self):
        """Initialize the dataset with automatic download if needed"""
        # Download dataset if needed
        self.dataset_dir = self._download_dataset_if_needed()
        # Re-scan for HDF5 files after potential download
        if hasattr(self, 'dataset_path_list') and len(self.dataset_path_list) == 0:
            self.dataset_path_list = self._find_all_hdf5(self.dataset_dir)
            self.episode_ids = np.arange(len(self.dataset_path_list))
        super().initialize()

    def get_language_instruction(self):
        """
        Get language instruction based on the task name.
        
        Returns:
            String containing the task instruction
        """
        task_name = os.path.split(self.get_dataset_dir())[-1]
        if 'transfer' in task_name:
            return 'Transfer the red cube from the right arm to the left arm.'
        elif 'insert' in task_name:
            return 'Insert the red peg into the blue socket.'
        else:
            raise ValueError("Unknown task")
        
    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """
        Load one-step data at start_ts from the episode specified by dataset_path.
        
        Args:
            dataset_path: Path to the dataset file
            start_ts: Starting timestep
            
        Returns:
            Dictionary containing the loaded data
        """
        root = self.loaded_data[dataset_path] if self.loaded_data is not None else h5py.File(dataset_path, 'r')
        # Load text
        raw_lang = self.get_language_instruction()
        # Load action & state
        action = root[f'/action'][start_ts:start_ts+self.chunk_size]
        # Load corresponding action data based on control type
        if self.ctrl_type == 'abs':
            state = root[f'/observations/qpos'][start_ts]
        elif self.ctrl_type == 'delta':
            states = root[f'/observations/qpos'][start_ts:start_ts+self.chunk_size]
            action = action - states
            state = states[0]
        elif self.ctrl_type == 'rel':
            raise NotImplementedError("relative action was not implemented")
        # Load images
        image_dict = dict(primary=root['/observations/images/top'][start_ts])
        if 'left_wrist' in self.camera_names:
            image_dict.update(
                dict(wrist_left=root['/observations/images/left_wrist'][start_ts])
            )
        if 'right_wrist' in self.camera_names:
            image_dict.update(
                dict(wrist_right=root['/observations/images/right_wrist'][start_ts])
            )
        if self.image_size is not None:
            for k in image_dict:
                if image_dict[k].shape[0]!=self.image_size[0] or image_dict[k].shape[1]!=self.image_size[1]:
                    image_dict[k] = cv2.resize(image_dict[k], (self.image_size[1], self.image_size[0])) # cv2.resize receive (W, H) and numpy.ndarray is (H, W, C)
        # Load reasoning information
        reasoning = ""
        if self.loaded_data is None: 
            root.close()
        return {
            'action': action,
            'image': image_dict,
            'state': state,
            'language_instruction': raw_lang,
            'reasoning': reasoning,
        }
        
    def load_feat_from_episode(self, dataset_path, feats=[]):
        """
        Load all steps data from the episode specified by dataset_path.
        
        Args:
            dataset_path: Path to the dataset file
            feats: List of features to load
            
        Returns:
            Dictionary containing the loaded data
        """
        data_dict = {}
        if isinstance(feats, str): 
            feats = [feats]
        with h5py.File(dataset_path, 'r') as root:
            if 'language_instruction' in feats or len(feats) == 0: 
                data_dict['language_instruction'] = self.get_language_instruction()  # Load text
            if 'state' in feats or len(feats) == 0: 
                data_dict['state'] = root[f'/observations/qpos'][()]  # Load state
            if 'action' in feats or len(feats) == 0:  # Load action
                data_dict['action'] = root[f'/action'][()]  # Load corresponding action data based on control type
                if self.ctrl_type == 'delta': 
                    data_dict['action'] = data_dict['action'] - data_dict.get('state', root[f'/observations/qpos'][()])
                elif self.ctrl_type == 'rel':
                    raise NotImplementedError("relative action was not implemented")
            if 'image' in feats or len(feats) == 0:  # Load images
                all_images = root[f'/observations/images/top'][()]
                image_dict = dict(primary=[cv2.resize(all_images[i], (self.image_size[1], self.image_size[0])) for i in range(all_images.shape[0])])
                data_dict['image'] = image_dict
            if 'image' in feats or 'image_wrist' in feats or len(feats) == 0:
                all_left_images = root[f'/observations/images/left_wrist'][()]
                left_dict = dict(wrist_left=[cv2.resize(all_left_images[i], (self.image_size[1], self.image_size[0])) for i in range(all_left_images.shape[0])])
                all_right_images = root[f'/observations/images/right_wrist'][()]
                right_dict = dict(wrist_right=[cv2.resize(all_right_images[i], (self.image_size[1], self.image_size[0])) for i in range(all_right_images.shape[0])])
                data_dict['image'].update(left_dict)
                data_dict['image'].update(right_dict)
        return data_dict
