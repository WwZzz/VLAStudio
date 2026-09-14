"""
AlohaSIIv2Dataset implementation.

This module contains the AlohaSIIv2Dataset class for loading ALOHA SII v2 data.
"""

import numpy as np
import cv2
import h5py
from .base import EpisodicDataset


class AlohaSIIv2Dataset(EpisodicDataset):
    """
    Dataset class for ALOHA SII v2 data.
    
    This class handles loading and processing of ALOHA SII v2 datasets,
    which include multi-object manipulation tasks.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the ALOHA SII v2 dataset with 25Hz frequency."""
        super().__init__(*args, **kwargs)
        self.freq = 25  # 25Hz 
    
    def get_freq(self):
        """Get the dataset frequency."""
        return self.freq
    
    def get_language_instruction(self):
        """
        Get language instruction for the multi-object task.
        
        Returns:
            String containing the task instruction
        """
        return "Put the green duck to the blue bowl and the orange square to the pink plate"
        
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
        action = root[f'/actions'][start_ts:start_ts+self.chunk_size]
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
        image_dict = dict(
            primary=cv2.resize(root[f'/observations/image/primary'][start_ts], self.image_size),
            wrist_left=cv2.resize(root[f'/observations/image/wrist_left'][start_ts], self.image_size),
            wrist_right=cv2.resize(root[f'/observations/image/wrist_right'][start_ts], self.image_size),
        )
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
            if 'qpos' in feats or len(feats) == 0: 
                data_dict['qpos'] = root[f'/observations/qpos'][()]
            if 'qvel' in feats or len(feats) == 0: 
                data_dict['qvel'] = root[f'/observations/qvel'][()]
            if 'action' in feats or len(feats) == 0:  # Load action
                data_dict['action'] = root[f'/actions'][()]  # Load corresponding action data based on control type
                if self.ctrl_type == 'delta': 
                    data_dict['action'] = data_dict['action'] - data_dict.get('state', root[f'/observations/qpos'][()])
                elif self.ctrl_type == 'rel':
                    raise NotImplementedError("relative action was not implemented")
            image_dict = dict()
            if 'image_primary' in feats or 'image' in feats or len(feats) == 0:  # Load images
                img_bytes = root[f'/observations/image/primary'][()]
                image_dict.update(
                    dict(
                        primary=np.stack([cv2.resize(img_byte, self.image_size) for img_byte in img_bytes]),
                    )
                )
            if 'image_wrist' in feats or 'image' in feats or len(feats) == 0:
                left_bytes = root[f'/observations/image/wrist_left'][()]
                image_dict.update(
                    dict(
                        wrist_left=np.stack([cv2.resize(lb, self.image_size) for lb in left_bytes]),
                    )
                )
                del left_bytes
                right_bytes = root[f'/observations/image/wrist_right'][()]
                image_dict.update(
                    dict(
                        wrist_right=np.stack([cv2.resize(rb, self.image_size) for rb in right_bytes]),
                    )
                )
                del right_bytes
            if len(image_dict) > 0: 
                data_dict['image'] = image_dict
        return data_dict
