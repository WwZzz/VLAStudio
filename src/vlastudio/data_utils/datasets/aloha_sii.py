"""
AlohaSIIDataset implementation.

This module contains the AlohaSIIDataset class for loading ALOHA SII (Simulated Interactive Intelligence) data.
"""

import numpy as np
import cv2
import h5py
from .base import EpisodicDataset


class AlohaSIIDataset(EpisodicDataset):
    """
    Dataset class for ALOHA SII data.
    
    This class handles loading and processing of ALOHA SII datasets,
    which include multi-camera data with specific image processing requirements.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the ALOHA SII dataset with 25Hz frequency."""
        super().__init__(*args, **kwargs)
        self.freq = 25  # 25Hz 
    
    def get_freq(self):
        """Get the dataset frequency."""
        return self.freq
    
    def get_language_instruction(self):
        """
        Get language instruction based on the dataset directory.
        
        Returns:
            String containing the task instruction
        """
        if 'red_cube' in self.get_dataset_dir():
            return "Pick up the red cube and put it on the top of the blue cup"
        return ""
        
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
        image_dict = dict(
            primary=cv2.resize(cv2.imdecode(np.frombuffer(root[f'/observations/images/cam_high'][start_ts], np.uint8), cv2.IMREAD_COLOR), self.image_size),
            wrist_left=cv2.resize(cv2.imdecode(np.frombuffer(root[f'/observations/images/cam_left_wrist'][start_ts], np.uint8), cv2.IMREAD_COLOR), self.image_size),
            wrist_right=cv2.resize(cv2.imdecode(np.frombuffer(root[f'/observations/images/cam_right_wrist'][start_ts], np.uint8), cv2.IMREAD_COLOR), self.image_size),
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
                data_dict['action'] = root[f'/action'][()]  # Load corresponding action data based on control type
                if self.ctrl_type == 'delta': 
                    data_dict['action'] = data_dict['action'] - data_dict.get('state', root[f'/observations/qpos'][()])
                elif self.ctrl_type == 'rel':
                    raise NotImplementedError("relative action was not implemented")
            image_dict = dict()
            if 'image_primary' in feats or 'image' in feats or len(feats) == 0:  # Load images
                img_bytes = root[f'/observations/images/cam_high'][()]
                image_dict.update(
                    dict(
                        primary=np.stack([cv2.resize(cv2.imdecode(np.frombuffer(img_byte, np.uint8), cv2.IMREAD_COLOR), self.image_size) for img_byte in img_bytes]),
                    )
                )
            if 'image_wrist' in feats or 'image' in feats or len(feats) == 0:
                left_bytes = root[f'/observations/images/cam_left_wrist'][()]
                image_dict.update(
                    dict(
                        wrist_left=np.stack([cv2.resize(cv2.imdecode(np.frombuffer(lb, np.uint8), cv2.IMREAD_COLOR), self.image_size) for lb in left_bytes]),
                    )
                )
                del left_bytes
                right_bytes = root[f'/observations/images/cam_right_wrist'][()]
                image_dict.update(
                    dict(
                        wrist_right=np.stack([cv2.resize(cv2.imdecode(np.frombuffer(rb, np.uint8), cv2.IMREAD_COLOR), self.image_size) for rb in right_bytes]),
                    )
                )
                del right_bytes
            if len(image_dict) > 0: 
                data_dict['image'] = image_dict
        return data_dict
    
    def get_joint_limits(self):
        """
        Get joint limits for the ALOHA SII robot.
        
        Returns:
            Dictionary containing upper and lower joint limits
        """
        lower = np.array([-2.687807, 0.0, -3.054326, -1.850049, -1.308997, -1.745329])
        upper = np.array([2.687807, 3.403392, 0.0, 1.850049, 1.308997, 1.745329])
        return {'upper': upper, 'lower': lower}
