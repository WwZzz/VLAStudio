"""
KochDataset implementation.
"""

import numpy as np
import cv2
import h5py
from .base import EpisodicDataset


class KochDataset(EpisodicDataset):
    """
    Dataset class for ALOHA SII v2 data.
    
    This class handles loading and processing of ALOHA SII v2 datasets,
    which include multi-object manipulation tasks.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = 13 
        self.qpos_keys = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

    def get_freq(self):
        """Get the dataset frequency."""
        return self.freq
    
    def get_language_instruction(self):
        """
        Returns:
            String containing the task instruction
        """
        return "Put the red cap to the red cup"
        
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
        qpos = np.stack([root[f'/observations/{qkey}.pos'][start_ts:start_ts+self.chunk_size] for qkey in self.qpos_keys]).T
        action = qpos
        # Load corresponding action data based on control type
        if self.ctrl_type == 'abs':
            state = qpos[0]
        elif self.ctrl_type == 'delta':
            action = action - action[0]
            state = qpos[0]
        elif self.ctrl_type == 'rel':
            raise NotImplementedError("relative action was not implemented")
        # Load images
        image_dict = dict(
            primary=cv2.resize(root[f'/observations/front_camera'][start_ts], self.image_size),
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
            qpos = np.stack([root[f'/observations/{qkey}.pos'][()] for qkey in self.qpos_keys]).T
            if 'language_instruction' in feats or len(feats) == 0: 
                data_dict['language_instruction'] = self.get_language_instruction()  # Load text
            if 'state' in feats or len(feats) == 0: 
                data_dict['state'] = qpos
            if 'qpos' in feats or len(feats) == 0: 
                data_dict['qpos'] = qpos
            if 'qvel' in feats or len(feats) == 0: 
                data_dict['qvel'] = np.zeros_like(qpos)
            if 'action' in feats or len(feats) == 0:  # Load action
                data_dict['action'] = qpos  
                if self.ctrl_type == 'delta': 
                    data_dict['action'] = np.concatenate([qpos[1:,:]-qpos[:-1,:], np.zeros_like(qpos[:1, :])], axis=0)
                elif self.ctrl_type == 'rel':
                    raise NotImplementedError("relative action was not implemented")
            image_dict = dict()
            if 'image_primary' in feats or 'image' in feats or len(feats) == 0:  # Load images
                img_bytes = root[f'/observations/front_camera'][()]
                image_dict.update(
                    dict(
                        primary=np.stack([cv2.resize(img_byte, self.image_size) for img_byte in img_bytes]),
                    )
                )
            if len(image_dict) > 0: 
                data_dict['image'] = image_dict
        return data_dict
