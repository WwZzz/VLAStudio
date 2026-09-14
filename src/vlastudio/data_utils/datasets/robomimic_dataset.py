"""
RobomimicDataset implementation.

This module contains the RobomimicDataset class for loading RoboMimic datasets.
"""

import numpy as np
import os
from collections import OrderedDict
from vlastudio.data_utils.pose_utils import quat2axisangle
from vlastudio.data_utils.datasets.base import EpisodicDataset
from typing import List, Union

MODALITIES = {
    "obs": {
        "low_dim": [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        "rgb": [],
        "depth": [],
        "scan": []
    },
    "goal": {
        "low_dim": [],
        "rgb": [],
        "depth": [],
        "scan": []
    }
}

class RobomimicDataset(EpisodicDataset):
    """
    Dataset class for RoboMimic data.
    
    This class handles loading and processing of RoboMimic datasets,
    which include various manipulation tasks from the RoboMimic benchmark.
    """
    def __init__(self, *args, use_low_dim: bool = False, use_img: bool = True, use_wrist_img: bool = False, filter_by_attribute:str|list[str]='train', **kwargs):
        self.use_low_dim = use_low_dim      
        self.use_img = use_img
        self.use_wrist_img = use_wrist_img
        self.filter_by_attribute = filter_by_attribute
        self.obs_key_shapes = None
        self.task_name = None
        self.low_dim_key = []
        self.img_primary_key = []
        self.img_wrist_key = []
        super().__init__(*args, **kwargs)   
        
    def initialize(self):
        """Initialize the RoboMimic dataset with sequence datasets."""
        from robomimic.utils.dataset import SequenceDataset
        import robomimic.utils.obs_utils as ObsUtils
        if self.use_low_dim: MODALITIES["obs"]["low_dim"] = ["object"] + MODALITIES["obs"]["low_dim"]
        if self.use_img: MODALITIES["obs"]["rgb"].append("agentview_image")
        if self.use_wrist_img: MODALITIES["obs"]["rgb"].append("robot0_eye_in_hand_image")
        ObsUtils.initialize_obs_utils_with_obs_specs(MODALITIES)
        keyname = 'image' if self.use_img or self.use_wrist_img else 'low_dim'
        if self.multi_dataset:
            all_h5 = sum([self._find_all_hdf5(di) for di in self.dataset_path_list], [])
            self.dataset_path_list = [di for di in all_h5 if keyname in di]
            if isinstance(self.filter_by_attribute, str):
                self.filter_by_attribute = [self.filter_by_attribute for di in self.dataset_path_list if keyname in di]
        else:
            if isinstance(self.filter_by_attribute, str):
                self.filter_by_attribute = [self.filter_by_attribute]
        self._datasets = [SequenceDataset(**self.create_config(di, fbai)) for di,fbai in zip(self.dataset_path_list, self.filter_by_attribute) if keyname in di]
        self._languages = [self.get_raw_lang(di) for di in self.dataset_path_list if 'image' in di or 'low_dim' in di]
        self._dataset_dir = os.path.dirname(self.dataset_path_list[0])
        self.episode_ids = np.arange(sum(d.n_demos for d in self._datasets))
        self.dataset_path_list = sum([[f"{idx}:{ep}" for ep in di.demos] for idx, di in enumerate(self._datasets)], [])
        self.episode_len = self.get_episode_len()  # Get length of each episode
        self.cumulative_len = np.cumsum(self.episode_len)  # Compute cumulative lengths
        self.max_episode_len = max(self.episode_len)  # Get maximum episode length
        self.state_keys = self.get_state_keys(self.task_name)
        self.camera_names = self.img_primary_key + self.img_wrist_key

    def get_dataset_dir(self):
        """Get the dataset directory path."""
        return self._dataset_dir
    
    def get_raw_lang(self, data_path):
        """
        Get raw language instruction based on the data path.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            String containing the language instruction
        """
        # Initialize raw language
        from vlastudio.benchmark.robomimic import ALL_ENV_LANGUAGES
        if 'can' in data_path: 
            return ALL_ENV_LANGUAGES['PickPlaceCan']
        elif 'lift' in data_path: 
            return ALL_ENV_LANGUAGES['Lift']
        elif 'tool_hang' in data_path: 
            return ALL_ENV_LANGUAGES['ToolHang']
        elif 'square' in data_path: 
            return ALL_ENV_LANGUAGES['NutAssemblySquare']
        elif 'transport' in data_path: 
            return ALL_ENV_LANGUAGES['TwoArmTransport']
        else:
            raise KeyError("Unknown language")
    
    def get_state_keys(self, task_name):
        state_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
        if task_name=='transport':
            state_keys = state_keys + ['robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos']
        if self.use_low_dim:
            state_keys = ['object'] + state_keys
        return state_keys
    
    def get_obs_key_shapes(self, data_path):
        obs_key_shapes_list = [
            ('robot0_eef_pos', [3]), 
            ('robot0_eef_quat', [4]), 
            ('robot0_gripper_qpos', [2])
        ]
        # lift: ('object', [10]),  can: ('object', [14]),  square: ('object', [14]), tool_hang: ('object', [44]), transport: ('object', [41])
        all_tasks = ['lift', 'can', 'square', 'tool_hang', 'transport']
        task_name = None
        for t in all_tasks:
            if t in data_path:
                task_name = t
                break
        if task_name is None: raise KeyError("Unknown task for obs_key_shapes")
        self.task_name = task_name
        if task_name=='transport':
            obs_key_shapes_list = obs_key_shapes_list + [('robot1_eef_pos', [3]), ('robot1_eef_quat', [4]), ('robot1_gripper_qpos', [2])]
            
        # Init object obs
        all_obj_shapes = {'lift': [10], 'can': [14], 'square': [14], 'tool_hang': [44], 'transport': [41]}
        if self.use_low_dim:
            obs_key_shapes_list = [('object', all_obj_shapes[task_name])] + obs_key_shapes_list
            self.low_dim_key = ['object']

        # Init image obs
        if self.use_img:
            if task_name in ['lift', 'can', 'square']:
                obs_key_shapes_list = obs_key_shapes_list +  [('agentview_image', [3, 84, 84])]
                self.img_primary_key = ['agentview_image']
            elif task_name=='tool_hang':
                obs_key_shapes_list = obs_key_shapes_list +  [('sideview_image', [3, 84, 84]),]
                self.img_primary_key = ['sideview_image']
            else:
                obs_key_shapes_list = obs_key_shapes_list +  [('shouldercamera0_image', [3, 84, 84]), ('shouldercamera1_image', [3, 84, 84])]
                self.img_primary_key = ['shouldercamera0_image', 'shouldercamera1_image']

            
        # Init wrist image obs
        if self.use_wrist_img:
            obs_key_shapes_list = obs_key_shapes_list + [('robot0_eye_in_hand_image', [3, 84, 84])]
            self.img_wrist_key = ['robot0_eye_in_hand_image']
            if task_name=='toolhang':
                obs_key_shapes_list = obs_key_shapes_list + [('robot1_eye_in_hand_image', [3, 84, 84])]
                self.img_wrist_key = ['robot0_eye_in_hand_image', 'robot1_eye_in_hand_image']
        return OrderedDict(obs_key_shapes_list)
    
    def create_config(self, data_path, filter_by_attribute='train', seq_length=1):
        """
        Create configuration for the RoboMimic sequence dataset.
        
        Args:
            data_path: Path to the data file
            filter_by_attribute: Attribute to filter by
            seq_length: Sequence length
            
        Returns:
            Dictionary containing the configuration
        """
        if self.obs_key_shapes is None:
            self.obs_key_shapes = self.get_obs_key_shapes(data_path)
        return {
            'hdf5_path': data_path,
            'obs_keys': list(self.obs_key_shapes.keys()),
            'dataset_keys': ('actions', 'rewards', 'dones'),
            'load_next_obs': False,
            'frame_stack': 1,
            'seq_length': seq_length,
            'pad_frame_stack': True,
            'pad_seq_length': True,
            'get_pad_mask': False,
            'goal_mode': None,
            'hdf5_cache_mode': 'all' if self.preload_data else 'low_dim',
            'hdf5_use_swmr': True,
            'hdf5_normalize_obs': False,
            'filter_by_attribute': filter_by_attribute,
        }
    
    def get_episode_len(self):
        """
        Get the length of each episode in the dataset.
        
        Returns:
            List of episode lengths
        """
        all_ep_lengths = []
        for dataset_path in self.dataset_path_list:
            idx, ep = dataset_path.split(':')
            ep_length = self._datasets[eval(idx)]._demo_id_to_demo_length[ep]
            all_ep_lengths.append(ep_length)
        return all_ep_lengths
        
    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """
        Load one-step data at start_ts from the episode specified by dataset_path.
        
        Args:
            dataset_path: Path to the dataset file
            start_ts: Starting timestep
            
        Returns:
            Dictionary containing the loaded data
        """
        dataset_idx, ep = dataset_path.split(':')
        dataset = self._datasets[eval(dataset_idx)]
        demo_start_index = dataset._demo_id_to_start_indices[ep]  # Demo start global index
        global_index = demo_start_index + start_ts
        data = dataset[global_index]
        # Load language 
        raw_lang = self._languages[eval(dataset_idx)]
        # Load state
        # 1. obj, 2. robot_k(xyz+axisangle+gripper)
        all_states = [data['obs'][k] if 'quat' not in k else quat2axisangle(data['obs'][k]) for k in self.state_keys]
        state = np.concatenate(all_states, axis=1)[0]
        # Load action
        if dataset.hdf5_cache_mode == 'all':
            demo_length = dataset._demo_id_to_demo_length[ep]
            chunk_size = min(self.chunk_size, demo_length - start_ts)
            action = np.concatenate([data['actions']] + [dataset[i]['actions'] for i in range(global_index + 1, global_index + chunk_size)], axis=0)
        else:
            action = data['actions'] if self.chunk_size == 1 else dataset.get_dataset_for_ep(ep=ep, key="actions")[start_ts:start_ts + self.chunk_size]
        # Load image
        image_dict = dict()
        if self.use_img:
            for i,k in enumerate(self.img_primary_key):
                image_dict[k] = data['obs'][k][0]
        if self.use_wrist_img:
            for i,k in enumerate(self.img_wrist_key):
                image_dict[k] = data['obs'][k][0]
        return dict(
            action=action,
            state=state,
            image=image_dict,
            language_instruction=raw_lang,
            reasoning="",
        )
       
    def load_feat_from_episode(self, dataset_path, feats=[]):
        """
        Load all steps data from the episode specified by dataset_path.
        
        Args:
            dataset_path: Path to the dataset file
            feats: List of features to load
            
        Returns:
            Dictionary containing the loaded data
        """
        dataset_idx, ep = dataset_path.split(':')
        dataset = self._datasets[eval(dataset_idx)]
        if dataset.hdf5_cache_mode == 'all':
            demo_start = dataset._demo_id_to_start_indices[ep]
            demo_length = dataset._demo_id_to_demo_length[ep]
            demo_end = demo_start + demo_length
            trajectory = {
                "obs": [dataset.getitem_cache[i]["obs"] for i in range(demo_start, demo_end)],
                "actions": [dataset.getitem_cache[i]["actions"] for i in range(demo_start, demo_end)],
            }
            trajectory["obs"] = {k: np.concatenate([obs[k] for obs in trajectory["obs"]], axis=0) for k in trajectory["obs"][0]}
            trajectory["actions"] = np.concatenate(trajectory["actions"], axis=0)
            trajectory_data = trajectory
        else:
            demo_index = dataset.demos.index(ep)
            trajectory_data = dataset.get_trajectory_at_index(demo_index)
        data_dict = {}
        if isinstance(feats, str): 
            feats = [feats]
        if 'language_instruction' in feats or len(feats) == 0: 
            data_dict['language_instruction'] = self._languages[dataset_idx]
        if 'state' in feats or len(feats) == 0: 
            all_states = [trajectory_data['obs'][k] if 'quat' not in k else quat2axisangle(trajectory_data['obs'][k]) for k in self.state_keys]
            state = np.concatenate(all_states, axis=1)[0]
            data_dict['state'] = state
        if 'action' in feats or len(feats) == 0:  # Load action
            action = trajectory_data['actions']
            data_dict['action'] = action
        if 'image' in feats or len(feats) == 0:  # Load images
            image_dict = dict()
            if self.use_img:
                for i,k in enumerate(self.img_primary_key):
                    image_dict[k] = data['obs'][k][0]
            if self.use_wrist_img:
                for i,k in enumerate(self.img_wrist_key):
                    image_dict[k] = data['obs'][k][0]
            data_dict['image'] = image_dict
        return data_dict

if __name__=='__main__':
    d1 = RobomimicDataset(['/inspire/hdd/project/robot-action/public/data/robomimic/transport/ph', ], use_low_dim=True, use_img=False, image_size=(256, 256))
    d2 = RobomimicDataset(['/inspire/hdd/project/robot-action/public/data/robomimic/transport/mh', ], use_low_dim=True, use_img=False, image_size=(256, 256))
    d12 = RobomimicDataset(['/inspire/hdd/project/robot-action/public/data/robomimic/transport/ph', '/inspire/hdd/project/robot-action/public/data/robomimic/transport/mh'], use_low_dim=True, use_img=False, image_size=(256, 256))
    print(f"Len ph: {len(d1)}, Len mh: {len(d2)}, Len mix: {len(d12)}")
    print('ok')