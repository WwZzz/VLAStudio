from pprint import pprint
import torch.utils.data as tud
import torch
from huggingface_hub import HfApi
from typing import List
import lerobot
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
import warnings
from vlastudio.benchmark.utils import resize_with_pad
from .lerobot_wrapper import WrappedLerobotDataset
from vlastudio.data_utils.utils import ensure_uint8_image

class LerobotLIBERO(WrappedLerobotDataset):
    def extract_from_episode(self, episode_idx, keyname=[]):
        dataset_idx = np.argmax(self.cumulative_num_episodes > episode_idx)
        inner_episode_idx = episode_idx - self.per_dataset_episode_start[dataset_idx]
        ds_meta = self.dataset_metas[dataset_idx]
        all_features = ds_meta.features
        preserved_keys = []
        if 'state' in keyname:
            preserved_keys.append('observation.state')
        if 'action' in keyname:
            preserved_keys.append('action')
        if 'image' in keyname or 'images' in keyname:
            preserved_keys.extend(ds_meta.camera_keys)
            ignore_image = False
        else:
            ignore_image = all([ckey not in keyname for ckey in ds_meta.camera_keys])
        ignore_keys = [feat for feat in all_features if feat not in preserved_keys and feat not in keyname]
        subdata = LeRobotDataset(
            self.dataset_path_list[dataset_idx], 
            episodes=[inner_episode_idx],
        )
        if ignore_image:
            for k,v in subdata.meta.features.items():
                if v['dtype']=='video': subdata.meta.info['features'][k]['dtype'] = 'hidden'
        extracted_feats = [{k:s[k].numpy() for k in preserved_keys} for s in subdata]
        if ignore_image:
            for k,v in subdata.meta.features.items():
                if v['dtype']=='hidden': subdata.meta.info['features'][k]['dtype'] = 'video'
        res_dict = {k: np.stack([efeat[k] for efeat in  extracted_feats]) if isinstance(extracted_feats[0][k], np.ndarray) else [efeat[k] for efeat in  extracted_feats] for k in preserved_keys}
        return res_dict
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing the sample data
        """
        # find dataset_id by index: start_index for the target dataset by the num_frames of each dataset
        # find sample_id in dataset: index-start_index
        
        dataset_idx, start_ts = self._locate_dataset_for_transition(index)
        sample = self.datasets[dataset_idx][start_ts]
        data_dict = {}
        episode_id = self.per_dataset_episode_start[dataset_idx] + sample['episode_index'].item()
        raw_lang = sample['task']
        action = sample['action']
        state = sample['observation.state']
        timestamp = sample['frame_index'].item()
        is_pad = sample['action_is_pad']
        # process image
        cam_keys = self.datasets[dataset_idx].meta.camera_keys if len(self.camera_names)==0 else self.camera_names
        if self.image_size is not None:
            images = torch.cat([resize_with_pad(sample[cam_key].unsqueeze(0), height=self.image_size[1], width=self.image_size[0]) for cam_key in cam_keys], dim=0)
        else:
            images = torch.stack([sample[cam_key] for cam_key in cam_keys])
        
        # Safety check: ensure images are uint8 with values in [0, 255]
        images = ensure_uint8_image(images)
        
        data_dict = {
            'image': images,
            'state': state,
            'action': action,
            'is_pad': is_pad,
            'raw_lang': raw_lang,
            'reasoning': {},
            'timestamp': timestamp,  
            'episode_id': episode_id,
        }  
        return data_dict
