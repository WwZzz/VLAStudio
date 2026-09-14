import numpy as np
import torch
from vlastudio.benchmark.utils import resize_with_pad
class OctoDataProcessor:
    def __init__(self, text_processor, use_wrist=False, image_size=(256,256), wrist_image_size=(128,128)):
        self.text_processor = text_processor
        self.use_wrist = use_wrist
        self.image_size = image_size
        self.wrist_image_size = wrist_image_size

    def _np2pt(self, data, dtype=None):
        """Transform dictionary with numpy arrays to torch tensors.
        Trnsform images to channel-first format: NHWC -> NCHW, NTHWC -> NTCHW
        """
        if isinstance(data, dict):
            return {key: self._np2pt(val) for key, val in data.items()}
        if isinstance(data, torch.Tensor):
            return data
        t = torch.from_numpy(data)
        return t
    
    def _pt2dev(self, data, dev):
        """Transform dictionary with numpy arrays to torch tensors.
        Trnsform images to channel-first format: NHWC -> NCHW, NTHWC -> NTCHW
        """
        if isinstance(data, dict):
            return {key: self._pt2dev(val, dev) for key, val in data.items()}
        if isinstance(data, torch.Tensor):
            return data.to(dev)
        else:
            return data
    
    def __call__(self, sample):
        data_dict = {}
        # task
        if 'raw_lang' in sample:
            task = {}
            task["language_instruction"] = self.text_processor.encode([s.decode("utf-8") for s in np.array([sample['raw_lang'].encode()])])
            task["pad_mask_dict"] = {"language_instruction": np.array([True])}
            task["language_instruction"]['input_ids'] = task["language_instruction"]['input_ids'][0]
            task["language_instruction"]['attention_mask'] = task["language_instruction"]['attention_mask'][0]
            data_dict['task'] = task 
        # observation
        obs = {}
        obs['proprio'] = sample['state'][np.newaxis,:] # T*D
        obs['timestep'] = np.array([sample['timestamp']])
        obs['timestep_pad_mask'] = np.array([True])
        if 'action' in sample:
            obs['task_completed'] = np.array([False for _ in range(sample['action'].shape[0])])[np.newaxis,:] # (1 ,chunk_size)
        sample['image'] = resize_with_pad(sample['image'], self.image_size[0], self.image_size[1])
        obs['image_primary'] = sample['image'][0:1,:] # k c h w -> k h w c
        obs['pad_mask_dict'] = {
            'image_primary': np.array([True]),
            'proprio': np.array([True]),
            'timestep': np.array([True]),
        }
        if self.use_wrist:
            assert sample['image'].shape[0]>1
            obs['image_wrist'] = resize_with_pad(sample['image'][1:2,:], self.wrist_image_size[0], self.wrist_image_size[1])
            obs['pad_mask_dict']['image_wrist'] = np.array([True])
        data_dict['observation'] = obs
        # action
        if 'action' in sample:
            data_dict['action'] = sample['action'][np.newaxis,:]
            is_pad = sample['is_pad'][:, None]                 # (k,1)
            mask = 1.0 - np.tile(is_pad, (1, data_dict['action'].shape[-1]))  # (k,d)
            mask = mask.astype(bool)                           # or np.bool_
            mask = mask[np.newaxis, :]        
            data_dict['action_pad_mask'] = mask
        data_dict = self._np2pt(data_dict)
        return data_dict

class OctoCollator:
    def recursive_stack(self, items):
        if isinstance(items[0], dict):
            return {key: self.recursive_stack([item[key] for item in items]) for key in items[0].keys()}
        else:
            return torch.stack(items)
        
    def __call__(self, instances):
        # recursively stack tensors in a dictionary for each instance
        batch = self.recursive_stack(instances)
        if 'task' in batch:
            batch["task"]["pad_mask_dict"]['language_instruction'] = batch["task"]["pad_mask_dict"]['language_instruction'][:, 0]
        return batch        
        
        
        
