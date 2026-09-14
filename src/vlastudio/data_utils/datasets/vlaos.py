import os
from torch.utils.data import IterableDataset
import tensorflow_datasets as tfds
import dlimp as dl
import io
from PIL import Image
import json
import numpy as np
import os
import h5py
from tqdm import tqdm
from vlastudio.policy.openvla.prismatic.vla.datasets.datasets import RLDSDataset
from torch.utils.data import IterableDataset
import torch
from vlastudio.data_utils.utils import ensure_uint8_image

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

class VLAOSDataset(IterableDataset):
    def __init__(self, dataset_dir:str, data_mix:str, image_size=(256, 256), chunk_size=16, ctrl_type='delta', ctrl_space='ee', use_state=True, use_depth=False, shuffle_buffer_size: int=10, *args, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.dataset_dir = dataset_dir
        if not isinstance(image_size, tuple): image_size = tuple(image_size)
        self.data_mix = data_mix
        self.dataset = RLDSDataset(
            data_root_dir=dataset_dir,
            data_mix=data_mix,
            batch_transform=lambda x: x,
            resize_resolution=image_size,
            load_proprio=use_state,
            load_depth=use_depth,
            *args,
            **kwargs
        )
        # load reasoning data
        self.reasoning_file = os.path.join(dataset_dir, data_mix, "reasoning.json")
        
    
    def __iter__(self):
        for data in self.dataset:
            # Get image data and ensure it's uint8 [0, 255]
            image_np = ensure_uint8_image(data['observation']['image_primary'])
            
            data_dict = dict(
                raw_lang=data["task"]["language_instruction"].decode(),
                action=torch.from_numpy(data["action"]),
                image = torch.einsum('k h w c -> k c h w', torch.from_numpy(image_np)),
                is_pad=torch.from_numpy(~data['action_pad_mask']),
                reasoning={},
                state=torch.from_numpy(data['observation']['proprio'][0]),
                timestamp=data['observation']['timestep'].item(),
                episode_id=data['traj_index'],
                dataset_id=data["dataset_name"].decode(),
            )
            yield data_dict
            
    def get_dataset_statistics(self, keyname=None):
        if keyname is None:
            keyname = self.data_mix
        stats = self.dataset.dataset_statistics[keyname]
        stats['state'] = stats['proprio']
        return stats
    
if __name__=='__main__':
    dataset = VLAOSDataset('/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero', data_mix='libero_object', image_size=(256, 256))
    d = next(iter(dataset))
    # dataset = VLAOSDataset('/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/peract2', data_mix='peract2', image_size=(256, 256))
    # d = next(iter(dataset)) # state_dim=74
    # dataset = VLAOSDataset('/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/colosseum', data_mix='colosseum', image_size=(256, 256))
    # d = next(iter(dataset)) # state_dim=37
    print('ok')