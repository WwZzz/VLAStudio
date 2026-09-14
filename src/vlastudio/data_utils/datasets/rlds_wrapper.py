from vlastudio.policy.openvla.prismatic.vla.datasets.datasets import RLDSDataset
from torch.utils.data import IterableDataset
from vlastudio.data_utils.utils import convert_rlds_sample
from typing import Optional
import torch
import tensorflow as tf


def f(x):
    return x

class WrappedRLDSDataset(IterableDataset):
    def __init__(self, dataset_dir:str, data_mix, image_size=(256, 256), chunk_size=16, ctrl_type='delta', ctrl_space='ee', use_state=True, use_depth=False, num_parallel_calls: Optional[int]=None, num_parallel_reads: Optional[int]=None,shuffle_buffer_size: int=256000, *args, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.dataset_dir = dataset_dir
        if not isinstance(image_size, tuple): image_size = tuple(image_size)
        self.data_mix = data_mix
        self.rlds_dataset = RLDSDataset(
            data_root_dir=dataset_dir,
            data_mix=data_mix,
            batch_transform=f,
            chunk_size=chunk_size,
            resize_resolution=image_size,
            load_proprio=use_state,
            load_depth=use_depth,
            shuffle_buffer_size=shuffle_buffer_size,
            num_parallel_calls=num_parallel_calls,
            num_parallel_reads=num_parallel_reads,
            *args,
            **kwargs
        )
        dataset = self.rlds_dataset.dataset
        def format_convert(frame):
            # Convert int64 to int32 for PyTorch compatibility
            frame["state"] = frame['observation']['proprio'][0]
            frame["action"] = tf.cast(frame["action"], tf.float32)
            frame["raw_lang"] = frame["task"]["language_instruction"]
            frame["is_pad"] = ~frame['action_pad_mask']
            frame['image'] = frame['observation']['image_primary']
            frame['timestamp'] = frame['observation']['timestep']
            frame['episode_id'] = frame['traj_index']
            frame["dataset_id"] = frame["dataset_name"]
            return frame
        dataset = dataset.map(format_convert, num_parallel_calls=num_parallel_calls if num_parallel_calls is not None else tf.data.AUTOTUNE)
        self.dataset = dataset
    
    def __iter__(self):
        for data in self.dataset.as_numpy_iterator():
            data_dict = convert_rlds_sample(data)
            yield data_dict
            
    def get_dataset_statistics(self, keyname=None):
        if keyname is None:
            keyname = self.data_mix
        stats = self.rlds_dataset.dataset_statistics[keyname]
        stats['state'] = stats['proprio']
        return stats
    
if __name__=='__main__':
    dataset = WrappedRLDSDataset('/inspire/hdd/global_public/public_datas/Robotics_Related/Open-X-Embodiment/openx/', data_mix='bc_z', image_size=(256, 256))
    d = next(iter(dataset))
    # dataset = WrappedRLDSDataset('/inspire/hdd/project/robot-action/public/data/libero/openvla', data_mix="libero_object_no_noops", image_size=(256, 256))
    # d = next(iter(dataset))
    print('ok')