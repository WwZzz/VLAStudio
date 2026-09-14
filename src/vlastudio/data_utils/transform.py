"""
This module contains classes for transforming datasets. Each transform is a callable 
that takes a standard ILSTD data sample and returns a transformed data sample.
The format of the standard ILSTD data sample is a dictionary with the following keys:
sample = {
    'image': image_data, # torch.Tensor(k, c, h, w)
    'state': state_data, # torch.Tensor(state_dim, )
    'action': action_data, # torch.Tensor(chunk_size, action_dim)
    'is_pad': is_pad, # torch.Tensor(chunk_size, action_dim), dtype=bool
    'raw_lang': raw_lang, # str
    'reasoning': reasoning, # Anything
    'timestamp': timestamp,  # float or int
    'episode_id': episode_id, # int
    'dataset_id': dataset_id, # str or int
    '__index__': index, # int
} 
"""

from torch.utils.data import Dataset, IterableDataset
from typing import Callable, Sequence
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class MapTransformPipeline(Dataset):
    def __init__(self, dataset: Dataset, transforms: Sequence[Callable]=None, add_dataset_id: bool=True):
        self.dataset = dataset
        self.transforms = transforms
        self.add_dataset_id = add_dataset_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transforms is not None:
            data = self.transforms(data)
        if self.add_dataset_id and 'dataset_id' not in data:
            data['dataset_id'] = getattr(self.dataset, 'dataset_id', 'unknown')
        return data

class IterableTransformPipeline(IterableDataset):
    def __init__(self, dataset: IterableDataset, transforms: Sequence[Callable]=None, add_dataset_id: bool=True):
        self.dataset = dataset
        self.transforms = transforms
        self.add_dataset_id = add_dataset_id

    def __iter__(self):
        for item in self.dataset:
            if self.transforms is not None:
                item = self.transforms(item)
            if self.add_dataset_id and 'dataset_id' not in item:
                item['dataset_id'] = getattr(self.dataset, 'dataset_id', 'unknown')
            yield item

class TransformPipeline:
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        return self._apply_transforms(x)
    
    def append(self, transform: Callable):
        self.transforms.append(transform)
    
    def _apply_transforms(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class PadStatesAndActionsTransform:
    def __init__(self, target_action_dim: int=None, target_state_dim: int=None, pad_mode: str = 'constant', pad_value: float | int = 0.0):
        self.target_action_dim = target_action_dim
        self.target_state_dim = target_state_dim 
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def __call__(self, x):
        # check if dim is already the target_state_dim
        if self.target_state_dim is not None and 'state' in x and x['state'].shape[-1] != self.target_state_dim:
            if isinstance(x.get('state', None), np.ndarray):
                x['state'] = np.pad(x['state'], (0, self.target_state_dim - x['state'].shape[-1]), mode=self.pad_mode, constant_values=self.pad_value)
            elif isinstance(x['state'], torch.Tensor):
                x['state'] = torch.nn.functional.pad(x['state'], (0, self.target_state_dim - x['state'].shape[-1]), mode=self.pad_mode, value=self.pad_value)
            else:
                raise ValueError(f"Unsupported state type: {type(x['state'])}")
            x['state_dim_pad'] = torch.zeros_like(x['state'], dtype=torch.bool)
            x['state_dim_pad'][-(self.target_state_dim-x['state'].shape[-1]):] = True

        # check if dim is already the target_action_dim
        if self.target_action_dim is not None and 'action' in x and x['action'].shape[-1] != self.target_action_dim:
            # Pad action along the last dim
            if isinstance(x['action'], np.ndarray):
                x['action'] = np.pad(x['action'], ((0, 0), (0, self.target_action_dim - x['action'].shape[-1])), mode=self.pad_mode, constant_values=self.pad_value)
            elif isinstance(x['action'], torch.Tensor):
                x['action'] = torch.nn.functional.pad(x['action'], (0, self.target_action_dim - x['action'].shape[-1]), mode=self.pad_mode, value=self.pad_value)
            else:
                raise ValueError(f"Unsupported action type: {type(x['action'])}")
            x['action_dim_pad'] = torch.zeros_like(x['action'], dtype=torch.bool)
            x['action_dim_pad'][:, -(self.target_action_dim-x['action'].shape[-1]):] = True
        return x

class ActionSmoothingTransform:
    def __init__(self, window_size: int):
        self.window_size = window_size

    def __call__(self, x):
        if 'action' in x: # assume action is of shape (time_horizon, action_dim)
            # action should be smoothed along the time_horizon dimension
            x['action'] = torch.nn.functional.avg_pool1d(x['action'], kernel_size=self.window_size, stride=1, padding=0)
        return x

class ImageToFloat:
    def __call__(self, x):
        if 'image' not in x:
            return x
        is_uint8 = (x['image'].dtype == torch.uint8 or x['image'].max()>1.0)
        if is_uint8: x['image'] = x['image'] / 255.0
        x['image'] = x['image'].float()
        return x

class ImageToUint8:
    def __call__(self, x):
        if 'image' not in x:
            return x
        if x['image'].dtype == torch.float32 or x['image'].dtype == torch.float64:
            x['image'] = (x['image'] * 255.0).clamp(0, 255)
        x['image'] = x['image'].to(torch.uint8)
        return x
        
class ImageColorJitterTransform:
    def __init__(self, brightness: float = 0.3, contrast: float = 0.4, saturation: float = 0.4, hue: float = 0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, x):
        if 'image' in x:
            x['image'] = F.adjust_brightness(x['image'], self.brightness)
            x['image'] = F.adjust_contrast(x['image'], self.contrast)
            x['image'] = F.adjust_saturation(x['image'], self.saturation)
            x['image'] = F.adjust_hue(x['image'], self.hue)
        return x    

class ImageRandomCropTransform:
    def __init__(self, size: int | tuple[int, int]):
        self.size = size
        self.transform = transforms.RandomCrop(size)

    def __call__(self, x):
        if 'image' in x:
            x['image'] = self.transform(x['image'])
        return x
        
class ImageResizeTransform:
    def __init__(self, size: int | tuple[int, int]):
        self.size = size
        self.transform = transforms.Resize(size)

    def __call__(self, x):
        if 'image' in x:
            x['image'] = self.transform(x['image'])
        return x

class ImagePadTransform:
    def __init__(self, size: int | tuple[int, int], padding_mode: str = 'constant', fill: int = 0):
        self.size = size
        self.transform = transforms.Pad(padding=size, fill=0, padding_mode='constant')

    def __call__(self, x):
        if 'image' in x:
            x['image'] = self.transform(x['image'])
        return x

class ImageRotateTransform:
    def __init__(self, degrees: int | tuple[int, int], expand: bool = True):
        self.degrees = degrees
        self.transform = transforms.RandomRotation(degrees=degrees, expand=expand)

    def __call__(self, x):
        if 'image' in x:
            x['image'] = self.transform(x['image'])
        return x

class ImageGaussianBlurTransform:
    def __init__(self, kernel_size: int | tuple[int, int], sigma: float = 0.1):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, x):
        if 'image' in x:
            x['image'] = self.transform(x['image'])
        return x

class ImageExpander:
    """
    Expand the image dimension by the number of views specified in the dataset_map.
    dataset_map is a dictionary that maps the dataset name to a list of target view indices.

    Example 1:
        x: {
            'image': torch.Tensor(2, c, h, w),
            'dataset_id': 'dataset1',
        }
        n = x['image'].shape[0]
        f = ImageExpander(num_views=3, dataset_map={'dataset1': [2,1]}, return_mask=True)
        x1 = f(x)
        x1: {
            'image': torch.Tensor(3, c, h, w),
            'dataset_id': 'dataset1',
            'image_mask': torch.Tensor([False, True, True])
        }
        
        # x1['image'][0] = torch.zeros(c,h,w) # this is because index 0 not in dataset_map['dataset1']
        # for i in range(n):
        #     x1['image'][dataset_map['dataset1'][i]] = x['image'][i] 
    """
    def __init__(self, num_views: int, dataset_map: dict[str, list[int]] = None, return_mask: bool = True):
        """
        Args:
            num_views: Total number of views in the output image tensor
            dataset_map: Dictionary mapping dataset_id to list of target view indices.
                        Each list should have length equal to the number of input views,
                        and values should be indices in [0, num_views-1].
                        Example: {'dataset1': [2, 1]} means:
                            - Input view 0 -> Output position 2
                            - Input view 1 -> Output position 1
            return_mask: If True, add 'image_mask' key indicating which views are real (True) vs padded (False)
        """
        self.num_views = num_views
        self.dataset_map = dataset_map or {}
        self.return_mask = return_mask

    def __call__(self, x):
        if 'image' not in x:
            return x
        
        image = x['image']  # (n, c, h, w)
        n = image.shape[0]
        c, h, w = image.shape[1:]
        
        # Get dataset_id, default to 'default' if not present
        dataset_id = x.get('dataset_id', 'default')
        
        # Get view mapping for this dataset
        if dataset_id in self.dataset_map:
            view_indices = self.dataset_map[dataset_id]
        else:
            # Default: place views sequentially starting from index 0
            view_indices = list(range(n))
        # Initialize output tensor with zeros
        expanded_image = torch.zeros(self.num_views, c, h, w, dtype=image.dtype, device=image.device)
        
        # Place input images at specified positions
        for i in range(n):
            target_idx = view_indices[i]
            expanded_image[target_idx] = image[i]
        
        x['image'] = expanded_image
        
        # Create mask if requested
        if self.return_mask:
            image_mask = torch.zeros(self.num_views, dtype=torch.bool, device=image.device)
            for idx in view_indices:
                image_mask[idx] = True
            x['image_mask'] = image_mask
        
        return x