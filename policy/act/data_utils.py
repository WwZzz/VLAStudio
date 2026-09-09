import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from benchmark.utils import resize_with_pad


class ACTDataProcessor:
    """TEMP: letterbox resize to match training dataset (lerobot resize_with_pad).

    Training applies this in the dataset wrapper; ACT inference previously skipped it.
    Remove once a proper shared image pipeline is wired for ACT.
    """

    def __init__(self, image_size=(256, 256), pad_value=0):
        # image_size: (width, height), same convention as task configs
        self.width = int(image_size[0])
        self.height = int(image_size[1])
        self.pad_value = pad_value

    def __call__(self, sample):
        if sample is None or sample.get("image") is None:
            return sample
        img = sample["image"]
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)
        # Expected: (N, C, H, W) or (C, H, W)
        if img.ndim == 3:
            img = img.unsqueeze(0)
            squeeze = True
        elif img.ndim == 4:
            squeeze = False
        else:
            raise ValueError(f"ACTDataProcessor expected image (N,C,H,W) or (C,H,W), got {tuple(img.shape)}")

        _, _, h, w = img.shape
        if h != self.height or w != self.width:
            # Same call pattern as lerobotv21_wrapper
            img = resize_with_pad(
                img,
                height=self.height,
                width=self.width,
                pad_value=self.pad_value,
            )
        if squeeze:
            img = img.squeeze(0)
        sample["image"] = img
        return sample


def data_collator(instances):
    """
    Collates a list of samples into a batch for training the DiffusionPolicyModel.
    
    Args:
        samples (list): A list of individual samples from the dataset's `__getitem__` method.
                        Each sample is a dictionary containing:
                            - image: tensor of shape [N, C, H, W]
                            - state (qpos): tensor of shape [state_dim]
                            - action: tensor of shape [Ta, action_dim]
                            - is_pad: Boolean or Integer indicating padding
                            - raw_lang: string (not used)
                            - reasoning: tensor/string/any (not used)
    
    Returns:
        A dictionary containing the batched data for the model:
            - image: tensor of shape [B, N, C, H, W]
            - state: tensor of shape [B, state_dim]
            - action: tensor of shape [B, Ta, action_dim]
            - is_pad: tensor of shape [B]
    """
    if 'action' in instances[0]:
        if not isinstance(instances[0]['action'], torch.Tensor):
            actions = torch.tensor(np.array([instance['action'] for instance in instances]))
        else:
            actions = torch.stack([instance['action'] for instance in instances])
    else:
        actions = None
    states = torch.tensor(np.array([instance['state'] for instance in instances])) if not isinstance(instances[0]['state'], torch.Tensor) else torch.stack([instance['state'] for instance in instances])
    is_pad_all = torch.stack([instance['is_pad'] for instance in instances]) if 'is_pad' in instances[0] else None
    images = torch.stack([instance['image'] for instance in instances])
    if images.dtype == torch.uint8 or images.max() > 1.0:
        images = images.float() / 255.0
    batch = dict(
        image=images,
        actions=actions,
        qpos=states,
        is_pad=is_pad_all
    )
    # Return batched data
    return batch