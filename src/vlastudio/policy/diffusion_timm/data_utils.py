"""
Data utilities for DiffusionUnetTimmPolicy.
"""
import torch
import torch.nn.functional as F
import numpy as np


def resize_image(image, target_size):
    """
    Resize image tensor to target size.
    
    Args:
        image: (N, C, H, W) or (C, H, W) tensor
        target_size: (H, W) tuple
    
    Returns:
        Resized image tensor
    """
    if image is None:
        return None
    
    squeeze = False
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        squeeze = True
    
    # (N, C, H, W) -> resize
    resized = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)
    
    if squeeze:
        resized = resized.squeeze(0)
    
    return resized


def data_collator(instances):
    """
    Collates a list of samples into a batch for training.
    
    Args:
        instances: List of samples, each containing:
            - image: tensor of shape [N, C, H, W] (N cameras)
            - state: tensor of shape [state_dim]
            - action: tensor of shape [T, action_dim]
            - is_pad: tensor of shape [T] indicating padding
    
    Returns:
        Batch dict with:
            - image: tensor of shape [B, N, C, H, W]
            - qpos: tensor of shape [B, state_dim]
            - actions: tensor of shape [B, T, action_dim]
            - is_pad: tensor of shape [B, T]
    """
    # Collate actions
    if 'action' in instances[0] and instances[0]['action'] is not None:
        if not isinstance(instances[0]['action'], torch.Tensor):
            actions = torch.tensor(np.array([inst['action'] for inst in instances]))
        else:
            actions = torch.stack([inst['action'] for inst in instances])
    else:
        actions = None
    
    # Collate states
    if not isinstance(instances[0]['state'], torch.Tensor):
        states = torch.tensor(np.array([inst['state'] for inst in instances]))
    else:
        states = torch.stack([inst['state'] for inst in instances])
    
    # Collate is_pad
    if 'is_pad' in instances[0] and instances[0]['is_pad'] is not None:
        is_pad = torch.stack([inst['is_pad'] for inst in instances])
    else:
        is_pad = None
    
    # Collate images
    images = torch.stack([inst['image'] for inst in instances])
    
    # Normalize images to [0, 1] if needed
    if images.dtype == torch.uint8 or images.max() > 1.0:
        images = images.float() / 255.0
    
    batch = {
        'image': images,      # (B, N, C, H, W)
        'qpos': states,       # (B, D)
        'actions': actions,   # (B, T, A) or None
        'is_pad': is_pad,     # (B, T) or None
    }
    
    return batch


class DataProcessor:
    """
    Data processor that prepares samples for the model.
    Maps 'state' to 'qpos' for compatibility.
    Handles image resizing to target image_size.
    """
    def __init__(self, image_size=None, normalize_image=False):
        """
        Args:
            image_size: Target image size (H, W) for resizing, or None for no resize
            normalize_image: Whether to normalize images to [0, 1]
        """
        # Normalize image_size format
        if image_size is not None:
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            elif isinstance(image_size, (list, tuple)):
                image_size = tuple(image_size)
        self.image_size = image_size
        self.normalize_image = normalize_image
    
    def __call__(self, sample):
        """
        Process a single sample.
        
        Args:
            sample: Dict with 'image', 'state', 'action', 'is_pad' keys
        
        Returns:
            Processed sample dict
        """
        # Map 'state' to 'qpos' for diffusion policy compatibility
        if 'state' in sample and 'qpos' not in sample:
            sample['qpos'] = sample['state']
        
        # Process image
        if 'image' in sample and self.image_size is not None:
            img = sample['image']
            
            # Convert to tensor if needed
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            
            # Normalize to [0, 1] if needed
            if self.normalize_image:
                if img.dtype == torch.uint8 or img.max() > 1.0:
                    img = img.float() / 255.0
            
            # Resize to target image_size
            target_h, target_w = self.image_size
            # img shape: (N, C, H, W) for multi-camera or (C, H, W) for single
            if len(img.shape) == 4:
                N, C, H, W = img.shape
                if H != target_h or W != target_w:
                    img = img.float() if img.dtype != torch.float32 else img
                    img = F.interpolate(img, size=(target_h, target_w), 
                                      mode='bilinear', align_corners=False)
            elif len(img.shape) == 3:
                C, H, W = img.shape
                if H != target_h or W != target_w:
                    img = img.float() if img.dtype != torch.float32 else img
                    img = img.unsqueeze(0)
                    img = F.interpolate(img, size=(target_h, target_w),
                                      mode='bilinear', align_corners=False)
                    img = img.squeeze(0)
            
            sample['image'] = img
        
        return sample


def get_data_processor(args, model_components):
    """
    Returns a data processor for the diffusion timm policy.
    
    Args:
        args: Configuration arguments
        model_components: Dict with 'model' key
    
    Returns:
        DataProcessor instance
    """
    image_size = getattr(args, 'image_size', None)
    
    return DataProcessor(
        image_size=image_size,
        normalize_image=False
    )

