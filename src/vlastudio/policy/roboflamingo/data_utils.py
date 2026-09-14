"""
Data utilities for RoboFlamingo model in ILStudio.

Handles:
  - Data processing: Convert ILStudio standard format to RoboFlamingo format
  - Data collation: Batch samples for training

Reference:
  - Original RoboFlamingo data processing from robot_flamingo/data/data.py
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Sequence, Optional, List, Any
from dataclasses import dataclass
from einops import rearrange

from loguru import logger


class RoboFlamingoDataProcessor:
    """
    Process ILStudio samples to RoboFlamingo model format.
    
    Converts standard ILStudio sample format:
        {
            'image': torch.Tensor(k, c, h, w),  # k camera views
            'state': torch.Tensor(state_dim,),
            'action': torch.Tensor(chunk_size, action_dim),
            'is_pad': torch.Tensor(chunk_size,),
            'raw_lang': str,
            ...
        }
    
    To RoboFlamingo input format with window of frames.
    """
    
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        window_size: int = 12,
        use_gripper: bool = True,
        rgb_pad: int = 10,
        gripper_pad: int = 4,
        max_seq_len: int = 512,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.use_gripper = use_gripper
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        self.max_seq_len = max_seq_len
        
        # CLIP image mean/std for normalization
        self.image_mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.image_std = np.array([0.26862954, 0.26130258, 0.27577711])

    def _preprocess_image(self, image: Any) -> torch.Tensor:
        """
        Preprocess image using CLIP's image processor.
        
        Args:
            image: PIL Image, numpy array, or torch tensor
            
        Returns:
            Processed image tensor (C, H, W)
        """
        if self.image_processor is not None:
            # Use CLIP's preprocessor
            if isinstance(image, torch.Tensor):
                # Convert to PIL
                if image.dim() == 3 and image.shape[0] in [1, 3, 4]:
                    image = image.permute(1, 2, 0)
                image = Image.fromarray(image.numpy().astype(np.uint8))
            elif isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                    image = np.transpose(image, (1, 2, 0))
                image = Image.fromarray(image.astype(np.uint8))
            
            processed = self.image_processor(image)
            return processed
        else:
            # Manual preprocessing if no processor
            if isinstance(image, Image.Image):
                image = np.array(image.resize((224, 224)))
            elif isinstance(image, torch.Tensor):
                image = image.numpy()
            
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                image = np.transpose(image, (1, 2, 0))
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            image = (image - self.image_mean) / self.image_std
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            
            return torch.tensor(image, dtype=torch.float32)

    def _apply_random_shift(self, image: torch.Tensor, pad: int) -> torch.Tensor:
        """
        Apply random shift augmentation.
        
        Args:
            image: (C, H, W) tensor
            pad: Amount of padding
            
        Returns:
            Shifted image tensor (C, H, W)
        """
        if pad <= 0:
            return image
        
        c, h, w = image.shape
        # Pad image
        padded = torch.zeros(c, h + 2*pad, w + 2*pad, dtype=image.dtype)
        padded[:, pad:pad+h, pad:pad+w] = image
        
        # Random crop
        start_h = np.random.randint(0, 2*pad)
        start_w = np.random.randint(0, 2*pad)
        cropped = padded[:, start_h:start_h+h, start_w:start_w+w]
        
        return cropped

    def _tokenize_language(self, instruction: str) -> Dict[str, torch.Tensor]:
        """Tokenize language instruction."""
        if self.tokenizer is None:
            # Return dummy tokens if no tokenizer
            return {
                'input_ids': torch.zeros(10, dtype=torch.long),
                'attention_mask': torch.ones(10, dtype=torch.long),
            }
        
        # Add special tokens
        text = f"<image>{instruction}<|endofchunk|>"
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sample.
        
        Args:
            sample: ILStudio standard format sample
            
        Returns:
            Processed sample ready for collation
        """
        # Extract images
        image_data = sample['image']
        
        # Handle multi-view images
        if isinstance(image_data, torch.Tensor):
            if image_data.dim() == 4:
                # (num_views, C, H, W)
                rgb_image = image_data[0]  # First view is RGB
                gripper_image = image_data[1] if image_data.shape[0] > 1 else None
            else:
                rgb_image = image_data
                gripper_image = None
        elif isinstance(image_data, (list, tuple)):
            rgb_image = image_data[0]
            gripper_image = image_data[1] if len(image_data) > 1 else None
        else:
            rgb_image = image_data
            gripper_image = None
        
        # Preprocess RGB image
        rgb_processed = self._preprocess_image(rgb_image)
        
        # Preprocess gripper image if available
        if self.use_gripper and gripper_image is not None:
            gripper_processed = self._preprocess_image(gripper_image)
        else:
            gripper_processed = None
        
        # Get language tokens
        instruction = sample.get('raw_lang', '')
        lang_tokens = self._tokenize_language(instruction)
        
        # Extract actions and states
        actions = sample.get('action')
        if actions is not None and isinstance(actions, torch.Tensor):
            if actions.dim() == 2:
                # (chunk_size, action_dim)
                # Separate arm actions and gripper
                arm_actions = actions[..., :self.window_size] if actions.shape[-1] > 6 else actions
                gripper_actions = actions[..., -1:] if actions.shape[-1] > 6 else None
            else:
                arm_actions = actions
                gripper_actions = None
        else:
            arm_actions = None
            gripper_actions = None
        
        state = sample.get('state')
        is_pad = sample.get('is_pad')
        
        # Build output dict
        data_dict = {
            'rgb_image': rgb_processed,  # (C, H, W)
            'gripper_image': gripper_processed,  # (C, H, W) or None
            'input_ids': lang_tokens['input_ids'],
            'attention_mask': lang_tokens['attention_mask'],
            'actions': arm_actions,
            'gripper_actions': gripper_actions,
            'state': state,
            'is_pad': is_pad,
        }
        
        return data_dict


@dataclass
class RoboFlamingoDataCollator:
    """
    Collate examples for RoboFlamingo supervised training.
    
    Handles batching and window organization.
    """
    
    tokenizer: Any = None
    window_size: int = 12
    use_gripper: bool = True
    dtype: torch.dtype = torch.float32
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of instances."""
        batch_size = len(instances)
        
        # Collate RGB images
        rgb_images = torch.stack([inst['rgb_image'] for inst in instances])
        # Reshape to (B, T=1, F=1, C, H, W) for single-frame
        rgb_images = rgb_images.unsqueeze(1).unsqueeze(2)
        
        # Collate gripper images if present
        if self.use_gripper and instances[0].get('gripper_image') is not None:
            gripper_images = torch.stack([inst['gripper_image'] for inst in instances])
            gripper_images = gripper_images.unsqueeze(1).unsqueeze(2)
        else:
            gripper_images = None
        
        # Collate language
        input_ids = torch.stack([inst['input_ids'] for inst in instances])
        attention_mask = torch.stack([inst['attention_mask'] for inst in instances])
        
        # Collate actions
        actions = self._collate_tensor_field(instances, 'actions', self.dtype)
        gripper_actions = self._collate_tensor_field(instances, 'gripper_actions', self.dtype)
        
        # Collate states
        states = self._collate_tensor_field(instances, 'state', self.dtype)
        
        # Collate is_pad
        is_pad_list = [inst['is_pad'] for inst in instances if inst.get('is_pad') is not None]
        if is_pad_list and is_pad_list[0] is not None:
            is_pad = torch.stack([
                inst['is_pad'] if isinstance(inst['is_pad'], torch.Tensor)
                else torch.tensor(inst['is_pad'])
                for inst in instances
            ])
        else:
            is_pad = None
        
        batch = {
            'vision_x': rgb_images.to(dtype=self.dtype),
            'lang_x': input_ids.long(),
            'attention_mask': attention_mask,
            'vision_gripper': gripper_images.to(dtype=self.dtype) if gripper_images is not None else None,
            'state_tensor': states,
            'actions': actions,
            'gripper_actions': gripper_actions,
            'is_pad': is_pad,
        }
        
        # Remove None values
        batch = {k: v for k, v in batch.items() if v is not None}
        
        return batch
    
    def _collate_tensor_field(
        self,
        instances: Sequence[Dict],
        field: str,
        dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Helper to collate a tensor field from instances."""
        values = [inst.get(field) for inst in instances]
        if values[0] is None:
            return None
        
        if isinstance(values[0], torch.Tensor):
            result = torch.stack(values)
        elif isinstance(values[0], np.ndarray):
            result = torch.tensor(np.stack(values))
        elif isinstance(values[0], list):
            result = torch.tensor(np.array(values))
        else:
            return None
        
        return result.to(dtype=dtype)


class RoboFlamingoWindowDataProcessor:
    """
    Data processor that organizes samples into windows for RoboFlamingo.
    
    This is for datasets that provide episode data where we need to
    sample windows of consecutive frames.
    """
    
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        window_size: int = 12,
        use_gripper: bool = True,
        rgb_pad: int = 10,
        gripper_pad: int = 4,
        max_seq_len: int = 512,
        apply_augmentation: bool = True,
    ):
        self.base_processor = RoboFlamingoDataProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            window_size=window_size,
            use_gripper=use_gripper,
            rgb_pad=rgb_pad,
            gripper_pad=gripper_pad,
            max_seq_len=max_seq_len,
        )
        self.window_size = window_size
        self.apply_augmentation = apply_augmentation
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a windowed sample.
        
        Expects sample to contain sequences:
        - 'image': (T, num_views, C, H, W) or list of frames
        - 'state': (T, state_dim)
        - 'action': (T, action_dim)
        """
        processed = self.base_processor(sample)
        
        # Apply augmentation during training
        if self.apply_augmentation:
            if 'rgb_image' in processed and processed['rgb_image'] is not None:
                processed['rgb_image'] = self.base_processor._apply_random_shift(
                    processed['rgb_image'], self.rgb_pad
                )
            if 'gripper_image' in processed and processed['gripper_image'] is not None:
                processed['gripper_image'] = self.base_processor._apply_random_shift(
                    processed['gripper_image'], self.gripper_pad
                )
        
        return processed

