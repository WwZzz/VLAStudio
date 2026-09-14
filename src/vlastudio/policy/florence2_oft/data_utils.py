"""
Data utilities for Florence2-OFT model in ILStudio.

Handles:
  - Data processing: Convert ILStudio standard format to Florence2 format
  - Data collation: Batch samples for training
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Sequence, Optional, List, Any
from dataclasses import dataclass
import transformers

from loguru import logger


class Florence2OFTDataProcessor:
    """
    Process ILStudio samples to Florence2-OFT model format.
    
    Converts standard ILStudio sample format:
        {
            'image': torch.Tensor(k, c, h, w),
            'state': torch.Tensor(state_dim,),
            'action': torch.Tensor(chunk_size, action_dim),
            'is_pad': torch.Tensor(chunk_size, action_dim),
            'raw_lang': str,
            ...
        }
    
    To Florence2 input format.
    
    Note: Florence2 currently only supports single image input per sample.
    Multi-view images will use the first view only.
    """
    
    def __init__(
        self,
        processor=None,
        chunk_size: int = 16,
        max_seq_len: int = 512,
        image_size: Optional[List[int]] = None,
        task_prompt: str = "Predict the next robot actions:",
        action_token: str = "🔍",
    ):
        self.processor = processor
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.task_prompt = task_prompt
        self.action_token = action_token
        
        # Configure processor image size if specified
        if image_size is not None and processor is not None:
            if isinstance(image_size, (list, tuple)):
                h, w = image_size[0], image_size[1] if len(image_size) > 1 else image_size[0]
            else:
                h = w = image_size
            
            if hasattr(processor, 'image_processor'):
                processor.image_processor.size = {'height': h, 'width': w}
                processor.image_processor.crop_size = {'height': h, 'width': w}
                logger.info(f"Set Florence2 image processor size to {h}x{w}")

    def _preprocess_image(self, image_tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        if isinstance(image_tensor, Image.Image):
            return image_tensor
        elif isinstance(image_tensor, np.ndarray):
            if image_tensor.ndim == 3 and image_tensor.shape[0] in [1, 3, 4]:
                # CHW -> HWC
                image_tensor = np.transpose(image_tensor, (1, 2, 0))
            return Image.fromarray(image_tensor.astype(np.uint8))
        elif isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:
                image_tensor = image_tensor.squeeze(0)
            if image_tensor.shape[0] in [1, 3, 4]:  # CHW format
                image_tensor = image_tensor.permute(1, 2, 0)
            return Image.fromarray(image_tensor.numpy().astype(np.uint8))
        else:
            raise ValueError(f"Unsupported image type: {type(image_tensor)}")

    def _build_prompt_with_action_tokens(self, instruction: str) -> str:
        """Build prompt with action prediction tokens."""
        # Append action tokens at the end (same as qwen_oft and llava_oft)
        action_tokens = self.action_token * self.chunk_size
        return f"{self.task_prompt} {instruction} {action_tokens}"

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sample.
        
        Args:
            sample: ILStudio standard format sample
            
        Returns:
            Processed sample ready for collation
        """
        # Extract and process images
        image_data = sample['image']
        if isinstance(image_data, torch.Tensor):
            if image_data.dim() == 4:
                # Multi-view: use first view only for Florence2
                images = [self._preprocess_image(image_data[0])]
                if image_data.shape[0] > 1:
                    logger.warning("Florence2 only supports single image, using first view only")
            else:
                images = [self._preprocess_image(image_data)]
        elif isinstance(image_data, list):
            # Use first image only
            images = [self._preprocess_image(image_data[0])]
            if len(image_data) > 1:
                logger.warning("Florence2 only supports single image, using first view only")
        else:
            images = [self._preprocess_image(image_data)]
        
        # Get instruction
        instruction = sample.get('raw_lang', '')
        prompt = self._build_prompt_with_action_tokens(instruction)
        
        # Process with Florence2 processor
        model_inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Build output dict
        data_dict = {
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs.get('attention_mask'),
            'pixel_values': model_inputs.get('pixel_values'),
            'action': sample.get('action'),
            'state': sample.get('state'),
            'is_pad': sample.get('is_pad'),
        }
        
        return data_dict


@dataclass
class Florence2OFTDataCollator:
    """
    Collate examples for Florence2-OFT supervised training.
    
    Handles padding and batching of processed samples.
    """
    
    processor: transformers.AutoProcessor = None
    dtype: torch.dtype = torch.float16
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of instances."""
        # Handle input_ids with padding
        input_ids = [instance['input_ids'].squeeze(0) for instance in instances]
        
        # Get pad token id from processor
        pad_token_id = 0  # Florence2 default
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            pad_token_id = self.processor.tokenizer.pad_token_id or 0
        
        # Pad input_ids
        max_len = max(ids.shape[0] for ids in input_ids)
        padded_input_ids = []
        attention_masks = []
        for ids in input_ids:
            padding_len = max_len - ids.shape[0]
            if padding_len > 0:
                padded_ids = torch.cat([
                    torch.full((padding_len,), pad_token_id, dtype=ids.dtype),
                    ids
                ])
            else:
                padded_ids = ids
            padded_input_ids.append(padded_ids)
            attention_masks.append(padded_ids.ne(pad_token_id))
        
        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(attention_masks)
        
        # Handle pixel values
        pixel_values_list = [inst['pixel_values'] for inst in instances if inst.get('pixel_values') is not None]
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = None
        
        # Handle actions
        actions = self._collate_tensor_field(instances, 'action', self.dtype)
        
        # Handle states
        states = self._collate_tensor_field(instances, 'state', self.dtype)
        
        # Handle is_pad
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
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'actions': actions,
            'states': states,
            'is_pad': is_pad,
        }
        
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

