"""
Data utilities for LLaVA-OFT model in ILStudio.

Handles:
  - Data processing: Convert ILStudio standard format to LLaVA format
  - Data collation: Batch samples for training
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Sequence, Optional, List, Any
from dataclasses import dataclass
import transformers

from loguru import logger


class LlavaOFTDataProcessor:
    """
    Process ILStudio samples to LLaVA-OFT model format.
    
    Converts standard ILStudio sample format:
        {
            'image': torch.Tensor(k, c, h, w),
            'state': torch.Tensor(state_dim,),
            'action': torch.Tensor(chunk_size, action_dim),
            'is_pad': torch.Tensor(chunk_size, action_dim),
            'raw_lang': str,
            ...
        }
    
    To LLaVA input format.
    """
    
    def __init__(
        self,
        processor=None,
        chunk_size: int = 16,
        max_seq_len: int = 512,
        image_size: Optional[List[int]] = None,
        prompt_template: str = "USER: <image>\n{instruction}\nASSISTANT:",
        action_token: str = "🔍",
    ):
        self.processor = processor
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.prompt_template = prompt_template
        self.action_token = action_token

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
        """Build prompt with action prediction tokens.
        
        The action tokens serve to extend the sequence length, providing positions
        for action prediction. The model uses the last chunk_size hidden states
        for action prediction, so the exact tokenization of action tokens is not critical.
        """
        # Add action tokens to extend sequence length
        # The model will use last chunk_size hidden states for action prediction
        action_tokens = self.action_token * self.chunk_size
        prompt_suffix = f" Please predict the next {self.chunk_size} robot actions: {action_tokens}"
        return instruction + prompt_suffix

    def _build_conversation(self, instruction: str) -> List[Dict]:
        """Build LLaVA conversation format."""
        # Add action tokens to instruction
        instruction_with_actions = self._build_prompt_with_action_tokens(instruction)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction_with_actions},
                ],
            },
        ]
        return conversation

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
                # Multi-view: use first view for LLaVA
                images = [self._preprocess_image(image_data[0])]
                if image_data.shape[0] > 1:
                    logger.debug("LLaVA using first view only for multi-view input")
            else:
                images = [self._preprocess_image(image_data)]
        elif isinstance(image_data, list):
            images = [self._preprocess_image(image_data[0])]
        else:
            images = [self._preprocess_image(image_data)]
        
        # Get instruction
        instruction = sample.get('raw_lang', '')
        
        # Build conversation
        conversation = self._build_conversation(instruction)
        
        # Apply chat template
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        
        # Process with LLaVA processor
        model_inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        
        # Build output dict
        data_dict = {
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs.get('attention_mask'),
            'pixel_values': model_inputs.get('pixel_values'),
            'image_sizes': model_inputs.get('image_sizes'),
            'action': sample.get('action'),
            'state': sample.get('state'),
            'is_pad': sample.get('is_pad'),
        }
        
        return data_dict


@dataclass
class LlavaOFTDataCollator:
    """
    Collate examples for LLaVA-OFT supervised training.
    
    Handles padding and batching of processed samples.
    """
    
    processor: transformers.AutoProcessor = None
    dtype: torch.dtype = torch.bfloat16
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of instances."""
        # Handle input_ids with left padding
        input_ids = [
            torch.flip(instance['input_ids'].squeeze(0), dims=[0]) 
            for instance in instances
        ]
        
        # Get pad token id
        pad_token_id = 0
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            pad_token_id = self.processor.tokenizer.pad_token_id or 0
        
        # Pad input_ids
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id
        )
        input_ids = torch.flip(input_ids, dims=[1])
        
        # Handle attention mask
        attention_mask = input_ids.ne(pad_token_id)
        
        # Handle pixel values
        pixel_values_list = [inst['pixel_values'] for inst in instances if inst.get('pixel_values') is not None]
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = None
        
        # Handle image_sizes
        image_sizes_list = [inst['image_sizes'] for inst in instances if inst.get('image_sizes') is not None]
        if image_sizes_list:
            image_sizes = torch.cat(image_sizes_list, dim=0)
        else:
            image_sizes = None
        
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
            'image_sizes': image_sizes,
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

