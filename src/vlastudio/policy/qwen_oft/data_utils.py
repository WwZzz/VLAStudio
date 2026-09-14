"""
Data utilities for Qwen-OFT model in ILStudio.

Handles:
  - Data processing: Convert ILStudio standard format to Qwen VL format
  - Supports both Qwen2.5-VL and Qwen3-VL
  - Data collation: Batch samples for training
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Sequence, Optional, List, Any
from dataclasses import dataclass
import transformers
from qwen_vl_utils import process_vision_info
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms

from loguru import logger


class QwenOFTDataProcessor:
    """
    Process ILStudio samples to Qwen-OFT model format.
    
    Supports both Qwen2.5-VL and Qwen3-VL.
    
    Converts standard ILStudio sample format:
        {
            'image': torch.Tensor(k, c, h, w),
            'state': torch.Tensor(state_dim,),
            'action': torch.Tensor(chunk_size, action_dim),
            'is_pad': torch.Tensor(chunk_size, action_dim),
            'raw_lang': str,
            ...
        }
    
    To Qwen VL input format with action tokens appended.
    """
    
    def __init__(
        self,
        tokenizer=None,
        multimodal_processor=None,
        chunk_size: int = 16,
        action_token: str = "🔍",
        max_seq_len: int = 512,
        image_size: Optional[List[int]] = None,
        qwen_version: str = 'qwen2.5',  # 'qwen2.5' or 'qwen3'
    ):
        self.tokenizer = tokenizer
        self.multimodal_processor = multimodal_processor
        self.chunk_size = chunk_size
        self.action_token = action_token
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.qwen_version = qwen_version

    def _preprocess_image(self, image_tensor: torch.Tensor, camera_name: str = None) -> Image.Image:
        """Convert tensor to PIL Image with optional Qwen-specific processing."""
        # Handle different input formats
        if isinstance(image_tensor, Image.Image):
            img_pil = image_tensor
        elif isinstance(image_tensor, np.ndarray):
            if image_tensor.ndim == 3 and image_tensor.shape[0] in [1, 3, 4]:
                # CHW -> HWC
                image_tensor = np.transpose(image_tensor, (1, 2, 0))
            img_pil = Image.fromarray(image_tensor.astype(np.uint8))
        elif isinstance(image_tensor, torch.Tensor):
            # Ensure proper format
            if image_tensor.dim() == 4:
                image_tensor = image_tensor.squeeze(0)
            if image_tensor.shape[0] in [1, 3, 4]:  # CHW format
                image_tensor = image_tensor.permute(1, 2, 0)
            img_pil = Image.fromarray(image_tensor.numpy().astype(np.uint8))
        else:
            raise ValueError(f"Unsupported image type: {type(image_tensor)}")
        
        # Resize if needed
        if self.image_size is not None:
            img_pil = img_pil.resize(tuple(self.image_size), Image.BILINEAR)
        
        return img_pil

    def _build_prompt_with_action_tokens(self, instruction: str) -> str:
        """Build prompt with action prediction tokens."""
        action_tokens = self.action_token * self.chunk_size
        prompt_suffix = f" Please predict the next {self.chunk_size} robot actions: <action>{action_tokens}</action>."
        return instruction + prompt_suffix

    def _build_messages(self, images: List[Image.Image], instruction: str) -> List[Dict]:
        """Build Qwen VL chat format messages."""
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": self._build_prompt_with_action_tokens(instruction)})
        
        messages = [{"role": "user", "content": content}]
        return messages

    def _process_qwen25(self, images: List[Image.Image], messages: List[Dict]) -> Dict[str, Any]:
        """
        Process for Qwen2.5-VL.
        
        Uses process_vision_info to extract image inputs separately.
        """
        # Get text prompt
        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Extract image and video inputs using qwen_vl_utils
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Process with multimodal processor
        model_inputs = self.multimodal_processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        return model_inputs

    def _process_qwen3(self, images: List[Image.Image], messages: List[Dict]) -> Dict[str, Any]:
        """
        Process for Qwen3-VL.
        
        Uses apply_chat_template with tokenize=True directly.
        """
        # Qwen3 can process directly with apply_chat_template
        model_inputs = self.multimodal_processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        return model_inputs

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
                # (K, C, H, W) -> list of PIL Images
                images = [self._preprocess_image(image_data[i]) for i in range(image_data.shape[0])]
            else:
                images = [self._preprocess_image(image_data)]
        elif isinstance(image_data, list):
            images = [self._preprocess_image(img) for img in image_data]
        else:
            images = [self._preprocess_image(image_data)]
        
        # Get instruction
        instruction = sample.get('raw_lang', '')
        
        # Build messages
        messages = self._build_messages(images, instruction)
        
        # Process based on Qwen version
        if self.qwen_version == 'qwen3':
            model_inputs = self._process_qwen3(images, messages)
        else:  # qwen2.5
            model_inputs = self._process_qwen25(images, messages)
        
        # Build output dict
        data_dict = {
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'pixel_values': model_inputs.get('pixel_values'),
            'image_grid_thw': model_inputs.get('image_grid_thw'),
            'action': sample.get('action'),
            'state': sample.get('state'),
            'is_pad': sample.get('is_pad'),
        }
        
        return data_dict


@dataclass
class QwenOFTDataCollator:
    """
    Collate examples for Qwen-OFT supervised training.
    
    Handles padding and batching of processed samples.
    """
    
    multimodal_processor: transformers.AutoProcessor = None
    tokenizer: transformers.AutoTokenizer = None
    dtype: torch.dtype = torch.bfloat16
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of instances."""
        # Handle input_ids with left padding (flip, pad, flip back)
        input_ids = [
            torch.flip(instance['input_ids'].squeeze(0), dims=[0]) 
            for instance in instances
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        input_ids = torch.flip(input_ids, dims=[1])
        
        # Handle attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        # Handle pixel values
        pixel_values_list = [inst['pixel_values'] for inst in instances if inst.get('pixel_values') is not None]
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = None
        
        # Handle image_grid_thw
        image_grid_thw_list = [inst['image_grid_thw'] for inst in instances if inst.get('image_grid_thw') is not None]
        if image_grid_thw_list:
            image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
        else:
            image_grid_thw = None
        
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
            'image_grid_thw': image_grid_thw,
            'pixel_values_videos': None,
            'video_grid_thw': None,
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