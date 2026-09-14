"""
Data utilities for OpenVLA-OFT Policy.

This module provides data processing and collation utilities for training
and inference with OpenVLA-OFT models.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List, Union
from PIL import Image

# Add openvla-oft to path
OPENVLA_OFT_PATH = os.path.join(os.path.dirname(__file__), 'openvla-oft')
if OPENVLA_OFT_PATH not in sys.path:
    sys.path.insert(0, OPENVLA_OFT_PATH)

from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer


IGNORE_INDEX = -100


class OpenVLAOFTCollator:
    """
    Collator for batching OpenVLA-OFT samples.
    
    Handles padding of variable-length sequences and proper formatting
    of pixel values, input IDs, labels, and actions for both training and inference.
    """
    
    def __init__(self, tokenizer, dtype=torch.bfloat16, num_actions_chunk: int = 8):
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.num_actions_chunk = num_actions_chunk
        self.collator = PaddedCollatorForActionPrediction(
            tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right"
        )
    
    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into a batch.
        
        Works for both training (with labels) and inference (without labels).
        
        Args:
            instances: List of dictionaries from data processor
        
        Returns:
            Batched dictionary with properly padded tensors
        """
        # Check if this is inference (no labels) or training (with labels)
        has_labels = 'labels' in instances[0] and instances[0]['labels'] is not None
        
        if has_labels:
            # Training mode: use the full collator
            batch = self.collator(instances)
        else:
            # Inference mode: manually collate without labels
            batch = self._collate_inference(instances)
        
        # Handle proprio if present
        if instances[0].get('proprio') is not None:
            proprios = [inst['proprio'] for inst in instances]
            batch['proprio'] = torch.stack(proprios)
        
        # Handle actions if present (for training)
        if instances[0].get('actions') is not None:
            actions = [inst['actions'] for inst in instances]
            batch['actions'] = torch.stack(actions)
        
        return batch
    
    def _collate_inference(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate samples for inference (without labels).
        
        Args:
            instances: List of sample dictionaries
        
        Returns:
            Batched dictionary with pixel_values, input_ids, attention_mask
        """
        # Stack pixel_values
        pixel_values = torch.stack([inst['pixel_values'] for inst in instances])
        
        # Pad input_ids to same length
        input_ids_list = [inst['input_ids'] for inst in instances]
        max_len = max(ids.shape[0] for ids in input_ids_list)
        
        padded_input_ids = []
        attention_masks = []
        
        for ids in input_ids_list:
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                # Pad on the right
                padded_ids = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=ids.dtype)])
                mask = torch.cat([torch.ones(ids.shape[0], dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            else:
                padded_ids = ids
                mask = torch.ones(ids.shape[0], dtype=torch.long)
            
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)
        
        batch = {
            'pixel_values': pixel_values,
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(attention_masks),
        }
        
        return batch


class OpenVLAOFTProcessor:
    """
    Data processor for OpenVLA-OFT training samples.
    
    Converts raw dataset samples into the format expected by OpenVLA-OFT,
    including image transformation, text tokenization, and action formatting.
    """
    
    def __init__(
        self,
        tokenizer,
        image_transform,
        num_actions_chunk: int = 8,
        action_dim: int = 7,
        use_proprio: bool = True,
        num_images_in_input: int = 2,
        camera_names: List[str] = None,
    ):
        """
        Initialize the processor.
        
        Args:
            tokenizer: HuggingFace tokenizer
            image_transform: Image transformation function
            num_actions_chunk: Number of actions in a chunk
            action_dim: Dimension of each action
            use_proprio: Whether to include proprioceptive state
            num_images_in_input: Number of images expected (1 or 2)
            camera_names: List of camera names to use
        """
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.num_actions_chunk = num_actions_chunk
        self.action_dim = action_dim
        self.use_proprio = use_proprio
        self.num_images_in_input = num_images_in_input
        self.camera_names = camera_names if camera_names is not None else ["primary"]
        
        self.prompt_builder_fn = PurePromptBuilder
        self.action_tokenizer = ActionTokenizer(tokenizer)
        self.predict_stop_token = True
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a single sample from the dataset.
        
        Args:
            sample: Dictionary containing:
                - image: Image tensor [num_cameras, C, H, W]
                - state: State tensor [state_dim]
                - action: Action tensor [chunk_size, action_dim]
                - raw_lang: Language instruction string
        
        Returns:
            Processed dictionary ready for collation
        """
        # Extract and validate action
        action = sample.get('action', None)
        action_chunk_string = ""
        if action is not None:
            # Handle chunk dimension
            if action.ndim == 1:
                action = action.unsqueeze(0)  # [action_dim] -> [1, action_dim]
            elif action.ndim == 2:
                pass  # [chunk_size, action_dim]
            
            # Pad or truncate to num_actions_chunk
            if action.shape[0] < self.num_actions_chunk:
                padding = torch.zeros(
                    self.num_actions_chunk - action.shape[0], 
                    action.shape[1],
                    dtype=action.dtype
                )
                action = torch.cat([action, padding], dim=0)
            elif action.shape[0] > self.num_actions_chunk:
                action = action[:self.num_actions_chunk]
            
            # Build action chunk string for ALL actions in the chunk
            # Each action gets tokenized to action_dim tokens
            action_numpy = action.numpy() if isinstance(action, torch.Tensor) else action
            # Tokenize first action
            action_chunk_string = self.action_tokenizer(action_numpy[0])
            # Tokenize remaining actions and concatenate
            for i in range(1, self.num_actions_chunk):
                action_chunk_string += self.action_tokenizer(action_numpy[i])
        
        # Handle image format: (num_cameras, C, H, W)
        image_tensor = sample['image']
        if image_tensor.ndim == 3:
            # Single image [C, H, W] -> add camera dimension
            image_tensor = image_tensor.unsqueeze(0)
        
        # Process primary image
        primary_image = image_tensor[0]  # Take first camera (primary)
        if isinstance(primary_image, torch.Tensor):
            # Convert from tensor to PIL Image
            if primary_image.dtype == torch.float32 or primary_image.dtype == torch.float64:
                if primary_image.max() <= 1.0:
                    primary_image = (primary_image * 255).byte()
                else:
                    primary_image = primary_image.byte()
            image_array = primary_image.permute(1, 2, 0).numpy().astype(np.uint8)
        else:
            image_array = np.array(primary_image).astype(np.uint8)
        primary_img = Image.fromarray(image_array)
        
        # Get language instruction
        lang = sample.get('raw_lang', 'perform the task')
        if lang is None or lang == '':
            lang = 'perform the task'
        
        # Construct Chat-based Prompt with full action chunk
        prompt_builder = self.prompt_builder_fn("openvla")
        if action is not None:
            # Use the full action chunk string (all chunk_size * action_dim tokens)
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": ""},
            ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        
        # Tokenize
        input_ids = self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        
        # Tensorize
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        
        # Transform primary image
        pixel_values = self.image_transform(primary_img)
        
        # Handle additional images (e.g., wrist camera)
        if self.num_images_in_input > 1 and image_tensor.shape[0] > 1:
            additional_images = []
            for i in range(1, min(self.num_images_in_input, image_tensor.shape[0])):
                add_img = image_tensor[i]
                if isinstance(add_img, torch.Tensor):
                    if add_img.dtype == torch.float32 or add_img.dtype == torch.float64:
                        if add_img.max() <= 1.0:
                            add_img = (add_img * 255).byte()
                        else:
                            add_img = add_img.byte()
                    add_array = add_img.permute(1, 2, 0).numpy().astype(np.uint8)
                else:
                    add_array = np.array(add_img).astype(np.uint8)
                add_pil = Image.fromarray(add_array)
                additional_images.append(self.image_transform(add_pil))
            
            # Concatenate along channel dimension
            if additional_images:
                pixel_values = torch.cat([pixel_values] + additional_images, dim=0)
        
        # Set labels: only compute loss for action tokens
        if action is not None:
            # Mask all tokens except action tokens
            # action_chunk_len = num_actions_chunk * action_dim (each action -> action_dim tokens)
            action_chunk_len = len(action_chunk_string)  # Number of characters = number of tokens
            labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
            if not self.predict_stop_token:
                labels[-1] = IGNORE_INDEX
        
        # Build output dictionary
        output = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'labels': labels,
        }
        
        # Add proprio if using
        if self.use_proprio:
            state = sample.get('state', None)
            if state is not None:
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state).float()
                output['proprio'] = state
        
        # Add full action chunk for continuous action training
        if action is not None:
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()
            output['actions'] = action
        
        return output


class OpenVLAOFTInferenceProcessor:
    """
    Data processor for OpenVLA-OFT inference.
    
    Simplified processor that handles observation formatting for inference,
    compatible with ILStudio's MetaPolicy interface.
    """
    
    def __init__(
        self,
        tokenizer,
        image_transform,
        use_proprio: bool = True,
        num_images_in_input: int = 2,
        camera_names: List[str] = None,
    ):
        """
        Initialize the inference processor.
        
        Args:
            tokenizer: HuggingFace tokenizer
            image_transform: Image transformation function
            use_proprio: Whether to include proprioceptive state
            num_images_in_input: Number of images expected
            camera_names: List of camera names to use
        """
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.use_proprio = use_proprio
        self.num_images_in_input = num_images_in_input
        self.camera_names = camera_names if camera_names is not None else ["primary"]
        self.prompt_builder_fn = PurePromptBuilder
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a single observation for inference.
        
        Args:
            sample: Dictionary containing:
                - image: Image tensor [num_cameras, C, H, W]
                - state: State tensor [state_dim]
                - raw_lang: Language instruction string
        
        Returns:
            Processed dictionary ready for model inference
        """
        # Handle image format
        image_tensor = sample['image']
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Process primary image
        primary_image = image_tensor[0]
        if isinstance(primary_image, torch.Tensor):
            if primary_image.dtype == torch.float32 or primary_image.dtype == torch.float64:
                if primary_image.max() <= 1.0:
                    primary_image = (primary_image * 255).byte()
                else:
                    primary_image = primary_image.byte()
            image_array = primary_image.permute(1, 2, 0).numpy().astype(np.uint8)
        else:
            image_array = np.array(primary_image).astype(np.uint8)
        primary_img = Image.fromarray(image_array)
        
        # Get language instruction
        lang = sample.get('raw_lang', 'perform the task')
        if lang is None or lang == '':
            lang = 'perform the task'
        
        # Construct prompt for inference (no response yet)
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": ""},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        
        # Tokenize
        input_ids = self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        input_ids = torch.tensor(input_ids)
        
        # Transform primary image
        pixel_values = self.image_transform(primary_img)
        
        # Handle additional images
        if self.num_images_in_input > 1 and image_tensor.shape[0] > 1:
            additional_images = []
            for i in range(1, min(self.num_images_in_input, image_tensor.shape[0])):
                add_img = image_tensor[i]
                if isinstance(add_img, torch.Tensor):
                    if add_img.dtype == torch.float32 or add_img.dtype == torch.float64:
                        if add_img.max() <= 1.0:
                            add_img = (add_img * 255).byte()
                        else:
                            add_img = add_img.byte()
                    add_array = add_img.permute(1, 2, 0).numpy().astype(np.uint8)
                else:
                    add_array = np.array(add_img).astype(np.uint8)
                add_pil = Image.fromarray(add_array)
                additional_images.append(self.image_transform(add_pil))
            
            if additional_images:
                pixel_values = torch.cat([pixel_values] + additional_images, dim=0)
        
        # Build output dictionary
        output = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids),
        }
        
        # Add proprio if using
        if self.use_proprio:
            state = sample.get('state', None)
            if state is not None:
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state).float()
                output['proprio'] = state
        
        return output

