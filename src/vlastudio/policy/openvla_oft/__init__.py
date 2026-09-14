"""
OpenVLA-OFT Policy Module for ILStudio.

This module provides integration for OpenVLA-OFT (Optimized Fine-Tuning),
which uses continuous action heads for improved robot policy performance.

Key features:
- L1 regression or diffusion-based action prediction
- Proprioceptive state input support
- Multi-camera input support
- LoRA fine-tuning support
- Compatible with ILStudio's training and evaluation pipeline

Usage:
    from policy.openvla_oft import load_model, get_data_processor, get_data_collator
    
    # Training
    model_components = load_model(args)
    data_processor = get_data_processor(args, model_components)
    data_collator = get_data_collator(args, model_components)
    
    # Inference
    model_components = load_model(args)
    action = model_components['model'].select_action(batch_obs)
"""

import os
import sys
from loguru import logger

# Add openvla-oft to path for imports
OPENVLA_OFT_PATH = os.path.join(os.path.dirname(__file__), 'openvla-oft')
if OPENVLA_OFT_PATH not in sys.path:
    sys.path.insert(0, OPENVLA_OFT_PATH)

from .modeling import OpenVLAOFTConfig, OpenVLAOFTPolicy
from .data_utils import OpenVLAOFTProcessor, OpenVLAOFTCollator, OpenVLAOFTInferenceProcessor
from .trainer import Trainer

from peft import LoraConfig, get_peft_model, PeftModel


def load_model(args):
    """
    Load OpenVLA-OFT model components.
    
    Args:
        args: Arguments containing model configuration.
            Required attributes:
                - is_training (bool): Whether loading for training or inference
            Optional attributes (for training):
                - pretrained_checkpoint (str): Path to pretrained checkpoint
                - use_l1_regression (bool): Use L1 regression action head
                - use_diffusion (bool): Use diffusion action head
                - use_proprio (bool): Use proprioceptive state
                - num_images_in_input (int): Number of input images
                - action_dim (int): Action dimension
                - state_dim (int): State dimension
                - lora_rank (int): LoRA rank
                - training_mode (str): 'lora' or 'full'
            For inference:
                - model_name_or_path (str): Path to trained checkpoint
    
    Returns:
        dict: Dictionary containing:
            - model: The OpenVLAOFTPolicy model
            - tokenizer: The tokenizer
            - processor: The VLA processor
    """
    if not args.is_training:
        # Load from trained checkpoint
        import torch
        
        config = OpenVLAOFTConfig.from_pretrained(args.model_name_or_path)
        
        if config.training_mode == "lora":
            # Load base policy first
            model = OpenVLAOFTPolicy(config)
            # Load LoRA adapter into the VLA backbone
            model.vla = PeftModel.from_pretrained(model.vla, args.model_name_or_path)
        else:
            # Load full model
            model = OpenVLAOFTPolicy.from_pretrained(args.model_name_or_path, config=config)
        
        # Load extra weights (action_head, projectors) if they exist
        extra_weights_path = os.path.join(args.model_name_or_path, 'extra_weights.bin')
        if os.path.exists(extra_weights_path):
            extra_state_dict = torch.load(extra_weights_path, map_location='cpu')
            if 'action_head' in extra_state_dict and model.action_head is not None:
                model.action_head.load_state_dict(extra_state_dict['action_head'])
                logger.info(f"   ✓ Loaded action_head weights")
            if 'proprio_projector' in extra_state_dict and model.proprio_projector is not None:
                model.proprio_projector.load_state_dict(extra_state_dict['proprio_projector'])
                logger.info(f"   ✓ Loaded proprio_projector weights")
            if 'noisy_action_projector' in extra_state_dict and model.noisy_action_projector is not None:
                model.noisy_action_projector.load_state_dict(extra_state_dict['noisy_action_projector'])
                logger.info(f"   ✓ Loaded noisy_action_projector weights")
        
        model.to('cuda')
        model.eval()
    else:
        # Create new model for training
        config = OpenVLAOFTConfig(
            pretrained_checkpoint=getattr(args, 'pretrained_checkpoint', 'openvla/openvla-7b'),
            use_l1_regression=getattr(args, 'use_l1_regression', True),
            use_diffusion=getattr(args, 'use_diffusion', False),
            num_diffusion_steps_train=getattr(args, 'num_diffusion_steps_train', 50),
            use_film=getattr(args, 'use_film', False),
            num_images_in_input=getattr(args, 'num_images_in_input', 2),
            use_proprio=getattr(args, 'use_proprio', True),
            center_crop=getattr(args, 'center_crop', True),
            action_dim=getattr(args, 'action_dim', 7),
            state_dim=getattr(args, 'state_dim', 8),
            chunk_size=getattr(args, 'chunk_size', 8),
            training_mode=getattr(args, 'training_mode', 'lora'),
            lora_rank=getattr(args, 'lora_rank', 32),
            lora_alpha=getattr(args, 'lora_alpha', 16),
            lora_dropout=getattr(args, 'lora_dropout', 0.0),
            use_quantization=getattr(args, 'use_quantization', False),
            load_in_8bit=getattr(args, 'load_in_8bit', False),
            load_in_4bit=getattr(args, 'load_in_4bit', False),
            camera_names=getattr(args, 'camera_names', ['primary']),
            num_open_loop_steps=getattr(args, 'num_open_loop_steps', 8),
        )
        
        model = OpenVLAOFTPolicy(config)
        
        # Apply LoRA if using lora training mode
        # LoRA is applied only to the VLA backbone, not the entire policy
        if config.training_mode == "lora":
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=min(config.lora_alpha, 16),
                lora_dropout=config.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            # Apply LoRA only to the VLA backbone (not action_head, proprio_projector, etc.)
            model.vla = get_peft_model(model.vla, lora_config)
            model.vla.print_trainable_parameters()
    
    # Initialize data_processor and data_collator for inference
    if not args.is_training:
        image_transform = model.processor.image_processor.apply_transform
        model.data_processor = OpenVLAOFTInferenceProcessor(
            tokenizer=model.tokenizer,
            image_transform=image_transform,
            use_proprio=config.use_proprio,
            num_images_in_input=config.num_images_in_input,
            camera_names=config.camera_names,
        )
        model.data_collator = OpenVLAOFTCollator(
            model.tokenizer,
            num_actions_chunk=config.num_actions_chunk,
        )
    
    return {
        'model': model,
        'tokenizer': model.tokenizer,
        'processor': model.processor,
    }


def get_data_collator(args, model_components):
    """
    Get data collator for OpenVLA-OFT.
    
    Args:
        args: Arguments containing configuration
        model_components: Dictionary from load_model containing model components
    
    Returns:
        OpenVLAOFTCollator: Collator for batching samples
    """
    tokenizer = model_components['tokenizer']
    model = model_components['model']
    
    # Get num_actions_chunk from model config
    if hasattr(model, 'config'):
        num_actions_chunk = getattr(model.config, 'num_actions_chunk', 8)
    else:
        num_actions_chunk = getattr(args, 'num_actions_chunk', 8)
    
    return OpenVLAOFTCollator(
        tokenizer=tokenizer,
        num_actions_chunk=num_actions_chunk,
    )


def get_data_processor(args, model_components):
    """
    Get data processor for OpenVLA-OFT.
    
    Args:
        args: Arguments containing configuration
        model_components: Dictionary from load_model containing model components
    
    Returns:
        OpenVLAOFTProcessor: Processor for transforming samples
    """
    tokenizer = model_components['tokenizer']
    processor = model_components['processor']
    model = model_components['model']
    image_transform = processor.image_processor.apply_transform
    
    # Get configuration from model or args
    if hasattr(model, 'config'):
        config = model.config
        num_actions_chunk = getattr(config, 'num_actions_chunk', 8)
        action_dim = getattr(config, 'action_dim', 7)
        use_proprio = getattr(config, 'use_proprio', True)
        num_images_in_input = getattr(config, 'num_images_in_input', 2)
        camera_names = getattr(config, 'camera_names', ['primary'])
    else:
        num_actions_chunk = getattr(args, 'num_actions_chunk', 8)
        action_dim = getattr(args, 'action_dim', 7)
        use_proprio = getattr(args, 'use_proprio', True)
        num_images_in_input = getattr(args, 'num_images_in_input', 2)
        camera_names = getattr(args, 'camera_names', ['primary'])
    
    return OpenVLAOFTProcessor(
        tokenizer=tokenizer,
        image_transform=image_transform,
        num_actions_chunk=num_actions_chunk,
        action_dim=action_dim,
        use_proprio=use_proprio,
        num_images_in_input=num_images_in_input,
        camera_names=camera_names,
    )

