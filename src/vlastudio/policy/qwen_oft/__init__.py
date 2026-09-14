"""
Qwen-OFT Policy for ILStudio.

A lightweight VLA implementation using Qwen VL backbone (supports both Qwen2.5-VL 
and Qwen3-VL) with action token prediction via L1 regression, inspired by OpenVLA-OFT.

This module provides three main interfaces:
  - load_model: Load model and processor
  - get_data_processor: Get data preprocessing function
  - get_data_collator: Get data collation function
"""

from .modeling import QwenOFTConfig, QwenOFTForPolicy, detect_qwen_version
from .data_utils import QwenOFTDataProcessor, QwenOFTDataCollator

import torch
from transformers import AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import save_file, load_file
import os
from .trainer import Trainer
from loguru import logger


def find_all_linear_names(model, lora_module=None):
    """Find all linear layer names for LoRA injection."""
    cls = torch.nn.Linear
    lora_module_names = set()
    no_lora_keywords = ['lm_head', 'action_head']
    
    if lora_module is None:
        lora_module = []
    if 'vit' not in lora_module:
        no_lora_keywords.append("visual")
    if 'llm' not in lora_module:
        no_lora_keywords.append("model")
    
    for name, module in model.named_modules():
        if any(kw in name for kw in no_lora_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    
    return list(lora_module_names)


def load_model(args):
    """
    Load Qwen-OFT model and processor.
    
    Supports both Qwen2.5-VL and Qwen3-VL. The version is auto-detected from
    vlm_model_name_or_path, or can be explicitly set via qwen_version arg.
    
    Args:
        args: Configuration namespace with:
            - vlm_model_name_or_path: Path to VLM model (Qwen2.5-VL or Qwen3-VL)
            - qwen_version: Optional, 'qwen2.5' or 'qwen3', auto-detected if not set
            - model_name_or_path: Path to checkpoint (for inference)
            - action_dim: Action dimension
            - state_dim: State dimension
            - chunk_size: Action prediction horizon
            - is_training: Whether loading for training
            - lora_enable: Whether to use LoRA
            - lora_r, lora_alpha, lora_dropout, lora_bias: LoRA config
            - lora_module: List of modules to apply LoRA to
            
    Returns:
        Dict with 'model', 'tokenizer', 'multimodal_processor', 'qwen_version' keys.
    """
    args.device = getattr(args, 'device', 'cuda')
    
    if not getattr(args, 'is_training', True):
        # === Inference Mode ===
        config = QwenOFTConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name_or_path)
        multimodal_processor = AutoProcessor.from_pretrained(config.vlm_model_name_or_path)
        
        # Get qwen version from config
        qwen_version = config.qwen_version
        
        # Check for PEFT adapter
        peft_path = os.path.join(args.model_name_or_path, 'adapter_config.json')
        if os.path.exists(peft_path):
            model = QwenOFTForPolicy(config=config)
            model = PeftModel.from_pretrained(model, args.model_name_or_path)
            
            # Load extra trainable parameters
            extra_path = os.path.join(args.model_name_or_path, 'extra_trainable.safetensors')
            if os.path.exists(extra_path):
                extra_state = load_file(extra_path, device="cpu")
                model.load_state_dict(extra_state, strict=False)
            
            model = model.merge_and_unload()
        else:
            model = QwenOFTForPolicy.from_pretrained(args.model_name_or_path, config=config)
        
        # Set processor
        model.set_processor(multimodal_processor)
        
        # Initialize data processor and collator for inference
        model.data_processor = QwenOFTDataProcessor(
            tokenizer=tokenizer,
            multimodal_processor=multimodal_processor,
            chunk_size=config.chunk_size,
            action_token=config.action_token,
            qwen_version=qwen_version,
        )
        model.data_collator = QwenOFTDataCollator(
            multimodal_processor=multimodal_processor,
            tokenizer=tokenizer,
            dtype=torch.bfloat16,
        )
        
    else:
        # === Training Mode ===
        tokenizer = AutoTokenizer.from_pretrained(args.vlm_model_name_or_path)
        multimodal_processor = AutoProcessor.from_pretrained(args.vlm_model_name_or_path)
        
        # Detect or use provided qwen version
        qwen_version = getattr(args, 'qwen_version', None) or detect_qwen_version(args.vlm_model_name_or_path)
        logger.info(f"Using Qwen version: {qwen_version}")
        
        config = QwenOFTConfig(
            vlm_model_name_or_path=args.vlm_model_name_or_path,
            qwen_version=qwen_version,
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            chunk_size=args.chunk_size,
        )
        model = QwenOFTForPolicy(config=config)
        
        # Set processor
        model.set_processor(multimodal_processor)
        
        # Apply LoRA if enabled
        if getattr(args, 'lora_enable', False):
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=find_all_linear_names(model, getattr(args, 'lora_module', None)),
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
            )
            
            if getattr(args, 'bits', 16) == 16:
                if getattr(args, 'bf16', False):
                    model.to(torch.bfloat16)
                elif getattr(args, 'fp16', False):
                    model.to(torch.float16)
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
    
    # Store references
    model.tokenizer = tokenizer
    model.multimodal_processor = multimodal_processor
    model.config.use_cache = False
    
    # Set requires_grad
    if hasattr(args, 'is_training') and args.is_training:
        model.set_requires_grad(args)
    
    # Enable gradient checkpointing if specified
    if getattr(args, 'gradient_checkpointing', False):
        if hasattr(model.vlm, "enable_input_require_grads"):
            model.vlm.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.vlm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # Move to device and set dtype
    if hasattr(args, 'bf16') and args.bf16:
        model.to(dtype=torch.bfloat16, device=args.device)
    elif hasattr(args, 'fp16') and args.fp16:
        model.to(dtype=torch.float16, device=args.device)
    else:
        model.to(dtype=torch.bfloat16, device=args.device)
    
    # Save config
    if hasattr(args, 'output_dir') and args.is_training:
        os.makedirs(args.output_dir, exist_ok=True)
        model.config.save_pretrained(args.output_dir)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'multimodal_processor': multimodal_processor,
        'qwen_version': model.qwen_version,
    }


def get_data_processor(args, model_components):
    """
    Get data processor for Qwen-OFT.
    
    Args:
        args: Configuration namespace
        model_components: Dict from load_model
        
    Returns:
        QwenOFTDataProcessor instance
    """
    # Get qwen version from model_components or detect from args
    qwen_version = model_components.get('qwen_version')
    if qwen_version is None:
        qwen_version = getattr(args, 'qwen_version', None)
        if qwen_version is None and hasattr(args, 'vlm_model_name_or_path'):
            qwen_version = detect_qwen_version(args.vlm_model_name_or_path)
        else:
            qwen_version = 'qwen2.5'  # Default
    
    return QwenOFTDataProcessor(
        tokenizer=model_components['tokenizer'],
        multimodal_processor=model_components['multimodal_processor'],
        chunk_size=args.chunk_size,
        action_token=getattr(args, 'action_token', '🔍'),
        qwen_version=qwen_version,
    )


def get_data_collator(args, model_components):
    """
    Get data collator for Qwen-OFT.
    
    Args:
        args: Configuration namespace
        model_components: Dict from load_model
        
    Returns:
        QwenOFTDataCollator instance
    """
    dtype = torch.bfloat16
    if getattr(args, 'fp16', False):
        dtype = torch.float16
    elif getattr(args, 'bf16', True):
        dtype = torch.bfloat16
    
    return QwenOFTDataCollator(
        multimodal_processor=model_components.get('multimodal_processor'),
        tokenizer=model_components.get('tokenizer'),
        dtype=dtype,
    )



