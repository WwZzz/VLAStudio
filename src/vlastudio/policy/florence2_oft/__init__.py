"""
Florence2-OFT Policy for ILStudio.

A lightweight VLA implementation using Florence2 encoder backbone with 
action prediction via L1 regression.

This module provides three main interfaces:
  - load_model: Load model and processor
  - get_data_processor: Get data preprocessing function
  - get_data_collator: Get data collation function
"""

from .modeling import Florence2OFTConfig, Florence2OFTForPolicy
from .data_utils import Florence2OFTDataProcessor, Florence2OFTDataCollator

import torch
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import save_file, load_file
import os

from loguru import logger


def find_all_linear_names(model, lora_module=None):
    """Find all linear layer names for LoRA injection."""
    cls = torch.nn.Linear
    lora_module_names = set()
    no_lora_keywords = ['lm_head', 'action_head', 'decoder']
    
    if lora_module is None:
        lora_module = []
    if 'vision' not in lora_module:
        no_lora_keywords.extend(["vision_tower", "image_projection", "vision"])
    if 'encoder' not in lora_module:
        no_lora_keywords.append("encoder")
    
    for name, module in model.named_modules():
        if any(kw in name for kw in no_lora_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    
    return list(lora_module_names)


def load_model(args):
    """
    Load Florence2-OFT model and processor.
    
    Args:
        args: Configuration namespace with:
            - vlm_model_name_or_path: Path to Florence2 model
            - model_name_or_path: Path to checkpoint (for inference)
            - action_dim: Action dimension
            - state_dim: State dimension
            - chunk_size: Action prediction horizon
            - is_training: Whether loading for training
            - lora_enable: Whether to use LoRA
            - lora_r, lora_alpha, lora_dropout, lora_bias: LoRA config
            - lora_module: List of modules to apply LoRA to
            
    Returns:
        Dict with 'model', 'processor' keys.
    """
    args.device = getattr(args, 'device', 'cuda')
    
    if not getattr(args, 'is_training', True):
        # === Inference Mode ===
        config = Florence2OFTConfig.from_pretrained(args.model_name_or_path)
        processor = AutoProcessor.from_pretrained(
            config.vlm_model_name_or_path,
            trust_remote_code=True
        )
        
        # Add action token to tokenizer BEFORE loading model to ensure vocab size matches
        # This is critical because checkpoint may contain the added token
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        action_token = getattr(config, 'action_token', '🔍')
        
        # Check if token needs to be added
        action_token_only = tokenizer(action_token, add_special_tokens=False)["input_ids"]
        tokens_per_action = len(action_token_only)
        num_added = 0
        
        if tokens_per_action != 1 and hasattr(tokenizer, 'add_tokens'):
            num_added = tokenizer.add_tokens([action_token], special_tokens=True)
            if num_added > 0:
                logger.info(
                    f"Added action token '{action_token}' as special token before loading model. "
                    f"Tokenizer vocabulary size increased by {num_added}."
                )
        
        # Check for PEFT adapter
        peft_path = os.path.join(args.model_name_or_path, 'adapter_config.json')
        if os.path.exists(peft_path):
            model = Florence2OFTForPolicy(config=config)
            # Resize embeddings if token was added
            if num_added > 0:
                if hasattr(model.vlm, 'resize_token_embeddings'):
                    model.vlm.resize_token_embeddings(len(tokenizer))
                elif hasattr(model.vlm, 'language_model') and hasattr(model.vlm.language_model, 'resize_token_embeddings'):
                    model.vlm.language_model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(model, args.model_name_or_path)
            
            # Load extra trainable parameters
            extra_path = os.path.join(args.model_name_or_path, 'extra_trainable.safetensors')
            if os.path.exists(extra_path):
                extra_state = load_file(extra_path, device="cpu")
                model.load_state_dict(extra_state, strict=False)
            
            model = model.merge_and_unload()
        else:
            # Create model first, then resize embeddings if token was added
            # This ensures vocab size matches checkpoint before loading weights
            model = Florence2OFTForPolicy(config=config)
            
            # Resize embeddings if token was added (before loading checkpoint)
            if num_added > 0:
                if hasattr(model.vlm, 'resize_token_embeddings'):
                    model.vlm.resize_token_embeddings(len(tokenizer))
                    logger.info("Resized VLM embeddings to match tokenizer vocabulary size")
                elif hasattr(model.vlm, 'language_model') and hasattr(model.vlm.language_model, 'resize_token_embeddings'):
                    model.vlm.language_model.resize_token_embeddings(len(tokenizer))
                    logger.info("Resized language_model embeddings to match tokenizer vocabulary size")
            
            # Now load checkpoint weights
            # Use from_pretrained but skip the model creation part by loading state dict manually
            try:
                # Try to load state dict directly
                state_dict_path = os.path.join(args.model_name_or_path, 'pytorch_model.bin')
                if os.path.exists(state_dict_path):
                    state_dict = torch.load(state_dict_path, map_location='cpu')
                else:
                    # Try safetensors format
                    state_dict_path = os.path.join(args.model_name_or_path, 'model.safetensors')
                    if os.path.exists(state_dict_path):
                        state_dict = load_file(state_dict_path)
                    else:
                        raise FileNotFoundError(f"No checkpoint file found in {args.model_name_or_path}")
                
                model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded checkpoint weights successfully")
            except Exception as e:
                # Fallback: use from_pretrained with ignore_mismatched_sizes
                logger.warning(f"Failed to load state dict directly: {e}. Using from_pretrained with ignore_mismatched_sizes")
                model = Florence2OFTForPolicy.from_pretrained(
                    args.model_name_or_path, 
                    config=config,
                    ignore_mismatched_sizes=True
                )
                # Resize again after from_pretrained
                if num_added > 0:
                    if hasattr(model.vlm, 'resize_token_embeddings'):
                        model.vlm.resize_token_embeddings(len(tokenizer))
                    elif hasattr(model.vlm, 'language_model') and hasattr(model.vlm.language_model, 'resize_token_embeddings'):
                        model.vlm.language_model.resize_token_embeddings(len(tokenizer))
        
        # Set processor (this will verify tokenization but won't add tokens again)
        model.set_processor(processor)
        
        # Get image_size from config
        image_size = getattr(config, 'image_size', None)
        
        # Initialize data processor and collator for inference
        model.data_processor = Florence2OFTDataProcessor(
            processor=processor,
            chunk_size=config.chunk_size,
            action_token=config.action_token,
            image_size=image_size,
        )
        model.data_collator = Florence2OFTDataCollator(
            processor=processor,
            dtype=torch.float16,
        )
        
    else:
        # === Training Mode ===
        processor = AutoProcessor.from_pretrained(
            args.vlm_model_name_or_path,
            trust_remote_code=True
        )
        
        config = Florence2OFTConfig(
            vlm_model_name_or_path=args.vlm_model_name_or_path,
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            chunk_size=args.chunk_size,
            action_token=getattr(args, 'action_token', '🔍'),
            image_size=getattr(args, 'image_size', None),
        )
        model = Florence2OFTForPolicy(config=config)
        
        # Set processor
        model.set_processor(processor)
        
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
                elif getattr(args, 'fp16', True):
                    model.to(torch.float16)
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
    
    # Store references
    model.multimodal_processor = processor
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
        model.to(dtype=torch.float16, device=args.device)
    
    # Keep action_head in float32 for numerical stability
    unwrapped_model = model.base_model if hasattr(model, 'base_model') else model
    if hasattr(unwrapped_model, 'action_head'):
        unwrapped_model.action_head.float()
    
    # Save config
    if hasattr(args, 'output_dir') and args.is_training:
        os.makedirs(args.output_dir, exist_ok=True)
        model.config.save_pretrained(args.output_dir)
    
    return {
        'model': model,
        'processor': processor,
        'multimodal_processor': processor,  # Alias for compatibility
    }


def get_data_processor(args, model_components):
    """
    Get data processor for Florence2-OFT.
    
    Args:
        args: Configuration namespace
        model_components: Dict from load_model
        
    Returns:
        Florence2OFTDataProcessor instance
    """
    # Get image_size from args
    image_size = getattr(args, 'image_size', None)
    
    return Florence2OFTDataProcessor(
        processor=model_components['processor'],
        chunk_size=args.chunk_size,
        task_prompt=getattr(args, 'task_prompt', "Predict the next robot actions:"),
        action_token=getattr(args, 'action_token', '🔍'),
        image_size=image_size,
    )


def get_data_collator(args, model_components):
    """
    Get data collator for Florence2-OFT.
    
    Args:
        args: Configuration namespace
        model_components: Dict from load_model
        
    Returns:
        Florence2OFTDataCollator instance
    """
    dtype = torch.float16
    if getattr(args, 'bf16', False):
        dtype = torch.bfloat16
    elif getattr(args, 'fp16', True):
        dtype = torch.float16
    
    return Florence2OFTDataCollator(
        processor=model_components.get('processor'),
        dtype=dtype,
    )


# Optional: Export Trainer if custom training logic needed
try:
    from vlastudio.policy.trainer import BaseTrainer
    
    class Trainer(BaseTrainer):
        """Florence2-OFT Trainer with custom loss computation."""
        
        EXTRA_FILE = "extra_trainable.safetensors"
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            loss = outputs["loss"]
            
            # Log metrics
            # logging_steps = self.args.logging_steps
            # if (self.state.global_step % logging_steps == 0) and (self.state.global_step != 0):
            #     log_dict = {}
            #     if "action_loss" in outputs:
            #         log_dict["action_loss"] = outputs["action_loss"].detach().cpu().item()
            #     if log_dict:
            #         self.log(log_dict)
            
            return (loss, outputs) if return_outputs else loss
        
        def save_model(self, output_dir=None, _internal_call=False):
            """Save model with extra trainable parameters."""
            output_dir = output_dir or self.args.output_dir
            super().save_model(output_dir, _internal_call)
            self.model.config.save_pretrained(output_dir)
            
            if not self.is_world_process_zero():
                return
            
            # Get non-LoRA trainable parameters
            trainable_keys = [
                n for n, p in self.accelerator.unwrap_model(self.model).named_parameters()
                if "lora_" not in n and p.requires_grad
            ]
            
            if self.is_deepspeed_enabled:
                if self.accelerator.deepspeed_config["zero_optimization"]["stage"] == 3:
                    state_dict = self.deepspeed._zero3_consolidated_16bit_state_dict()
                else:
                    from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
                    state_dict = clone_tensors_for_torch_save(
                        self.accelerator.unwrap_model(self.deepspeed).state_dict()
                    )
            else:
                state_dict = self.model.state_dict()
            
            extra_state = {k: v for k, v in state_dict.items() if k in trainable_keys}
            if extra_state:
                os.makedirs(output_dir, exist_ok=True)
                save_file(extra_state, os.path.join(output_dir, self.EXTRA_FILE))
                
except ImportError:
    # BaseTrainer not available, skip Trainer export
    pass

