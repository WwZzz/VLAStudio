"""
LLaVA-OFT Policy for ILStudio.

A lightweight VLA implementation using LLaVA-OneVision backbone with 
action prediction via L1 regression on last token hidden states.

Reference: UniAct implementation

This module provides three main interfaces:
  - load_model: Load model and processor
  - get_data_processor: Get data preprocessing function
  - get_data_collator: Get data collation function
"""

from .modeling import LlavaOFTConfig, LlavaOFTForPolicy
from .data_utils import LlavaOFTDataProcessor, LlavaOFTDataCollator

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
    no_lora_keywords = ['lm_head', 'action_head']
    
    if lora_module is None:
        lora_module = []
    if 'vision' not in lora_module:
        no_lora_keywords.extend(["vision_tower", "vision_model", "image_newline"])
    if 'llm' not in lora_module:
        no_lora_keywords.append("language_model")
    if 'projector' not in lora_module:
        no_lora_keywords.append("multi_modal_projector")
    
    for name, module in model.named_modules():
        if any(kw in name for kw in no_lora_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    
    return list(lora_module_names)


def load_model(args):
    """
    Load LLaVA-OFT model and processor.
    
    Args:
        args: Configuration namespace with:
            - vlm_model_name_or_path: Path to LLaVA model
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
        config = LlavaOFTConfig.from_pretrained(args.model_name_or_path)
        processor = AutoProcessor.from_pretrained(
            config.vlm_model_name_or_path,
            trust_remote_code=True
        )
        
        # Check for PEFT adapter
        peft_path = os.path.join(args.model_name_or_path, 'adapter_config.json')
        if os.path.exists(peft_path):
            model = LlavaOFTForPolicy(config=config)
            model = PeftModel.from_pretrained(model, args.model_name_or_path)
            
            # Load extra trainable parameters
            extra_path = os.path.join(args.model_name_or_path, 'extra_trainable.safetensors')
            if os.path.exists(extra_path):
                extra_state = load_file(extra_path, device="cpu")
                model.load_state_dict(extra_state, strict=False)
            
            model = model.merge_and_unload()
        else:
            model = LlavaOFTForPolicy.from_pretrained(args.model_name_or_path, config=config)
        
        # Set processor
        model.set_processor(processor)
        
        # Initialize data processor and collator for inference
        model.data_processor = LlavaOFTDataProcessor(
            processor=processor,
            chunk_size=config.chunk_size,
            action_token=config.action_token,
        )
        model.data_collator = LlavaOFTDataCollator(
            processor=processor,
            dtype=torch.bfloat16,
        )
        
    else:
        # === Training Mode ===
        processor = AutoProcessor.from_pretrained(
            args.vlm_model_name_or_path,
            trust_remote_code=True
        )
        
        config = LlavaOFTConfig(
            vlm_model_name_or_path=args.vlm_model_name_or_path,
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            chunk_size=args.chunk_size,
            action_token=getattr(args, 'action_token', '🔍'),
        )
        model = LlavaOFTForPolicy(config=config)
        
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
                if getattr(args, 'bf16', True):
                    model.to(torch.bfloat16)
                elif getattr(args, 'fp16', False):
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
        model.to(dtype=torch.bfloat16, device=args.device)
    
    # Save config
    if hasattr(args, 'output_dir') and args.is_training:
        os.makedirs(args.output_dir, exist_ok=True)
        model.config.save_pretrained(args.output_dir)
    
    return {
        'model': model,
        'processor': processor,
        'multimodal_processor': processor,
    }


def get_data_processor(args, model_components):
    """
    Get data processor for LLaVA-OFT.
    
    Args:
        args: Configuration namespace
        model_components: Dict from load_model
        
    Returns:
        LlavaOFTDataProcessor instance
    """
    return LlavaOFTDataProcessor(
        processor=model_components['processor'],
        chunk_size=args.chunk_size,
        action_token=getattr(args, 'action_token', '🔍'),
    )


def get_data_collator(args, model_components):
    """
    Get data collator for LLaVA-OFT.
    
    Args:
        args: Configuration namespace
        model_components: Dict from load_model
        
    Returns:
        LlavaOFTDataCollator instance
    """
    dtype = torch.bfloat16
    if getattr(args, 'fp16', False):
        dtype = torch.float16
    elif getattr(args, 'bf16', True):
        dtype = torch.bfloat16
    
    return LlavaOFTDataCollator(
        processor=model_components.get('processor'),
        dtype=dtype,
    )


# Optional: Export Trainer if custom training logic needed
try:
    from vlastudio.policy.trainer import BaseTrainer
    
    class Trainer(BaseTrainer):
        """LLaVA-OFT Trainer with custom loss computation."""
        
        EXTRA_FILE = "extra_trainable.safetensors"
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            loss = outputs["loss"]
            
            # Log metrics
            logging_steps = self.args.logging_steps
            if (self.state.global_step % logging_steps == 0) and (self.state.global_step != 0):
                log_dict = {}
                if "action_loss" in outputs:
                    log_dict["action_loss"] = outputs["action_loss"].detach().cpu().item()
                if log_dict:
                    self.log(log_dict)
            
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

