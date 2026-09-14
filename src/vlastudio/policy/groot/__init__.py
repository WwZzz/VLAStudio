"""
GR00T Policy for ILStudio.

NVIDIA's GR00T-N1.5 foundation model for robotics, featuring:
  - Eagle2 vision-language backbone
  - Flow matching diffusion action head
  - Multi-embodiment support

Reference:
  - https://huggingface.co/nvidia/GR00T-N1.5-3B
  - Isaac-GR00T

This module provides three main interfaces:
  - load_model: Load model and processor
  - get_data_processor: Get data preprocessing function
  - get_data_collator: Get data collation function
"""

import sys
from pathlib import Path

# Add third_party/lerobot to path for imports
LEROBOT_PATH = Path(__file__).resolve().parents[2] / "third_party" / "lerobot" / "src"
if str(LEROBOT_PATH) not in sys.path:
    sys.path.insert(0, str(LEROBOT_PATH))

from .modeling import GrootOFTConfig, GrootOFTForPolicy
from .data_utils import GrootDataProcessor, GrootDataCollator, _build_eagle_processor
from .trainer import Trainer
import torch
from safetensors.torch import save_file, load_file
import os
from loguru import logger


def load_model(args):
    """
    Load GR00T model and processor.
    
    Args:
        args: Configuration namespace with:
            - base_model_path: Path to GR00T model (default: nvidia/GR00T-N1.5-3B)
            - model_name_or_path: Path to checkpoint (for inference)
            - action_dim: Action dimension
            - state_dim: State dimension
            - chunk_size: Action prediction horizon
            - is_training: Whether loading for training
            - tune_llm, tune_visual, tune_projector, tune_diffusion_model: Fine-tuning flags
            
    Returns:
        Dict with 'model', 'processor' keys.
    """
    args.device = getattr(args, 'device', 'cuda')
    tokenizer_assets_repo = getattr(args, 'tokenizer_assets_repo', 'lerobot/eagle2hg-processor-groot-n1p5')
    
    # Get Eagle processor - use the custom builder that handles cache properly
    eagle_processor = None
    try:
        eagle_processor = _build_eagle_processor(tokenizer_assets_repo)
        logger.info(f"Loaded Eagle processor from {tokenizer_assets_repo}")
    except Exception as e:
        logger.warning(f"Could not load Eagle processor: {e}")
    
    if not getattr(args, 'is_training', True):
        # === Inference Mode ===
        config = GrootOFTConfig.from_pretrained(args.model_name_or_path)
        model = GrootOFTForPolicy(config=config)
        
        # Load checkpoint weights
        checkpoint_path = os.path.join(args.model_name_or_path, 'model.safetensors')
        if os.path.exists(checkpoint_path):
            state_dict = load_file(checkpoint_path, device="cpu")
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        # Set processor
        model.set_processor(eagle_processor)
        
        # Initialize data processor and collator for inference
        model.data_processor = GrootDataProcessor(
            eagle_processor=eagle_processor,
            chunk_size=config.chunk_size,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            embodiment_tag=config.embodiment_tag,
        )
        model.data_collator = GrootDataCollator(
            eagle_processor=eagle_processor,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            chunk_size=config.chunk_size,
            dtype=torch.bfloat16,
        )
        
    else:
        # === Training Mode ===
        config = GrootOFTConfig(
            base_model_path=getattr(args, 'base_model_path', 'nvidia/GR00T-N1.5-3B'),
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            chunk_size=min(args.chunk_size, 16),  # GR00T max is 16
            n_action_steps=getattr(args, 'n_action_steps', min(args.chunk_size, 16)),
            max_state_dim=getattr(args, 'max_state_dim', 64),
            max_action_dim=getattr(args, 'max_action_dim', 32),
            embodiment_tag=getattr(args, 'embodiment_tag', 'new_embodiment'),
            tokenizer_assets_repo=getattr(args, 'tokenizer_assets_repo', 'lerobot/eagle2hg-processor-groot-n1p5'),
            tune_llm=getattr(args, 'tune_llm', False),
            tune_visual=getattr(args, 'tune_visual', False),
            tune_projector=getattr(args, 'tune_projector', True),
            tune_diffusion_model=getattr(args, 'tune_diffusion_model', True),
            use_bf16=getattr(args, 'use_bf16', True),
        )
        model = GrootOFTForPolicy(config=config)
        
        # Set processor
        model.set_processor(eagle_processor)
    
    # Store references
    model.processor = eagle_processor
    model.config.use_cache = False
    
    # Move to device
    if getattr(args, 'use_bf16', True):
        model.to(dtype=torch.bfloat16, device=args.device)
    else:
        model.to(device=args.device)
    
    # Save config
    if hasattr(args, 'output_dir') and getattr(args, 'is_training', True):
        os.makedirs(args.output_dir, exist_ok=True)
        model.config.save_pretrained(args.output_dir)
    
    return {
        'model': model,
        'processor': eagle_processor,
        'eagle_processor': eagle_processor,
    }


def get_data_processor(args, model_components):
    """
    Get data processor for GR00T.
    
    Args:
        args: Configuration namespace
        model_components: Dict from load_model
        
    Returns:
        GrootDataProcessor instance
    """
    tokenizer_assets_repo = getattr(args, 'tokenizer_assets_repo', 'lerobot/eagle2hg-processor-groot-n1p5')
    
    return GrootDataProcessor(
        eagle_processor=model_components.get('eagle_processor'),
        chunk_size=min(args.chunk_size, 16),
        max_state_dim=getattr(args, 'max_state_dim', 64),
        max_action_dim=getattr(args, 'max_action_dim', 32),
        embodiment_tag=getattr(args, 'embodiment_tag', 'new_embodiment'),
        tokenizer_assets_repo=tokenizer_assets_repo,
    )


def get_data_collator(args, model_components):
    """
    Get data collator for GR00T.
    
    Args:
        args: Configuration namespace
        model_components: Dict from load_model
        
    Returns:
        GrootDataCollator instance
    """
    dtype = torch.bfloat16 if getattr(args, 'use_bf16', True) else torch.float32
    tokenizer_assets_repo = getattr(args, 'tokenizer_assets_repo', 'lerobot/eagle2hg-processor-groot-n1p5')
    
    return GrootDataCollator(
        eagle_processor=model_components.get('eagle_processor'),
        max_state_dim=getattr(args, 'max_state_dim', 64),
        max_action_dim=getattr(args, 'max_action_dim', 32),
        chunk_size=min(args.chunk_size, 16),
        dtype=dtype,
        tokenizer_assets_repo=tokenizer_assets_repo,
    )




