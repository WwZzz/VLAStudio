"""
DiffusionTimmPolicy - Diffusion Policy with Timm Vision Encoder

This policy uses timm pretrained vision encoders (ResNet, ViT, etc.)
with a diffusion model for action generation.

Supports two backbone types (configurable via `backbone_type`):
    - 'unet': 1D U-Net (default)
    - 'transformer': Transformer decoder

Interfaces:
    - load_model(args) -> dict with 'model' key
    - get_data_collator(args, model_components) -> collator function
    - get_data_processor(args, model_components) -> processor callable (optional)
"""
from .modeling import DiffusionUnetTimmConfig, DiffusionUnetTimmModel
from .data_utils import data_collator, DataProcessor, get_data_processor

import torch
from loguru import logger


def _get_arg(args, model_args, key, default):
    """Get argument from model_args first, then from args directly."""
    if model_args and key in model_args:
        return model_args[key]
    return getattr(args, key, default)


def load_model(args):
    """
    Load or create a DiffusionUnetTimmModel.
    
    Args:
        args: Configuration namespace with:
            - is_training: bool
            - action_dim: int
            - state_dim: int
            - chunk_size: int
            - camera_names: list
            - image_size: tuple/list
            - model_args: dict (optional, for model-specific args)
            - Or model args directly under args (vision_model_name, etc.)
            - model_name_or_path: str (for loading pretrained)
    
    Returns:
        dict with 'model' key
    """
    if args.is_training:
        # Get model arguments - support both nested model_args and direct args
        model_args = getattr(args, 'model_args', None)
        if model_args is not None:
            # Convert DictConfig to dict for proper .get() behavior
            if hasattr(model_args, 'to_container'):
                model_args = dict(model_args)
            elif not isinstance(model_args, dict):
                model_args = dict(model_args)
        else:
            model_args = {}
        
        # Helper to get arg from model_args or directly from args
        def get_arg(key, default):
            return _get_arg(args, model_args, key, default)
        
        vision_model_name = get_arg('vision_model_name', 'resnet18')
        logger.info(f"Using backbone model: {vision_model_name}")
        
        # Get image size
        image_size = args.image_size
        if isinstance(image_size, str):
            image_size = eval(image_size)
        if isinstance(image_size, (list, tuple)):
            image_size = tuple(image_size)
        else:
            image_size = (image_size, image_size)
        
        # Get number of cameras
        camera_names = getattr(args, 'camera_names', ['primary'])
        num_cameras = len(camera_names)
        
        # Get backbone type
        backbone_type = get_arg('backbone_type', 'unet')
        logger.info(f"Using backbone type: {backbone_type}")
        
        # Create config
        config = DiffusionUnetTimmConfig(
            # Vision encoder
            vision_model_name=vision_model_name,
            vision_pretrained=get_arg('vision_pretrained', True),
            vision_frozen=get_arg('vision_frozen', False),
            vision_feature_dim=get_arg('vision_feature_dim', 64),
            feature_aggregation=get_arg('feature_aggregation', 'spatial_softmax'),
            num_kp=get_arg('num_kp', 32),
            imagenet_norm=get_arg('imagenet_norm', True),
            share_rgb_model=get_arg('share_rgb_model', False),
            # Data dimensions
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            num_cameras=num_cameras,
            image_size=image_size,
            # Diffusion
            chunk_size=args.chunk_size,
            num_inference_steps=get_arg('num_inference_steps', 10),
            num_train_timesteps=get_arg('num_train_timesteps', 100),
            beta_start=get_arg('beta_start', 0.0001),
            beta_end=get_arg('beta_end', 0.02),
            beta_schedule=get_arg('beta_schedule', 'squaredcos_cap_v2'),
            prediction_type=get_arg('prediction_type', 'epsilon'),
            clip_sample=get_arg('clip_sample', True),
            clip_sample_range=get_arg('clip_sample_range', 1.0),
            # Backbone type
            backbone_type=backbone_type,
            # UNet specific
            diffusion_step_embed_dim=get_arg('diffusion_step_embed_dim', 256),
            down_dims=tuple(get_arg('down_dims', [256, 512, 1024])),
            kernel_size=get_arg('kernel_size', 5),
            n_groups=get_arg('n_groups', 8),
            cond_predict_scale=get_arg('cond_predict_scale', True),
            # Transformer specific
            n_layer=get_arg('n_layer', 8),
            n_head=get_arg('n_head', 8),
            n_emb=get_arg('n_emb', 256),
            p_drop_attn=get_arg('p_drop_attn', 0.1),
            # Training
            input_pertub=get_arg('input_pertub', 0.1),
            train_diffusion_n_samples=get_arg('train_diffusion_n_samples', 1),
            ema_power=get_arg('ema_power', 0.75),
        )
        
        model = DiffusionUnetTimmModel(config)
    
    else:
        # Load pretrained model
        model = DiffusionUnetTimmModel.from_pretrained(
            args.model_name_or_path, 
            trust_remote_code=True
        )
        
        model_args = getattr(args, 'model_args', {})
        
        # Apply EMA weights if configured
        if model_args.get('using_ema', False) and model.ema is not None:
            model.ema.copy_to(model.parameters())
        
        # Update inference steps if specified
        if 'num_inference_steps' in model_args:
            model.config.num_inference_steps = model_args['num_inference_steps']
        
        model.to('cuda')
        
        # Attach data collator for inference
        model.data_collator = data_collator
    
    return {'model': model}


def get_data_collator(args, model_components):
    """
    Returns the data collator function.
    
    Args:
        args: Configuration arguments
        model_components: Dict with 'model' key
    
    Returns:
        data_collator function
    """
    return data_collator

