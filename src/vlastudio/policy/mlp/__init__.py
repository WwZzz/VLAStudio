from .mlp import MLPPolicy, MLPPolicyConfig
from .data_utils import data_collator, MLPDataProcessor
import torch
import numpy as np


def load_model(args):
    """
    Load MLP model according to the framework requirements.
    
    This function provides two functionalities:
    1) Load original model directly
    2) Load checkpoint model trained by the framework
    
    Args:
        args: Arguments containing model configuration
        
    Returns:
        dict: Dictionary containing at least 'model' key with the model instance
    """
    if not args.is_training:
        # Load from pretrained checkpoint
        model = MLPPolicy.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model.to(args.device if hasattr(args, 'device') else 'cuda')
        # Initialize processor and collator for inference
        model.data_processor = MLPDataProcessor(
            state_dim=model.config.state_dim,
            use_camera=model.config.use_camera
        )
        model.data_collator = data_collator
    else:
        # Create new model from configuration
        model_args = getattr(args, 'model_args', {})
        
        # Calculate image dimension if using camera
        image_dim = None
        if getattr(args, 'use_camera', False):
            # Calculate flattened image dimension
            # This should be set based on the actual image dimensions from the environment
            image_shapes = getattr(args, 'image_sizes', None)
            image_channels = getattr(args, 'image_channels', 3)
            if image_shapes:
                # Sum up all camera image dimensions
                image_dim = int(sum(np.prod(shape) for shape in image_shapes)) * image_channels
            else:
                # Default assumption for debugging (can be overridden)
                image_dim = 3 * 224 * 224  # Default RGB image
        model_args['image_dim'] = image_dim
        # Extract configuration parameters
        config = MLPPolicyConfig(
            **model_args
        )
        
        model = MLPPolicy(config=config)
        
        # Move to device
        device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    
    return {'model': model}


def get_data_collator(args, model_components):
    """
    Get data collator for MLP policy.
    
    This function returns a callable object that can organize multiple samples
    processed by get_data_processor into a batch format that the model can accept.
    This shields the differences between different models and makes it easier
    to integrate into IL-Studio with minimal changes.
    
    Args:
        args: Arguments containing configuration
        model_components: Dictionary containing model components from load_model
        
    Returns:
        Callable: Data collator function
    """
    return data_collator


def get_data_processor(args, model_components):
    """
    Get data processor for MLP policy.
    
    This function returns a callable object that can convert data returned by
    dataset.__getitem__ into a format that the model can accept through format
    conversion, adapting to different input models. It returns a Callable object
    that can perform sample-level transformations.
    
    Args:
        args: Arguments containing configuration
        model_components: Dictionary containing model components from load_model
        
    Returns:
        Callable: Data processor object for sample-level transformation
    """
    # Extract configuration from model if available
    model = model_components.get('model')
    if hasattr(model, 'config'):
        state_dim = model.config.state_dim
    else:
        state_dim = getattr(args, 'state_dim', None)
    
    # Get use_camera setting
    use_camera = getattr(args, 'use_camera', False)
    if hasattr(model, 'config'):
        use_camera = model.config.use_camera
    
    # Create and return the processor
    processor = MLPDataProcessor(
        state_dim=state_dim,
        use_camera=use_camera
    )
    
    return processor


# Export the required interfaces for the framework
__all__ = [
    'load_model',
    'get_data_collator', 
    'get_data_processor',
    'MLPPolicy',
    'MLPPolicyConfig'
]
