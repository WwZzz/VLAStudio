"""
Flow Matching Policy Module

A simple and efficient Flow Matching policy for imitation learning.
"""

from .modeling import (
    FlowMatchingPolicy,
    FlowMatchingConfig,
    SinusoidalTimeEmbedding,
    VelocityModel
)
from .data_utils import FlowMatchingDataProcessor, flow_matching_collator
from vlastudio.policy.trainer import BaseTrainer


def load_model(args):
    """
    Load Flow Matching model according to the framework requirements.
    
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
        model = FlowMatchingPolicy.from_pretrained(args.model_name_or_path)
        model.to(args.device if hasattr(args, 'device') else 'cuda')
        # Initialize processor and collator for inference
        model.data_processor = FlowMatchingDataProcessor(model.config)
        model.data_collator = flow_matching_collator
    else:
        # Create new model from configuration
        model_args = getattr(args, 'model_args', {})
        
        # Create configuration
        config = FlowMatchingConfig(
            state_dim=getattr(args, 'state_dim', model_args.get('state_dim', 10)),
            action_dim=getattr(args, 'action_dim', model_args.get('action_dim', 7)),
            chunk_size=getattr(args, 'chunk_size', model_args.get('chunk_size', 1)),
            time_dim=model_args.get('time_dim', 64),
            hidden_dim=model_args.get('hidden_dim', 256),
            num_layers=model_args.get('num_layers', 3),
            learning_rate=model_args.get('learning_rate', 1e-4),
            num_sampling_steps=model_args.get('num_sampling_steps', 100),
            use_ode_solver=model_args.get('use_ode_solver', True),
            ode_atol=model_args.get('ode_atol', 1e-5),
            ode_rtol=model_args.get('ode_rtol', 1e-5),
        )
        
        # Create model
        model = FlowMatchingPolicy(config)
        model.to(args.device if hasattr(args, 'device') else 'cuda')
    
    return {
        'model': model,
        'config': model.config
    }


def get_data_processor(args, model_components):
    """
    Get data processor for Flow Matching policy.
    
    The processor converts raw dataset samples to model input format.
    Delegates to FlowMatchingDataProcessor in data_utils.py.
    
    Args:
        args: Arguments
        model_components: Dictionary containing model and config
        
    Returns:
        FlowMatchingDataProcessor instance (callable)
    """
    model = model_components['model']
    return FlowMatchingDataProcessor(model.config)


def get_data_collator(args, model_components):
    """
    Get data collator for Flow Matching policy.
    
    The collator batches multiple processed samples together.
    Delegates to flow_matching_collator in data_utils.py.
    
    Args:
        args: Arguments
        model_components: Dictionary containing model and config
        
    Returns:
        flow_matching_collator function (callable)
    """
    return flow_matching_collator


# Use BaseTrainer for training
Trainer = BaseTrainer


__all__ = [
    'FlowMatchingPolicy',
    'FlowMatchingConfig',
    'SinusoidalTimeEmbedding',
    'VelocityModel',
    'load_model',
    'get_data_processor',
    'get_data_collator',
    'Trainer'
]

