"""
Training Configuration Loader

This module provides utilities to load training configurations from YAML files
and convert them to transformers.TrainingArguments.
"""

import yaml
import os
from loguru import logger
from typing import Dict, Any, Optional
from pathlib import Path
import transformers
from ..utils import resolve_yaml


class TrainingConfig:
    """Training configuration loaded from YAML file - flexible parameter handling."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with configuration dictionary."""
        self.config_dict = config_dict or {}
        
        # Special handling for non-TrainingArguments parameters
        self.preload_data = self.config_dict.pop('preload_data', False)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Load training configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        
        # Handle type conversions for common parameters
        processed_config = {}
        for key, value in config_data.items():
                # Handle scientific notation strings that should be floats
            if key in ['adam_epsilon', 'learning_rate', 'weight_decay', 'warmup_ratio'] and isinstance(value, str):
                try:
                    processed_config[key] = float(value)
                except ValueError:
                    logger.warning(f"Could not convert {key}='{value}' to float, keeping as string")
                    processed_config[key] = value
            else:
                processed_config[key] = value
        
        return cls(processed_config)
    
    def to_training_arguments(self, hyper_args, **overrides) -> transformers.TrainingArguments:
        """Convert to transformers.TrainingArguments using HyperArguments and optional overrides."""
        # Start with a copy of the config dictionary
        config_dict = self.config_dict.copy()
        
        # Add required parameters from hyper_args
        config_dict['output_dir'] = hyper_args.output_dir
        
        # Set default logging_dir if not explicitly set or if set to default value
        if 'logging_dir' not in config_dict:
            import os
            config_dict['logging_dir'] = os.path.join(hyper_args.output_dir, 'log')
        
        # Apply overrides
        config_dict.update(overrides)

        # This project-level switch is intentionally not passed to
        # transformers.TrainingArguments. When enabled, the trainer removes
        # older checkpoints only after the newest checkpoint has been fully
        # written. save_total_limit=1 also gives trainers that do not inherit
        # from BaseTrainer the closest equivalent built-in behavior.
        save_latest_checkpoint_only = bool(
            config_dict.pop('save_latest_checkpoint_only', False)
        )
        if save_latest_checkpoint_only:
            if config_dict.get('load_best_model_at_end', False):
                raise ValueError(
                    "save_latest_checkpoint_only cannot be combined with "
                    "load_best_model_at_end because older best checkpoints "
                    "are intentionally removed"
                )
            if config_dict.get('save_total_limit') != 1:
                logger.info(
                    "save_latest_checkpoint_only=True: forcing save_total_limit=1"
                )
            config_dict['save_total_limit'] = 1
        
        # Auto-set eval_strategy if do_eval is True but eval_strategy is not set
        # This ensures evaluation actually runs when do_eval=True
        do_eval = config_dict.get('do_eval', False)
        if do_eval:
            # Check if eval_strategy is already set
            has_eval_strategy = 'eval_strategy' in config_dict
            if not has_eval_strategy:
                # Set eval_strategy to "steps" if eval_steps is set, otherwise "epoch"
                if 'eval_steps' in config_dict and config_dict.get('eval_steps') is not None:
                    config_dict['eval_strategy'] = 'steps'
                else:
                    config_dict['eval_strategy'] = 'epoch'
        
        # Create TrainingArguments - it will use default values for any missing parameters
        try:
            training_args = transformers.TrainingArguments(**config_dict)
        except TypeError as e:
            # If there are invalid parameters, filter them out and try again
            import inspect
            valid_params = set(inspect.signature(transformers.TrainingArguments.__init__).parameters.keys())
            valid_params.discard('self')  # Remove 'self' parameter
            
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}
            invalid_params = set(config_dict.keys()) - valid_params
            
            
            training_args = transformers.TrainingArguments(**filtered_config)
            if invalid_params:
                for k in invalid_params:
                    setattr(training_args, k, config_dict[k])

        setattr(
            training_args,
            'save_latest_checkpoint_only',
            save_latest_checkpoint_only,
        )
        return training_args


def load_training_config(config_path: str = "configs/training/default.yaml") -> TrainingConfig:
    """Load training configuration from YAML file. Accepts name or path."""
    base_dir = os.path.join(Path(__file__).resolve().parent)
    try:
        resolved = resolve_yaml(config_path, base_dir)
    except FileNotFoundError:
        resolved = config_path
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Training configuration not found: {resolved}")
    return TrainingConfig.from_yaml(resolved)


def create_training_arguments(config_path: str = "configs/training/default.yaml", hyper_args=None, **overrides) -> transformers.TrainingArguments:
    """Load training configuration and convert to TrainingArguments with overrides."""
    config = load_training_config(config_path)
    if hyper_args is None:
        raise ValueError("hyper_args must be provided to create TrainingArguments")
    return config.to_training_arguments(hyper_args, **overrides)
