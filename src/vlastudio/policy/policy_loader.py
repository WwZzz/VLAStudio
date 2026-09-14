"""
Policy Configuration Loader

This module provides utilities to load model configurations from YAML files
and dynamically import and instantiate model modules.
"""

import yaml
import importlib
from vlastudio.utils.extensions import resolve
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PolicyConfig:
    """Configuration for a policy model."""
    name: str
    module_path: str
    config_class: str
    model_class: str
    pretrained_config: Optional[Dict[str, Any]] = None
    model_args: Optional[Dict[str, Any]] = None
    config_params: Optional[Dict[str, Any]] = None  # Parameters for config class initialization
    data_processor: Optional[str] = None
    data_collator: Optional[str] = None
    trainer_class: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PolicyConfig':
        """Load policy configuration from YAML file.
        Supports both old and new unified formats.
        """
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Handle new unified format
        name = config_data.get('name')
        
        # Extract module_path from either 'type' or 'module_path'
        module_path = config_data.get('type') or config_data.get('module_path')
        
        # Extract model_args from either 'args' or 'model_args'
        model_args = config_data.get('args') or config_data.get('model_args', {})
        
        return cls(
            name=name,
            module_path=module_path,
            config_class=config_data.get('config_class'),
            model_class=config_data.get('model_class'),
            pretrained_config=config_data.get('pretrained_config'),
            model_args=model_args,
            config_params=config_data.get('config_params', {}),
            data_processor=config_data.get('data_processor'),
            data_collator=config_data.get('data_collator'),
            trainer_class=config_data.get('trainer_class')
        )


class PolicyLoader:
    """Dynamic policy loader that loads models based on YAML configurations."""
    
    def __init__(self, policy_dir: str = "configs/policy"):
        self.policy_dir = Path(policy_dir)
        self._loaded_modules = {}
    
    def load_policy_config(self, policy_name_or_path: str) -> PolicyConfig:
        """Load policy configuration from YAML file."""
        # Check if it's a full path or just a policy name
        if policy_name_or_path.endswith('.yaml') or '/' in policy_name_or_path:
            # It's a full path
            yaml_path = Path(policy_name_or_path)
            if not yaml_path.exists():
                raise FileNotFoundError(f"Policy configuration not found: {yaml_path}")
        else:
            # It's just a policy name, construct the path
            yaml_path = self.policy_dir / f"{policy_name_or_path}.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"Policy configuration not found: {yaml_path}")
        
        return PolicyConfig.from_yaml(str(yaml_path))
    
    def load_model_module(self, policy_config: PolicyConfig):
        """Dynamically load the model module."""
        if policy_config.module_path in self._loaded_modules:
            return self._loaded_modules[policy_config.module_path]
        
        try:
            module = resolve(policy_config.module_path, kind="policy", module=True)
            self._loaded_modules[policy_config.module_path] = module
            return module
        except ImportError as e:
            raise ImportError(f"Failed to import module {policy_config.module_path}: {e}")
    
    def create_config_instance(self, policy_config: PolicyConfig, args=None, task_config=None, phase='training'):
        """Create a config instance using parameters from args.
        
        NOTE: This method is deprecated. Parameter processing should be done by ConfigLoader.
        This method is kept only for backward compatibility.
        """
        # Load the module to get access to the config class
        model_module = self.load_model_module(policy_config)
        
        # Get the config class
        if policy_config.config_class is None:
            # If no config class specified, skip config creation
            return None
            
        if not hasattr(model_module, policy_config.config_class):
            raise AttributeError(f"Module {policy_config.module_path} does not have config class {policy_config.config_class}")
        
        config_class = getattr(model_module, policy_config.config_class)
        
        # ConfigLoader should have already processed all parameters and put them in args
        # Extract config parameters from args
        config_kwargs = {}
        
        # Get all attributes from args that could be config parameters
        if args:
            for attr_name in dir(args):
                if not attr_name.startswith('_'):  # Skip private attributes
                    value = getattr(args, attr_name)
                    if value is not None:
                        config_kwargs[attr_name] = value
        
        # Filter to only include parameters that the config class actually accepts
        import inspect
        try:
            sig = inspect.signature(config_class.__init__)
            valid_params = set(sig.parameters.keys()) - {'self'}
            config_kwargs = {k: v for k, v in config_kwargs.items() if k in valid_params}
        except Exception:
            # If we can't inspect the signature, just try with all parameters
            pass
        
        # Create and return the config instance
        try:
            config_instance = config_class(**config_kwargs)
            return config_instance
        except Exception as e:
            raise ValueError(f"Failed to create config instance with parameters {config_kwargs}: {e}")
    
    def load_model_with_config(self, policy_name_or_path: str, args, task_config=None, phase='training') -> Dict[str, Any]:
        """Load model using policy configuration - simplified to use direct load_model."""
        return self.load_model(policy_name_or_path, args)
    
    def load_model_for_training(self, policy_name_or_path: str, args, task_config=None) -> Dict[str, Any]:
        """Load model for training phase - parameter processing should be done by ConfigLoader."""
        return self.load_model(policy_name_or_path, args)
    
    def load_model_for_evaluation(self, policy_name_or_path: str, args) -> Dict[str, Any]:
        """Load model for evaluation phase - parameter processing should be done by ConfigLoader."""
        return self.load_model(policy_name_or_path, args)
    
    def load_model(self, policy_name_or_path: str, args) -> Dict[str, Any]:
        """Load model using policy configuration.
        
        Note: All parameter processing should be done by ConfigLoader before calling this method.
        This method should only load the module and call its load_model function.
        """
        # Load policy configuration
        policy_config = self.load_policy_config(policy_name_or_path)
        
        # Load model module
        model_module = self.load_model_module(policy_config)
        
        # Verify required methods exist
        if not hasattr(model_module, 'load_model'):
            raise AttributeError(f"Module {policy_config.module_path} must provide 'load_model' function")
        
        # Only set args.model_args if it doesn't exist yet
        # ConfigLoader.merge_all_parameters() may have already set this with merged parameters
        if not hasattr(args, 'model_args') or args.model_args is None:
            args.model_args = policy_config.model_args
        
        # Call the module's load_model function directly with processed args
        # All parameter processing should have been done by ConfigLoader already
        model_components = model_module.load_model(args)
        
        return model_components
    
    def get_data_processor(self, policy_name_or_path: str, args, model_components):
        """Get data processor for the policy."""
        policy_config = self.load_policy_config(policy_name_or_path)
        model_module = self.load_model_module(policy_config)
        
        # Always try to get get_data_processor function from the module
        if hasattr(model_module, 'get_data_processor'):
            processor_func = getattr(model_module, 'get_data_processor')
            return processor_func(args, model_components)
        
        # Fallback to config-specified processor
        if policy_config.data_processor and hasattr(model_module, policy_config.data_processor):
            processor_func = getattr(model_module, policy_config.data_processor)
            return processor_func(args, model_components)
        
        return None
    
    def get_data_collator(self, policy_name_or_path: str, args, model_components):
        """Get data collator for the policy."""
        policy_config = self.load_policy_config(policy_name_or_path)
        model_module = self.load_model_module(policy_config)
        
        # Always try to get get_data_collator function from the module
        if hasattr(model_module, 'get_data_collator'):
            collator_func = getattr(model_module, 'get_data_collator')
            return collator_func(args, model_components)
        
        # Fallback to config-specified collator
        if policy_config.data_collator and hasattr(model_module, policy_config.data_collator):
            collator_func = getattr(model_module, policy_config.data_collator)
            return collator_func(args, model_components)
        
        return None
    
    def get_trainer_class(self, policy_name_or_path: str):
        """Get trainer class for the policy."""
        policy_config = self.load_policy_config(policy_name_or_path)
        model_module = self.load_model_module(policy_config)
        
        if policy_config.trainer_class and hasattr(model_module, policy_config.trainer_class):
            return getattr(model_module, policy_config.trainer_class)
        
        return None
    
    def list_available_policies(self) -> list:
        """List all available policy configurations."""
        yaml_files = list(self.policy_dir.glob("*.yaml"))
        return [f.stem for f in yaml_files]


# Global policy loader instance
policy_loader = PolicyLoader()


def load_policy_model(policy_name_or_path: str, args) -> Dict[str, Any]:
    """Convenience function to load a policy model."""
    return policy_loader.load_model(policy_name_or_path, args)


def load_policy_model_with_config(policy_name_or_path: str, args, task_config=None, phase='training') -> Dict[str, Any]:
    """Convenience function to load a policy model with custom config from YAML."""
    return policy_loader.load_model_with_config(policy_name_or_path, args, task_config, phase)

def load_policy_model_for_training(policy_name_or_path: str, args, task_config=None) -> Dict[str, Any]:
    """Convenience function to load a policy model for training - uses task config parameters."""
    return policy_loader.load_model_for_training(policy_name_or_path, args, task_config)

def load_policy_model_for_evaluation(policy_name_or_path: str, args) -> Dict[str, Any]:
    """Convenience function to load a policy model for evaluation - uses saved model config parameters."""
    return policy_loader.load_model_for_evaluation(policy_name_or_path, args)


def get_policy_data_processor(policy_name_or_path: str, args, model_components):
    """Convenience function to get data processor."""
    return policy_loader.get_data_processor(policy_name_or_path, args, model_components)


def get_policy_data_collator(policy_name_or_path: str, args, model_components):
    """Convenience function to get data collator."""
    return policy_loader.get_data_collator(policy_name_or_path, args, model_components)


def get_policy_trainer_class(policy_name_or_path: str):
    """Convenience function to get trainer class."""
    return policy_loader.get_trainer_class(policy_name_or_path)
