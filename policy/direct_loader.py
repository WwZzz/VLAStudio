"""
Direct Policy Loader

This module provides utilities to directly load policy models from checkpoint paths
without requiring YAML configuration files. It automatically detects the policy type
from the checkpoint's config.json file.
"""

import json
import os
import importlib
from utils.extensions import resolve
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch


class DirectPolicyLoader:
    """Direct policy loader that loads models from checkpoint paths.

    Note: This implementation requires a policy_metadata.json saved alongside the
    checkpoint directory. Heuristic string-based detection has been removed to
    ensure extensibility. Train scripts save this metadata before training starts.
    """
    
    def __init__(self):
        self._loaded_modules = {}
    
    def detect_policy_type(self, checkpoint_path: str) -> str:
        """
        Detect the policy type strictly from checkpoint metadata (policy_metadata.json).
        Heuristic fallbacks have been removed to avoid hard-coded coupling.

        Args:
            checkpoint_path: Path to checkpoint directory or specific checkpoint

        Returns:
            Policy type string (e.g., 'act')
        """
        checkpoint_path = Path(checkpoint_path)
        
        # If it's a specific checkpoint subdirectory, go up to parent
        if checkpoint_path.name.startswith('checkpoint-'):
            checkpoint_path = checkpoint_path.parent
        
        # Require: saved policy metadata
        metadata_path = checkpoint_path / 'policy_metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                policy_module = metadata.get('policy_module')
                if policy_module:
                    # Extract policy name from module path (e.g., 'policy.act' -> 'act')
                    if policy_module.startswith('policy.') and policy_module.count('.') == 1:
                        return policy_module.split('.')[-1]
                    return policy_module if any(c in policy_module for c in '.:@') else policy_module + ':'
            except Exception as e:
                print(f"Warning: Failed to load policy metadata: {e}")

        raise ValueError(
            f"Could not detect policy type for {checkpoint_path}. "
            f"Missing policy_metadata.json. Please retrain (train.py now saves it before training) "
            f"or create the file with keys: {{'policy_module': 'policy.<name>', 'policy_name': '<name>'}}."
        )
    
    def load_policy_module(self, policy_type: str):
        """Load the policy module dynamically."""
        if policy_type in self._loaded_modules:
            return self._loaded_modules[policy_type]
        
        module_path = policy_type if any(c in policy_type for c in ".:@") else f"policy.{policy_type}"
        try:
            module = resolve(module_path, kind="policy", module=True)
            self._loaded_modules[policy_type] = module
            return module
        except ImportError as e:
            raise ImportError(f"Failed to import policy module {module_path}: {e}")
    
    def load_model_from_checkpoint(self, checkpoint_path: str, args) -> Dict[str, Any]:
        """
        Load a model directly from checkpoint path.
        
        Args:
            checkpoint_path: Path to checkpoint directory or specific checkpoint
            args: Arguments object with model_name_or_path set to checkpoint_path
            
        Returns:
            Dictionary containing 'model' and optionally 'config'
        """
        # Detect policy type
        policy_type = self.detect_policy_type(checkpoint_path)
        print(f"Detected policy type: {policy_type}")
        
        # Load checkpoint config and update args
        self._update_args_from_checkpoint(checkpoint_path, args)
        
        # Load policy module
        policy_module = self.load_policy_module(policy_type)
        
        # Set args.model_name_or_path to the checkpoint path
        args.model_name_or_path = checkpoint_path
        args.is_pretrained = True
        
        # Call the module's load_model function
        if not hasattr(policy_module, 'load_model'):
            raise AttributeError(f"Policy module {policy_type} must provide 'load_model' function")
        
        model_components = policy_module.load_model(args)
        
        return model_components
    
    def _update_args_from_checkpoint(self, checkpoint_path: str, args):
        """Update args object with parameters from checkpoint config.json."""
        checkpoint_path = Path(checkpoint_path)
        
        # If it's a specific checkpoint subdirectory, go up to parent
        if checkpoint_path.name.startswith('checkpoint-'):
            checkpoint_path = checkpoint_path.parent
        
        # Look for config.json in the checkpoint directory
        config_path = checkpoint_path / 'config.json'
        if not config_path.exists():
            print(f"Warning: config.json not found in {checkpoint_path}, using default args")
            return
        
        # Load config.json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update args with config parameters. Respect an explicit positive chunk_size
        # override from CLI; otherwise inherit the checkpoint default.
        if 'chunk_size' in config and getattr(args, 'chunk_size', -1) <= 0:
            args.chunk_size = config['chunk_size']
        if 'camera_names' in config:
            args.camera_names = config['camera_names']
        if 'image_size_primary' in config:
            args.image_size = config['image_size_primary']
        if 'image_size_wrist' in config:
            args.image_size = config['image_size_wrist']
        if 'action_dim' in config:
            args.action_dim = config['action_dim']
        if 'state_dim' in config:
            args.state_dim = config['state_dim']
        
        print(f"Updated args from checkpoint config: chunk_size={getattr(args, 'chunk_size', 'N/A')}, "
              f"camera_names={getattr(args, 'camera_names', 'N/A')}, "
              f"action_dim={getattr(args, 'action_dim', 'N/A')}")
    
    def get_data_processor(self, checkpoint_path: str, args, model_components):
        """Get data processor for the policy."""
        policy_type = self.detect_policy_type(checkpoint_path)
        policy_module = self.load_policy_module(policy_type)
        
        # Try to get data processor function
        if hasattr(policy_module, 'get_data_processor'):
            return policy_module.get_data_processor(args, model_components)
        
        return None
    
    def get_data_collator(self, checkpoint_path: str, args, model_components):
        """Get data collator for the policy."""
        policy_type = self.detect_policy_type(checkpoint_path)
        policy_module = self.load_policy_module(policy_type)
        
        # Try to get data collator function
        if hasattr(policy_module, 'get_data_collator'):
            return policy_module.get_data_collator(args, model_components)
        
        return None


# Global direct policy loader instance
direct_policy_loader = DirectPolicyLoader()


def load_model_from_checkpoint(checkpoint_path: str, args) -> Dict[str, Any]:
    """
    Convenience function to load a model directly from checkpoint path.
    
    Args:
        checkpoint_path: Path to checkpoint directory or specific checkpoint
        args: Arguments object
        
    Returns:
        Dictionary containing 'model' and optionally 'config'
    """
    return direct_policy_loader.load_model_from_checkpoint(checkpoint_path, args)


def get_data_processor_from_checkpoint(checkpoint_path: str, args, model_components):
    """Convenience function to get data processor from checkpoint."""
    return direct_policy_loader.get_data_processor(checkpoint_path, args, model_components)


def get_data_collator_from_checkpoint(checkpoint_path: str, args, model_components):
    """Convenience function to get data collator from checkpoint."""
    return direct_policy_loader.get_data_collator(checkpoint_path, args, model_components)
