"""
Action Manager Loader

This module provides utilities to load action manager configurations from YAML files
and instantiate the corresponding manager class.

Supports dynamic class loading from config files via the 'type' field.
"""

import yaml
import os
import importlib
from pathlib import Path
from typing import Dict, Any, Union, Optional
from loguru import logger


# Config search paths
CONFIG_SEARCH_PATHS = [
    Path(__file__).parent.parent.parent / "configs" / "action_manager",  # ILStudio/configs/action_manager
]


def _find_config_file(name: str) -> Optional[Path]:
    """Find config file by name in search paths."""
    # If it's already a path, return it
    if name.endswith('.yaml') or name.endswith('.yml'):
        path = Path(name)
        if path.exists():
            return path
        # Try search paths
        for search_path in CONFIG_SEARCH_PATHS:
            full_path = search_path / name
            if full_path.exists():
                return full_path
        return None
    
    # Try adding .yaml extension
    for search_path in CONFIG_SEARCH_PATHS:
        for ext in ['.yaml', '.yml']:
            full_path = search_path / f"{name}{ext}"
            if full_path.exists():
                return full_path
    return None


def _load_class_from_path(type_path: str):
    """
    Dynamically import a class from a full module path.
    
    Args:
        type_path: Full path like 'deploy.action_manager.base.BasicActionManager'
        
    Returns:
        The class object
    """
    from utils.extensions import resolve
    return resolve(type_path, kind="action_manager")


def load_action_manager(manager_name_or_path: str = None, config: Dict[str, Any] = None):
    """
    Load and instantiate an action manager.
    
    Args:
        manager_name_or_path: Can be:
            - A config name (e.g., 'basic', 'older_first') - looks up in configs/action_manager/
            - A path to a YAML config file
            - None (will use config parameter)
        config: Configuration dict containing manager settings.
                Must have 'type' field with full class path.
    
    Returns:
        An instantiated action manager instance
    
    Examples:
        # Load from config name (recommended)
        manager = load_action_manager('basic')
        manager = load_action_manager('temporal_agg')
        
        # Load from YAML file path
        manager = load_action_manager('configs/action_manager/truncated.yaml')
        
        # Load from config dict
        config = {
            'type': 'deploy.action_manager.truncated.TruncatedManager',
            'args': {
                'start_ratio': 0.1,
                'end_ratio': 0.2
            }
        }
        manager = load_action_manager(config=config)
    """
    manager_config = {}
    manager_class = None
    
    # Priority 1: Load from config dict
    if config is not None:
        manager_config = config.copy()
    
    # Priority 2: Load from name or path
    elif manager_name_or_path is not None:
        config_path = _find_config_file(manager_name_or_path)
        if config_path is None:
            raise FileNotFoundError(
                f"Config file not found: {manager_name_or_path}\n"
                f"Searched in: {[str(p) for p in CONFIG_SEARCH_PATHS]}"
            )
        
        with open(config_path, 'r') as f:
            manager_config = yaml.safe_load(f) or {}
        logger.info("Loaded action manager config from: {}", config_path)
    
    else:
        # Default to basic
        manager_name_or_path = 'basic'
        config_path = _find_config_file(manager_name_or_path)
        if config_path:
            with open(config_path, 'r') as f:
                manager_config = yaml.safe_load(f) or {}
            logger.info("No action manager specified, using default: basic")
        else:
            # Fallback to hardcoded BasicActionManager
            from .base import BasicActionManager
            logger.info("No action manager specified, using default: BasicActionManager")
            return BasicActionManager()
    
    # Get the type field (required)
    type_path = manager_config.get('type')
    if not type_path:
        raise ValueError(
            f"Config must have 'type' field with full class path.\n"
            f"Example: type: deploy.action_manager.base.BasicActionManager\n"
            f"Config: {manager_config}"
        )
    
    # Load the class
    manager_class = _load_class_from_path(type_path)
    
    # Get args for initialization
    args = manager_config.get('args', {})
    
    # Remove internal fields that shouldn't be passed to constructor
    args.pop('module_path', None)
    args.pop('class_name', None)
    
    # Instantiate manager with args as kwargs
    try:
        manager = manager_class(**args)
        logger.info("Successfully loaded action manager: {}", manager_class.__name__)
        return manager
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate {manager_class.__name__} with args {args}: {e}"
        ) from e
