"""
RL Algorithms Module

This module provides various RL algorithm implementations.

Available algorithms:
- PPO: Proximal Policy Optimization
- SAC: Soft Actor-Critic
- TD3: Twin Delayed DDPG
- DPO: Direct Preference Optimization (for VLA)
- GRPO: Group Relative Policy Optimization (for VLA)
- REINFORCE: Basic policy gradient

Note: Implementations are provided in separate files.
This __init__.py provides factory functions for creating algorithms.
"""

from typing import Type, Dict, Any, Optional, Tuple

# Registry for algorithm classes and their config classes
_ALGORITHM_REGISTRY: Dict[str, Type] = {}
_CONFIG_REGISTRY: Dict[str, Type] = {}


def register_algorithm(name: str, algorithm_class: Type, config_class: Type = None) -> None:
    """
    Register an algorithm class and its config class.
    
    Args:
        name: Algorithm name (e.g., 'ppo', 'sac')
        algorithm_class: Algorithm class to register
        config_class: Config class for the algorithm (optional)
    """
    _ALGORITHM_REGISTRY[name.lower()] = algorithm_class
    if config_class is not None:
        _CONFIG_REGISTRY[name.lower()] = config_class


def get_algorithm_class(name_or_type: str) -> Type:
    """
    Get algorithm class by name or type string.
    
    Args:
        name_or_type: Algorithm name (e.g., 'ppo') or full type path 
                     (e.g., 'rl.algorithms.ppo.PPOAlgorithm')
    
    Returns:
        Algorithm class
    
    Raises:
        ValueError: If algorithm not found
    """
    # First check registry
    if name_or_type.lower() in _ALGORITHM_REGISTRY:
        return _ALGORITHM_REGISTRY[name_or_type.lower()]
    
    # Try to import from type path
    if '.' in name_or_type:
        try:
            parts = name_or_type.rsplit('.', 1)
            module_path = parts[0]
            class_name = parts[1]
            
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot import algorithm from '{name_or_type}': {e}")
    
    raise ValueError(f"Unknown algorithm: '{name_or_type}'. Available: {list(_ALGORITHM_REGISTRY.keys())}")


def get_config_class(name: str) -> Optional[Type]:
    """
    Get config class for an algorithm.
    
    Args:
        name: Algorithm name (e.g., 'td3')
    
    Returns:
        Config class or None if not registered
    """
    return _CONFIG_REGISTRY.get(name.lower())


def get_algorithm_and_config(name: str) -> Tuple[Type, Optional[Type]]:
    """
    Get both algorithm class and config class.
    
    Args:
        name: Algorithm name
    
    Returns:
        Tuple of (algorithm_class, config_class)
    """
    return get_algorithm_class(name), get_config_class(name)


def list_algorithms() -> list:
    """List all registered algorithms."""
    return list(_ALGORITHM_REGISTRY.keys())


__all__ = [
    'register_algorithm',
    'get_algorithm_class',
    'get_config_class',
    'get_algorithm_and_config',
    'list_algorithms',
]


# Auto-register built-in algorithms when this module is imported
def _register_builtin_algorithms():
    """Import algorithm submodules to trigger their registration."""
    try:
        from . import td3  # noqa: F401
    except ImportError:
        pass
    # Add more algorithms here as they are implemented
    # try:
    #     from . import sac
    # except ImportError:
    #     pass


_register_builtin_algorithms()
