"""
Reward Functions Module

This module provides modular reward functions for RL training.

Design Philosophy:
- Modular: Reward functions are independent modules, easy to replace and extend
- Composable: Support combining multiple reward functions
- Language-conditioned: Support VLA's language-conditioned rewards

Available reward functions:
- SparseReward: Only give reward when task is completed
- DenseReward: Distance-based reward
- LearnedReward: Learned reward model
- LanguageReward: Language-conditioned reward (for VLA)
- CompositeReward: Combine multiple reward functions

Note: Implementations are provided in separate files.
This __init__.py provides factory functions for creating reward functions.
"""

from typing import Type, Dict, Any

from .base_reward import BaseReward

# Registry for reward classes
_REWARD_REGISTRY: Dict[str, Type] = {}


def register_reward(name: str, reward_class: Type) -> None:
    """
    Register a reward function class.
    
    Args:
        name: Reward function name (e.g., 'sparse', 'dense')
        reward_class: Reward function class to register
    """
    _REWARD_REGISTRY[name.lower()] = reward_class


def get_reward_class(name_or_type: str) -> Type:
    """
    Get reward function class by name or type string.
    
    Args:
        name_or_type: Reward name (e.g., 'sparse') or full type path
                     (e.g., 'rl.rewards.sparse_reward.SparseReward')
    
    Returns:
        Reward function class
    
    Raises:
        ValueError: If reward function not found
    """
    # First check registry
    if name_or_type.lower() in _REWARD_REGISTRY:
        return _REWARD_REGISTRY[name_or_type.lower()]
    
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
            raise ValueError(f"Cannot import reward function from '{name_or_type}': {e}")
    
    raise ValueError(f"Unknown reward function: '{name_or_type}'. Available: {list(_REWARD_REGISTRY.keys())}")


def list_rewards() -> list:
    """List all registered reward functions."""
    return list(_REWARD_REGISTRY.keys())


__all__ = [
    'BaseReward',
    'register_reward',
    'get_reward_class',
    'list_rewards',
]

