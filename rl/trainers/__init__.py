"""
Trainers Module

This module provides trainers for coordinating RL training.

Design Philosophy:
- Coordinate environment, policy, and algorithm for training loop
- Support single algorithm and multiple algorithms training
- Support custom reward functions (applied during training, not data collection)
- Support evaluation during training

Available trainers:
- SimpleTrainer: Single machine trainer
- ParallelTrainer: Parallel environment trainer
- DistributedTrainer: Distributed training

Note: Implementations are provided in separate files.
This __init__.py provides factory functions for creating trainers.
"""

from typing import Type, Dict, Any

from .base_trainer import BaseTrainer
from .offpolicy_trainer import OffPolicyTrainer, OffPolicyTrainerConfig

# Registry for trainer classes
_TRAINER_REGISTRY: Dict[str, Type] = {
    'offpolicy': OffPolicyTrainer,
}


def register_trainer(name: str, trainer_class: Type) -> None:
    """
    Register a trainer class.
    
    Args:
        name: Trainer name (e.g., 'simple', 'parallel')
        trainer_class: Trainer class to register
    """
    _TRAINER_REGISTRY[name.lower()] = trainer_class


def get_trainer_class(name_or_type: str) -> Type:
    """
    Get trainer class by name or type string.
    
    Args:
        name_or_type: Trainer name (e.g., 'simple') or full type path
                     (e.g., 'rl.trainers.simple_trainer.SimpleTrainer')
    
    Returns:
        Trainer class
    
    Raises:
        ValueError: If trainer not found
    """
    # First check registry
    if name_or_type.lower() in _TRAINER_REGISTRY:
        return _TRAINER_REGISTRY[name_or_type.lower()]
    
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
            raise ValueError(f"Cannot import trainer from '{name_or_type}': {e}")
    
    raise ValueError(f"Unknown trainer: '{name_or_type}'. Available: {list(_TRAINER_REGISTRY.keys())}")


def list_trainers() -> list:
    """List all registered trainers."""
    return list(_TRAINER_REGISTRY.keys())


__all__ = [
    'BaseTrainer',
    'OffPolicyTrainer',
    'OffPolicyTrainerConfig',
    'register_trainer',
    'get_trainer_class',
    'list_trainers',
]

