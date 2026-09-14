"""
Data Collectors Module

This module provides data collectors for gathering experience from environments.

Design Philosophy:
- Responsibility separation: Separate data collection logic from trainer
- Environment abstraction: Support single environment, parallel environments, multiple environment types
- Raw data storage: Only store raw environment rewards, no reward function computation
- Statistics: Collect and return episode statistics

Available collectors:
- SimCollector: Collector for simulation environments
- RealCollector: Collector for real robot environments

Note: Implementations are provided in separate files.
This __init__.py provides factory functions for creating collectors.
"""

from typing import Type, Dict, Any

from .base_collector import BaseCollector, DummyCollector

# Registry for collector classes
_COLLECTOR_REGISTRY: Dict[str, Type] = {}


def register_collector(name: str, collector_class: Type) -> None:
    """
    Register a collector class.
    
    Args:
        name: Collector name (e.g., 'sim', 'real')
        collector_class: Collector class to register
    """
    _COLLECTOR_REGISTRY[name.lower()] = collector_class


def get_collector_class(name_or_type: str) -> Type:
    """
    Get collector class by name or type string.
    
    Args:
        name_or_type: Collector name (e.g., 'sim') or full type path
                     (e.g., 'rl.collectors.sim_collector.SimCollector')
    
    Returns:
        Collector class
    
    Raises:
        ValueError: If collector not found
    """
    # First check registry
    if name_or_type.lower() in _COLLECTOR_REGISTRY:
        return _COLLECTOR_REGISTRY[name_or_type.lower()]
    
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
            raise ValueError(f"Cannot import collector from '{name_or_type}': {e}")
    
    raise ValueError(f"Unknown collector: '{name_or_type}'. Available: {list(_COLLECTOR_REGISTRY.keys())}")


def list_collectors() -> list:
    """List all registered collectors."""
    return list(_COLLECTOR_REGISTRY.keys())


__all__ = [
    'BaseCollector',
    'DummyCollector',
    'register_collector',
    'get_collector_class',
    'list_collectors',
]

