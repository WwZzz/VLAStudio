"""
RL Infrastructure Module

This module provides infrastructure components for reproducibility and stability:
- SeedManager: Unified random seed management
- Logger: Training logging system (TensorBoard, WandB, etc.)
- Checkpoint: Model and training state checkpoint management
- Callback: Training callback system for hooks and monitoring
- Distributed: Distributed training support utilities

Design Philosophy:
- Reproducibility: Ensure experiments can be reproduced exactly
- Stability: Provide robust training infrastructure
- Modularity: Each component can be used independently
- Extensibility: Easy to add new logging backends, callbacks, etc.
"""

from .seed_manager import SeedManager, set_global_seed, get_global_seed
from .logger import BaseLogger, ConsoleLogger, TensorBoardLogger, CompositeLogger
from .checkpoint import CheckpointManager
from .callback import (
    Callback, CallbackList,
    ProgressCallback, EvalCallback, CheckpointCallback, EarlyStoppingCallback
)
from .distributed import DistributedContext, get_world_size, get_rank, is_main_process

__all__ = [
    # Seed management
    'SeedManager',
    'set_global_seed',
    'get_global_seed',
    
    # Logging
    'BaseLogger',
    'ConsoleLogger',
    'TensorBoardLogger',
    'CompositeLogger',
    
    # Checkpoint
    'CheckpointManager',
    
    # Callbacks
    'Callback',
    'CallbackList',
    'ProgressCallback',
    'EvalCallback',
    'CheckpointCallback',
    'EarlyStoppingCallback',
    
    # Distributed
    'DistributedContext',
    'get_world_size',
    'get_rank',
    'is_main_process',
]

