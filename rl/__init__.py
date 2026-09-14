"""
ILStudio Reinforcement Learning Module

This module provides a modular RL framework for robot learning, supporting:
- Traditional RL algorithms (PPO, SAC, TD3, etc.)
- VLA fine-tuning algorithms (DPO, GRPO)
- Flexible replay buffers for experience storage
- Modular reward functions
- Data collectors for various environments
- Trainers for different training scenarios
- Infrastructure for reproducibility and stability

The framework is designed to:
1. Be highly abstract - only core interfaces, no specific implementations
2. Be universal - support various RL algorithms and training modes
3. Be compatible - directly use MetaEnv and MetaPolicy without adapters
4. Be extensible - easy to extend for parallel and distributed training
5. Be modular - reward functions and config systems are independently replaceable
6. Be reproducible - comprehensive infrastructure for reproducibility

Directory Structure:
    rl/
    ├── __init__.py              # This file
    ├── algorithms/              # RL algorithm implementations
    │   ├── __init__.py          # Algorithm registry
    │   └── base.py              # BaseAlgorithm class
    │   └── __init__.py          # Algorithm registry
    ├── buffer/                  # Replay buffer implementations
    │   ├── __init__.py
    │   └── base_replay.py       # BaseReplay class
    ├── rewards/                 # Reward function implementations
    │   ├── __init__.py
    │   └── base_reward.py       # BaseReward class
    ├── collectors/              # Data collector implementations
    │   ├── __init__.py
    │   └── base_collector.py    # BaseCollector class
    ├── trainers/                # Trainer implementations
    │   ├── __init__.py
    │   └── base_trainer.py      # BaseTrainer class
    ├── infra/                   # Infrastructure for reproducibility
    │   ├── __init__.py
    │   ├── seed_manager.py      # Random seed management
    │   ├── logger.py            # Logging system
    │   ├── checkpoint.py        # Checkpoint management
    │   ├── callback.py          # Callback system
    │   └── distributed.py       # Distributed training support
    └── utils/                   # Utility functions
        └── __init__.py
"""

# Base classes
from .algorithms.base import BaseAlgorithm
from .buffer import BaseReplay
from .rewards import BaseReward
from .collectors import BaseCollector
from .trainers import BaseTrainer

# Factory functions
from .algorithms import get_algorithm_class, register_algorithm, list_algorithms
from .rewards import get_reward_class, register_reward, list_rewards
from .collectors import get_collector_class, register_collector, list_collectors
from .trainers import get_trainer_class, register_trainer, list_trainers

# Utility functions
from .utils import (
    compute_gae,
    compute_returns,
    RunningMeanStd,
    explained_variance,
    polyak_update,
    hard_update,
)

# Infrastructure - Seed management
from .infra import (
    SeedManager,
    set_global_seed,
    get_global_seed,
)

# Infrastructure - Logging
from .infra import (
    BaseLogger,
    ConsoleLogger,
    TensorBoardLogger,
    CompositeLogger,
)

# Infrastructure - Checkpoint
from .infra import CheckpointManager

# Infrastructure - Callbacks
from .infra import (
    Callback,
    CallbackList,
    ProgressCallback,
    EvalCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
)

# Infrastructure - Distributed
from .infra import (
    DistributedContext,
    get_world_size,
    get_rank,
    is_main_process,
)

__all__ = [
    # Base classes
    'BaseAlgorithm',
    'BaseReplay',
    'BaseReward',
    'BaseCollector',
    'BaseTrainer',
    
    # Factory functions for algorithms
    'get_algorithm_class',
    'register_algorithm',
    'list_algorithms',
    
    # Factory functions for rewards
    'get_reward_class',
    'register_reward',
    'list_rewards',
    
    # Factory functions for collectors
    'get_collector_class',
    'register_collector',
    'list_collectors',
    
    # Factory functions for trainers
    'get_trainer_class',
    'register_trainer',
    'list_trainers',
    
    # Utility functions
    'compute_gae',
    'compute_returns',
    'RunningMeanStd',
    'explained_variance',
    'polyak_update',
    'hard_update',
    
    # Infrastructure - Seed management
    'SeedManager',
    'set_global_seed',
    'get_global_seed',
    
    # Infrastructure - Logging
    'BaseLogger',
    'ConsoleLogger',
    'TensorBoardLogger',
    'CompositeLogger',
    
    # Infrastructure - Checkpoint
    'CheckpointManager',
    
    # Infrastructure - Callbacks
    'Callback',
    'CallbackList',
    'ProgressCallback',
    'EvalCallback',
    'CheckpointCallback',
    'EarlyStoppingCallback',
    
    # Infrastructure - Distributed
    'DistributedContext',
    'get_world_size',
    'get_rank',
    'is_main_process',
]
