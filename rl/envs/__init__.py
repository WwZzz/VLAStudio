"""
RL Environments Module

Provides protocols and utilities for vectorized environments.
"""

from .protocols import VectorEnvProtocol, VectorEnv, EnvsType
from .utils import make_vector_env, get_env_info

__all__ = [
    # Protocols
    'VectorEnvProtocol',
    'VectorEnv',
    'EnvsType',
    # Utilities
    'make_vector_env',
    'get_env_info',
]

