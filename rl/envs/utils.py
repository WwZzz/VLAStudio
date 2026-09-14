"""
Environment Utilities

Provides helper functions for creating and managing vectorized environments.
"""

from typing import Callable, Any, Optional
from .protocols import VectorEnvProtocol


def make_vector_env(
    env_fn: Callable[[], Any],
    num_envs: int = 1,
    vector_type: str = 'sequential',
    **kwargs
) -> VectorEnvProtocol:
    """
    Create a vectorized environment from an environment factory function.
    
    This is a convenience function for creating vectorized environments.
    For more control, create the vectorized environment directly.
    
    Args:
        env_fn: Factory function that returns a single environment (e.g., MetaEnv).
                This function will be called num_envs times to create parallel envs.
        num_envs: Number of parallel environments to create. Default is 1.
        vector_type: Type of vectorization to use:
            - 'sequential': SequentialVectorEnv (no multiprocessing, safe for daemon processes)
            - 'subproc': SubprocVectorEnv (multiprocessing, faster for CPU-bound envs)
            - 'dummy': DummyVectorEnv (tianshou's sequential implementation)
            - 'shmem': ShmemVectorEnv (shared memory, fastest for large observations)
        **kwargs: Additional arguments passed to the vector environment constructor
    
    Returns:
        VectorEnvProtocol: A vectorized environment instance
    
    Examples:
        >>> from benchmark.aloha import create_env
        >>> 
        >>> # Create 4 parallel environments using sequential vectorization
        >>> vec_env = make_vector_env(
        ...     env_fn=lambda: create_env(config),
        ...     num_envs=4,
        ...     vector_type='sequential'
        ... )
        >>> 
        >>> # Create 8 parallel environments using subprocesses
        >>> vec_env = make_vector_env(
        ...     env_fn=lambda: create_env(config),
        ...     num_envs=8,
        ...     vector_type='subproc'
        ... )
    
    Raises:
        ValueError: If vector_type is not recognized
        ImportError: If required module for vector_type is not available
    """
    env_fns = [env_fn for _ in range(num_envs)]
    
    if vector_type == 'sequential':
        from benchmark.utils import SequentialVectorEnv
        return SequentialVectorEnv(env_fns)
    
    elif vector_type == 'subproc':
        from tianshou.env import SubprocVectorEnv
        return SubprocVectorEnv(env_fns, **kwargs)
    
    elif vector_type == 'dummy':
        from tianshou.env import DummyVectorEnv
        return DummyVectorEnv(env_fns)
    
    elif vector_type == 'shmem':
        from tianshou.env import ShmemVectorEnv
        return ShmemVectorEnv(env_fns, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown vector_type: {vector_type}. "
            f"Supported types: 'sequential', 'subproc', 'dummy', 'shmem'"
        )


def get_env_info(envs: VectorEnvProtocol) -> dict:
    """
    Get information about a vectorized environment.
    
    Args:
        envs: A vectorized environment
    
    Returns:
        Dictionary containing environment information:
        - env_num: Number of parallel environments
        - type: Type name of the vectorized environment
    """
    return {
        'env_num': envs.env_num if hasattr(envs, 'env_num') else len(envs),
        'type': type(envs).__name__,
    }

