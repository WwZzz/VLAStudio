"""
Vector Environment Protocols

Defines the protocol (interface) for vectorized environments.
Supports SequentialVectorEnv, SubprocVectorEnv, and custom implementations.
"""

from typing import Protocol, runtime_checkable, Any, Optional, Union, List, Dict
import numpy as np


@runtime_checkable
class VectorEnvProtocol(Protocol):
    """
    Vectorized environment protocol.
    
    Any class implementing this protocol can be used as a vectorized environment
    in the RL framework. This includes:
    - benchmark.utils.SequentialVectorEnv
    - tianshou.env.SubprocVectorEnv
    - tianshou.env.DummyVectorEnv
    - tianshou.env.ShmemVectorEnv
    - Any custom implementation satisfying this interface
    
    Attributes:
        env_num: Number of parallel environments
    """
    env_num: int
    
    def reset(
        self, 
        id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> Any:
        """
        Reset environment(s).
        
        Args:
            id: Optional environment index(es) to reset.
                - None: Reset all environments
                - int: Reset single environment
                - List[int] or np.ndarray: Reset specific environments
        
        Returns:
            Observations from reset environment(s)
        """
        ...
    
    def step(
        self, 
        action: Any, 
        id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> tuple:
        """
        Execute action(s) in environment(s).
        
        Args:
            action: Action(s) to execute
            id: Optional environment index(es) to step
        
        Returns:
            Tuple of (obs, reward, done, info)
        """
        ...
    
    def close(self) -> None:
        """Close all environments and release resources."""
        ...
    
    def __len__(self) -> int:
        """Return number of environments."""
        ...


# Type aliases for convenience
VectorEnv = VectorEnvProtocol
EnvsType = Union[VectorEnvProtocol, Dict[str, VectorEnvProtocol]]

