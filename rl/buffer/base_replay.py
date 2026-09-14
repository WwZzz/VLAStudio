"""
Base Replay Buffer Class

This module defines the base class for all replay buffers in the RL framework.

Design Philosophy:
- Store raw Meta data: Store original MetaObs and MetaAction in buffer, keeping data in its raw form
- Complete information storage: Support storing all fields of MetaObs and MetaAction
- Extensibility: Support storing additional custom fields (value, log_prob, advantage, trajectory_id, etc.)
- Conversion during sampling: Convert to ILStudio data pipeline format (normalization, etc.) during sampling
- Compatibility: Maintain data integrity while being compatible with ILStudio's normalization pipeline
- Vectorized environment support: Each buffer corresponds to one environment type with multiple parallel envs
  - Storage shape: (capacity, n_envs, ...) where capacity is number of time steps
  - Each add() stores data from all parallel envs at one time step
  - Sample supports selecting specific env indices
"""

import torch
import numpy as np
import pickle
from typing import Dict, Any, Optional, Union, Callable, List, Set
from abc import ABC, abstractmethod


class BaseReplay(ABC):
    """
    Base class for Replay Buffer.
    
    This class defines the interface for all replay buffers in the RL framework.
    Replay buffers store experience data (transitions) for training RL algorithms.
    
    Supports vectorized environments where one buffer stores data from multiple parallel
    environments of the same type. Storage shape is (capacity, n_envs, ...).
    
    Attributes:
        capacity: Maximum number of time steps to store (not total transitions)
        device: Device to store data on ('cpu' or 'cuda')
        env_type: Environment type identifier for this buffer
        n_envs: Number of parallel environments (default 1 for non-vectorized)
    """
    
    def __init__(
        self,
        capacity: int = 1000000,
        device: Union[str, torch.device] = 'cpu',
        env_type: Optional[str] = None,
        n_envs: int = 1,
        **kwargs
    ):
        """
        Initialize the Replay Buffer.
        
        Args:
            capacity: Buffer capacity (maximum number of time steps to store)
                     - Total transitions stored = capacity * n_envs
            device: Data storage device ('cpu' or 'cuda', default 'cpu')
                   - 'cpu': Store data in CPU memory
                   - 'cuda' or 'cuda:0': Store data in GPU memory
            env_type: Environment type identifier (e.g., 'sim', 'real', 'indoor', 'outdoor')
                     - Used to distinguish buffers for different environment types
                     - If None, defaults to 'default'
            n_envs: Number of parallel environments for this buffer
                   - Storage shape will be (capacity, n_envs, ...)
                   - Default 1 for non-vectorized single environment
            **kwargs: Other initialization parameters
        """
        self.capacity = capacity
        self.device = torch.device(device) if isinstance(device, str) else device
        self.env_type = env_type if env_type is not None else 'default'
        self.n_envs = n_envs
        self._size = 0
        self._position = 0
    
    @abstractmethod
    def add(self, transition: Dict[str, Any]) -> None:
        """
        Add a transition (one time step from all parallel envs) to the buffer.
        
        Stores raw Meta data (MetaObs, MetaAction) without any normalization.
        Supports storing all fields of MetaObs and MetaAction, plus additional custom information.
        
        For vectorized environments (n_envs > 1):
            - Each field should have shape (n_envs, ...) 
            - e.g., state: (n_envs, state_dim), action: (n_envs, action_dim)
            - reward: (n_envs,), done: (n_envs,), truncated: (n_envs,)
        
        For single environment (n_envs = 1):
            - Fields can be either (1, ...) or (...) shape
            - Will be automatically expanded to (1, ...) if needed
        
        Note on done vs truncated (Gymnasium API):
            - done (terminated): True if episode ended naturally (goal reached, failure, etc.)
            - truncated: True if episode was cut off due to time limit or other external reasons
            - For bootstrap value calculation:
              - If done=True: V(next_state) = 0 (true terminal state)
              - If truncated=True: V(next_state) should be bootstrapped (not a true terminal)
        
        Args:
            transition: Dictionary containing the following fields:
                - state: Current state, shape (n_envs, state_dim) or dict with MetaObs fields
                - action: Action, shape (n_envs, action_dim) or dict with MetaAction fields
                - reward: Reward, shape (n_envs,)
                - next_state: Next state, shape (n_envs, state_dim) or dict with MetaObs fields
                - done: Terminated flag, shape (n_envs,) - True if episode ended naturally
                - truncated: Truncated flag, shape (n_envs,) - True if episode was cut off
                - info: Optional, additional information (list of dicts, one per env)
                - **other custom fields**: Can store any additional information
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample(
        self, 
        batch_size: int,
        env_indices: Optional[Union[List[int], np.ndarray]] = None,
        keys: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Sample a batch from the buffer (raw data).
        
        Sampling is done in two steps:
        1. Select which parallel environments to sample from (env_indices)
        2. Sample batch_size transitions from (time_idx, env_idx) combinations
        
        Args:
            batch_size: Number of transitions to sample
            env_indices: Optional, indices of parallel environments to sample from
                        - If None, sample from all environments uniformly
                        - e.g., [0, 2] means only sample from env 0 and env 2
            keys: Optional set of keys to sample. If None, uses default keys
                 (typically state, action, next_state, reward, done, truncated).
                 Subclasses may define their own default keys and available keys.
        
        Returns:
            Dictionary containing raw Meta data (without normalization)
            - All fields have shape (batch_size, ...) with env dimension flattened
            - e.g., state: (batch_size, state_dim), action: (batch_size, action_dim)
        """
        raise NotImplementedError
    
    def sample_for_training(
        self, 
        batch_size: int,
        env_indices: Optional[Union[List[int], np.ndarray]] = None,
        keys: Optional[Set[str]] = None,
        data_processor: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Sample and convert to ILStudio training format.
        
        Args:
            batch_size: Batch size
            env_indices: Optional, indices of parallel environments to sample from
            keys: Optional set of keys to sample
            data_processor: Optional data processing function to align with ILStudio pipeline
                          - If None, return raw data
                          - If provided, should be a function: batch -> processed_batch
        
        Returns:
            Processed batch data (conforming to ILStudio training format)
        """
        batch = self.sample(batch_size, env_indices=env_indices, keys=keys)
        if data_processor is not None:
            batch = data_processor(batch)
        return batch
    
    def __len__(self) -> int:
        """Return current buffer size (number of time steps stored)."""
        return self._size
    
    @property
    def total_transitions(self) -> int:
        """
        Total number of transitions stored (size * n_envs).
        
        Returns:
            Total transition count across all parallel environments
        """
        return self._size * self.n_envs
    
    def get_env_type(self) -> str:
        """Get the environment type identifier."""
        return self.env_type
    
    def get_n_envs(self) -> int:
        """Get the number of parallel environments."""
        return self.n_envs
    
    @abstractmethod
    def clear(self) -> None:
        """Clear the buffer."""
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path: str, **kwargs) -> None:
        """
        Save buffer data to file.
        
        Args:
            path: Save path (can be file path or directory path)
            **kwargs: Save options
                - format: Save format (e.g., 'pkl', 'hdf5', 'npz', optional)
                - compress: Whether to compress (optional)
        """
        raise NotImplementedError
    
    @abstractmethod
    def load(self, path: str, **kwargs) -> None:
        """
        Load data from file into buffer.
        
        Args:
            path: Load path (can be file path or directory path)
            **kwargs: Load options
                - format: Load format (optional, can auto-detect)
                - append: Whether to append to existing buffer (default False, clear before load)
        """
        raise NotImplementedError
    
    def is_full(self) -> bool:
        """Check if buffer is full (time steps, not total transitions)."""
        return self._size >= self.capacity
    
    def get_all(
        self, 
        env_indices: Optional[Union[List[int], np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Get all data in the buffer.
        
        Args:
            env_indices: Optional, indices of parallel environments to get data from
                        - If None, get data from all environments
        
        Returns:
            Dictionary containing all stored data
        """
        return self.sample(self.total_transitions, env_indices=env_indices) if self._size > 0 else {}
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(capacity={self.capacity}, size={self._size}, "
                f"n_envs={self.n_envs}, total_transitions={self.total_transitions}, "
                f"env_type='{self.env_type}', device={self.device})")


if __name__ == '__main__':
    """
    Test code for BaseReplay class.
    
    Since BaseReplay is abstract, we create a simple concrete implementation for testing.
    Tests include both single-env and vectorized (multi-env) scenarios.
    """
    import sys
    sys.path.insert(0, '/home/zhang/robot/126/ILStudio')
    
    from benchmark.base import MetaObs, MetaAction
    from dataclasses import asdict
    
    # Simple concrete implementation for testing (supports vectorized envs)
    class SimpleReplay(BaseReplay):
        """Simple in-memory replay buffer for testing with vectorized env support."""
        
        def __init__(
            self, 
            capacity: int = 1000, 
            device: str = 'cpu',
            env_type: Optional[str] = None,
            n_envs: int = 1,
            state_dim: int = 10,
            action_dim: int = 7,
            **kwargs
        ):
            super().__init__(
                capacity=capacity, 
                device=device, 
                env_type=env_type, 
                n_envs=n_envs, 
                **kwargs
            )
            self.state_dim = state_dim
            self.action_dim = action_dim
            
            # Pre-allocate storage: (capacity, n_envs, dim)
            self._state = np.zeros((capacity, n_envs, state_dim), dtype=np.float32)
            self._action = np.zeros((capacity, n_envs, action_dim), dtype=np.float32)
            self._reward = np.zeros((capacity, n_envs), dtype=np.float32)
            self._next_state = np.zeros((capacity, n_envs, state_dim), dtype=np.float32)
            self._done = np.zeros((capacity, n_envs), dtype=np.bool_)
            self._truncated = np.zeros((capacity, n_envs), dtype=np.bool_)
        
        def add(self, transition: Dict[str, Any]) -> None:
            """Add transition from all parallel envs at one time step."""
            idx = self._position
            
            # Store data (expecting shape (n_envs, dim))
            self._state[idx] = transition['state']
            self._action[idx] = transition['action']
            self._reward[idx] = transition['reward']
            self._next_state[idx] = transition['next_state']
            self._done[idx] = transition['done']
            self._truncated[idx] = transition.get('truncated', np.zeros(self.n_envs, dtype=bool))
            
            self._position = (self._position + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
        
        # Default sample keys
        DEFAULT_SAMPLE_KEYS = {'state', 'action', 'next_state', 'reward', 'done', 'truncated'}
        
        def sample(
            self, 
            batch_size: int,
            env_indices: Optional[Union[List[int], np.ndarray]] = None,
            keys: Optional[Set[str]] = None
        ) -> Dict[str, Any]:
            """Sample batch_size transitions from buffer.
            
            Args:
                batch_size: Number of transitions to sample
                env_indices: Optional, indices of parallel environments to sample from
                keys: Optional set of keys to sample. If None, uses DEFAULT_SAMPLE_KEYS
            """
            if self._size == 0:
                return {}
            
            if keys is None:
                keys = self.DEFAULT_SAMPLE_KEYS
            
            # Determine which envs to sample from
            if env_indices is None:
                env_indices = np.arange(self.n_envs)
            env_indices = np.asarray(env_indices)
            
            # Sample (time_idx, env_idx) pairs
            time_indices = np.random.randint(0, self._size, size=batch_size)
            env_sample_indices = np.random.choice(env_indices, size=batch_size)
            
            batch = {}
            if 'state' in keys:
                batch['state'] = self._state[time_indices, env_sample_indices]      # (batch_size, state_dim)
            if 'action' in keys:
                batch['action'] = self._action[time_indices, env_sample_indices]    # (batch_size, action_dim)
            if 'reward' in keys:
                batch['reward'] = self._reward[time_indices, env_sample_indices]    # (batch_size,)
            if 'next_state' in keys:
                batch['next_state'] = self._next_state[time_indices, env_sample_indices]
            if 'done' in keys:
                batch['done'] = self._done[time_indices, env_sample_indices]
            if 'truncated' in keys:
                batch['truncated'] = self._truncated[time_indices, env_sample_indices]
            
            # Always include indices for debugging
            batch['time_indices'] = time_indices
            batch['env_indices'] = env_sample_indices
            
            return batch
        
        def clear(self) -> None:
            self._state.fill(0)
            self._action.fill(0)
            self._reward.fill(0)
            self._next_state.fill(0)
            self._done.fill(False)
            self._truncated.fill(False)
            self._size = 0
            self._position = 0
        
        def save(self, path: str, **kwargs) -> None:
            data = {
                'state': self._state[:self._size],
                'action': self._action[:self._size],
                'reward': self._reward[:self._size],
                'next_state': self._next_state[:self._size],
                'done': self._done[:self._size],
                'truncated': self._truncated[:self._size],
                'size': self._size,
                'env_type': self.env_type,
                'n_envs': self.n_envs,
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {self._size} time steps ({self.total_transitions} transitions) to {path}")
        
        def load(self, path: str, **kwargs) -> None:
            append = kwargs.get('append', False)
            if not append:
                self.clear()
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            size = data['size']
            for i in range(size):
                self.add({
                    'state': data['state'][i],
                    'action': data['action'][i],
                    'reward': data['reward'][i],
                    'next_state': data['next_state'][i],
                    'done': data['done'][i],
                    'truncated': data.get('truncated', np.zeros(self.n_envs, dtype=bool))[i] if 'truncated' in data else np.zeros(self.n_envs, dtype=bool),
                })
            print(f"Loaded {self._size} time steps ({self.total_transitions} transitions) from {path}")
    
    # Test the implementation
    print("=" * 60)
    print("Testing BaseReplay (SimpleReplay with Vectorized Env Support)")
    print("=" * 60)
    
    # Test 1: Single environment (n_envs=1)
    print("\n" + "-" * 40)
    print("Test 1: Single Environment (n_envs=1)")
    print("-" * 40)
    
    buffer = SimpleReplay(capacity=100, device='cpu', env_type='single', n_envs=1)
    print(f"\n1. Created buffer: {buffer}")
    print(f"   env_type: {buffer.get_env_type()}")
    print(f"   n_envs: {buffer.get_n_envs()}")
    
    # Add transitions (n_envs=1, so shapes are (1, dim))
    print("\n2. Adding transitions...")
    for i in range(10):
        transition = {
            'state': np.random.randn(1, 10).astype(np.float32),
            'action': np.random.randn(1, 7).astype(np.float32),
            'reward': np.array([np.random.randn()]),
            'next_state': np.random.randn(1, 10).astype(np.float32),
            'done': np.array([i == 9]),        # True terminal state
            'truncated': np.array([False]),    # Not truncated
        }
        buffer.add(transition)
    
    print(f"   Buffer size (time steps): {len(buffer)}")
    print(f"   Total transitions: {buffer.total_transitions}")
    
    # Sample
    print("\n3. Sampling from buffer...")
    batch = buffer.sample(batch_size=5)
    print(f"   Batch keys: {batch.keys()}")
    print(f"   State shape: {batch['state'].shape}")
    print(f"   Reward shape: {batch['reward'].shape}")
    print(f"   Done shape: {batch['done'].shape}")
    print(f"   Truncated shape: {batch['truncated'].shape}")
    
    # Test 2: Vectorized environment (n_envs=4)
    print("\n" + "-" * 40)
    print("Test 2: Vectorized Environment (n_envs=4)")
    print("-" * 40)
    
    vec_buffer = SimpleReplay(
        capacity=100, 
        device='cpu', 
        env_type='sim', 
        n_envs=4,
        state_dim=10,
        action_dim=7
    )
    print(f"\n1. Created buffer: {vec_buffer}")
    
    # Add transitions from all 4 envs at once
    print("\n2. Adding transitions from 4 parallel envs...")
    for i in range(20):
        transition = {
            'state': np.random.randn(4, 10).astype(np.float32),      # (n_envs, state_dim)
            'action': np.random.randn(4, 7).astype(np.float32),      # (n_envs, action_dim)
            'reward': np.random.randn(4).astype(np.float32),         # (n_envs,)
            'next_state': np.random.randn(4, 10).astype(np.float32),
            'done': np.random.choice([True, False], size=4),         # terminated
            'truncated': np.random.choice([True, False], size=4, p=[0.1, 0.9]),  # truncated (10% chance)
        }
        vec_buffer.add(transition)
    
    print(f"   Buffer size (time steps): {len(vec_buffer)}")
    print(f"   Total transitions: {vec_buffer.total_transitions}")
    
    # Sample from all envs
    print("\n3. Sampling from all envs...")
    batch = vec_buffer.sample(batch_size=16)
    print(f"   State shape: {batch['state'].shape}")  # (16, 10)
    print(f"   Sampled from envs: {np.unique(batch['env_indices'])}")
    
    # Sample from specific envs only
    print("\n4. Sampling from specific envs [0, 2] only...")
    batch = vec_buffer.sample(batch_size=16, env_indices=[0, 2])
    print(f"   State shape: {batch['state'].shape}")
    print(f"   Sampled from envs: {np.unique(batch['env_indices'])}")
    assert all(e in [0, 2] for e in batch['env_indices']), "Should only sample from envs 0 and 2"
    
    # Test sample_for_training with env_indices
    print("\n5. Testing sample_for_training with env_indices and processor...")
    def simple_processor(batch):
        batch['processed'] = True
        return batch
    
    processed_batch = vec_buffer.sample_for_training(
        batch_size=8, 
        env_indices=[1, 3],
        data_processor=simple_processor
    )
    print(f"   Processed: {processed_batch.get('processed', False)}")
    print(f"   Sampled from envs: {np.unique(processed_batch['env_indices'])}")
    
    # Test save and load
    print("\n6. Testing save and load...")
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'vec_buffer.pkl')
        vec_buffer.save(save_path)
        
        new_buffer = SimpleReplay(capacity=100, n_envs=4, state_dim=10, action_dim=7)
        new_buffer.load(save_path)
        print(f"   Loaded buffer size: {len(new_buffer)}")
        print(f"   Loaded total transitions: {new_buffer.total_transitions}")
    
    # Test 3: Multiple buffers for different env types
    print("\n" + "-" * 40)
    print("Test 3: Multiple Buffers for Different Env Types")
    print("-" * 40)
    
    buffers = {
        'indoor': SimpleReplay(capacity=50, env_type='indoor', n_envs=2),
        'outdoor': SimpleReplay(capacity=50, env_type='outdoor', n_envs=4),
        'sim': SimpleReplay(capacity=100, env_type='sim', n_envs=8),
    }
    
    print("\nCreated buffers for different env types:")
    for name, buf in buffers.items():
        print(f"   {name}: env_type='{buf.env_type}', n_envs={buf.n_envs}, capacity={buf.capacity}")
    
    # Add some data to each
    for name, buf in buffers.items():
        for _ in range(10):
            buf.add({
                'state': np.random.randn(buf.n_envs, 10).astype(np.float32),
                'action': np.random.randn(buf.n_envs, 7).astype(np.float32),
                'reward': np.random.randn(buf.n_envs).astype(np.float32),
                'next_state': np.random.randn(buf.n_envs, 10).astype(np.float32),
                'done': np.zeros(buf.n_envs, dtype=bool),
                'truncated': np.zeros(buf.n_envs, dtype=bool),
            })
    
    print("\nBuffer statistics:")
    for name, buf in buffers.items():
        print(f"   {name}: {len(buf)} time steps, {buf.total_transitions} total transitions")
    
    # Test clear
    print("\n7. Testing clear...")
    vec_buffer.clear()
    print(f"   Buffer size after clear: {len(vec_buffer)}")
    print(f"   Total transitions after clear: {vec_buffer.total_transitions}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

