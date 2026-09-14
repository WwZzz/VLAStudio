"""
Base Algorithm Class

This module defines the base class for all RL algorithms in the framework.

Design Philosophy (inspired by SKRL):
- Replay buffer is placed inside the algorithm, allowing:
  - Each algorithm to have its own replay configuration
  - A single Trainer to train multiple different algorithms (each with own replay)
  - More flexibility, supporting multi-agent scenarios
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List, Callable
from abc import ABC, abstractmethod
from dataclasses import asdict, fields

# Type hints for Meta classes (imported at runtime to avoid circular imports)
from benchmark.base import MetaObs, MetaAction, MetaPolicy
from rl.buffer.transition import RLTransition

class BaseAlgorithm(ABC):
    """
    Base class for RL algorithms.
    
    This class defines the core interface for all RL algorithms.
    It holds a reference to the policy (MetaPolicy) and optionally a replay buffer.
    
    Attributes:
        meta_policy: The MetaPolicy instance used for action selection
        replay: Optional replay buffer(s) for experience storage
    """
    
    def __init__(
        self, 
        meta_policy: MetaPolicy,
        replay: Optional[Union['BaseReplay', Dict[str, 'BaseReplay']]] = None,
        **kwargs
    ):
        """
        Initialize the algorithm.
        
        Args:
            meta_policy: ILStudio's MetaPolicy instance (required)
            replay: Supports two formats:
                   - BaseReplay instance: Single replay buffer (shared by all environments)
                   - Dict[str, BaseReplay]: Multiple replay buffers (separated by environment type)
                   - None: No replay buffer (for on-policy algorithms)
            **kwargs: Algorithm-specific parameters
        """
        self.meta_policy = meta_policy  # Required attribute
        self.replay = replay  # Optional attribute (off-policy algorithms need this)
        self._kwargs = kwargs
    
    @abstractmethod
    def update(self, batch: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Update the policy using a batch of data.
        
        Args:
            batch: Optional, batch data
                  - If None and replay exists, sample from replay
                  - If provided, use the provided batch directly
            **kwargs: Update parameters, can include:
                     - batch_size: Batch size when sampling from replay
                     - env_types: Specify which environment types to sample from
                                 (when replay is Dict[str, BaseReplay])
                                 - e.g., ['indoor', 'outdoor']
                                 - If None, sample from all environment types
                     - env_weights: Data weights for different environment types
                                   (when using multiple replay buffers)
                                   - e.g., {'indoor': 0.6, 'outdoor': 0.4}
                                   - If None, use uniform weights
        
        Returns:
            Dictionary containing loss, metrics, and other information
        
        Examples:
            # Scenario 1: Single replay buffer
            algorithm.update(batch_size=256)
            
            # Scenario 2: Multiple replay buffers (by environment type)
            algorithm.update(
                batch_size=256,
                env_types=['indoor', 'outdoor'],
                env_weights={'indoor': 0.6, 'outdoor': 0.4}
            )
        """
        raise NotImplementedError
    
    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss (optional, some algorithms may need this).
        
        Args:
            batch: Batch data
        
        Returns:
            Loss value
        """
        raise NotImplementedError("Subclass should implement compute_loss if needed")
    
    def select_action(
        self, 
        obs: Union[MetaObs, List[MetaObs], np.ndarray], 
        **kwargs
    ) -> Union[MetaAction, List[MetaAction]]:
        """
        Select action(s) for given observation(s).
        
        Supports both single and batched inputs for vectorized environments.
        The recommended way for vectorized envs is to use organized batch obs
        (a single MetaObs with batched fields like (n_envs, state_dim)).
        
        Args:
            obs: Observation(s), can be:
                - MetaObs: Single observation OR organized batch obs
                  (with batched fields like state: (n_envs, state_dim))
                - List[MetaObs] or np.ndarray: List of individual observations
                  (will be organized into batch internally)
            **kwargs: Other parameters (e.g., t for timestep)
        
        Returns:
            MetaAction: Single action or batched action 
                       (with action field shape (n_envs, action_dim) for batch)
        
        Note:
            For vectorized environments, it's recommended to:
            1. Use organize_obs() to convert List[MetaObs] -> batched MetaObs
            2. Pass the organized obs to this method
            3. The policy handles batched inference internally
        """
        # Check if input is a list of observations that needs organizing
        if isinstance(obs, (list, np.ndarray)) and len(obs) > 0:
            first = obs[0] if isinstance(obs, list) else obs.flat[0]
            if hasattr(first, '__dataclass_fields__'):  # List of MetaObs
                # Organize into batched MetaObs
                obs = self._organize_obs(obs)
        
        # Pass to meta_policy (handles both single and batch)
        return self.meta_policy.select_action(obs, **kwargs)
    
    def _organize_obs(self, obs_list: Union[List[MetaObs], np.ndarray]) -> MetaObs:
        """
        Organize list of MetaObs into a single batched MetaObs.
        
        Converts List[MetaObs] -> MetaObs with batched fields.
        e.g., [MetaObs(state=(10,)), MetaObs(state=(10,))] -> MetaObs(state=(2, 10))
        
        Args:
            obs_list: List or array of MetaObs objects
        
        Returns:
            MetaObs with batched fields
        """
        if len(obs_list) == 0:
            return None
        
        # Convert to list of dicts
        obs_dicts = []
        for o in obs_list:
            if hasattr(o, '__dataclass_fields__'):
                obs_dicts.append(asdict(o))
            elif isinstance(o, dict):
                obs_dicts.append(o)
            else:
                obs_dicts.append(vars(o) if hasattr(o, '__dict__') else {})
        
        # Stack each field
        all_keys = list(obs_dicts[0].keys())
        batched = {}
        for k in all_keys:
            values = [d[k] for d in obs_dicts]
            if values[0] is None:
                batched[k] = None
            elif isinstance(values[0], np.ndarray):
                batched[k] = np.stack(values)
            elif isinstance(values[0], (int, float)):
                batched[k] = np.array(values)
            elif isinstance(values[0], str):
                batched[k] = values  # Keep as list for strings
            else:
                batched[k] = values  # Keep as list for other types
        
        # Convert back to MetaObs
        return MetaObs(**{k: v for k, v in batched.items() if k in [f.name for f in fields(MetaObs)]})
    
    def record_transition(
        self,
        transition: 'RLTransition',
        **kwargs
    ) -> None:
        """
        Record transition to replay buffer (if exists).

        Args:
            transition: RLTransition created by collector
            **kwargs: Additional fields:
                - env_type: Environment type (required for multi-buffer)
        """
        if self.replay is None:
            raise ValueError("Replay buffer is not set")

        from rl.buffer.transition import RLTransition
        if not isinstance(transition, RLTransition):
            raise TypeError("record_transition expects RLTransition")

        env_type = kwargs.get('env_type', None)
        if isinstance(self.replay, dict):
            if env_type is None:
                raise ValueError("env_type must be provided when using multiple replay buffers")
            if env_type not in self.replay:
                raise ValueError(f"env_type '{env_type}' not found in replay buffers")
            target_replay = self.replay[env_type]
        else:
            target_replay = self.replay

        target_replay.add(transition)
    
    def _stack_dicts(self, dict_list: List[Dict]) -> Dict:
        """
        Stack a list of dicts into a single dict with batched values.
        
        Args:
            dict_list: List of dicts with same keys
        
        Returns:
            Dict with stacked values (numpy arrays where applicable)
        """
        if not dict_list:
            return {}
        
        result = {}
        for key in dict_list[0].keys():
            values = [d.get(key) for d in dict_list]
            if values[0] is None:
                result[key] = None
            elif isinstance(values[0], np.ndarray):
                result[key] = np.stack(values)
            elif isinstance(values[0], (int, float, bool)):
                result[key] = np.array(values)
            elif isinstance(values[0], str):
                result[key] = values  # Keep strings as list
            else:
                result[key] = values  # Keep other types as list
        
        return result
    
    def get_policy(self) -> MetaPolicy:
        """Get the underlying MetaPolicy."""
        return self.meta_policy
    
    def train_mode(self) -> None:
        """Set policy to training mode."""
        if hasattr(self.meta_policy, 'policy') and hasattr(self.meta_policy.policy, 'train'):
            self.meta_policy.policy.train()
    
    def eval_mode(self) -> None:
        """Set policy to evaluation mode."""
        if hasattr(self.meta_policy, 'policy') and hasattr(self.meta_policy.policy, 'eval'):
            self.meta_policy.policy.eval()
    
    def save(self, path: str, **kwargs) -> None:
        """
        Save algorithm state (model, optimizer, etc.).
        
        Args:
            path: Save path
            **kwargs: Additional save options
        """
        raise NotImplementedError("Subclass should implement save")
    
    def load(self, path: str, **kwargs) -> None:
        """
        Load algorithm state.
        
        Args:
            path: Load path
            **kwargs: Additional load options
        """
        raise NotImplementedError("Subclass should implement load")
    
    def __repr__(self) -> str:
        replay_info = "None"
        if self.replay is not None:
            if isinstance(self.replay, dict):
                replay_info = f"Dict with {len(self.replay)} buffers"
            else:
                replay_info = repr(self.replay)
        return f"{self.__class__.__name__}(meta_policy={self.meta_policy.__class__.__name__}, replay={replay_info})"


if __name__ == '__main__':
    """
    Test code for BaseAlgorithm class.
    
    Since BaseAlgorithm is abstract, we create a simple concrete implementation for testing.
    """
    import sys
    sys.path.insert(0, '/home/zhang/robot/126/ILStudio')
    
    from benchmark.base import MetaObs, MetaAction, MetaPolicy
    from rl.buffer.base_replay import BaseReplay
    from rl.algorithms.base import BaseAlgorithm
    from dataclasses import asdict
    
    # Simple replay buffer for testing
    class SimpleReplay(BaseReplay):
        def __init__(self, capacity=1000, device='cpu', **kwargs):
            super().__init__(capacity=capacity, device=device, **kwargs)
            self._storage = []
        
        def add(self, transition):
            if self._size < self.capacity:
                self._storage.append(transition)
                self._size += 1
            else:
                self._storage[self._position] = transition
            self._position = (self._position + 1) % self.capacity
        
        def sample(self, batch_size):
            if self._size == 0:
                return {}
            indices = np.random.randint(0, self._size, size=min(batch_size, self._size))
            return {
                'states': [self._storage[i]['state'] for i in indices],
                'actions': [self._storage[i]['action'] for i in indices],
                'rewards': np.array([self._storage[i]['reward'] for i in indices]),
                'next_states': [self._storage[i]['next_state'] for i in indices],
                'dones': np.array([self._storage[i]['done'] for i in indices]),
            }
        
        def clear(self):
            self._storage = []
            self._size = 0
            self._position = 0
        
        def save(self, path, **kwargs):
            pass
        
        def load(self, path, **kwargs):
            pass
    
    # Simple policy for testing
    class DummyPolicy:
        def select_action(self, obs):
            return MetaAction(
                action=np.random.randn(7).astype(np.float32),
                ctrl_space='ee',
                ctrl_type='delta'
            )
        
        def train(self):
            pass
        
        def eval(self):
            pass
    
    # Simple MetaPolicy wrapper for testing
    class DummyMetaPolicy(MetaPolicy):
        def __init__(self):
            self.policy = DummyPolicy()
            self.chunk_size = 1
            self.ctrl_space = 'ee'
            self.ctrl_type = 'delta'
            self.action_queue = []
            self.action_normalizer = None
            self.state_normalizer = None
        
        def select_action(self, mobs, t=0, **kwargs):
            return self.policy.select_action(mobs)
    
    # Simple concrete algorithm for testing
    class SimpleAlgorithm(BaseAlgorithm):
        def __init__(self, meta_policy, replay=None, learning_rate=1e-3, **kwargs):
            super().__init__(meta_policy=meta_policy, replay=replay, **kwargs)
            self.learning_rate = learning_rate
            self.update_count = 0
        
        def update(self, batch=None, **kwargs):
            batch_size = kwargs.get('batch_size', 32)
            
            if batch is None and self.replay is not None:
                if isinstance(self.replay, dict):
                    # Sample from multiple replay buffers
                    env_types = kwargs.get('env_types', list(self.replay.keys()))
                    env_weights = kwargs.get('env_weights', {k: 1.0/len(env_types) for k in env_types})
                    
                    batches = {}
                    for env_type in env_types:
                        n_samples = int(batch_size * env_weights.get(env_type, 1.0/len(env_types)))
                        batches[env_type] = self.replay[env_type].sample(n_samples)
                    batch = batches
                else:
                    batch = self.replay.sample(batch_size)
            
            self.update_count += 1
            
            return {
                'loss': np.random.randn(),
                'update_count': self.update_count,
                'batch_size': batch_size
            }
        
        def compute_loss(self, batch):
            return torch.tensor(np.random.randn())
    
    # Test the implementation
    print("=" * 60)
    print("Testing BaseAlgorithm (SimpleAlgorithm implementation)")
    print("=" * 60)
    
    # Test 1: Single replay buffer
    print("\n1. Testing with single replay buffer...")
    meta_policy = DummyMetaPolicy()
    replay = SimpleReplay(capacity=100)
    algorithm = SimpleAlgorithm(meta_policy=meta_policy, replay=replay, learning_rate=1e-4)
    print(f"   Created algorithm: {algorithm}")
    
    # Add transitions
    print("\n2. Recording transitions...")
    for i in range(10):
        state = MetaObs(
            state=np.random.randn(10).astype(np.float32),
            state_ee=np.random.randn(7).astype(np.float32),
            raw_lang="test instruction"
        )
        action = MetaAction(
            action=np.random.randn(7).astype(np.float32),
            ctrl_space='ee',
            ctrl_type='delta'
        )
        next_state = MetaObs(
            state=np.random.randn(10).astype(np.float32),
            state_ee=np.random.randn(7).astype(np.float32),
            raw_lang="test instruction"
        )
        
        transition = RLTransition(
            obs=state,
            action=action,
            next_obs=next_state,
            reward=np.random.randn(),
            done=(i == 9),
            info={'step': i, 'value': np.random.randn()},
        )
        algorithm.record_transition(transition)
    print(f"   Replay buffer size: {len(algorithm.replay)}")
    
    # Update algorithm
    print("\n3. Testing update...")
    result = algorithm.update(batch_size=5)
    print(f"   Update result: {result}")
    
    # Test 2: Multiple replay buffers (by environment type)
    print("\n4. Testing with multiple replay buffers...")
    multi_replay = {
        'indoor': SimpleReplay(capacity=100),
        'outdoor': SimpleReplay(capacity=100)
    }
    multi_algorithm = SimpleAlgorithm(meta_policy=meta_policy, replay=multi_replay)
    
    # Add transitions to different environment types
    for i in range(5):
        state = MetaObs(state=np.random.randn(10).astype(np.float32))
        action = MetaAction(action=np.random.randn(7).astype(np.float32))
        next_state = MetaObs(state=np.random.randn(10).astype(np.float32))
        
        transition = RLTransition(
            obs=state,
            action=action,
            next_obs=next_state,
            reward=1.0,
            done=False,
        )
        multi_algorithm.record_transition(transition, env_type='indoor')
        transition = RLTransition(
            obs=state,
            action=action,
            next_obs=next_state,
            reward=0.5,
            done=False,
        )
        multi_algorithm.record_transition(transition, env_type='outdoor')
    
    print(f"   Indoor replay size: {len(multi_algorithm.replay['indoor'])}")
    print(f"   Outdoor replay size: {len(multi_algorithm.replay['outdoor'])}")
    
    # Update with weighted sampling
    result = multi_algorithm.update(
        batch_size=8,
        env_types=['indoor', 'outdoor'],
        env_weights={'indoor': 0.6, 'outdoor': 0.4}
    )
    print(f"   Update with weighted sampling: {result}")
    
    # Test train/eval mode
    print("\n5. Testing train/eval mode...")
    algorithm.train_mode()
    print("   Set to train mode")
    algorithm.eval_mode()
    print("   Set to eval mode")
    
    # Test select_action
    print("\n6. Testing select_action...")
    obs = MetaObs(state=np.random.randn(10).astype(np.float32))
    action = algorithm.select_action(obs)
    print(f"   Selected action shape: {action.action.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

