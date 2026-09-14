"""
Base Collector Class

This module defines the base class for all data collectors in the RL framework.

Design Philosophy:
- Responsibility separation: Separate data collection logic from trainer
  so that trainer can focus on training loop coordination
- Environment abstraction: Support vectorized environments (SequentialVectorEnv, SubprocVectorEnv, etc.)
- Raw data storage: Only store raw environment rewards, no reward function computation,
  ensuring data integrity
- Statistics: Collect and return episode statistics
- Exploration support: Support random exploration phase and noise-based exploration
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, TYPE_CHECKING
from abc import ABC, abstractmethod

# Type hints for Meta classes
from benchmark.base import MetaObs, MetaAction
from benchmark.base import MetaObs, MetaAction
from benchmark.utils import organize_obs
from rl.buffer.transition import RLTransition
# Vector environment protocol
from rl.envs import VectorEnvProtocol, EnvsType

if TYPE_CHECKING:
    from utils.exploration import ExplorationScheduler


class BaseCollector(ABC):
    """
    Base class for data collectors.
    
    This class defines the interface for all data collectors in the RL framework.
    Collectors gather experience data from vectorized environments by interacting with them
    using the algorithm's policy.
    
    Attributes:
        envs: Vectorized environment(s) to collect data from
        algorithm: Algorithm instance for action selection and transition recording
        exploration: Optional exploration strategy for action exploration
    
    Note: Collector only stores raw environment rewards, no reward function computation.
          Reward functions are applied in the trainer during training time.
    """
    
    def __init__(
        self,
        envs: Union[VectorEnvProtocol, Dict[str, VectorEnvProtocol]],
        algorithm: 'BaseAlgorithm',
        exploration: Optional['ExplorationScheduler'] = None,
        **kwargs
    ):
        """
        Initialize the collector.
        
        Args:
            envs: Vectorized environment(s), supports:
                - VectorEnvProtocol: Single vectorized environment
                  (SequentialVectorEnv, SubprocVectorEnv, DummyVectorEnv, etc.)
                - Dict[str, VectorEnvProtocol]: Multi-environment dict for different env types
                  e.g., {'sim': sim_vec_env, 'real': real_vec_env}
            algorithm: BaseAlgorithm instance (required)
                      - Used for action selection and transition recording
            exploration: Optional ExplorationScheduler for action exploration
                        - If None, no exploration is applied (use policy actions directly)
                        - Supports initial random exploration + noise-based exploration
                        - Example:
                          exploration = ExplorationScheduler(
                              exploration_strategy=GaussianNoise(sigma=0.1),
                              random_steps=10000,
                              action_low=np.array([-1.0] * 7),
                              action_high=np.array([1.0] * 7),
                          )
            **kwargs: Collector-specific parameters
        
        Note: Collector only stores raw environment rewards, no reward function computation.
              Reward functions are applied in trainer during training time.
        """
        self.envs = envs
        self.algorithm = algorithm
        self.exploration = exploration
        self._kwargs = kwargs
        
        # Total steps counter for exploration scheduling
        self._total_steps = 0
        
        # Normalize environment storage: always use dict internally
        if isinstance(envs, dict):
            self._envs_dict: Dict[str, VectorEnvProtocol] = envs
            self._is_multi_env = True
        else:
            self._envs_dict = {'default': envs}
            self._is_multi_env = False
    
    @abstractmethod
    def collect(self, n_steps: int, env_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect n_steps of interaction data.
        
        Args:
            n_steps: Number of steps to collect
            env_type: Optional, environment type identifier (for multi-environment scenarios)
                     - If provided, will be passed to record_transition with env_type
                     - Used to support a single algorithm storing data from multiple different environments
        
        Returns:
            Dictionary containing statistics, such as:
            - episode_rewards: List of episode rewards
            - episode_lengths: List of episode lengths
            - total_steps: Total number of steps collected
            - env_type_stats: Statistics grouped by environment type (if multi-environment is supported)
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self, **kwargs) -> None:
        """
        Reset collector state (e.g., reset environments).
        
        Args:
            **kwargs: Reset parameters
        """
        raise NotImplementedError
    
    def get_env(self, env_type: Optional[str] = None) -> VectorEnvProtocol:
        """
        Get the vectorized environment by type.
        
        Args:
            env_type: Environment type identifier. If None, returns 'default' env
                     or the first env if 'default' doesn't exist.
        
        Returns:
            The vectorized environment
        
        Raises:
            KeyError: If specified env_type is not found
        """
        if env_type is not None:
            return self._envs_dict[env_type]
        
        # Return 'default' if exists, otherwise return first env
        if 'default' in self._envs_dict:
            return self._envs_dict['default']
        return list(self._envs_dict.values())[0]
    
    def get_total_env_num(self) -> int:
        """
        Get total number of parallel environments across all types.
        
        Returns:
            Total count of parallel environments
        """
        return sum(len(env) for env in self._envs_dict.values())
    
    def get_env_types(self) -> List[str]:
        """
        Get all environment type identifiers.
        
        Returns:
            List of environment type strings
        """
        return list(self._envs_dict.keys())
    
    def get_envs(self) -> Union[VectorEnvProtocol, Dict[str, VectorEnvProtocol]]:
        """Get the underlying environment(s)."""
        return self.envs
    
    # ==================== Exploration Methods ====================
    
    def set_exploration(self, exploration: Optional['ExplorationScheduler']) -> None:
        """
        Set or update the exploration strategy.
        
        Args:
            exploration: ExplorationScheduler instance or None to disable exploration
        """
        self.exploration = exploration
    
    def apply_exploration(
        self, 
        action: Any, 
        obs: Any = None,
        **kwargs
    ) -> Any:
        """
        Apply exploration to action(s).
        
        Supports:
        - MetaAction: Single action or batched (action.action shape: (action_dim,) or (n_envs, action_dim))
        - List[MetaAction]: List of individual actions
        - np.ndarray: Raw action array
        - dict with 'action' key
        
        Args:
            action: Action(s) from policy
            obs: Optional observation (for uncertainty-based exploration)
            **kwargs: Additional arguments for exploration strategy
        
        Returns:
            Explored action(s) (same type/structure as input)
        """
        if self.exploration is None:
            return action
        
        # Case 1: List of MetaAction objects
        if isinstance(action, list) and len(action) > 0 and hasattr(action[0], 'action'):
            # Stack actions, apply exploration, then update each
            action_arrays = [a.action for a in action]
            stacked = np.stack(action_arrays)  # (n_envs, action_dim)
            explored = self.exploration(
                stacked,
                step=self._total_steps,
                obs=obs,
                **kwargs
            )
            # Update each MetaAction
            for i, a in enumerate(action):
                a.action = explored[i]
            return action
        
        # Case 2: Single MetaAction (possibly with batched action field)
        elif hasattr(action, 'action') and action.action is not None:
            action_array = action.action
            explored_array = self.exploration(
                action_array, 
                step=self._total_steps,
                obs=obs,
                **kwargs
            )
            action.action = explored_array
            return action
        
        # Case 3: Dict with 'action' key
        elif isinstance(action, dict) and 'action' in action:
            action_array = action['action']
            explored_array = self.exploration(
                action_array,
                step=self._total_steps,
                obs=obs,
                **kwargs
            )
            action['action'] = explored_array
            return action
        
        # Case 4: Raw numpy array
        elif isinstance(action, np.ndarray):
            return self.exploration(
                action,
                step=self._total_steps,
                obs=obs,
                **kwargs
            )
        
        # Case 5: Unknown type, return as-is
        else:
            return action
    
    def reset_exploration(self, env_idx: Optional[int] = None) -> None:
        """
        Reset exploration state (e.g., for OU noise).
        
        Args:
            env_idx: Optional specific environment index to reset
        """
        if self.exploration is not None:
            self.exploration.reset(env_idx)
    
    @property
    def total_steps(self) -> int:
        """Get total steps collected so far."""
        return self._total_steps
    
    @property
    def is_exploring_randomly(self) -> bool:
        """Check if currently in random exploration phase."""
        if self.exploration is None:
            return False
        return self.exploration.is_random_phase
    
    def _unpack_actions(self, actions: Any, n_envs: int) -> List[Any]:
        """
        Unpack batched actions into a list for vec_env.step.
        
        Handles:
        - MetaAction with batched action field: (n_envs, action_dim) -> List[MetaAction]
        - List[MetaAction]: Return as-is
        - np.ndarray: (n_envs, action_dim) -> List[np.ndarray]
        
        Args:
            actions: Batched actions from policy
            n_envs: Number of environments
        
        Returns:
            List of individual actions, one per environment
        """
        # Case 1: Already a list
        if isinstance(actions, list):
            return actions
        
        # Case 2: MetaAction with batched action field
        if hasattr(actions, 'action') and actions.action is not None:
            action_array = actions.action
            if isinstance(action_array, np.ndarray) and action_array.ndim >= 2:
                # Batched: (n_envs, action_dim) or (n_envs, chunk_size, action_dim)
                unpacked = []
                for i in range(min(n_envs, len(action_array))):
                    # Create new MetaAction for each env
                    single_action = MetaAction(
                        action=action_array[i],
                        ctrl_space=getattr(actions, 'ctrl_space', 'ee'),
                        ctrl_type=getattr(actions, 'ctrl_type', 'delta'),
                        gripper_continuous=getattr(actions, 'gripper_continuous', False),
                    )
                    unpacked.append(single_action)
                return unpacked
            else:
                # Single action, replicate for all envs
                return [actions] * n_envs
        
        # Case 3: np.ndarray (batched)
        if isinstance(actions, np.ndarray):
            if actions.ndim >= 2:
                return [actions[i] for i in range(min(n_envs, len(actions)))]
            else:
                return [actions] * n_envs
        
        # Case 4: Unknown, replicate for all envs
        return [actions] * n_envs
    
    def get_algorithm(self) -> 'BaseAlgorithm':
        """Get the algorithm instance."""
        return self.algorithm
    
    @property
    def env_num(self) -> int:
        """
        Number of environments in the default (or first) environment.
        
        Returns:
            Number of parallel environments
        """
        if 'default' in self._envs_dict:
            return len(self._envs_dict['default'])
        return len(list(self._envs_dict.values())[0])
    
    def __repr__(self) -> str:
        env_info = f"{self.get_total_env_num()} envs" if self._is_multi_env else f"{self.env_num} envs"
        return f"{self.__class__.__name__}(envs={env_info}, algorithm={self.algorithm.__class__.__name__})"


class DummyCollector(BaseCollector):
    """
    Simple collector implementation for off-policy RL algorithms.
    
    This collector:
    - Works with vectorized environments (SequentialVectorEnv, SubprocVectorEnv, etc.)
    - Handles batched observations and actions
    - Creates RLTransition objects for storage in replay buffer
    - Tracks episode statistics
    - Supports both step-by-step and batch collection
    """
    
    def __init__(self, envs, algorithm, ctrl_space='joint', action_dim=None, **kwargs):
        super().__init__(envs, algorithm, **kwargs)
        self.vec_env = self.get_env()
        self._last_obs = None
        self.ctrl_space = ctrl_space
        self.action_dim = action_dim
        
        # Episode tracking
        self._episode_rewards = None
        self._episode_lengths = None
    
    def reset(self, **kwargs):
        """Reset the collector and environments."""
        self.vec_env = self.get_env()
        self._last_obs = self.vec_env.reset()
        
        # Initialize episode tracking
        num_envs = len(self.vec_env)
        self._episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(num_envs, dtype=np.int32)
    
    def collect_step(
        self, 
        noise_scale: float = 0.0, 
        use_random: bool = False,
        env_type: str = None
    ) -> Dict[str, Any]:
        """
        Collect one step of interaction data.
        
        This is the primary method for off-policy training where we collect
        one step at a time and interleave with policy updates.
        
        Args:
            noise_scale: Exploration noise scale (for policy actions)
            use_random: If True, use random actions (for initial exploration)
            env_type: Optional environment type identifier
            
        Returns:
            Dictionary with statistics for this step
        """
        if self._last_obs is None:
            self.reset()
        
        from benchmark.base import MetaObs, MetaAction
        from benchmark.utils import organize_obs
        from rl.buffer.transition import RLTransition
        
        stats = {'episode_rewards': [], 'episode_lengths': [], 'total_steps': 0}
        
        # Organize observations into batched MetaObs
        obs = self._last_obs
        if not isinstance(obs, MetaObs):
            obs = organize_obs(obs, self.ctrl_space)
        
        num_envs = len(self.vec_env)
        
        # Select action
        if use_random:
            # Random exploration
            action_dim = self.action_dim
            if action_dim is None:
                # Try to infer from environment
                single_env = self.vec_env.envs[0] if hasattr(self.vec_env, 'envs') else self.vec_env
                if hasattr(single_env, 'action_space'):
                    action_dim = single_env.action_space.shape[0]
                elif hasattr(single_env, 'action_dim'):
                    action_dim = single_env.action_dim
                else:
                    raise ValueError("Cannot determine action_dim for random exploration")
            action_array = np.random.uniform(-1, 1, (num_envs, action_dim)).astype(np.float32)
            action = MetaAction(action=action_array)
        else:
            # Policy action with exploration noise
            action = self.algorithm.select_action(obs, noise_scale=noise_scale, env=self.vec_env)
        
        # Unpack action to numpy array
        if hasattr(action, 'action'):
            action_array = action.action
        else:
            action_array = action
        
        if action_array.ndim == 1:
            action_array = action_array[np.newaxis, :]
        
        # Convert to list of dicts for vec_env.step
        step_actions = [{'action': action_array[i]} for i in range(num_envs)]
        
        # Step environment
        next_obs, rewards, dones, infos = self.vec_env.step(step_actions)
        
        # Handle infos which may be None, list, or numpy array
        if infos is not None and len(infos) > 0:
            truncated = np.array([
                info.get('TimeLimit.truncated', False) if isinstance(info, dict) else False 
                for info in infos
            ])
        else:
            truncated = np.zeros_like(dones)
        
        # Organize next observations
        if not isinstance(next_obs, MetaObs):
            next_obs = organize_obs(next_obs, self.ctrl_space)
        
        # Ensure action is MetaAction
        if not isinstance(action, MetaAction):
            action = MetaAction(action=action_array)
        
        # Create transition and record
        transition = RLTransition(
            obs=obs,
            action=action,
            next_obs=next_obs,
            reward=rewards,
            done=dones,
            truncated=truncated,
        )
        
        kwargs_trans = {'env_type': env_type} if env_type else {}
        self.algorithm.record_transition(transition, **kwargs_trans)
        
        # Update episode statistics
        self._episode_rewards += rewards
        self._episode_lengths += 1
        stats['total_steps'] = num_envs
        self._total_steps += num_envs
        
        # Handle done episodes
        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            stats['episode_rewards'] = self._episode_rewards[done_indices].tolist()
            stats['episode_lengths'] = self._episode_lengths[done_indices].tolist()
            
            # Reset tracking for done envs
            self._episode_rewards[done_indices] = 0
            self._episode_lengths[done_indices] = 0
            
            # Reset done environments and update next_obs
            reset_obs = self.vec_env.reset(id=done_indices)
            if reset_obs is not None:
                reset_obs_organized = organize_obs(reset_obs, self.ctrl_space) if not isinstance(reset_obs, MetaObs) else reset_obs
                if isinstance(next_obs, MetaObs) and next_obs.state is not None:
                    if hasattr(reset_obs_organized, 'state') and reset_obs_organized.state is not None:
                        next_obs.state[done_indices] = reset_obs_organized.state
        
        # Update last observation
        self._last_obs = next_obs
        
        return stats
    
    def collect(self, n_steps, env_type=None):
        """
        Collect n_steps of interaction data.
        
        Args:
            n_steps: Number of steps to collect
            env_type: Optional environment type identifier
        
        Returns:
            Dictionary with statistics:
            - episode_rewards: List of episode rewards
            - episode_lengths: List of episode lengths
            - total_steps: Total number of steps collected
            - env_type: Environment type identifier
        """
        stats = {'episode_rewards': [], 'episode_lengths': [], 'total_steps': 0, 'env_type': env_type}
        
        for _ in range(n_steps):
            step_stats = self.collect_step(env_type=env_type)
            stats['episode_rewards'].extend(step_stats.get('episode_rewards', []))
            stats['episode_lengths'].extend(step_stats.get('episode_lengths', []))
            stats['total_steps'] += step_stats.get('total_steps', 0)
        
        return stats

