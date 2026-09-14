"""
Base Trainer Class

This module defines the base class for all RL trainers in the framework.

Design Philosophy:
- Coordinate algorithm and collectors for executing training loop
- Support a single algorithm collecting data from multiple different environment types
- Each environment type has its own Collector and Replay Buffer
- Collectors manage environments internally (Trainer doesn't need envs directly)
- Support custom reward functions (applied during training, not data collection)
- Support evaluation during training

Architecture:
    Trainer
    ├── algorithm: BaseAlgorithm (single algorithm)
    │   └── replay: Dict[env_type, BaseReplay]  (multiple replays, one per env type)
    ├── collectors: Dict[env_type, BaseCollector]  (multiple collectors, one per env type)
    │   ├── collector['env_type_A']: manages VectorEnv A and stores to replay['env_type_A']
    │   ├── collector['env_type_B']: manages VectorEnv B and stores to replay['env_type_B']
    │   └── ...
    └── reward_fn: Optional[BaseReward]
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Union, List, Callable, TYPE_CHECKING
from abc import ABC, abstractmethod

# Type hints for Meta classes
from benchmark.base import MetaObs, MetaAction

# Vector environment protocol
from rl.envs import VectorEnvProtocol, EnvsType

if TYPE_CHECKING:
    from rl.rewards.base_reward import BaseReward
else:
    # Import at runtime to avoid potential circular imports
    from rl.rewards.base_reward import BaseReward


class BaseTrainer(ABC):
    """
    Base class for RL trainers.
    
    This class defines the interface for all trainers in the RL framework.
    Trainers coordinate the training loop by managing:
    - Algorithm (with policy and multiple replay buffers for different env types)
    - Collectors (for data collection from different env types)
    - Reward functions (applied during training)
    
    Key Design:
    - Trainer does NOT directly manage environments
    - Each Collector manages its own VectorEnv internally
    - Algorithm has Dict[env_type, Replay] for storing data from different env types
    - Number of collectors must match the number of env types in algorithm's replay dict
    
    Attributes:
        algorithm: Algorithm for training
        collectors: Dict of collectors, keyed by env_type
        reward_fn: Optional custom reward function
    """
    
    def __init__(
        self,
        algorithm: 'BaseAlgorithm',
        collectors: Union['BaseCollector', Dict[str, 'BaseCollector']],
        reward_fn: Optional[Union['BaseReward', Callable]] = None,
        **kwargs
    ):
        """
        Initialize the trainer.
        
        Args:
            algorithm: BaseAlgorithm instance (required)
                      - Should have replay as Dict[env_type, BaseReplay] for multi-env scenarios
                      - Each replay buffer stores data from its corresponding env type
            collectors: Data collectors, supports:
                      - BaseCollector: Single collector (for single env type, uses 'default')
                      - Dict[str, BaseCollector]: Multi-collector dict for different env types
                        e.g., {'sim': sim_collector, 'real': real_collector}
                      - Each collector manages its own VectorEnv
                      - Keys should match algorithm's replay dict keys
            reward_fn: Optional reward function (if None, use raw environment reward)
                      - Can be BaseReward instance or Callable function
                      - Applied during training for algorithm updates
                      - Note: Replay buffer stores raw rewards, reward function only applied during training
            **kwargs: Trainer-specific parameters
        """
        self.algorithm = algorithm
        self.reward_fn = reward_fn
        self._kwargs = kwargs
        
        # Normalize collector storage: always use dict internally
        if isinstance(collectors, dict):
            self._collectors_dict: Dict[str, 'BaseCollector'] = collectors
        else:
            self._collectors_dict = {'default': collectors}
        
        # Store reference for easier access
        self.collectors = self._collectors_dict
    
    def get_collector(self, env_type: Optional[str] = None) -> 'BaseCollector':
        """
        Get the collector by env_type.
        
        Args:
            env_type: Environment type identifier. If None, returns 'default' collector
                     or the first collector if 'default' doesn't exist.
        
        Returns:
            The collector for the specified env type
        
        Raises:
            KeyError: If specified env_type is not found
        """
        if env_type is not None:
            return self._collectors_dict[env_type]
        
        # Return 'default' if exists, otherwise return first collector
        if 'default' in self._collectors_dict:
            return self._collectors_dict['default']
        return list(self._collectors_dict.values())[0]
    
    def get_env(self, env_type: Optional[str] = None) -> VectorEnvProtocol:
        """
        Get the vectorized environment by type (from collector).
        
        Args:
            env_type: Environment type identifier.
        
        Returns:
            The vectorized environment managed by the collector
        """
        collector = self.get_collector(env_type)
        return collector.get_env()
    
    def get_total_env_num(self) -> int:
        """
        Get total number of parallel environments across all types.
        
        Returns:
            Total count of parallel environments across all collectors
        """
        return sum(col.get_total_env_num() for col in self._collectors_dict.values())
    
    def get_env_types(self) -> List[str]:
        """
        Get all environment type identifiers.
        
        Returns:
            List of environment type strings (collector keys)
        """
        return list(self._collectors_dict.keys())
    
    @property
    def env_num(self) -> int:
        """
        Number of environments in the default (or first) collector.
        
        Returns:
            Number of parallel environments
        """
        collector = self.get_collector()
        return collector.env_num
    
    @abstractmethod
    def train(self, **kwargs) -> None:
        """
        Execute training loop.
        
        Args:
            **kwargs: Training parameters, can include:
                - total_steps: Total training steps (optional)
                - total_episodes: Total episodes (optional)
                - max_time: Maximum training time (optional)
                - log_interval: Logging interval (optional)
                - save_interval: Model save interval (optional)
                - eval_interval: Evaluation interval (optional)
        """
        raise NotImplementedError
    
    def compute_reward(
        self,
        state: MetaObs,
        action: MetaAction,
        next_state: MetaObs,
        env_reward: float,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute reward (support custom reward function).
        
        Used during training for algorithm update reward computation.
        Replay buffer stores raw rewards, reward function only applied during training.
        
        Args:
            state: Current state
            action: Action
            next_state: Next state
            env_reward: Environment raw reward
            info: Additional information dictionary
        
        Returns:
            Computed reward value
        """
        if self.reward_fn is not None:
            if isinstance(self.reward_fn, BaseReward):
                return self.reward_fn.compute(state, action, next_state, env_reward, info)
            else:
                # Assume it's a callable
                return self.reward_fn(state, action, next_state, env_reward, info)
        return env_reward
    
    def collect_rollout(
        self, 
        n_steps: int, 
        env_type: Optional[str] = None
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Collect rollout data (using collectors).
        
        Args:
            n_steps: Number of steps to collect
            env_type: Optional environment type identifier
                     - If specified, only collect from that env type's collector
                     - If None, collect from ALL collectors
        
        Returns:
            - If env_type specified: Single stats dict from that collector
            - If env_type is None: Dict[env_type, stats] from all collectors
        """
        if env_type is not None:
            # Collect from specific env type
            collector = self.get_collector(env_type)
            return collector.collect(n_steps, env_type=env_type)
        else:
            # Collect from all collectors
            all_stats = {}
            for env_type_key, collector in self._collectors_dict.items():
                all_stats[env_type_key] = collector.collect(n_steps, env_type=env_type_key)
            return all_stats
    
    def collect_rollout_all(self, n_steps: int) -> Dict[str, Dict[str, Any]]:
        """
        Collect rollout data from ALL collectors.
        
        Args:
            n_steps: Number of steps to collect from each collector
        
        Returns:
            Dict[env_type, stats] from all collectors
        """
        return self.collect_rollout(n_steps, env_type=None)
    
    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False,
        env_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate policy performance.
        
        Default implementation uses collector's environment and algorithm's select_action.
        Subclasses can override for custom evaluation logic.
        
        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render environment (optional)
            env_type: Optional, specify evaluation environment type
            **kwargs: Other evaluation parameters (vec_env, max_timesteps, ctrl_space, etc.)
        
        Returns:
            Dictionary containing evaluation metrics
        """
        from benchmark.utils import organize_obs
        
        # Get evaluation environment
        vec_env = kwargs.get('vec_env', None)
        if vec_env is None:
            # Try to get from collector
            collector = self.get_collector(env_type)
            if collector is not None:
                vec_env = collector.get_env()
            else:
                return {'error': 'No evaluation environment available'}
        
        max_timesteps = kwargs.get('max_timesteps', 1000)
        ctrl_space = kwargs.get('ctrl_space', 'joint')
        
        num_envs = len(vec_env)
        all_returns = []
        all_lengths = []
        episodes_completed = 0
        
        # Set algorithm to eval mode
        self.algorithm.eval_mode()
        
        while episodes_completed < n_episodes:
            obs = vec_env.reset()
            obs = organize_obs(obs, ctrl_space)
            
            episode_returns = np.zeros(num_envs, dtype=np.float32)
            episode_lengths = np.zeros(num_envs, dtype=np.int32)
            done_mask = np.zeros(num_envs, dtype=bool)
            
            for t in range(max_timesteps):
                with torch.no_grad():
                    action = self.algorithm.select_action(obs, noise_scale=0.0, env=vec_env)
                
                # Unpack action
                if hasattr(action, 'action'):
                    action_array = action.action
                else:
                    action_array = action
                
                if action_array.ndim == 1:
                    action_array = action_array[np.newaxis, :]
                
                # Convert to list of dicts for vec_env.step
                step_actions = [{'action': action_array[i]} for i in range(len(action_array))]
                
                # Step environment
                next_obs, rewards, dones, infos = vec_env.step(step_actions)
                next_obs = organize_obs(next_obs, ctrl_space)
                
                # Accumulate rewards for active episodes
                episode_returns += rewards * (~done_mask)
                episode_lengths += (~done_mask).astype(np.int32)
                
                # Check for newly done episodes
                newly_done = dones & (~done_mask)
                if newly_done.any():
                    for idx in np.where(newly_done)[0]:
                        all_returns.append(episode_returns[idx])
                        all_lengths.append(episode_lengths[idx])
                        episodes_completed += 1
                
                done_mask = done_mask | dones
                
                # Reset done environments
                if dones.any():
                    done_indices = np.where(dones)[0]
                    reset_obs = vec_env.reset(id=done_indices)
                    if reset_obs is not None:
                        reset_obs_organized = organize_obs(reset_obs, ctrl_space)
                        if isinstance(next_obs, MetaObs) and next_obs.state is not None:
                            if hasattr(reset_obs_organized, 'state') and reset_obs_organized.state is not None:
                                next_obs.state[done_indices] = reset_obs_organized.state
                    episode_returns[done_indices] = 0
                    episode_lengths[done_indices] = 0
                    done_mask[done_indices] = False
                
                obs = next_obs
                
                if episodes_completed >= n_episodes:
                    break
            
            # Handle truncated episodes
            for idx in range(num_envs):
                if not done_mask[idx] and episode_lengths[idx] > 0:
                    all_returns.append(episode_returns[idx])
                    all_lengths.append(episode_lengths[idx])
                    episodes_completed += 1
                    if episodes_completed >= n_episodes:
                        break
        
        # Set back to train mode
        self.algorithm.train_mode()
        
        # Compute statistics
        all_returns = np.array(all_returns[:n_episodes])
        all_lengths = np.array(all_lengths[:n_episodes])
        
        return {
            'returns': all_returns.tolist(),
            'mean_return': float(np.mean(all_returns)),
            'std_return': float(np.std(all_returns)),
            'min_return': float(np.min(all_returns)),
            'max_return': float(np.max(all_returns)),
            'episode_lengths': all_lengths.tolist(),
            'mean_length': float(np.mean(all_lengths)),
            'num_episodes': len(all_returns),
        }
    
    def save(self, path: str) -> None:
        """
        Save model and training state.
        
        Default implementation saves the algorithm's model.
        Subclasses can override to save additional state.
        
        Args:
            path: Path to save the model
        """
        self.algorithm.save(path)
    
    def load(self, path: str) -> None:
        """
        Load model and training state.
        
        Default implementation loads the algorithm's model.
        Subclasses can override to load additional state.
        
        Args:
            path: Path to load the model from
        """
        self.algorithm.load(path)
    
    def save_checkpoint(self, path: str, step: int) -> None:
        """
        Save training checkpoint with step information.
        
        Args:
            path: Directory or full path to save checkpoint
            step: Current training step number
        """
        import os
        if os.path.isdir(path):
            ckpt_path = os.path.join(path, f'checkpoint_{step}.pt')
        else:
            ckpt_path = path
        self.save(ckpt_path)
    
    def load_checkpoint(self, path: str) -> int:
        """
        Load checkpoint and return the step number.
        
        Extracts step number from checkpoint filename (e.g., checkpoint_10000.pt -> 10000).
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Step number extracted from checkpoint filename, or 0 if not found
        """
        import os
        self.load(path)
        # Try to extract step from checkpoint name
        try:
            step = int(os.path.basename(path).split('_')[-1].split('.')[0])
        except (ValueError, IndexError):
            step = 0
        return step
    
    def get_algorithm(self) -> 'BaseAlgorithm':
        """Get the algorithm."""
        return self.algorithm
    
    def get_collectors(self) -> Dict[str, 'BaseCollector']:
        """Get all collectors as a dict."""
        return self._collectors_dict
    
    def get_replay(self, env_type: Optional[str] = None) -> 'BaseReplay':
        """
        Get the replay buffer for a specific env type.
        
        Args:
            env_type: Environment type. If None, returns default or first replay.
        
        Returns:
            The replay buffer
        """
        replay = self.algorithm.replay
        if isinstance(replay, dict):
            if env_type is not None:
                return replay[env_type]
            elif 'default' in replay:
                return replay['default']
            else:
                return list(replay.values())[0]
        return replay
    
    def set_reward_fn(self, reward_fn: Union['BaseReward', Callable]) -> None:
        """
        Set or update the reward function.
        
        Args:
            reward_fn: New reward function to use
        """
        self.reward_fn = reward_fn
    
    def reset_collectors(self) -> None:
        """Reset all collectors."""
        for collector in self._collectors_dict.values():
            collector.reset()
    
    def __repr__(self) -> str:
        algo_info = self.algorithm.__class__.__name__
        env_types = list(self._collectors_dict.keys())
        collector_info = f"Dict[{len(self._collectors_dict)}]: {env_types}"
        reward_info = "None" if self.reward_fn is None else self.reward_fn.__class__.__name__
        total_envs = self.get_total_env_num()
        return (f"{self.__class__.__name__}(algorithm={algo_info}, "
                f"collectors={collector_info}, total_envs={total_envs}, reward_fn={reward_info})")


if __name__ == '__main__':
    """
    Test code for BaseTrainer class.
    
    Since BaseTrainer is abstract, we create a simple concrete implementation for testing.
    Tests the new architecture where Trainer uses Collectors (which manage envs internally).
    """
    import sys
    sys.path.insert(0, '/home/zhang/robot/126/ILStudio')
    
    from benchmark.base import MetaObs, MetaAction, MetaEnv, MetaPolicy
    from rl.buffer.base_replay import BaseReplay
    from rl.algorithms.base import BaseAlgorithm
    from rl.rewards.base_reward import BaseReward, IdentityReward, ScaledReward
    from rl.collectors.base_collector import BaseCollector
    from rl.envs import VectorEnvProtocol
    from dataclasses import asdict
    
    # Use MetaReplay for efficient env-first storage with vectorized sampling
    from rl.buffer.meta_replay import MetaReplay
    
    # Simple dummy environment for testing
    class DummyEnv:
        def __init__(self, state_dim=10, action_dim=7, max_steps=100):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self._step_count = 0
            self._max_steps = max_steps
        
        def reset(self):
            self._step_count = 0
            return {'state': np.random.randn(self.state_dim).astype(np.float32)}
        
        def step(self, action):
            self._step_count += 1
            obs = {'state': np.random.randn(self.state_dim).astype(np.float32)}
            reward = np.random.randn()
            done = self._step_count >= self._max_steps
            info = {'step': self._step_count}
            return obs, reward, done, info
        
        def close(self):
            pass
    
    # Simple MetaEnv wrapper for testing
    class DummyMetaEnv(MetaEnv):
        def __init__(self, state_dim=10, action_dim=7, max_steps=100):
            self.env = DummyEnv(state_dim=state_dim, action_dim=action_dim, max_steps=max_steps)
            self.prev_obs = None
        
        def obs2meta(self, raw_obs):
            return MetaObs(state=raw_obs['state'], raw_lang="test")
        
        def meta2act(self, action, *args):
            if hasattr(action, 'action'):
                return action.action
            return action
        
        def reset(self):
            init_obs = self.env.reset()
            self.prev_obs = self.obs2meta(init_obs)
            return self.prev_obs
        
        def step(self, action):
            act = self.meta2act(action)
            obs, reward, done, info = self.env.step(act)
            self.prev_obs = self.obs2meta(obs)
            return self.prev_obs, reward, done, info
    
    # Use SequentialVectorEnv from benchmark/utils.py for testing
    from benchmark.utils import SequentialVectorEnv
    
    # Simple policy for testing
    class DummyPolicy:
        def __init__(self, action_dim=7):
            self.action_dim = action_dim
        
        def select_action(self, obs, n_envs=1):
            # Return batched actions (n_envs, action_dim)
            return MetaAction(action=np.random.randn(n_envs, self.action_dim).astype(np.float32))
        
        def train(self):
            pass
        
        def eval(self):
            pass
    
    # Simple MetaPolicy wrapper for testing
    class DummyMetaPolicy(MetaPolicy):
        def __init__(self, action_dim=7):
            self.policy = DummyPolicy(action_dim=action_dim)
            self.chunk_size = 1
            self.ctrl_space = 'ee'
            self.ctrl_type = 'delta'
            self.action_queue = []
            self.action_normalizer = None
            self.state_normalizer = None
        
        def select_action(self, mobs, t=0, **kwargs):
            # Infer batch size from obs
            n_envs = 1
            if hasattr(mobs, 'state') and mobs.state is not None:
                n_envs = mobs.state.shape[0] if mobs.state.ndim > 1 else 1
            
            # Get MetaAction from policy
            mact = self.policy.select_action(mobs, n_envs=n_envs)
            
            # Simulate MetaPolicy.inference output format:
            # It returns a numpy array of dicts (one dict per env)
            # We must convert MetaAction to this format
            action_array = mact.action
            if action_array.ndim == 1:
                action_array = action_array[np.newaxis, :]
            
            # Create list of dicts, then convert to object array
            action_dicts = [{'action': action_array[i]} for i in range(len(action_array))]
            return np.array(action_dicts, dtype=object)
    
    # Simple algorithm for testing (supports Dict[env_type, replay])
    class DummyAlgorithm(BaseAlgorithm):
        def __init__(self, meta_policy, replay=None, **kwargs):
            super().__init__(meta_policy=meta_policy, replay=replay, **kwargs)
            self._timestep = 0
            self._update_count = 0
        
        def update(self, batch=None, **kwargs):
            self._update_count += 1
            batch_size = kwargs.get('batch_size', 32)
            env_type = kwargs.get('env_type', None)
            
            if batch is None and self.replay is not None:
                if isinstance(self.replay, dict):
                    if env_type:
                        batch = self.replay[env_type].sample(batch_size)
                    else:
                        # Sample from all replays
                        batch = {}
                        for k, v in self.replay.items():
                            batch[k] = v.sample(batch_size)
                else:
                    batch = self.replay.sample(batch_size)
            return {'loss': np.random.randn(), 'update_count': self._update_count}
        
        def select_action(self, obs, **kwargs):
            return self.meta_policy.select_action(obs, t=self._timestep)
    
    # Import DummyCollector from base_collector
    from rl.collectors.base_collector import DummyCollector
    
    # Simple trainer implementation for testing (new architecture)
    class SimpleTrainer(BaseTrainer):
        """Simple trainer for testing - uses collectors (which manage envs internally)."""
        
        def __init__(
            self,
            algorithm,
            collectors,
            reward_fn=None,
            **kwargs
        ):
            super().__init__(algorithm, collectors, reward_fn, **kwargs)
            self._total_steps = 0
            self._training_logs = []
        
        def train(self, **kwargs):
            total_steps = kwargs.get('total_steps', 1000)
            log_interval = kwargs.get('log_interval', 100)
            update_interval = kwargs.get('update_interval', 50)
            batch_size = kwargs.get('batch_size', 32)
            
            print(f"Starting training for {total_steps} steps...")
            
            while self._total_steps < total_steps:
                # Collect data from all env types
                all_stats = self.collect_rollout_all(n_steps=update_interval)
                
                # Sum up total steps from all collectors
                total_collected = sum(stats['total_steps'] for stats in all_stats.values())
                self._total_steps += total_collected
                
                # Update algorithm (can sample from any/all replays)
                result = self.algorithm.update(batch_size=batch_size)
                
                # Log
                if self._total_steps % log_interval == 0:
                    print(f"  Step {self._total_steps}: loss={result.get('loss', 0.0):.4f}")
                    self._training_logs.append({
                        'step': self._total_steps,
                        'loss': result.get('loss', 0.0)
                    })
            
            print(f"Training completed. Total steps: {self._total_steps}")
        
        def evaluate(self, n_episodes=10, render=False, env_type=None, **kwargs):
            print(f"Evaluating for {n_episodes} episodes on env_type={env_type}...")
            
            # Get env from collector
            vec_env = self.get_env(env_type)
            alg = self.algorithm
            
            episode_rewards = []
            episode_lengths = []
            
            for ep in range(n_episodes):
                obs = vec_env.reset(id=0)
                episode_reward = 0.0
                episode_length = 0
                done = False
                
                while not done:
                    action = alg.select_action(obs)
                    obs, reward, done, info = vec_env.step(action, id=0)
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            results = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths
            }
            print(f"Evaluation: mean_reward={results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            return results
        
        def save(self, path):
            print(f"Saving to {path} (mock)")
        
        def load(self, path):
            print(f"Loading from {path} (mock)")
    
    # ==========================================================================
    # Test the implementation
    # ==========================================================================
    print("=" * 60)
    print("Testing BaseTrainer with New Architecture")
    print("(Trainer uses Collectors which manage Envs internally)")
    print("=" * 60)
    
    # Test 1: Single environment type
    print("\n" + "-" * 40)
    print("Test 1: Single Environment Type")
    print("-" * 40)
    
    # Create env, replay, algorithm, collector
    env_fns = [lambda: DummyMetaEnv(state_dim=10, action_dim=7, max_steps=20) for _ in range(4)]
    vec_env = SequentialVectorEnv(env_fns)
    print(f"\n1. Created SequentialVectorEnv with {vec_env.env_num} parallel envs")
    
    meta_policy = DummyMetaPolicy(action_dim=7)
    replay = MetaReplay(capacity=10000, env_type='default', n_envs=4, state_dim=10, action_dim=7)
    algorithm = DummyAlgorithm(meta_policy=meta_policy, replay=replay)
    
    # Create collector (it manages the env)
    collector = DummyCollector(envs=vec_env, algorithm=algorithm)
    
    # Create trainer (NO envs parameter - collector manages it!)
    trainer = SimpleTrainer(
        algorithm=algorithm,
        collectors=collector,  # Single collector -> becomes {'default': collector}
        reward_fn=None
    )
    print(f"\n2. Created trainer: {trainer}")
    print(f"   env_types: {trainer.get_env_types()}")
    print(f"   total_env_num: {trainer.get_total_env_num()}")
    
    # Test compute_reward
    print("\n3. Testing compute_reward...")
    state = MetaObs(state=np.random.randn(10).astype(np.float32))
    action = MetaAction(action=np.random.randn(7).astype(np.float32))
    next_state = MetaObs(state=np.random.randn(10).astype(np.float32))
    env_reward = 1.5
    computed_reward = trainer.compute_reward(state, action, next_state, env_reward, {})
    print(f"   Env reward: {env_reward}, Computed reward: {computed_reward}")
    
    # Test training
    print("\n4. Testing training...")
    trainer.train(total_steps=200, log_interval=100, update_interval=20, batch_size=16)
    print(f"   Replay buffer size: {len(algorithm.replay)}")
    
    # Test 2: Multiple environment types (the main use case!)
    print("\n" + "-" * 40)
    print("Test 2: Multiple Environment Types")
    print("(One algorithm, multiple env types, each with own collector & replay)")
    print("-" * 40)
    
    # Create envs for different types
    sim_env_fns = [lambda: DummyMetaEnv(state_dim=10, action_dim=7, max_steps=20) for _ in range(4)]
    real_env_fns = [lambda: DummyMetaEnv(state_dim=10, action_dim=7, max_steps=10) for _ in range(2)]
    
    sim_vec_env = SequentialVectorEnv(sim_env_fns)
    real_vec_env = SequentialVectorEnv(real_env_fns)
    print(f"\n1. Created sim env with {len(sim_vec_env)} parallel envs")
    print(f"   Created real env with {len(real_vec_env)} parallel envs")
    
    # Create replays for each env type
    replays = {
        'sim': MetaReplay(capacity=5000, env_type='sim', n_envs=4, state_dim=10, action_dim=7),
        'real': MetaReplay(capacity=2000, env_type='real', n_envs=2, state_dim=10, action_dim=7),
    }
    
    # Create algorithm with multi-replay
    meta_policy2 = DummyMetaPolicy(action_dim=7)
    algorithm2 = DummyAlgorithm(meta_policy=meta_policy2, replay=replays)
    
    # Create collectors for each env type
    collectors = {
        'sim': DummyCollector(envs=sim_vec_env, algorithm=algorithm2),
        'real': DummyCollector(envs=real_vec_env, algorithm=algorithm2),
    }
    print(f"\n2. Created collectors for env types: {list(collectors.keys())}")
    
    # Create trainer with multiple collectors
    multi_trainer = SimpleTrainer(
        algorithm=algorithm2,
        collectors=collectors,
        reward_fn=IdentityReward()
    )
    print(f"\n3. Created multi-env trainer: {multi_trainer}")
    print(f"   env_types: {multi_trainer.get_env_types()}")
    print(f"   total_env_num: {multi_trainer.get_total_env_num()}")
    print(f"   sim env_num: {len(multi_trainer.get_env('sim'))}")
    print(f"   real env_num: {len(multi_trainer.get_env('real'))}")
    
    # Test collect from specific env type
    print("\n4. Testing collect from specific env type...")
    sim_stats = multi_trainer.collect_rollout(n_steps=50, env_type='sim')
    print(f"   Sim collected: {sim_stats['total_steps']} steps")
    print(f"   Sim replay size: {len(replays['sim'])}")
    print(f"   Real replay size: {len(replays['real'])}")
    
    # Test collect from all env types
    print("\n5. Testing collect from ALL env types...")
    all_stats = multi_trainer.collect_rollout_all(n_steps=30)
    print(f"   Collected from: {list(all_stats.keys())}")
    for env_type, stats in all_stats.items():
        print(f"   {env_type}: {stats['total_steps']} steps")
    print(f"   Sim replay size: {len(replays['sim'])}")
    print(f"   Real replay size: {len(replays['real'])}")
    
    # Test training with multi-env
    print("\n6. Testing training with multi-env...")
    multi_trainer.train(total_steps=200, log_interval=100, update_interval=20, batch_size=16)
    print(f"   Final sim replay size: {len(replays['sim'])}")
    print(f"   Final real replay size: {len(replays['real'])}")
    
    # Test evaluation on specific env type
    print("\n7. Testing evaluation on 'sim' env...")
    eval_results = multi_trainer.evaluate(n_episodes=3, env_type='sim')
    print(f"   Mean reward: {eval_results['mean_reward']:.2f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

