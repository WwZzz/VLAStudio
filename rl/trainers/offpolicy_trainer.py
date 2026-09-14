"""
Off-Policy Trainer

Trainer for off-policy RL algorithms (TD3, SAC, DDPG, etc.)

Design:
- Separates data collection (Collector) from policy updates (Trainer)
- Supports evaluation during training
- Handles checkpointing and logging
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, Callable, Type
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm

from .base_trainer import BaseTrainer
from rl.collectors import BaseCollector
from rl.algorithms.base import BaseAlgorithm
from benchmark.utils import SequentialVectorEnv


@dataclass
class OffPolicyTrainerConfig:
    """Configuration for OffPolicyTrainer."""
    total_steps: int = 1000000
    start_steps: int = 25000  # Random exploration steps
    update_after: int = 1000  # Start updating after this many steps
    update_every: int = 50    # Update every N steps
    batch_size: int = 256
    
    # Logging and evaluation
    log_freq: int = 1000
    eval_freq: int = 10000
    eval_episodes: int = 10
    save_freq: int = 50000
    
    # Output
    output_dir: str = 'ckpt/rl_training'
    
    # Environment settings
    max_timesteps: int = 1000
    ctrl_space: str = 'joint'
    
    # Exploration
    expl_noise: float = 0.1


class OffPolicyTrainer(BaseTrainer):
    """
    Trainer for off-policy RL algorithms.
    
    Coordinates:
    - Collector: Gathers experience from environment
    - Algorithm: Updates policy using collected data
    - Evaluation: Periodically evaluates policy performance
    """
    
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        collector: BaseCollector,
        config: Optional[OffPolicyTrainerConfig] = None,
        eval_env_fn: Optional[Callable] = None,
        eval_vec_env_cls: Optional[Type] = None,
        **kwargs
    ):
        """
        Initialize OffPolicyTrainer.
        
        Args:
            algorithm: RL algorithm (TD3, SAC, etc.)
            collector: Data collector for environment interaction
            config: Trainer configuration
            eval_env_fn: Factory function for creating evaluation environments
            eval_vec_env_cls: Vector environment class for evaluation (default: SequentialVectorEnv)
            **kwargs: Additional arguments
        """
        super().__init__(algorithm=algorithm, collectors=collector, **kwargs)
        
        self.config = config or OffPolicyTrainerConfig()
        self.collector = collector
        self.eval_env_fn = eval_env_fn
        self.eval_vec_env_cls = eval_vec_env_cls or SequentialVectorEnv
        
        # Training state
        self._current_step = 0
        self._episode_count = 0
        self._recent_rewards = []
        
    def train(self, resume_step: int = 0) -> Dict[str, Any]:
        """
        Run the training loop.
        
        Args:
            resume_step: Step to resume from (for checkpoint resumption)
            
        Returns:
            Training statistics
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(self.config.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            config_dict = {k: v for k, v in vars(self.config).items()}
            json.dump(config_dict, f, indent=2, default=str)
        
        self._current_step = resume_step
        
        logger.info("=" * 60)
        logger.info(f"Starting training for {self.config.total_steps} steps")
        logger.info("=" * 60)
        
        # Reset collector
        self.collector.reset()
        
        pbar = tqdm(
            range(resume_step, self.config.total_steps), 
            desc="Training", 
            initial=resume_step, 
            total=self.config.total_steps
        )
        
        for step in pbar:
            self._current_step = step
            
            # Collect one step of data
            # Use random exploration for initial steps
            use_random = step < self.config.start_steps
            collect_stats = self.collector.collect_step(
                noise_scale=self.config.expl_noise if not use_random else None,
                use_random=use_random,
            )
            
            # Update episode tracking
            if collect_stats.get('episode_rewards'):
                self._recent_rewards.extend(collect_stats['episode_rewards'])
                self._episode_count += len(collect_stats['episode_rewards'])
                # Keep only recent 100 rewards
                self._recent_rewards = self._recent_rewards[-100:]
            
            # Policy update
            if step >= self.config.update_after and step % self.config.update_every == 0:
                for _ in range(self.config.update_every):
                    self.algorithm.update(
                        batch_size=self.config.batch_size, 
                        env=self.collector.get_env()
                    )
            
            # Logging
            if step > 0 and step % self.config.log_freq == 0:
                avg_reward = np.mean(self._recent_rewards) if self._recent_rewards else 0.0
                pbar.set_postfix({
                    'episodes': self._episode_count,
                    'avg_reward': f'{avg_reward:.2f}',
                    'buffer': len(self.algorithm.replay) if self.algorithm.replay else 0,
                })
            
            # Evaluation
            if step > 0 and step % self.config.eval_freq == 0:
                self._run_evaluation(step)
            
            # Save checkpoint
            if step > 0 and step % self.config.save_freq == 0:
                self._save_checkpoint(step)
        
        # Final save
        final_path = os.path.join(self.config.output_dir, 'final_model.pt')
        self.algorithm.save(final_path)
        logger.info(f"Training complete. Final model saved to {final_path}")
        
        return {
            'total_steps': self.config.total_steps,
            'episode_count': self._episode_count,
            'final_avg_reward': np.mean(self._recent_rewards) if self._recent_rewards else 0.0,
        }
    
    def _run_evaluation(self, step: int) -> Dict[str, Any]:
        """Run evaluation and log results."""
        logger.info(f"Step {step}: Running evaluation...")
        
        if self.eval_env_fn is None:
            logger.warning("No eval_env_fn provided, skipping evaluation")
            return {}
        
        # Run evaluation (evaluate method handles env creation/closing)
        eval_result = self.evaluate(n_episodes=self.config.eval_episodes)
        
        if not eval_result or 'error' in eval_result:
            return eval_result
        
        # Log results
        logger.info(
            f"Step {step}: Eval return = {eval_result['mean_return']:.2f} ± {eval_result['std_return']:.2f}, "
            f"length = {eval_result['mean_length']:.1f}, "
            f"episodes = {eval_result['num_episodes']}"
        )
        
        # Save eval results
        eval_path = os.path.join(self.config.output_dir, 'eval_results.json')
        eval_data = {'step': step, **eval_result}
        with open(eval_path, 'a') as f:
            f.write(json.dumps(eval_data, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x) + '\n')
        
        return eval_result
    
    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False,
        env_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate policy performance.
        
        Overrides base class to support creating eval env from eval_env_fn.
        
        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render environment (not used currently)
            env_type: Optional environment type (not used currently)
            **kwargs: Additional arguments (e.g., vec_env, max_timesteps)
        
        Returns:
            Dictionary containing evaluation metrics
        """
        vec_env = kwargs.get('vec_env', None)
        should_close = False
        
        # Create evaluation environment if not provided
        if vec_env is None and self.eval_env_fn is not None:
            eval_env_fns = [self.eval_env_fn for _ in range(n_episodes)]
            vec_env = self.eval_vec_env_cls(eval_env_fns)
            should_close = True
        
        # Set default max_timesteps and ctrl_space from config
        if 'max_timesteps' not in kwargs:
            kwargs['max_timesteps'] = self.config.max_timesteps
        if 'ctrl_space' not in kwargs:
            kwargs['ctrl_space'] = self.config.ctrl_space
        
        # Pass vec_env to base class evaluate
        kwargs['vec_env'] = vec_env
        result = super().evaluate(n_episodes=n_episodes, render=render, env_type=env_type, **kwargs)
        
        if should_close and vec_env is not None:
            vec_env.close()
        
        return result
    
    def _save_checkpoint(self, step: int) -> None:
        """Save training checkpoint."""
        self.save_checkpoint(self.config.output_dir, step)
        logger.info(f"Saved checkpoint to {self.config.output_dir}/checkpoint_{step}.pt")

