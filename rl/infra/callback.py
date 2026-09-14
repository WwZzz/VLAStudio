"""
Callback System for RL Training

This module provides a flexible callback system for hooking into the training loop:
- Progress tracking and logging
- Evaluation during training
- Checkpoint saving
- Early stopping
- Custom callbacks

Design Philosophy:
- Non-intrusive: Callbacks don't modify training logic
- Composable: Multiple callbacks can be combined
- Extensible: Easy to add custom callbacks
- Event-driven: Callbacks respond to training events
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field


@dataclass
class TrainingState:
    """
    Container for training state passed to callbacks.
    
    This provides a standardized interface for callbacks to access training info.
    """
    step: int = 0
    episode: int = 0
    total_timesteps: int = 0
    
    # Episode info
    episode_reward: float = 0.0
    episode_length: int = 0
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    
    # Training info
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    
    # Timing
    fps: float = 0.0
    time_elapsed: float = 0.0
    
    # Extra info
    info: Dict[str, Any] = field(default_factory=dict)
    
    # Control flags
    should_stop: bool = False


class Callback(ABC):
    """
    Base class for training callbacks.
    
    Callbacks can hook into various points in the training loop:
    - on_training_start: Called once at the beginning of training
    - on_training_end: Called once at the end of training
    - on_step_start: Called before each training step
    - on_step_end: Called after each training step
    - on_episode_start: Called at the start of each episode
    - on_episode_end: Called at the end of each episode
    - on_rollout_start: Called before data collection
    - on_rollout_end: Called after data collection
    - on_update_start: Called before policy update
    - on_update_end: Called after policy update
    
    Override only the methods you need.
    """
    
    def __init__(self):
        self.training_state: Optional[TrainingState] = None
        self.trainer = None
        self.logger = None
    
    def set_trainer(self, trainer) -> None:
        """Set reference to the trainer."""
        self.trainer = trainer
    
    def set_logger(self, logger) -> None:
        """Set reference to the logger."""
        self.logger = logger
    
    def on_training_start(self, state: TrainingState) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_training_end(self, state: TrainingState) -> None:
        """Called at the end of training."""
        pass
    
    def on_step_start(self, state: TrainingState) -> None:
        """Called before each training step."""
        pass
    
    def on_step_end(self, state: TrainingState) -> bool:
        """
        Called after each training step.
        
        Returns:
            True to continue training, False to stop
        """
        return True
    
    def on_episode_start(self, state: TrainingState) -> None:
        """Called at the start of each episode."""
        pass
    
    def on_episode_end(self, state: TrainingState) -> None:
        """Called at the end of each episode."""
        pass
    
    def on_rollout_start(self, state: TrainingState) -> None:
        """Called before data collection (rollout)."""
        pass
    
    def on_rollout_end(self, state: TrainingState) -> None:
        """Called after data collection (rollout)."""
        pass
    
    def on_update_start(self, state: TrainingState) -> None:
        """Called before policy update."""
        pass
    
    def on_update_end(self, state: TrainingState) -> None:
        """Called after policy update."""
        pass


class CallbackList(Callback):
    """
    Container for multiple callbacks.
    
    Forwards all events to contained callbacks in order.
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        super().__init__()
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def set_trainer(self, trainer) -> None:
        """Set trainer for all callbacks."""
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)
    
    def set_logger(self, logger) -> None:
        """Set logger for all callbacks."""
        self.logger = logger
        for callback in self.callbacks:
            callback.set_logger(logger)
    
    def on_training_start(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_training_start(state)
    
    def on_training_end(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_training_end(state)
    
    def on_step_start(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_step_start(state)
    
    def on_step_end(self, state: TrainingState) -> bool:
        continue_training = True
        for callback in self.callbacks:
            if not callback.on_step_end(state):
                continue_training = False
        return continue_training
    
    def on_episode_start(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_episode_start(state)
    
    def on_episode_end(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_episode_end(state)
    
    def on_rollout_start(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start(state)
    
    def on_rollout_end(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end(state)
    
    def on_update_start(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_update_start(state)
    
    def on_update_end(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_update_end(state)


class ProgressCallback(Callback):
    """
    Callback for logging training progress.
    
    Logs metrics at specified intervals.
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        verbose: int = 1
    ):
        """
        Initialize progress callback.
        
        Args:
            log_interval: Steps between log outputs
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed)
        """
        super().__init__()
        self.log_interval = log_interval
        self.verbose = verbose
        self._start_time = None
        self._last_log_step = 0
    
    def on_training_start(self, state: TrainingState) -> None:
        self._start_time = time.time()
        if self.verbose >= 1:
            print("=" * 60)
            print("Training started")
            print("=" * 60)
    
    def on_training_end(self, state: TrainingState) -> None:
        if self.verbose >= 1:
            elapsed = time.time() - self._start_time
            print("=" * 60)
            print(f"Training completed in {elapsed:.1f}s")
            print(f"Total steps: {state.step}")
            print(f"Total episodes: {state.episode}")
            if state.episode_rewards:
                print(f"Final mean reward: {np.mean(state.episode_rewards[-100:]):.2f}")
            print("=" * 60)
    
    def on_step_end(self, state: TrainingState) -> bool:
        if state.step % self.log_interval == 0 and state.step > self._last_log_step:
            self._last_log_step = state.step
            self._log_progress(state)
        return True
    
    def _log_progress(self, state: TrainingState) -> None:
        """Log current progress."""
        if self.verbose == 0:
            return
        
        elapsed = time.time() - self._start_time
        fps = state.step / elapsed if elapsed > 0 else 0
        
        # Build log message
        parts = [f"Step: {state.step:>8}"]
        
        if state.episode_rewards:
            mean_reward = np.mean(state.episode_rewards[-100:])
            parts.append(f"Reward: {mean_reward:>8.2f}")
        
        if state.loss is not None:
            parts.append(f"Loss: {state.loss:>8.4f}")
        
        parts.append(f"FPS: {fps:>6.0f}")
        parts.append(f"Time: {elapsed:>6.0f}s")
        
        print(" | ".join(parts))
        
        if self.logger:
            metrics = {
                'progress/fps': fps,
                'progress/time_elapsed': elapsed
            }
            if state.episode_rewards:
                metrics['progress/mean_reward'] = np.mean(state.episode_rewards[-100:])
            self.logger.log_scalars(metrics, step=state.step)


class EvalCallback(Callback):
    """
    Callback for periodic evaluation during training.
    
    Evaluates the policy on a separate environment at specified intervals.
    """
    
    def __init__(
        self,
        eval_fn: Callable[[int], Dict[str, float]],
        eval_interval: int = 10000,
        n_eval_episodes: int = 10,
        verbose: int = 1
    ):
        """
        Initialize evaluation callback.
        
        Args:
            eval_fn: Function that takes step and returns eval metrics
            eval_interval: Steps between evaluations
            n_eval_episodes: Number of episodes for evaluation
            verbose: Verbosity level
        """
        super().__init__()
        self.eval_fn = eval_fn
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose
        
        self._last_eval_step = 0
        self.eval_results: List[Dict[str, Any]] = []
    
    def on_step_end(self, state: TrainingState) -> bool:
        if state.step >= self._last_eval_step + self.eval_interval:
            self._last_eval_step = state.step
            self._evaluate(state)
        return True
    
    def _evaluate(self, state: TrainingState) -> None:
        """Run evaluation."""
        if self.verbose >= 1:
            print(f"\n[Eval @ step {state.step}]", end=" ")
        
        # Run evaluation
        eval_metrics = self.eval_fn(state.step)
        
        # Store results
        result = {
            'step': state.step,
            **eval_metrics
        }
        self.eval_results.append(result)
        
        # Log
        if self.verbose >= 1:
            metrics_str = " | ".join([f"{k}: {v:.2f}" for k, v in eval_metrics.items()])
            print(metrics_str)
        
        if self.logger:
            self.logger.log_scalars(
                {f"eval/{k}": v for k, v in eval_metrics.items()},
                step=state.step
            )


class CheckpointCallback(Callback):
    """
    Callback for saving checkpoints during training.
    
    Saves checkpoints at specified intervals and keeps best checkpoint.
    """
    
    def __init__(
        self,
        checkpoint_manager: 'CheckpointManager',
        save_interval: int = 10000,
        save_on_best: bool = True,
        metric_name: str = "episode_reward",
        verbose: int = 1
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            checkpoint_manager: CheckpointManager instance
            save_interval: Steps between checkpoint saves
            save_on_best: Whether to save when best metric is achieved
            metric_name: Metric to track for best model
            verbose: Verbosity level
        """
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.save_interval = save_interval
        self.save_on_best = save_on_best
        self.metric_name = metric_name
        self.verbose = verbose
        
        self._last_save_step = 0
        self._best_metric = None
    
    def on_step_end(self, state: TrainingState) -> bool:
        # Save at interval
        if state.step >= self._last_save_step + self.save_interval:
            self._last_save_step = state.step
            self._save_checkpoint(state, is_best=False)
        
        # Save on best
        if self.save_on_best:
            metric = self._get_metric(state)
            if metric is not None:
                if self._best_metric is None or metric > self._best_metric:
                    self._best_metric = metric
                    self._save_checkpoint(state, is_best=True)
        
        return True
    
    def on_training_end(self, state: TrainingState) -> None:
        """Save final checkpoint."""
        self._save_checkpoint(state, is_best=False)
    
    def _get_metric(self, state: TrainingState) -> Optional[float]:
        """Get the metric value for best model tracking."""
        if self.metric_name == "episode_reward" and state.episode_rewards:
            return np.mean(state.episode_rewards[-10:])
        elif self.metric_name in state.info:
            return state.info[self.metric_name]
        return None
    
    def _save_checkpoint(self, state: TrainingState, is_best: bool) -> None:
        """Save checkpoint using the trainer's save method."""
        if self.trainer is None:
            return
        
        metric = self._get_metric(state)
        
        # Get model and optimizer state from trainer
        save_dict = {}
        if hasattr(self.trainer, 'algorithm'):
            alg = self.trainer.algorithm
            if hasattr(alg, 'meta_policy') and hasattr(alg.meta_policy, 'policy'):
                policy = alg.meta_policy.policy
                if hasattr(policy, 'state_dict'):
                    save_dict['model'] = policy.state_dict()
        
        path = self.checkpoint_manager.save(
            step=state.step,
            episode=state.episode,
            total_timesteps=state.total_timesteps,
            reward=metric,
            is_best=is_best,
            **save_dict
        )
        
        if self.verbose >= 1 and is_best:
            print(f"\n[Checkpoint] Saved best model @ step {state.step} (metric: {metric:.2f})")


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping based on a metric.
    
    Stops training if metric doesn't improve for a specified number of steps.
    """
    
    def __init__(
        self,
        patience: int = 50000,
        min_delta: float = 0.0,
        metric_name: str = "episode_reward",
        mode: str = "max",
        verbose: int = 1
    ):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of steps to wait for improvement
            min_delta: Minimum change to qualify as improvement
            metric_name: Metric to monitor
            mode: 'max' or 'min' - whether higher or lower is better
            verbose: Verbosity level
        """
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.mode = mode
        self.verbose = verbose
        
        self._best_metric = None
        self._steps_without_improvement = 0
        self._last_check_step = 0
    
    def on_step_end(self, state: TrainingState) -> bool:
        # Check every 1000 steps
        if state.step < self._last_check_step + 1000:
            return True
        self._last_check_step = state.step
        
        metric = self._get_metric(state)
        if metric is None:
            return True
        
        improved = False
        if self._best_metric is None:
            improved = True
        elif self.mode == "max" and metric > self._best_metric + self.min_delta:
            improved = True
        elif self.mode == "min" and metric < self._best_metric - self.min_delta:
            improved = True
        
        if improved:
            self._best_metric = metric
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1000
        
        # Check for early stopping
        if self._steps_without_improvement >= self.patience:
            if self.verbose >= 1:
                print(f"\n[Early Stopping] No improvement for {self.patience} steps. Stopping training.")
            state.should_stop = True
            return False
        
        return True
    
    def _get_metric(self, state: TrainingState) -> Optional[float]:
        """Get the metric value."""
        if self.metric_name == "episode_reward" and state.episode_rewards:
            return np.mean(state.episode_rewards[-10:])
        elif self.metric_name in state.info:
            return state.info[self.metric_name]
        return None


if __name__ == '__main__':
    """
    Test code for Callback module.
    """
    print("=" * 60)
    print("Testing Callback Module")
    print("=" * 60)
    
    # Test 1: TrainingState
    print("\n1. Testing TrainingState...")
    state = TrainingState(
        step=100,
        episode=5,
        episode_reward=50.0,
        episode_rewards=[40.0, 45.0, 50.0]
    )
    print(f"   Step: {state.step}, Episode: {state.episode}")
    print(f"   Episode rewards: {state.episode_rewards}")
    
    # Test 2: ProgressCallback
    print("\n2. Testing ProgressCallback...")
    progress_cb = ProgressCallback(log_interval=50, verbose=1)
    
    # Simulate training
    state = TrainingState()
    progress_cb.on_training_start(state)
    
    for step in range(200):
        state.step = step
        state.episode_rewards.append(np.random.randn() * 10 + 50)
        state.loss = 1.0 - step * 0.001
        progress_cb.on_step_end(state)
    
    progress_cb.on_training_end(state)
    
    # Test 3: CallbackList
    print("\n3. Testing CallbackList...")
    
    class CountingCallback(Callback):
        def __init__(self):
            super().__init__()
            self.step_count = 0
            self.episode_count = 0
        
        def on_step_end(self, state):
            self.step_count += 1
            return True
        
        def on_episode_end(self, state):
            self.episode_count += 1
    
    cb1 = CountingCallback()
    cb2 = CountingCallback()
    callback_list = CallbackList([cb1, cb2])
    
    state = TrainingState()
    for _ in range(10):
        callback_list.on_step_end(state)
    
    callback_list.on_episode_end(state)
    callback_list.on_episode_end(state)
    
    print(f"   CB1 step count: {cb1.step_count}, episode count: {cb1.episode_count}")
    print(f"   CB2 step count: {cb2.step_count}, episode count: {cb2.episode_count}")
    assert cb1.step_count == 10 and cb2.step_count == 10
    assert cb1.episode_count == 2 and cb2.episode_count == 2
    
    # Test 4: EarlyStoppingCallback
    print("\n4. Testing EarlyStoppingCallback...")
    early_stop = EarlyStoppingCallback(patience=5000, verbose=0)
    
    state = TrainingState()
    # Simulate no improvement
    for step in range(10):
        state.step = step * 1000
        state.episode_rewards = [50.0] * 10  # Constant reward
        result = early_stop.on_step_end(state)
    
    print(f"   Steps without improvement: {early_stop._steps_without_improvement}")
    print(f"   Should continue: {result}")
    
    # Test 5: EvalCallback
    print("\n5. Testing EvalCallback...")
    
    def dummy_eval_fn(step):
        return {'mean_reward': 50 + step * 0.001, 'success_rate': 0.8}
    
    eval_cb = EvalCallback(
        eval_fn=dummy_eval_fn,
        eval_interval=5000,
        verbose=1
    )
    
    state = TrainingState(step=10000)
    eval_cb.on_step_end(state)
    
    print(f"   Eval results: {eval_cb.eval_results}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

