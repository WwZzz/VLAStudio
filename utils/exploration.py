"""
Exploration Strategies for RL

This module defines exploration strategies for action selection during training.
Supports:
- Random exploration (for initial exploration phase)
- Gaussian noise
- Ornstein-Uhlenbeck noise
- Custom noise functions (e.g., uncertainty-based)

Usage:
    # Gaussian noise with initial random exploration
    exploration = ExplorationScheduler(
        exploration_strategy=GaussianNoise(sigma=0.1),
        random_steps=10000,
        action_low=np.array([-1.0] * 7),
        action_high=np.array([1.0] * 7),
    )
    
    # Apply exploration to action
    explored_action = exploration(action, step=current_step)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any


class BaseExploration(ABC):
    """
    Base class for exploration strategies.
    
    Exploration strategies modify actions to encourage exploration during training.
    """
    
    @abstractmethod
    def __call__(
        self, 
        action: np.ndarray, 
        step: int = 0,
        **kwargs
    ) -> np.ndarray:
        """
        Apply exploration to action.
        
        Args:
            action: Original action from policy, shape (action_dim,) or (n_envs, action_dim)
            step: Current training step (for annealing)
            **kwargs: Additional arguments (e.g., obs for uncertainty-based exploration)
        
        Returns:
            Explored action with same shape as input
        """
        raise NotImplementedError
    
    def reset(self, env_idx: Optional[int] = None) -> None:
        """Reset exploration state (e.g., for OU noise)."""
        pass


class NoExploration(BaseExploration):
    """No exploration - return action as-is."""
    
    def __call__(self, action: np.ndarray, step: int = 0, **kwargs) -> np.ndarray:
        return action


class GaussianNoise(BaseExploration):
    """
    Gaussian noise exploration.
    
    Adds zero-mean Gaussian noise to actions, with optional annealing.
    Used in TD3, SAC, etc.
    
    Example:
        noise = GaussianNoise(sigma=0.1, sigma_min=0.01, decay_steps=100000)
        noisy_action = noise(action, step=current_step)
    """
    
    def __init__(
        self,
        sigma: float = 0.1,
        sigma_min: float = 0.01,
        sigma_decay: float = 1.0,  # No decay by default
        decay_steps: int = 0,      # 0 means no decay
    ):
        """
        Args:
            sigma: Initial noise standard deviation
            sigma_min: Minimum noise standard deviation (for annealing)
            sigma_decay: Decay factor per step (exponential decay, < 1.0 to enable)
            decay_steps: Total steps for linear decay (> 0 to enable, takes priority over sigma_decay)
        """
        self.sigma_init = sigma
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.decay_steps = decay_steps
    
    def __call__(self, action: np.ndarray, step: int = 0, **kwargs) -> np.ndarray:
        # Anneal sigma
        if self.decay_steps > 0:
            # Linear decay
            decay_ratio = max(0.0, 1.0 - step / self.decay_steps)
            self.sigma = self.sigma_min + (self.sigma_init - self.sigma_min) * decay_ratio
        elif self.sigma_decay < 1.0:
            # Exponential decay
            self.sigma = max(self.sigma_min, self.sigma_init * (self.sigma_decay ** step))
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.sigma, size=action.shape)
        noisy_action = action + noise
        
        return noisy_action.astype(action.dtype)
    
    def reset(self, env_idx: Optional[int] = None) -> None:
        pass  # Gaussian noise is stateless
    
    def __repr__(self) -> str:
        return (f"GaussianNoise(sigma={self.sigma:.4f}, sigma_init={self.sigma_init}, "
                f"sigma_min={self.sigma_min}, decay_steps={self.decay_steps})")


class OUNoise(BaseExploration):
    """
    Ornstein-Uhlenbeck noise exploration.
    
    Temporally correlated noise, often used in continuous control tasks.
    Used in DDPG, etc.
    
    The OU process: dx = theta * (mu - x) * dt + sigma * dW
    
    Example:
        noise = OUNoise(action_dim=7, n_envs=4, sigma=0.2, theta=0.15)
        noisy_action = noise(action, step=current_step)
    """
    
    def __init__(
        self,
        action_dim: int,
        n_envs: int = 1,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        sigma_min: float = 0.01,
        sigma_decay: float = 1.0,
        decay_steps: int = 0,
    ):
        """
        Args:
            action_dim: Dimension of action space
            n_envs: Number of parallel environments
            mu: Mean of the noise (typically 0)
            theta: Rate of mean reversion (how fast noise returns to mu)
            sigma: Volatility of the noise
            sigma_min: Minimum sigma for annealing
            sigma_decay: Exponential decay factor
            decay_steps: Steps for linear decay
        """
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.mu = mu
        self.theta = theta
        self.sigma_init = sigma
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.decay_steps = decay_steps
        
        # Initialize state for each environment
        self._state = np.ones((n_envs, action_dim)) * mu
    
    def __call__(self, action: np.ndarray, step: int = 0, **kwargs) -> np.ndarray:
        # Anneal sigma
        if self.decay_steps > 0:
            decay_ratio = max(0.0, 1.0 - step / self.decay_steps)
            self.sigma = self.sigma_min + (self.sigma_init - self.sigma_min) * decay_ratio
        elif self.sigma_decay < 1.0:
            self.sigma = max(self.sigma_min, self.sigma_init * (self.sigma_decay ** step))
        
        # Handle both single and batched actions
        is_batched = action.ndim == 2
        if not is_batched:
            action = action[np.newaxis, :]  # (1, action_dim)
        
        batch_size = action.shape[0]
        
        # Update OU state: dx = theta * (mu - x) + sigma * noise
        dx = (self.theta * (self.mu - self._state[:batch_size]) + 
              self.sigma * np.random.randn(batch_size, self.action_dim))
        self._state[:batch_size] += dx
        
        # Add noise to action
        noisy_action = action + self._state[:batch_size]
        
        if not is_batched:
            noisy_action = noisy_action[0]
        
        return noisy_action.astype(action.dtype)
    
    def reset(self, env_idx: Optional[int] = None) -> None:
        """Reset OU state."""
        if env_idx is None:
            self._state = np.ones((self.n_envs, self.action_dim)) * self.mu
        else:
            self._state[env_idx] = self.mu
    
    def __repr__(self) -> str:
        return (f"OUNoise(action_dim={self.action_dim}, n_envs={self.n_envs}, "
                f"theta={self.theta}, sigma={self.sigma:.4f})")


class CustomNoise(BaseExploration):
    """
    Custom noise exploration using a user-provided function.
    
    Allows implementing advanced exploration strategies like:
    - Uncertainty-based exploration (using model epistemic uncertainty)
    - Curiosity-driven exploration
    - Parameter noise
    - Any other custom exploration logic
    
    Example:
        def uncertainty_noise_fn(action, step, obs=None, model=None, **kwargs):
            if model is not None and obs is not None:
                uncertainty = model.get_uncertainty(obs)
                noise_scale = uncertainty * 0.5
            else:
                noise_scale = 0.1
            noise = np.random.normal(0, noise_scale, size=action.shape)
            return action + noise
        
        noise = CustomNoise(noise_fn=uncertainty_noise_fn)
        noisy_action = noise(action, step=step, obs=obs, model=model)
    """
    
    def __init__(
        self,
        noise_fn: Callable[[np.ndarray, int, Any], np.ndarray],
        reset_fn: Optional[Callable[[Optional[int]], None]] = None,
    ):
        """
        Args:
            noise_fn: Custom noise function with signature:
                      noise_fn(action, step, **kwargs) -> noisy_action
                      - action: Original action from policy
                      - step: Current training step
                      - **kwargs: Additional info (obs, model uncertainty, etc.)
            reset_fn: Optional reset function for stateful noise
        """
        self.noise_fn = noise_fn
        self.reset_fn = reset_fn
    
    def __call__(self, action: np.ndarray, step: int = 0, **kwargs) -> np.ndarray:
        noisy_action = self.noise_fn(action, step, **kwargs)
        return noisy_action.astype(action.dtype)
    
    def reset(self, env_idx: Optional[int] = None) -> None:
        if self.reset_fn is not None:
            self.reset_fn(env_idx)
    
    def __repr__(self) -> str:
        return f"CustomNoise(noise_fn={self.noise_fn.__name__})"


class RandomExploration(BaseExploration):
    """
    Random action exploration.
    
    Samples random actions from action space, completely ignoring policy output.
    Used for initial exploration phase.
    
    Example:
        random_explore = RandomExploration(
            action_low=np.array([-1.0] * 7),
            action_high=np.array([1.0] * 7),
        )
        random_action = random_explore(policy_action)  # policy_action is ignored
    """
    
    def __init__(
        self,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ):
        """
        Args:
            action_low: Lower bound of action space
            action_high: Upper bound of action space
        """
        self.action_low = np.asarray(action_low)
        self.action_high = np.asarray(action_high)
    
    def __call__(self, action: np.ndarray, step: int = 0, **kwargs) -> np.ndarray:
        # Completely ignore input action, sample random action
        random_action = np.random.uniform(
            self.action_low, 
            self.action_high, 
            size=action.shape
        )
        return random_action.astype(action.dtype)
    
    def __repr__(self) -> str:
        return f"RandomExploration(action_low={self.action_low}, action_high={self.action_high})"


class ExplorationScheduler:
    """
    Scheduler for switching between exploration strategies.
    
    Supports:
    - Initial random exploration phase (before a certain step)
    - Transition to noise-based exploration
    - Annealing exploration over time
    
    Example:
        # Create scheduler with 10000 random steps, then Gaussian noise
        exploration = ExplorationScheduler(
            exploration_strategy=GaussianNoise(sigma=0.1, decay_steps=100000),
            random_steps=10000,
            action_low=np.array([-1.0] * 7),
            action_high=np.array([1.0] * 7),
        )
        
        # In training loop:
        for step in range(total_steps):
            action = policy(obs)
            explored_action = exploration(action, step=step)
            # or use internal counter:
            explored_action = exploration(action)
    """
    
    def __init__(
        self,
        exploration_strategy: BaseExploration,
        random_steps: int = 0,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
    ):
        """
        Args:
            exploration_strategy: Main exploration strategy (e.g., GaussianNoise, OUNoise)
            random_steps: Number of initial steps to use pure random exploration
                         - Set to 0 to disable random exploration phase
            action_low: Lower bound for random exploration (required if random_steps > 0)
            action_high: Upper bound for random exploration (required if random_steps > 0)
        """
        self.exploration_strategy = exploration_strategy
        self.random_steps = random_steps
        self.action_low = action_low
        self.action_high = action_high
        
        # Create random exploration for initial phase
        if random_steps > 0:
            if action_low is None or action_high is None:
                raise ValueError("action_low and action_high required when random_steps > 0")
            self.random_exploration = RandomExploration(action_low, action_high)
        else:
            self.random_exploration = None
        
        self._current_step = 0
    
    def __call__(
        self, 
        action: np.ndarray, 
        step: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Apply exploration based on current step.
        
        Args:
            action: Original action from policy
            step: Optional step override (if None, uses internal counter)
            **kwargs: Additional arguments for exploration strategy
        
        Returns:
            Explored action
        """
        if step is None:
            step = self._current_step
            self._current_step += 1
        
        # Initial random exploration phase
        if step < self.random_steps and self.random_exploration is not None:
            return self.random_exploration(action, step, **kwargs)
        
        # Main exploration strategy (pass adjusted step for proper annealing)
        return self.exploration_strategy(action, step - self.random_steps, **kwargs)
    
    def reset(self, env_idx: Optional[int] = None) -> None:
        """Reset exploration state."""
        self.exploration_strategy.reset(env_idx)
        if self.random_exploration is not None:
            self.random_exploration.reset(env_idx)
    
    def reset_step_counter(self) -> None:
        """Reset internal step counter."""
        self._current_step = 0
    
    @property
    def is_random_phase(self) -> bool:
        """Check if still in random exploration phase."""
        return self._current_step < self.random_steps
    
    @property
    def current_step(self) -> int:
        """Get current step count."""
        return self._current_step
    
    def __repr__(self) -> str:
        return (f"ExplorationScheduler(strategy={self.exploration_strategy}, "
                f"random_steps={self.random_steps}, current_step={self._current_step})")


