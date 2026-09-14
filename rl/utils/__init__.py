"""
RL Utilities Module

This module provides utility functions for the RL framework.

Available utilities:
- DataProcessor: Data processor for aligning with ILStudio pipeline
- Running statistics (mean, variance) for normalization
- Advantage computation (GAE)
- Discount reward computation
- Action post-processing helpers (ensure/clip actions)
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union



def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Array of rewards [T]
        values: Array of value estimates [T]
        dones: Array of done flags [T]
        next_value: Value estimate for the next state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        Array of advantages [T]
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    
    return advantages


def compute_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    next_value: float = 0.0,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Compute discounted returns.
    
    Args:
        rewards: Array of rewards [T]
        dones: Array of done flags [T]
        next_value: Value estimate for the next state
        gamma: Discount factor
    
    Returns:
        Array of discounted returns [T]
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    running_return = next_value
    
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return * (1.0 - dones[t])
        returns[t] = running_return
    
    return returns


class RunningMeanStd:
    """
    Running mean and standard deviation tracker.
    
    Useful for observation normalization in RL.
    """
    
    def __init__(self, shape: tuple = (), epsilon: float = 1e-8):
        """
        Initialize running statistics.
        
        Args:
            shape: Shape of the data to track
            epsilon: Small value for numerical stability
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with new data.
        
        Args:
            x: New data batch [batch_size, *shape]
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self, 
        batch_mean: np.ndarray, 
        batch_var: np.ndarray, 
        batch_count: int
    ) -> None:
        """Update from batch moments."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize data using running statistics.
        
        Args:
            x: Data to normalize
        
        Returns:
            Normalized data
        """
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Denormalize data using running statistics.
        
        Args:
            x: Normalized data
        
        Returns:
            Denormalized data
        """
        return x * np.sqrt(self.var + self.epsilon) + self.mean


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute explained variance.
    
    Args:
        y_pred: Predicted values
        y_true: True values
    
    Returns:
        Explained variance (1.0 is perfect prediction)
    """
    var_y = np.var(y_true)
    if var_y == 0:
        return np.nan
    return 1.0 - np.var(y_true - y_pred) / var_y


def polyak_update(
    source_params: List[torch.nn.Parameter],
    target_params: List[torch.nn.Parameter],
    tau: float = 0.005
) -> None:
    """
    Perform Polyak (soft) update of target network parameters.
    
    target = tau * source + (1 - tau) * target
    
    Args:
        source_params: Source network parameters
        target_params: Target network parameters
        tau: Interpolation factor (0.0 = no update, 1.0 = full copy)
    """
    with torch.no_grad():
        for source_param, target_param in zip(source_params, target_params):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)


def hard_update(
    source_params: List[torch.nn.Parameter],
    target_params: List[torch.nn.Parameter]
) -> None:
    """
    Perform hard update of target network parameters (full copy).
    
    Args:
        source_params: Source network parameters
        target_params: Target network parameters
    """
    polyak_update(source_params, target_params, tau=1.0)


__all__ = [
    'compute_gae',
    'compute_returns',
    'RunningMeanStd',
    'explained_variance',
    'polyak_update',
    'hard_update',
]


if __name__ == '__main__':
    """
    Test code for RL utilities.
    """
    print("=" * 60)
    print("Testing RL Utilities")
    print("=" * 60)
    
    # Test compute_gae
    print("\n1. Testing compute_gae...")
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    dones = np.array([0, 0, 0, 0, 1], dtype=np.float32)
    next_value = 0.0
    
    advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95)
    print(f"   Rewards: {rewards}")
    print(f"   Values: {values}")
    print(f"   Advantages: {advantages}")
    print(f"   Advantages shape: {advantages.shape}")
    
    # Test compute_returns
    print("\n2. Testing compute_returns...")
    returns = compute_returns(rewards, dones, next_value=0.0, gamma=0.99)
    print(f"   Returns: {returns}")
    print(f"   Returns shape: {returns.shape}")
    
    # Test RunningMeanStd
    print("\n3. Testing RunningMeanStd...")
    rms = RunningMeanStd(shape=(5,))
    
    # Update with random data
    for i in range(10):
        data = np.random.randn(32, 5)
        rms.update(data)
    
    print(f"   Mean: {rms.mean}")
    print(f"   Var: {rms.var}")
    print(f"   Count: {rms.count}")
    
    # Test normalization
    test_data = np.random.randn(10, 5)
    normalized = rms.normalize(test_data)
    denormalized = rms.denormalize(normalized)
    print(f"   Normalization error: {np.max(np.abs(test_data - denormalized)):.2e}")
    
    # Test explained_variance
    print("\n4. Testing explained_variance...")
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.1, 4.1, 4.9])
    ev = explained_variance(y_pred, y_true)
    print(f"   y_true: {y_true}")
    print(f"   y_pred: {y_pred}")
    print(f"   Explained variance: {ev:.4f}")
    
    # Test polyak_update
    print("\n5. Testing polyak_update...")
    
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 5)
    
    source_net = SimpleNet()
    target_net = SimpleNet()
    
    # Initialize target differently
    with torch.no_grad():
        target_net.fc.weight.fill_(0.0)
        target_net.fc.bias.fill_(0.0)
    
    print(f"   Source weight mean: {source_net.fc.weight.mean().item():.4f}")
    print(f"   Target weight mean before update: {target_net.fc.weight.mean().item():.4f}")
    
    polyak_update(
        list(source_net.parameters()),
        list(target_net.parameters()),
        tau=0.5
    )
    print(f"   Target weight mean after polyak update (tau=0.5): {target_net.fc.weight.mean().item():.4f}")
    
    # Test hard_update
    print("\n6. Testing hard_update...")
    with torch.no_grad():
        target_net.fc.weight.fill_(0.0)
    
    hard_update(
        list(source_net.parameters()),
        list(target_net.parameters())
    )
    
    diff = (source_net.fc.weight - target_net.fc.weight).abs().max().item()
    print(f"   Weight difference after hard update: {diff:.2e}")
    assert diff < 1e-6, "Hard update should copy exactly"
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

