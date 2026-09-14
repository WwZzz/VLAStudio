"""
Action utility helpers for RL algorithms.

These helpers keep action post-processing outside MetaEnv, so algorithms can
apply different refinement strategies when needed.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from loguru import logger

def _get_action_space(env) -> Optional[object]:
    """Best-effort extraction of an action_space from env or wrappers."""
    if env is None:
        return None
    if hasattr(env, "action_space"):
        return env.action_space
    if hasattr(env, "env") and hasattr(env.env, "action_space"):
        return env.env.action_space
    if hasattr(env, "envs") and env.envs:
        first_env = env.envs[0]
        if hasattr(first_env, "action_space"):
            return first_env.action_space
        if hasattr(first_env, "env") and hasattr(first_env.env, "action_space"):
            return first_env.env.action_space
    return None


def _get_action_bounds(env) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    action_space = _get_action_space(env)
    if action_space is None:
        return None, None
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return None, None
    return np.asarray(low), np.asarray(high)


def clip_action_to_space(env, action):
    """Clip action to env action space bounds if available."""
    low, high = _get_action_bounds(env)
    if low is None or high is None:
        return action
    if torch.is_tensor(action):
        low_t = torch.as_tensor(low, device=action.device, dtype=action.dtype)
        high_t = torch.as_tensor(high, device=action.device, dtype=action.dtype)
        return torch.clamp(action, low_t, high_t)
    return np.clip(action, low, high)

def tanh_action_to_space(env, action):
    """
    Apply tanh to action and scale to env action space bounds.
    
    Maps tanh output [-1, 1] to [action_space.low, action_space.high].
    If no action space bounds available, just applies tanh.
    """
    # Apply tanh to bound to [-1, 1]
    if torch.is_tensor(action):
        tanh_action = torch.tanh(action)
    else:
        tanh_action = np.tanh(action)
    
    # Scale to action space bounds
    low, high = _get_action_bounds(env)
    if low is None or high is None:
        return tanh_action
    
    # Map [-1, 1] -> [low, high]: scaled = low + (tanh + 1) * (high - low) / 2
    if torch.is_tensor(tanh_action):
        low_t = torch.as_tensor(low, device=tanh_action.device, dtype=tanh_action.dtype)
        high_t = torch.as_tensor(high, device=tanh_action.device, dtype=tanh_action.dtype)
        return low_t + (tanh_action + 1.0) * (high_t - low_t) / 2.0
    else:
        return low + (tanh_action + 1.0) * (high - low) / 2.0

def ensure_action(
    env,
    action,
    refine_fn: Optional[Callable[[object, object], object]] = None,
):
    """
    Ensure actions are valid for the environment.

    - Always applies the refine function (e.g., tanh_action_to_space) to map 
      network outputs to valid action space.
    - Clips to env action space bounds if available.

    Args:
        env: Environment (or wrapper) providing action_space.
        action: Tensor or numpy array action.
        refine_fn: Optional callable (env, action) -> action for custom refinement.
    """
    try:
        reasonable = env.envs[0].ensure_action_reasonable(action)
    except ValueError:
        logger.warning(f"Environment {env.__class__.__name__} does not support ensure_action_reasonable, using tanh_action_to_space")
    # Always apply refine_fn to map network outputs to action space
    if refine_fn is not None:
        action = refine_fn(env, action)
    
    # Final clip to ensure bounds (safety measure)
    action = clip_action_to_space(env, action)
    
    return action