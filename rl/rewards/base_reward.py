"""
Base Reward Function Class

This module defines the base class for all reward functions in the RL framework.

Design Philosophy:
- Modular: Reward functions are independent modules, easy to replace and extend
- Composable: Support combining multiple reward functions
- Language-conditioned: Support VLA's language-conditioned rewards
"""

import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

# Type hints for Meta classes
from benchmark.base import MetaObs, MetaAction


class BaseReward(ABC):
    """
    Base class for reward functions.
    
    This class defines the interface for all reward functions in the RL framework.
    Reward functions compute custom rewards based on state, action, and other information.
    
    Note: The reward function is used in the Trainer during training time.
    The replay buffer stores raw environment rewards, and the reward function
    is applied during the update step for computing the training reward.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the reward function.
        
        Args:
            **kwargs: Reward function specific parameters
        """
        self._kwargs = kwargs
    
    @abstractmethod
    def compute(
        self,
        state: MetaObs,
        action: MetaAction,
        next_state: MetaObs,
        env_reward: float,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute the reward.
        
        Args:
            state: Current state (MetaObs)
            action: Action (MetaAction)
            next_state: Next state (MetaObs)
            env_reward: Environment's raw reward
            info: Additional information dictionary
        
        Returns:
            Computed reward value
        """
        raise NotImplementedError
    
    def reset(self, **kwargs) -> None:
        """
        Reset reward function state (if needed).
        
        Some reward functions may have internal state (e.g., running statistics,
        episode counters) that need to be reset at the beginning of an episode.
        
        Args:
            **kwargs: Reset parameters
        """
        pass
    
    def __call__(
        self,
        state: MetaObs,
        action: MetaAction,
        next_state: MetaObs,
        env_reward: float,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Callable interface for computing reward.
        
        This allows the reward function to be used as a callable:
            reward = reward_fn(state, action, next_state, env_reward, info)
        
        Args:
            state: Current state (MetaObs)
            action: Action (MetaAction)
            next_state: Next state (MetaObs)
            env_reward: Environment's raw reward
            info: Additional information dictionary
        
        Returns:
            Computed reward value
        """
        return self.compute(state, action, next_state, env_reward, info)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._kwargs})"


class IdentityReward(BaseReward):
    """
    Identity reward function - returns the environment reward unchanged.
    
    This is the default reward function when no custom reward is specified.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def compute(
        self,
        state: MetaObs,
        action: MetaAction,
        next_state: MetaObs,
        env_reward: float,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Return the environment reward unchanged."""
        return env_reward


class ScaledReward(BaseReward):
    """
    Scaled reward function - scales the environment reward by a factor.
    """
    
    def __init__(self, scale: float = 1.0, offset: float = 0.0, **kwargs):
        """
        Initialize scaled reward.
        
        Args:
            scale: Scaling factor for the reward
            offset: Offset to add to the scaled reward
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.scale = scale
        self.offset = offset
    
    def compute(
        self,
        state: MetaObs,
        action: MetaAction,
        next_state: MetaObs,
        env_reward: float,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Return scaled and offset reward."""
        return env_reward * self.scale + self.offset


class CompositeReward(BaseReward):
    """
    Composite reward function - combines multiple reward functions.
    
    This allows combining different reward signals with weights.
    """
    
    def __init__(
        self,
        reward_fns: list,
        weights: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize composite reward.
        
        Args:
            reward_fns: List of reward function instances
            weights: Optional list of weights for each reward function.
                    If None, uses uniform weights.
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.reward_fns = reward_fns
        if weights is None:
            weights = [1.0] * len(reward_fns)
        assert len(weights) == len(reward_fns), "Number of weights must match number of reward functions"
        self.weights = weights
    
    def compute(
        self,
        state: MetaObs,
        action: MetaAction,
        next_state: MetaObs,
        env_reward: float,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute weighted sum of all reward functions."""
        total_reward = 0.0
        for reward_fn, weight in zip(self.reward_fns, self.weights):
            total_reward += weight * reward_fn.compute(
                state, action, next_state, env_reward, info
            )
        return total_reward
    
    def reset(self, **kwargs) -> None:
        """Reset all component reward functions."""
        for reward_fn in self.reward_fns:
            reward_fn.reset(**kwargs)


if __name__ == '__main__':
    """
    Test code for BaseReward class and its implementations.
    """
    import sys
    sys.path.insert(0, '/home/zhang/robot/126/ILStudio')
    
    from benchmark.base import MetaObs, MetaAction
    
    # Test IdentityReward
    print("=" * 60)
    print("Testing BaseReward and implementations")
    print("=" * 60)
    
    # Create sample states and actions
    state = MetaObs(
        state=np.random.randn(10).astype(np.float32),
        state_ee=np.random.randn(7).astype(np.float32),
        raw_lang="pick up the red block"
    )
    action = MetaAction(
        action=np.random.randn(7).astype(np.float32),
        ctrl_space='ee',
        ctrl_type='delta'
    )
    next_state = MetaObs(
        state=np.random.randn(10).astype(np.float32),
        state_ee=np.random.randn(7).astype(np.float32),
        raw_lang="pick up the red block"
    )
    
    # Test 1: IdentityReward
    print("\n1. Testing IdentityReward...")
    identity_reward = IdentityReward()
    env_reward = 1.5
    reward = identity_reward.compute(state, action, next_state, env_reward, {})
    print(f"   IdentityReward: {identity_reward}")
    print(f"   Env reward: {env_reward}, Computed reward: {reward}")
    assert reward == env_reward, "IdentityReward should return env_reward unchanged"
    
    # Test callable interface
    reward_callable = identity_reward(state, action, next_state, env_reward, {})
    print(f"   Callable interface result: {reward_callable}")
    assert reward_callable == env_reward, "Callable interface should work the same"
    
    # Test 2: ScaledReward
    print("\n2. Testing ScaledReward...")
    scaled_reward = ScaledReward(scale=2.0, offset=0.5)
    env_reward = 1.0
    reward = scaled_reward.compute(state, action, next_state, env_reward, {})
    expected = 1.0 * 2.0 + 0.5  # 2.5
    print(f"   ScaledReward: {scaled_reward}")
    print(f"   Env reward: {env_reward}, Scale: 2.0, Offset: 0.5")
    print(f"   Computed reward: {reward}, Expected: {expected}")
    assert abs(reward - expected) < 1e-6, "ScaledReward should scale and offset correctly"
    
    # Test 3: CompositeReward
    print("\n3. Testing CompositeReward...")
    reward_fn1 = IdentityReward()
    reward_fn2 = ScaledReward(scale=0.5, offset=0.0)
    composite_reward = CompositeReward(
        reward_fns=[reward_fn1, reward_fn2],
        weights=[0.6, 0.4]
    )
    env_reward = 2.0
    reward = composite_reward.compute(state, action, next_state, env_reward, {})
    # Expected: 0.6 * 2.0 + 0.4 * (2.0 * 0.5) = 1.2 + 0.4 = 1.6
    expected = 0.6 * 2.0 + 0.4 * (2.0 * 0.5)
    print(f"   CompositeReward: {composite_reward}")
    print(f"   Env reward: {env_reward}")
    print(f"   Component 1 (IdentityReward, weight=0.6): {reward_fn1.compute(state, action, next_state, env_reward, {})}")
    print(f"   Component 2 (ScaledReward*0.5, weight=0.4): {reward_fn2.compute(state, action, next_state, env_reward, {})}")
    print(f"   Computed reward: {reward}, Expected: {expected}")
    assert abs(reward - expected) < 1e-6, "CompositeReward should compute weighted sum correctly"
    
    # Test 4: Custom reward function (abstract class implementation)
    print("\n4. Testing custom reward function...")
    
    class SuccessBonus(BaseReward):
        """Give bonus reward when task is successful."""
        
        def __init__(self, bonus: float = 10.0, **kwargs):
            super().__init__(**kwargs)
            self.bonus = bonus
        
        def compute(self, state, action, next_state, env_reward, info):
            if info and info.get('success', False):
                return env_reward + self.bonus
            return env_reward
    
    success_bonus = SuccessBonus(bonus=5.0)
    
    # Without success
    info_no_success = {'success': False}
    reward_no_success = success_bonus.compute(state, action, next_state, 1.0, info_no_success)
    print(f"   SuccessBonus (no success): reward = {reward_no_success}")
    assert reward_no_success == 1.0
    
    # With success
    info_success = {'success': True}
    reward_success = success_bonus.compute(state, action, next_state, 1.0, info_success)
    print(f"   SuccessBonus (success): reward = {reward_success}")
    assert reward_success == 6.0  # 1.0 + 5.0
    
    # Test 5: Reset functionality
    print("\n5. Testing reset functionality...")
    
    class StatefulReward(BaseReward):
        """Reward function with internal state."""
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.step_count = 0
        
        def compute(self, state, action, next_state, env_reward, info):
            self.step_count += 1
            return env_reward + self.step_count * 0.01
        
        def reset(self, **kwargs):
            self.step_count = 0
    
    stateful_reward = StatefulReward()
    
    # Compute a few rewards
    for i in range(5):
        r = stateful_reward.compute(state, action, next_state, 1.0, {})
    print(f"   After 5 steps, step_count = {stateful_reward.step_count}")
    
    # Reset
    stateful_reward.reset()
    print(f"   After reset, step_count = {stateful_reward.step_count}")
    assert stateful_reward.step_count == 0, "Reset should clear step_count"
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

