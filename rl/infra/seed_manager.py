"""
Seed Manager for Reproducibility

This module provides unified random seed management to ensure reproducibility
across different libraries (NumPy, PyTorch, Python random, etc.).

Features:
- Unified seed setting for all random number generators
- Support for deterministic operations in PyTorch
- Worker seed management for parallel data loading
- Global seed state tracking
"""

import os
import random
import numpy as np
import torch
from typing import Optional, Callable
from dataclasses import dataclass


# Global seed state
_GLOBAL_SEED: Optional[int] = None


@dataclass
class SeedState:
    """Container for random state from different libraries."""
    python_state: tuple
    numpy_state: dict
    torch_state: torch.Tensor
    torch_cuda_state: Optional[list] = None


class SeedManager:
    """
    Unified random seed manager for reproducibility.
    
    This class manages random seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's CPU and GPU random number generators
    - CUDA deterministic operations
    
    Usage:
        # Simple usage
        SeedManager.set_seed(42)
        
        # With deterministic mode (slower but fully reproducible)
        SeedManager.set_seed(42, deterministic=True)
        
        # Save and restore state
        state = SeedManager.get_state()
        # ... do something ...
        SeedManager.set_state(state)
    """
    
    @staticmethod
    def set_seed(
        seed: int,
        deterministic: bool = False,
        benchmark: bool = True,
        warn_only: bool = False
    ) -> None:
        """
        Set random seed for all libraries.
        
        Args:
            seed: Random seed value
            deterministic: If True, enable deterministic algorithms in PyTorch
                          (may reduce performance but ensures reproducibility)
            benchmark: If True, enable cuDNN benchmark mode for faster training
                      (disable when deterministic=True for full reproducibility)
            warn_only: If True, only warn instead of error when deterministic
                      operations are not available
        """
        global _GLOBAL_SEED
        _GLOBAL_SEED = seed
        
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        
        # PyTorch CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        # Deterministic mode
        if deterministic:
            # Disable benchmark for determinism
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            # Use deterministic algorithms
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
            elif hasattr(torch, 'set_deterministic'):
                torch.set_deterministic(True)
        else:
            # Enable benchmark for performance
            torch.backends.cudnn.benchmark = benchmark
            torch.backends.cudnn.deterministic = False
        
        # Environment variable for hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    @staticmethod
    def get_seed() -> Optional[int]:
        """Get the current global seed."""
        return _GLOBAL_SEED
    
    @staticmethod
    def get_state() -> SeedState:
        """
        Get current random state from all libraries.
        
        Returns:
            SeedState containing states from all RNGs
        """
        cuda_state = None
        if torch.cuda.is_available():
            cuda_state = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
        
        return SeedState(
            python_state=random.getstate(),
            numpy_state=np.random.get_state(),
            torch_state=torch.get_rng_state(),
            torch_cuda_state=cuda_state
        )
    
    @staticmethod
    def set_state(state: SeedState) -> None:
        """
        Restore random state for all libraries.
        
        Args:
            state: SeedState to restore
        """
        random.setstate(state.python_state)
        np.random.set_state(state.numpy_state)
        torch.set_rng_state(state.torch_state)
        
        if state.torch_cuda_state is not None and torch.cuda.is_available():
            for i, cuda_state in enumerate(state.torch_cuda_state):
                if i < torch.cuda.device_count():
                    torch.cuda.set_rng_state(cuda_state, i)
    
    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """
        Worker initialization function for PyTorch DataLoader.
        
        Use this as the `worker_init_fn` argument in DataLoader to ensure
        each worker has a different but reproducible random seed.
        
        Args:
            worker_id: Worker ID (0, 1, 2, ...)
        
        Usage:
            DataLoader(..., worker_init_fn=SeedManager.worker_init_fn)
        """
        global _GLOBAL_SEED
        if _GLOBAL_SEED is not None:
            worker_seed = _GLOBAL_SEED + worker_id
        else:
            worker_seed = torch.initial_seed() % 2**32
        
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    
    @staticmethod
    def fork_rng(devices: Optional[list] = None, enabled: bool = True) -> 'RNGForkContext':
        """
        Fork the RNG state for a context (useful for dropout in eval mode).
        
        Args:
            devices: List of CUDA devices to fork RNG state for
            enabled: Whether to actually fork (useful for conditional forking)
        
        Returns:
            Context manager that restores RNG state on exit
        
        Usage:
            with SeedManager.fork_rng():
                # Operations here won't affect global RNG state
                pass
        """
        return RNGForkContext(devices=devices, enabled=enabled)


class RNGForkContext:
    """Context manager for forking RNG state."""
    
    def __init__(self, devices: Optional[list] = None, enabled: bool = True):
        self.devices = devices
        self.enabled = enabled
        self._state: Optional[SeedState] = None
    
    def __enter__(self):
        if self.enabled:
            self._state = SeedManager.get_state()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self._state is not None:
            SeedManager.set_state(self._state)
        return False


# Convenience functions
def set_global_seed(
    seed: int,
    deterministic: bool = False,
    benchmark: bool = True
) -> None:
    """
    Set global random seed for reproducibility.
    
    Convenience function that calls SeedManager.set_seed().
    
    Args:
        seed: Random seed value
        deterministic: Enable deterministic mode (slower but fully reproducible)
        benchmark: Enable cuDNN benchmark (disable when deterministic=True)
    """
    SeedManager.set_seed(seed, deterministic=deterministic, benchmark=benchmark)


def get_global_seed() -> Optional[int]:
    """Get the current global seed."""
    return SeedManager.get_seed()


if __name__ == '__main__':
    """
    Test code for SeedManager.
    """
    print("=" * 60)
    print("Testing SeedManager")
    print("=" * 60)
    
    # Test 1: Basic seed setting
    print("\n1. Testing basic seed setting...")
    SeedManager.set_seed(42)
    
    # Generate some random numbers
    py_random1 = [random.random() for _ in range(5)]
    np_random1 = np.random.rand(5).tolist()
    torch_random1 = torch.rand(5).tolist()
    
    # Reset and generate again
    SeedManager.set_seed(42)
    py_random2 = [random.random() for _ in range(5)]
    np_random2 = np.random.rand(5).tolist()
    torch_random2 = torch.rand(5).tolist()
    
    print(f"   Python random match: {py_random1 == py_random2}")
    print(f"   NumPy random match: {np_random1 == np_random2}")
    print(f"   PyTorch random match: {torch_random1 == torch_random2}")
    
    assert py_random1 == py_random2, "Python random not reproducible"
    assert np_random1 == np_random2, "NumPy random not reproducible"
    assert torch_random1 == torch_random2, "PyTorch random not reproducible"
    
    # Test 2: Get/Set state
    print("\n2. Testing state save/restore...")
    SeedManager.set_seed(123)
    
    # Generate some numbers
    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(1)
    
    # Save state
    state = SeedManager.get_state()
    
    # Generate more numbers
    val1 = random.random()
    val2 = np.random.rand()
    val3 = torch.rand(1).item()
    
    # Restore state
    SeedManager.set_state(state)
    
    # Generate again - should match
    val1_restored = random.random()
    val2_restored = np.random.rand()
    val3_restored = torch.rand(1).item()
    
    print(f"   Python state restored: {val1 == val1_restored}")
    print(f"   NumPy state restored: {val2 == val2_restored}")
    print(f"   PyTorch state restored: {val3 == val3_restored}")
    
    assert val1 == val1_restored
    assert val2 == val2_restored
    assert val3 == val3_restored
    
    # Test 3: Fork RNG context
    print("\n3. Testing RNG fork context...")
    SeedManager.set_seed(456)
    
    before = random.random()
    
    # Save current position
    SeedManager.set_seed(456)
    _ = random.random()  # Advance to same position
    
    with SeedManager.fork_rng():
        # This should not affect the outer state
        for _ in range(10):
            random.random()
    
    after = random.random()
    
    # Reset and check
    SeedManager.set_seed(456)
    _ = random.random()
    expected_after = random.random()
    
    print(f"   Fork context preserved state: {after == expected_after}")
    
    # Test 4: Global seed tracking
    print("\n4. Testing global seed tracking...")
    SeedManager.set_seed(789)
    print(f"   Global seed: {SeedManager.get_seed()}")
    assert SeedManager.get_seed() == 789
    
    # Test convenience functions
    set_global_seed(999)
    print(f"   After set_global_seed: {get_global_seed()}")
    assert get_global_seed() == 999
    
    # Test 5: Deterministic mode
    print("\n5. Testing deterministic mode...")
    SeedManager.set_seed(42, deterministic=True)
    print("   Deterministic mode enabled (slower but fully reproducible)")
    
    # Test 6: Worker init function
    print("\n6. Testing worker init function...")
    SeedManager.set_seed(42)
    
    # Simulate worker initialization
    for worker_id in range(3):
        SeedManager.worker_init_fn(worker_id)
        worker_val = random.random()
        print(f"   Worker {worker_id} random value: {worker_val:.6f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

