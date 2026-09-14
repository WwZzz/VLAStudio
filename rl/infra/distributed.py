"""
Distributed Training Support for RL

This module provides utilities for distributed and parallel training:
- Multi-GPU training support
- Multi-process data collection
- Weight synchronization
- Gradient aggregation

Design Philosophy:
- Transparent: Code works with or without distributed setup
- Compatible: Works with PyTorch DDP and other backends
- Flexible: Support various parallelism strategies
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, List, Any, Dict, Union, Callable
from contextlib import contextmanager


# Global distributed state
_DISTRIBUTED_CONTEXT: Optional['DistributedContext'] = None


class DistributedContext:
    """
    Context manager for distributed training.
    
    Handles initialization and cleanup of distributed training environment.
    Supports PyTorch DDP and manual multi-process setups.
    
    Usage:
        # Single process (no distribution)
        ctx = DistributedContext()
        
        # Multi-GPU DDP
        ctx = DistributedContext(
            backend='nccl',
            init_method='env://'
        )
        
        with ctx:
            # Training code
            pass
    """
    
    def __init__(
        self,
        backend: str = 'nccl',
        init_method: Optional[str] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        timeout_minutes: int = 30,
        auto_init: bool = True,
        **kwargs
    ):
        """
        Initialize distributed context.
        
        Args:
            backend: Distributed backend ('nccl', 'gloo', 'mpi')
            init_method: URL specifying how to initialize process group
            world_size: Total number of processes (auto-detect from env if None)
            rank: Global rank of this process (auto-detect from env if None)
            local_rank: Local rank on this node (auto-detect from env if None)
            timeout_minutes: Timeout for distributed operations
            auto_init: Whether to auto-initialize if env vars are set
        """
        self.backend = backend
        self.init_method = init_method
        self.timeout_minutes = timeout_minutes
        self._initialized = False
        self._kwargs = kwargs
        
        # Try to get from environment variables
        self.world_size = world_size or self._get_env_int('WORLD_SIZE', 1)
        self.rank = rank or self._get_env_int('RANK', 0)
        self.local_rank = local_rank or self._get_env_int('LOCAL_RANK', 0)
        
        # Auto-initialize if running in distributed mode
        if auto_init and self.world_size > 1:
            self.init()
    
    @staticmethod
    def _get_env_int(key: str, default: int) -> int:
        """Get integer from environment variable."""
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
        return default
    
    def init(self) -> None:
        """Initialize distributed process group."""
        if self._initialized:
            return
        
        if self.world_size <= 1:
            # Single process, no distribution needed
            self._initialized = True
            return
        
        if dist.is_initialized():
            # Already initialized elsewhere
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self._initialized = True
            return
        
        # Initialize process group
        init_method = self.init_method or 'env://'
        timeout = torch.distributed.default_pg_timeout
        if hasattr(torch.distributed, 'init_process_group'):
            from datetime import timedelta
            timeout = timedelta(minutes=self.timeout_minutes)
        
        dist.init_process_group(
            backend=self.backend,
            init_method=init_method,
            world_size=self.world_size,
            rank=self.rank,
            timeout=timeout
        )
        
        # Set CUDA device if available
        if torch.cuda.is_available() and self.local_rank < torch.cuda.device_count():
            torch.cuda.set_device(self.local_rank)
        
        self._initialized = True
        
        # Set global context
        global _DISTRIBUTED_CONTEXT
        _DISTRIBUTED_CONTEXT = self
    
    def cleanup(self) -> None:
        """Cleanup distributed process group."""
        if self._initialized and dist.is_initialized():
            dist.destroy_process_group()
        self._initialized = False
        
        global _DISTRIBUTED_CONTEXT
        if _DISTRIBUTED_CONTEXT is self:
            _DISTRIBUTED_CONTEXT = None
    
    def __enter__(self):
        self.init()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if distributed is initialized."""
        return self._initialized
    
    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.world_size > 1
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0
    
    @property
    def device(self) -> torch.device:
        """Get the device for this process."""
        if torch.cuda.is_available() and self.local_rank < torch.cuda.device_count():
            return torch.device(f'cuda:{self.local_rank}')
        return torch.device('cpu')
    
    def barrier(self) -> None:
        """Synchronization barrier across all processes."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier()
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from src to all processes."""
        if self.is_distributed and dist.is_initialized():
            dist.broadcast(tensor, src=src)
        return tensor
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM
    ) -> torch.Tensor:
        """All-reduce tensor across all processes."""
        if self.is_distributed and dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensors from all processes."""
        if not self.is_distributed or not dist.is_initialized():
            return [tensor]
        
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list
    
    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce tensor with mean across all processes."""
        if self.is_distributed and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / self.world_size
        return tensor
    
    def sync_params(self, model: torch.nn.Module, src: int = 0) -> None:
        """Synchronize model parameters from src to all processes."""
        if not self.is_distributed or not dist.is_initialized():
            return
        
        for param in model.parameters():
            dist.broadcast(param.data, src=src)
    
    def sync_gradients(self, model: torch.nn.Module) -> None:
        """Synchronize gradients across all processes (average)."""
        if not self.is_distributed or not dist.is_initialized():
            return
        
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad = param.grad / self.world_size


# Convenience functions
def get_distributed_context() -> Optional[DistributedContext]:
    """Get the global distributed context."""
    return _DISTRIBUTED_CONTEXT


def get_world_size() -> int:
    """Get world size (total number of processes)."""
    if _DISTRIBUTED_CONTEXT is not None:
        return _DISTRIBUTED_CONTEXT.world_size
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get global rank of current process."""
    if _DISTRIBUTED_CONTEXT is not None:
        return _DISTRIBUTED_CONTEXT.rank
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Get local rank of current process."""
    if _DISTRIBUTED_CONTEXT is not None:
        return _DISTRIBUTED_CONTEXT.local_rank
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return get_world_size() > 1


def barrier() -> None:
    """Synchronization barrier across all processes."""
    if _DISTRIBUTED_CONTEXT is not None:
        _DISTRIBUTED_CONTEXT.barrier()
    elif dist.is_initialized():
        dist.barrier()


@contextmanager
def main_process_first():
    """
    Context manager to ensure main process runs first.
    
    Useful for downloading files or creating directories.
    
    Usage:
        with main_process_first():
            # Only main process runs this first, others wait
            download_dataset()
    """
    if not is_main_process():
        barrier()
    
    yield
    
    if is_main_process():
        barrier()


def print_once(*args, **kwargs) -> None:
    """Print only on main process."""
    if is_main_process():
        print(*args, **kwargs)


def reduce_dict(
    data: Dict[str, float],
    average: bool = True
) -> Dict[str, float]:
    """
    Reduce a dictionary of values across all processes.
    
    Args:
        data: Dictionary with scalar values
        average: If True, compute average; if False, compute sum
    
    Returns:
        Reduced dictionary
    """
    if not is_distributed() or not dist.is_initialized():
        return data
    
    world_size = get_world_size()
    
    # Stack values into tensor
    keys = sorted(data.keys())
    values = torch.tensor([data[k] for k in keys], dtype=torch.float32)
    
    if torch.cuda.is_available():
        values = values.cuda()
    
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    
    if average:
        values = values / world_size
    
    return {k: v.item() for k, v in zip(keys, values)}


class DistributedSampler:
    """
    Simple distributed sampler for replay buffers.
    
    Ensures each process samples different data.
    """
    
    def __init__(
        self,
        dataset_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0
    ):
        """
        Initialize distributed sampler.
        
        Args:
            dataset_size: Total size of the dataset
            num_replicas: Number of processes (auto-detect if None)
            rank: Rank of current process (auto-detect if None)
            shuffle: Whether to shuffle indices
            seed: Random seed for shuffling
        """
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas or get_world_size()
        self.rank = rank or get_rank()
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
    
    def __iter__(self):
        """Generate indices for this process."""
        if self.shuffle:
            # Deterministic shuffling based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=g).tolist()
        else:
            indices = list(range(self.dataset_size))
        
        # Subsample for this replica
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)
    
    def __len__(self):
        """Return number of samples for this process."""
        return (self.dataset_size + self.num_replicas - 1) // self.num_replicas
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch


if __name__ == '__main__':
    """
    Test code for Distributed module.
    """
    print("=" * 60)
    print("Testing Distributed Module")
    print("=" * 60)
    
    # Test 1: Basic context (single process)
    print("\n1. Testing DistributedContext (single process)...")
    ctx = DistributedContext(auto_init=False)
    ctx.init()
    
    print(f"   World size: {ctx.world_size}")
    print(f"   Rank: {ctx.rank}")
    print(f"   Local rank: {ctx.local_rank}")
    print(f"   Is distributed: {ctx.is_distributed}")
    print(f"   Is main process: {ctx.is_main_process}")
    print(f"   Device: {ctx.device}")
    
    ctx.cleanup()
    
    # Test 2: Convenience functions
    print("\n2. Testing convenience functions...")
    print(f"   get_world_size(): {get_world_size()}")
    print(f"   get_rank(): {get_rank()}")
    print(f"   get_local_rank(): {get_local_rank()}")
    print(f"   is_main_process(): {is_main_process()}")
    print(f"   is_distributed(): {is_distributed()}")
    
    # Test 3: print_once
    print("\n3. Testing print_once...")
    print_once("   This should print (main process)")
    
    # Test 4: reduce_dict (single process - no-op)
    print("\n4. Testing reduce_dict (single process)...")
    data = {'loss': 0.5, 'reward': 100.0}
    reduced = reduce_dict(data)
    print(f"   Input: {data}")
    print(f"   Output: {reduced}")
    
    # Test 5: DistributedSampler
    print("\n5. Testing DistributedSampler...")
    sampler = DistributedSampler(
        dataset_size=100,
        num_replicas=4,
        rank=0,
        shuffle=True,
        seed=42
    )
    
    indices = list(sampler)
    print(f"   Dataset size: 100, Replicas: 4")
    print(f"   Samples for rank 0: {len(indices)}")
    print(f"   First 10 indices: {indices[:10]}")
    
    # Test different rank
    sampler_rank1 = DistributedSampler(
        dataset_size=100,
        num_replicas=4,
        rank=1,
        shuffle=True,
        seed=42
    )
    indices_rank1 = list(sampler_rank1)
    print(f"   First 10 indices (rank 1): {indices_rank1[:10]}")
    
    # Verify no overlap
    overlap = set(indices) & set(indices_rank1)
    print(f"   Overlap between rank 0 and 1: {len(overlap)} (should be 0)")
    assert len(overlap) == 0, "Samplers should not overlap!"
    
    # Test 6: Context manager
    print("\n6. Testing context manager...")
    with DistributedContext(auto_init=False) as ctx:
        print(f"   Inside context: rank={ctx.rank}")
    print("   Context exited successfully")
    
    # Test 7: main_process_first context
    print("\n7. Testing main_process_first...")
    with main_process_first():
        print("   Main process first block executed")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

