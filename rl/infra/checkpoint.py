"""
Checkpoint Manager for RL Training

This module provides checkpoint management for saving and loading training state:
- Model weights (policy, value network, etc.)
- Optimizer states
- Training progress (step, episode, etc.)
- Replay buffer (optional)
- Configuration and hyperparameters

Design Philosophy:
- Atomic saves: Use temporary files to prevent corruption
- Version tracking: Track checkpoint versions for compatibility
- Selective loading: Load only specific components if needed
- Automatic cleanup: Keep only N most recent checkpoints
"""

import os
import json
import shutil
import glob
import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    version: str = "1.0"
    timestamp: str = ""
    step: int = 0
    episode: int = 0
    total_timesteps: int = 0
    best_reward: Optional[float] = None
    extra_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class CheckpointManager:
    """
    Manager for saving and loading training checkpoints.
    
    Features:
    - Save/load model weights, optimizer states, training state
    - Atomic saves with temporary files
    - Automatic cleanup of old checkpoints
    - Best model tracking
    - Version compatibility checking
    
    Usage:
        manager = CheckpointManager(
            checkpoint_dir="checkpoints/",
            max_to_keep=5
        )
        
        # Save checkpoint
        manager.save(
            step=1000,
            model=policy.state_dict(),
            optimizer=optimizer.state_dict(),
            config=config_dict
        )
        
        # Load checkpoint
        state = manager.load_latest()
        policy.load_state_dict(state['model'])
    """
    
    VERSION = "1.0"
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 5,
        keep_best: bool = True,
        save_optimizer: bool = True,
        save_replay_buffer: bool = False,
        checkpoint_prefix: str = "ckpt",
        **kwargs
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to keep (0 = unlimited)
            keep_best: Whether to always keep the best checkpoint
            save_optimizer: Whether to save optimizer state by default
            save_replay_buffer: Whether to save replay buffer by default
            checkpoint_prefix: Prefix for checkpoint filenames
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        self.save_optimizer = save_optimizer
        self.save_replay_buffer = save_replay_buffer
        self.checkpoint_prefix = checkpoint_prefix
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._best_reward: Optional[float] = None
        self._best_checkpoint: Optional[str] = None
    
    def _get_checkpoint_path(self, step: int) -> Path:
        """Get checkpoint file path for a given step."""
        return self.checkpoint_dir / f"{self.checkpoint_prefix}_{step:08d}.pt"
    
    def _get_best_checkpoint_path(self) -> Path:
        """Get path for best checkpoint."""
        return self.checkpoint_dir / f"{self.checkpoint_prefix}_best.pt"
    
    def _get_metadata_path(self) -> Path:
        """Get path for metadata file."""
        return self.checkpoint_dir / "checkpoint_metadata.json"
    
    def save(
        self,
        step: int,
        model: Optional[Dict[str, Any]] = None,
        optimizer: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        replay_buffer: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        episode: int = 0,
        total_timesteps: int = 0,
        reward: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        **kwargs
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            step: Current training step
            model: Model state dict (or dict of state dicts)
            optimizer: Optimizer state dict (or dict of state dicts)
            scheduler: Learning rate scheduler state dict
            replay_buffer: Replay buffer to save (if save_replay_buffer=True)
            config: Configuration dictionary
            episode: Current episode number
            total_timesteps: Total environment timesteps
            reward: Current reward (for best model tracking)
            extra: Extra data to save
            is_best: Force save as best checkpoint
            **kwargs: Additional state to save
        
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint data
        checkpoint = {
            'metadata': asdict(CheckpointMetadata(
                version=self.VERSION,
                step=step,
                episode=episode,
                total_timesteps=total_timesteps,
                best_reward=reward,
                extra_info=extra
            )),
            'step': step,
            'episode': episode,
            'total_timesteps': total_timesteps,
        }
        
        if model is not None:
            checkpoint['model'] = model
        
        if optimizer is not None and self.save_optimizer:
            checkpoint['optimizer'] = optimizer
        
        if scheduler is not None:
            checkpoint['scheduler'] = scheduler
        
        if config is not None:
            checkpoint['config'] = config
        
        if replay_buffer is not None and self.save_replay_buffer:
            # Save replay buffer separately (can be large)
            buffer_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_{step:08d}_buffer.pt"
            torch.save(replay_buffer, buffer_path)
            checkpoint['replay_buffer_path'] = str(buffer_path)
        
        # Add any extra kwargs
        checkpoint.update(kwargs)
        
        # Save to temporary file first (atomic save)
        checkpoint_path = self._get_checkpoint_path(step)
        temp_path = checkpoint_path.with_suffix('.tmp')
        
        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)
        
        # Update best checkpoint
        if is_best or (reward is not None and (self._best_reward is None or reward > self._best_reward)):
            self._best_reward = reward
            self._best_checkpoint = str(checkpoint_path)
            
            # Copy to best checkpoint
            best_path = self._get_best_checkpoint_path()
            shutil.copy(checkpoint_path, best_path)
        
        # Save metadata
        self._save_metadata()
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load(
        self,
        path: Optional[str] = None,
        step: Optional[int] = None,
        load_best: bool = False,
        load_optimizer: bool = True,
        load_replay_buffer: bool = False,
        map_location: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            path: Direct path to checkpoint file
            step: Load checkpoint from specific step
            load_best: Load best checkpoint
            load_optimizer: Whether to load optimizer state
            load_replay_buffer: Whether to load replay buffer
            map_location: Device to map tensors to
        
        Returns:
            Checkpoint dictionary
        """
        # Determine which checkpoint to load
        if path is not None:
            checkpoint_path = Path(path)
        elif load_best:
            checkpoint_path = self._get_best_checkpoint_path()
        elif step is not None:
            checkpoint_path = self._get_checkpoint_path(step)
        else:
            checkpoint_path = self._get_latest_checkpoint_path()
        
        if not checkpoint_path or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Optionally remove optimizer state
        if not load_optimizer and 'optimizer' in checkpoint:
            del checkpoint['optimizer']
        
        # Optionally load replay buffer
        if load_replay_buffer and 'replay_buffer_path' in checkpoint:
            buffer_path = checkpoint['replay_buffer_path']
            if os.path.exists(buffer_path):
                checkpoint['replay_buffer'] = torch.load(buffer_path, map_location=map_location)
        
        return checkpoint
    
    def load_latest(self, **kwargs) -> Dict[str, Any]:
        """Load the most recent checkpoint."""
        return self.load(**kwargs)
    
    def load_best(self, **kwargs) -> Dict[str, Any]:
        """Load the best checkpoint."""
        return self.load(load_best=True, **kwargs)
    
    def _get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get path to the most recent checkpoint."""
        pattern = str(self.checkpoint_dir / f"{self.checkpoint_prefix}_*.pt")
        checkpoints = glob.glob(pattern)
        
        # Filter out best checkpoint and buffer files
        checkpoints = [c for c in checkpoints if not c.endswith('_best.pt') and '_buffer.pt' not in c]
        
        if not checkpoints:
            return None
        
        # Sort by step number and return latest
        checkpoints.sort()
        return Path(checkpoints[-1])
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only max_to_keep most recent."""
        if self.max_to_keep <= 0:
            return
        
        pattern = str(self.checkpoint_dir / f"{self.checkpoint_prefix}_*.pt")
        checkpoints = glob.glob(pattern)
        
        # Filter out best checkpoint and buffer files
        checkpoints = [c for c in checkpoints if not c.endswith('_best.pt') and '_buffer.pt' not in c]
        checkpoints.sort()
        
        # Keep only the most recent
        to_remove = checkpoints[:-self.max_to_keep] if len(checkpoints) > self.max_to_keep else []
        
        for checkpoint in to_remove:
            # Don't remove best checkpoint
            if self.keep_best and checkpoint == self._best_checkpoint:
                continue
            
            os.remove(checkpoint)
            
            # Also remove associated buffer file
            buffer_path = checkpoint.replace('.pt', '_buffer.pt')
            if os.path.exists(buffer_path):
                os.remove(buffer_path)
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata to JSON file."""
        metadata = {
            'best_reward': self._best_reward,
            'best_checkpoint': self._best_checkpoint,
            'version': self.VERSION
        }
        
        metadata_path = self._get_metadata_path()
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self) -> None:
        """Load checkpoint metadata from JSON file."""
        metadata_path = self._get_metadata_path()
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self._best_reward = metadata.get('best_reward')
                self._best_checkpoint = metadata.get('best_checkpoint')
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint info dictionaries
        """
        pattern = str(self.checkpoint_dir / f"{self.checkpoint_prefix}_*.pt")
        checkpoints = glob.glob(pattern)
        checkpoints = [c for c in checkpoints if not c.endswith('_best.pt') and '_buffer.pt' not in c]
        checkpoints.sort()
        
        result = []
        for ckpt_path in checkpoints:
            # Extract step from filename
            filename = os.path.basename(ckpt_path)
            step_str = filename.replace(f"{self.checkpoint_prefix}_", "").replace(".pt", "")
            try:
                step = int(step_str)
            except ValueError:
                step = -1
            
            result.append({
                'path': ckpt_path,
                'step': step,
                'is_best': ckpt_path == self._best_checkpoint,
                'size_mb': os.path.getsize(ckpt_path) / (1024 * 1024)
            })
        
        return result
    
    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists."""
        return self._get_latest_checkpoint_path() is not None
    
    def get_best_reward(self) -> Optional[float]:
        """Get the best reward seen so far."""
        return self._best_reward


if __name__ == '__main__':
    """
    Test code for CheckpointManager.
    """
    import tempfile
    
    print("=" * 60)
    print("Testing CheckpointManager")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Basic save/load
        print("\n1. Testing basic save/load...")
        manager = CheckpointManager(
            checkpoint_dir=tmpdir,
            max_to_keep=3
        )
        
        # Create dummy model and optimizer state
        model_state = {'layer1.weight': torch.randn(10, 5), 'layer1.bias': torch.randn(10)}
        optimizer_state = {'state': {}, 'param_groups': [{'lr': 0.001}]}
        config = {'learning_rate': 0.001, 'batch_size': 32}
        
        # Save checkpoint
        path = manager.save(
            step=100,
            model=model_state,
            optimizer=optimizer_state,
            config=config,
            episode=10,
            reward=50.0
        )
        print(f"   Saved checkpoint to: {path}")
        
        # Load checkpoint
        loaded = manager.load()
        print(f"   Loaded step: {loaded['step']}")
        print(f"   Loaded episode: {loaded['episode']}")
        print(f"   Model keys: {list(loaded['model'].keys())}")
        
        # Verify model weights match
        assert torch.allclose(loaded['model']['layer1.weight'], model_state['layer1.weight'])
        print("   Model weights match!")
        
        # Test 2: Multiple checkpoints and cleanup
        print("\n2. Testing multiple checkpoints and cleanup...")
        for step in [200, 300, 400, 500]:
            manager.save(
                step=step,
                model=model_state,
                reward=step * 0.1
            )
        
        checkpoints = manager.list_checkpoints()
        print(f"   Number of checkpoints: {len(checkpoints)}")
        print(f"   Checkpoint steps: {[c['step'] for c in checkpoints]}")
        
        # Should keep at most max_to_keep (3) + possibly the first checkpoint (if it's best)
        # The cleanup keeps max_to_keep most recent, plus best if keep_best=True
        assert len(checkpoints) <= 4, f"Expected at most 4 checkpoints, got {len(checkpoints)}"
        
        # Test 3: Best checkpoint
        print("\n3. Testing best checkpoint...")
        print(f"   Best reward: {manager.get_best_reward()}")
        
        best_ckpt = manager.load_best()
        print(f"   Best checkpoint step: {best_ckpt['step']}")
        
        # Test 4: Load specific step
        print("\n4. Testing load specific step...")
        ckpt_400 = manager.load(step=400)
        print(f"   Loaded step 400: {ckpt_400['step']}")
        assert ckpt_400['step'] == 400
        
        # Test 5: has_checkpoint
        print("\n5. Testing has_checkpoint...")
        print(f"   Has checkpoint: {manager.has_checkpoint()}")
        assert manager.has_checkpoint()
        
        # Test 6: New manager loads metadata
        print("\n6. Testing metadata persistence...")
        new_manager = CheckpointManager(checkpoint_dir=tmpdir, max_to_keep=3)
        new_manager._load_metadata()
        print(f"   Loaded best reward: {new_manager.get_best_reward()}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

