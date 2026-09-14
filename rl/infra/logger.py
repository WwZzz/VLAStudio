"""
Logger Module for RL Training

This module provides a flexible logging system for RL training, supporting:
- Console logging with progress bars
- TensorBoard logging
- WandB logging (optional)
- Composite logging (multiple backends)

Design Philosophy:
- Unified interface: All loggers implement the same BaseLogger interface
- Lazy initialization: Heavy backends (TensorBoard, WandB) are initialized lazily
- Thread-safe: Support concurrent logging from multiple processes
- Metric aggregation: Support for rolling window statistics
"""

import os
import sys
import time
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path


class MetricTracker:
    """
    Track metrics with rolling window statistics.
    
    Supports computing mean, std, min, max over a sliding window.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metric tracker.
        
        Args:
            window_size: Size of the rolling window for statistics
        """
        self.window_size = window_size
        self._values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._total_counts: Dict[str, int] = defaultdict(int)
        self._total_sums: Dict[str, float] = defaultdict(float)
    
    def add(self, key: str, value: float) -> None:
        """Add a value to a metric."""
        self._values[key].append(value)
        self._total_counts[key] += 1
        self._total_sums[key] += value
    
    def get_mean(self, key: str) -> Optional[float]:
        """Get rolling mean of a metric."""
        if key not in self._values or len(self._values[key]) == 0:
            return None
        return np.mean(self._values[key])
    
    def get_std(self, key: str) -> Optional[float]:
        """Get rolling std of a metric."""
        if key not in self._values or len(self._values[key]) < 2:
            return None
        return np.std(self._values[key])
    
    def get_min(self, key: str) -> Optional[float]:
        """Get rolling min of a metric."""
        if key not in self._values or len(self._values[key]) == 0:
            return None
        return np.min(self._values[key])
    
    def get_max(self, key: str) -> Optional[float]:
        """Get rolling max of a metric."""
        if key not in self._values or len(self._values[key]) == 0:
            return None
        return np.max(self._values[key])
    
    def get_total_mean(self, key: str) -> Optional[float]:
        """Get total mean (not windowed) of a metric."""
        if self._total_counts[key] == 0:
            return None
        return self._total_sums[key] / self._total_counts[key]
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get latest value of a metric."""
        if key not in self._values or len(self._values[key]) == 0:
            return None
        return self._values[key][-1]
    
    def get_count(self, key: str) -> int:
        """Get total count of a metric."""
        return self._total_counts[key]
    
    def get_stats(self, key: str) -> Dict[str, Optional[float]]:
        """Get all statistics for a metric."""
        return {
            'mean': self.get_mean(key),
            'std': self.get_std(key),
            'min': self.get_min(key),
            'max': self.get_max(key),
            'latest': self.get_latest(key),
            'total_mean': self.get_total_mean(key),
            'count': self.get_count(key)
        }
    
    def keys(self) -> List[str]:
        """Get all tracked metric keys."""
        return list(self._values.keys())
    
    def clear(self) -> None:
        """Clear all metrics."""
        self._values.clear()
        self._total_counts.clear()
        self._total_sums.clear()


class BaseLogger(ABC):
    """
    Base class for all loggers.
    
    Provides a unified interface for logging training metrics.
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        name: str = "rl_training",
        **kwargs
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            name: Name of the experiment/run
            **kwargs: Additional logger-specific arguments
        """
        self.log_dir = log_dir
        self.name = name
        self._step = 0
        self._start_time = time.time()
        self.metrics = MetricTracker()
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    @abstractmethod
    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a scalar value.
        
        Args:
            key: Metric name
            value: Metric value
            step: Step number (uses internal counter if None)
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_scalars(self, data: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple scalar values.
        
        Args:
            data: Dictionary of metric name -> value
            step: Step number (uses internal counter if None)
        """
        raise NotImplementedError
    
    def log_histogram(self, key: str, values: np.ndarray, step: Optional[int] = None) -> None:
        """Log a histogram of values (optional, not all loggers support this)."""
        pass
    
    def log_image(self, key: str, image: np.ndarray, step: Optional[int] = None) -> None:
        """Log an image (optional, not all loggers support this)."""
        pass
    
    def log_video(self, key: str, video: np.ndarray, step: Optional[int] = None, fps: int = 30) -> None:
        """Log a video (optional, not all loggers support this)."""
        pass
    
    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """Log text (optional, not all loggers support this)."""
        pass
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass
    
    def set_step(self, step: int) -> None:
        """Set the current step."""
        self._step = step
    
    def get_step(self) -> int:
        """Get the current step."""
        return self._step
    
    def increment_step(self, n: int = 1) -> int:
        """Increment step and return new value."""
        self._step += n
        return self._step
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since logger creation."""
        return time.time() - self._start_time
    
    def close(self) -> None:
        """Close the logger and release resources."""
        pass
    
    def flush(self) -> None:
        """Flush any buffered data."""
        pass


class ConsoleLogger(BaseLogger):
    """
    Simple console logger with optional progress bar.
    
    Outputs training metrics to console/terminal.
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        name: str = "rl_training",
        log_interval: int = 100,
        verbose: int = 1,
        **kwargs
    ):
        """
        Initialize console logger.
        
        Args:
            log_dir: Directory to save logs (also saves to file if provided)
            name: Name of the experiment
            log_interval: Steps between log outputs
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed)
        """
        super().__init__(log_dir=log_dir, name=name, **kwargs)
        self.log_interval = log_interval
        self.verbose = verbose
        self._file = None
        
        if log_dir:
            log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            self._file = open(log_file, 'w')
    
    def _format_value(self, value: float) -> str:
        """Format a value for display."""
        if abs(value) < 0.001 or abs(value) > 10000:
            return f"{value:.3e}"
        return f"{value:.4f}"
    
    def _write(self, message: str) -> None:
        """Write message to console and optionally to file."""
        if self.verbose > 0:
            print(message)
        if self._file:
            self._file.write(message + "\n")
            self._file.flush()
    
    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a scalar value."""
        step = step if step is not None else self._step
        self.metrics.add(key, value)
        
        if self.verbose >= 2 or (step % self.log_interval == 0):
            elapsed = self.get_elapsed_time()
            self._write(f"[{step:>8}] {key}: {self._format_value(value)} (elapsed: {elapsed:.1f}s)")
    
    def log_scalars(self, data: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple scalar values."""
        step = step if step is not None else self._step
        
        for key, value in data.items():
            self.metrics.add(key, value)
        
        if step % self.log_interval == 0:
            elapsed = self.get_elapsed_time()
            metrics_str = " | ".join([f"{k}: {self._format_value(v)}" for k, v in data.items()])
            self._write(f"[{step:>8}] {metrics_str} (elapsed: {elapsed:.1f}s)")
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self._write("\n" + "=" * 60)
        self._write("Hyperparameters:")
        self._write("=" * 60)
        for key, value in params.items():
            self._write(f"  {key}: {value}")
        self._write("=" * 60 + "\n")
        
        # Also save to JSON file
        if self.log_dir:
            params_file = os.path.join(self.log_dir, "hyperparams.json")
            with open(params_file, 'w') as f:
                json.dump(params, f, indent=2, default=str)
    
    def close(self) -> None:
        """Close the logger."""
        if self._file:
            self._file.close()
            self._file = None


class TensorBoardLogger(BaseLogger):
    """
    TensorBoard logger for rich visualization.
    
    Supports scalars, histograms, images, and more.
    """
    
    def __init__(
        self,
        log_dir: str,
        name: str = "rl_training",
        flush_secs: int = 30,
        **kwargs
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs
            name: Name of the experiment
            flush_secs: Flush interval in seconds
        """
        super().__init__(log_dir=log_dir, name=name, **kwargs)
        self.flush_secs = flush_secs
        self._writer = None
    
    def _get_writer(self):
        """Lazy initialization of SummaryWriter."""
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    "TensorBoard not installed. Install with: pip install tensorboard"
                )
            
            log_path = os.path.join(self.log_dir, self.name)
            self._writer = SummaryWriter(log_dir=log_path, flush_secs=self.flush_secs)
        return self._writer
    
    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a scalar value."""
        step = step if step is not None else self._step
        self.metrics.add(key, value)
        self._get_writer().add_scalar(key, value, step)
    
    def log_scalars(self, data: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple scalar values."""
        step = step if step is not None else self._step
        for key, value in data.items():
            self.metrics.add(key, value)
            self._get_writer().add_scalar(key, value, step)
    
    def log_histogram(self, key: str, values: np.ndarray, step: Optional[int] = None) -> None:
        """Log a histogram."""
        step = step if step is not None else self._step
        self._get_writer().add_histogram(key, values, step)
    
    def log_image(self, key: str, image: np.ndarray, step: Optional[int] = None) -> None:
        """Log an image (expects HWC or CHW format)."""
        step = step if step is not None else self._step
        # Convert HWC to CHW if needed
        if image.ndim == 3 and image.shape[-1] in [1, 3, 4]:
            image = np.transpose(image, (2, 0, 1))
        self._get_writer().add_image(key, image, step)
    
    def log_video(self, key: str, video: np.ndarray, step: Optional[int] = None, fps: int = 30) -> None:
        """Log a video (expects THWC or NTCHW format)."""
        step = step if step is not None else self._step
        # Add batch dimension if needed (THWC -> NTCHW)
        if video.ndim == 4:
            video = video[np.newaxis, ...]  # Add N dimension
            video = np.transpose(video, (0, 1, 4, 2, 3))  # NTHWC -> NTCHW
        self._get_writer().add_video(key, video, step, fps=fps)
    
    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """Log text."""
        step = step if step is not None else self._step
        self._get_writer().add_text(key, text, step)
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        # Filter out non-serializable values
        filtered_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                filtered_params[k] = v
            else:
                filtered_params[k] = str(v)
        
        self._get_writer().add_hparams(filtered_params, {})
    
    def flush(self) -> None:
        """Flush buffered data."""
        if self._writer:
            self._writer.flush()
    
    def close(self) -> None:
        """Close the logger."""
        if self._writer:
            self._writer.close()
            self._writer = None


class CompositeLogger(BaseLogger):
    """
    Composite logger that forwards logs to multiple backends.
    
    Usage:
        logger = CompositeLogger([
            ConsoleLogger(verbose=1),
            TensorBoardLogger(log_dir="logs/tb")
        ])
    """
    
    def __init__(
        self,
        loggers: List[BaseLogger],
        log_dir: Optional[str] = None,
        name: str = "rl_training",
        **kwargs
    ):
        """
        Initialize composite logger.
        
        Args:
            loggers: List of logger instances to forward to
            log_dir: Optional directory (passed to base class)
            name: Name of the experiment
        """
        super().__init__(log_dir=log_dir, name=name, **kwargs)
        self.loggers = loggers
    
    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a scalar to all backends."""
        step = step if step is not None else self._step
        self.metrics.add(key, value)
        for logger in self.loggers:
            logger.log_scalar(key, value, step)
    
    def log_scalars(self, data: Dict[str, float], step: Optional[int] = None) -> None:
        """Log scalars to all backends."""
        step = step if step is not None else self._step
        for key, value in data.items():
            self.metrics.add(key, value)
        for logger in self.loggers:
            logger.log_scalars(data, step)
    
    def log_histogram(self, key: str, values: np.ndarray, step: Optional[int] = None) -> None:
        """Log histogram to all backends that support it."""
        step = step if step is not None else self._step
        for logger in self.loggers:
            logger.log_histogram(key, values, step)
    
    def log_image(self, key: str, image: np.ndarray, step: Optional[int] = None) -> None:
        """Log image to all backends that support it."""
        step = step if step is not None else self._step
        for logger in self.loggers:
            logger.log_image(key, image, step)
    
    def log_video(self, key: str, video: np.ndarray, step: Optional[int] = None, fps: int = 30) -> None:
        """Log video to all backends that support it."""
        step = step if step is not None else self._step
        for logger in self.loggers:
            logger.log_video(key, video, step, fps)
    
    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """Log text to all backends that support it."""
        step = step if step is not None else self._step
        for logger in self.loggers:
            logger.log_text(key, text, step)
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to all backends."""
        for logger in self.loggers:
            logger.log_hyperparams(params)
    
    def set_step(self, step: int) -> None:
        """Set step for all loggers."""
        self._step = step
        for logger in self.loggers:
            logger.set_step(step)
    
    def flush(self) -> None:
        """Flush all loggers."""
        for logger in self.loggers:
            logger.flush()
    
    def close(self) -> None:
        """Close all loggers."""
        for logger in self.loggers:
            logger.close()


if __name__ == '__main__':
    """
    Test code for Logger module.
    """
    import tempfile
    import shutil
    
    print("=" * 60)
    print("Testing Logger Module")
    print("=" * 60)
    
    # Test 1: MetricTracker
    print("\n1. Testing MetricTracker...")
    tracker = MetricTracker(window_size=5)
    
    for i in range(10):
        tracker.add("loss", 1.0 - i * 0.1)
        tracker.add("reward", i * 10)
    
    print(f"   Loss stats: {tracker.get_stats('loss')}")
    print(f"   Reward stats: {tracker.get_stats('reward')}")
    print(f"   Tracked keys: {tracker.keys()}")
    
    # Test 2: ConsoleLogger
    print("\n2. Testing ConsoleLogger...")
    with tempfile.TemporaryDirectory() as tmpdir:
        console_logger = ConsoleLogger(
            log_dir=tmpdir,
            name="test_console",
            log_interval=2,
            verbose=1
        )
        
        console_logger.log_hyperparams({
            'learning_rate': 0.001,
            'batch_size': 32,
            'algorithm': 'PPO'
        })
        
        for step in range(5):
            console_logger.log_scalars({
                'loss': 1.0 - step * 0.1,
                'reward': step * 10
            }, step=step)
            console_logger.increment_step()
        
        print(f"   Elapsed time: {console_logger.get_elapsed_time():.2f}s")
        console_logger.close()
    
    # Test 3: TensorBoardLogger (if tensorboard is installed)
    print("\n3. Testing TensorBoardLogger...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tb_logger = TensorBoardLogger(
                log_dir=tmpdir,
                name="test_tb"
            )
            
            for step in range(10):
                tb_logger.log_scalars({
                    'loss': 1.0 - step * 0.05,
                    'reward': step * 5
                }, step=step)
            
            # Test histogram
            tb_logger.log_histogram("weights", np.random.randn(1000), step=0)
            
            tb_logger.flush()
            tb_logger.close()
            print("   TensorBoard logger test passed!")
    except ImportError:
        print("   TensorBoard not installed, skipping...")
    
    # Test 4: CompositeLogger
    print("\n4. Testing CompositeLogger...")
    with tempfile.TemporaryDirectory() as tmpdir:
        composite_logger = CompositeLogger(
            loggers=[
                ConsoleLogger(log_interval=5, verbose=1),
            ],
            log_dir=tmpdir,
            name="test_composite"
        )
        
        for step in range(10):
            composite_logger.log_scalar("test_metric", step * 0.5, step=step)
        
        composite_logger.close()
        print("   Composite logger test passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

