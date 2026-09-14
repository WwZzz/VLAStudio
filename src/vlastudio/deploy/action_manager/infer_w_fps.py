import time
from .base import BasicActionManager


class InferWithFPSManager(BasicActionManager):
    """
    Action manager that triggers inference at a minimum FPS rate.
    
    Unlike BasicActionManager which only infers when buffer is empty,
    this manager ensures inference happens at least `fps` times per second,
    enabling asynchronous/pipelined inference where new chunks are requested
    before the current buffer is exhausted.
    
    This is useful for:
    - Hiding inference latency by overlapping inference with action execution
    - Maintaining a consistent inference rate regardless of chunk consumption speed
    - Simulating real-time inference constraints
    
    Example:
        fps=10 → inference triggered at least every 100ms
        fps=50 → inference triggered at least every 20ms
    """
    
    def __init__(self, fps: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.fps = fps
        self._min_interval = 1.0 / self.fps if self.fps > 0 else float('inf')
        self._last_infer_time: float = 0.0

    def should_infer(self) -> bool:
        """
        Return True if enough time has passed since last inference.
        
        This ensures inference is triggered at least `fps` times per second,
        even if the buffer still has actions to dispatch.
        Uses wall-clock time internally — works for both sim and real.
        """
        now = time.perf_counter()
        elapsed = now - self._last_infer_time
        return elapsed >= self._min_interval

    def put(self, chunk, timestamp: float = None):
        """Record inference time when new chunk arrives."""
        super().put(chunk, timestamp)
        self._last_infer_time = time.perf_counter()

    def reset(self):
        super().reset()
        self._last_infer_time = 0.0
