import time
from .base import BasicActionManager


class DelayFreeManager(BasicActionManager):
    """Remove the outdated actions from each chunk"""
    
    def __init__(self, duration: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.duration = duration

    def put(self, chunk, timestamp: float = None):
        if self._chunk_buffer is None: 
            super().put(chunk, timestamp)
            return
        else:
            delay_time = time.perf_counter() - timestamp
            delayed_start_idx = int(delay_time // self.duration)

            if delayed_start_idx < len(chunk):
                chunk = chunk[delayed_start_idx:]
                with self._lock:
                    self._chunk_buffer = chunk
                    self.current_step = 0
            else:
                self._stats["chunks_discarded"] += 1
