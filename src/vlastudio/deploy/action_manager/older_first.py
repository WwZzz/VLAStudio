from .base import BasicActionManager


class OlderFirstManager(BasicActionManager):
    """Refuse newly coming chunks unless the last chunk ends x%"""
    
    def __init__(self, coef: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.coef = coef

    def put(self, chunk, timestamp: float = None):
        if self._chunk_buffer is None:
            super().put(chunk, timestamp)
        else:
            with self._lock:
                current_len = len(self._chunk_buffer)
                threshold = int(len(self._chunk_buffer) * self.coef)
                if self.current_step < threshold:
                    self._stats["chunks_discarded"] += 1
                    self._log_debug(
                        "discard incoming chunk_len={} current_step={}/{} threshold={}",
                        len(chunk) if chunk is not None else 0,
                        self.current_step,
                        current_len,
                        threshold,
                    )
                    return
            self._log_debug(
                "accept incoming chunk_len={} after finishing step {}/{}",
                len(chunk) if chunk is not None else 0,
                self.current_step,
                current_len,
            )
            super().put(chunk, timestamp)
