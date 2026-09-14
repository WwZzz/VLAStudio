from .temporal_agg import TemporalAggManager


class TemporalOlderManager(TemporalAggManager):
    """Refuse newly coming chunks until the last chunk ends x%"""
    
    def __init__(self, coef: float = 0.1, older_coef: float = 0.75, **kwargs):
        super().__init__(coef=coef, **kwargs)
        self.older_coef = older_coef

    def put(self, chunk, timestamp: float = None):
        if self._chunk_buffer is None:
            with self._lock:
                self._chunk_buffer = chunk
                self.current_step = 0
        else:
            with self._lock:
                threshold = int(len(self._chunk_buffer) * self.older_coef)
                if self.current_step < threshold:
                    self._stats["chunks_discarded"] += 1
                    return
            super().put(chunk, timestamp)
