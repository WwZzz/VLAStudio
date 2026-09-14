from .base import BasicActionManager


class TemporalAggManager(BasicActionManager):
    """Exponentially average the last and the new chunks for better smoothness"""
    
    def __init__(self, coef: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.coef = coef

    def put(self, chunk, timestamp: float = None):
        if self._chunk_buffer is None: 
            super().put(chunk, timestamp)
            return
        else:
            with self._lock:
                prev_step = self.current_step
                prev_len = len(self._chunk_buffer)
                remain_len = prev_len - prev_step
                if remain_len > 0:
                    for idx in range(remain_len):
                        chunk[idx]['action'] = (1. - self.coef) * chunk[idx]['action'] + self.coef * self._chunk_buffer[idx + prev_step]['action']
                self._chunk_buffer = chunk
                self.current_step = 0
