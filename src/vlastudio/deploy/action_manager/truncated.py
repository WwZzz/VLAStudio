from .base import BasicActionManager


class TruncatedManager(BasicActionManager):
    """
    Truncate action chunks by dropping the first and last portions.
    
    This manager discards:
    - First len(chunk) * start_ratio actions (warm-up phase) - NOT applied to the first chunk
    - Last len(chunk) * end_ratio actions (uncertain predictions)
    
    Special behavior: The first chunk keeps all beginning actions (start_ratio is ignored)
    to ensure smooth startup. Subsequent chunks apply both start_ratio and end_ratio.
    
    Then executes the remaining middle portion like OlderFirstManager,
    refusing new chunks until the current chunk is sufficiently executed.
    """
    
    def __init__(self, start_ratio: float = 0.0, end_ratio: float = 0.0, older_coef: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.older_coef = older_coef

    def put(self, chunk, timestamp: float = None):
        # Truncate the chunk first
        if chunk is not None and len(chunk) > 0:
            total_len = len(chunk)
            
            # For the first chunk, don't drop the beginning (start_idx = 0)
            # For subsequent chunks, apply start_ratio normally
            is_first_chunk = (self._chunk_buffer is None)
            start_idx = 0 if is_first_chunk else int(total_len * self.start_ratio)
            end_idx = total_len - int(total_len * self.end_ratio)
            
            # Ensure we have at least one action
            if end_idx <= start_idx:
                mid_idx = total_len // 2
                chunk = [chunk[mid_idx]]
            else:
                chunk = chunk[start_idx:end_idx]
        
        # Now apply OlderFirst logic
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
            
            with self._lock:
                self._chunk_buffer = chunk
                self.current_step = 0
