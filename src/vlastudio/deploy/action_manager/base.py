import numpy as np
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from vlastudio.deploy.inference import InferenceContext


def _env_flag(name: str) -> bool:
    value = str(os.getenv(name, "")).strip().lower()
    return value not in ("", "0", "false", "no", "off")


class AbstractActionManager(ABC):
    """Abstract base class for action managers.
    
    Action managers manage the buffering and aggregation of action chunks
    from policy inference. They support:
    - select_action(): Main interface - get single-step action,
      triggering inference when buffer is empty or should_infer() returns True.
    - put(): Store a new action chunk into the buffer
    - get(): Retrieve a single-step action from the buffer
    - should_infer(): Determine if new inference is needed (beyond buffer-empty condition)
    - reset(): Reset the action manager state
    
    The step counter `t` is maintained internally and auto-incremented on each
    select_action() call. Subclasses can access it via `self.t`.
    """
    
    @abstractmethod
    def select_action(self):
        """
        Get single-step action, triggering inference if needed.
        
        No parameters — the step counter is maintained internally (self.t).
        Obs is written to SHM externally (sim) or read by the worker (real).
            
        Returns:
            Single-step action
        """
        pass
    
    @abstractmethod
    def put(self, chunk: np.ndarray, timestamp: float = None):
        """Put action chunk into local cache"""
        pass

    @abstractmethod
    def get(self, timestamp: float = None):
        """Get one-step action from local cache"""
        pass

    @abstractmethod
    def should_infer(self) -> bool:
        """
        User-overridable condition for triggering inference.
        
        This is checked IN ADDITION to the buffer-empty condition.
        When this returns True, a new inference will be triggered even if
        the buffer still has actions to dispatch.
        
        Subclasses can use self.t (step count) or time.perf_counter()
        for their decisions.
        
        Default returns False, meaning inference only happens when buffer is empty,
        which naturally forms synchronous execution.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the action manager state."""
        pass


class BasicActionManager(AbstractActionManager):
    """
    Basic action manager: drops the previous chunk whenever a new one arrives.
    
    Inference is always delegated to an inference_worker subprocess via
    InferenceContext (SHM). The flow is:
    
    1. Poll SHM for new action chunks from inference subprocess
    2. If buffer empty or should_infer(): send trigger to subprocess via ctrl_shm
    3. If buffer empty: block-wait for inference result
    4. Return single-step action via get()
    
    The action_manager never touches obs — obs is handled externally:
    - Sim: main process writes obs to obs_shm
    - Real: inference worker reads device SHMs directly
    """
    
    def __init__(self, debug: bool = False, **kwargs):
        self._lock = threading.Lock()
        self.t = 0              # Step counter, auto-incremented
        self.current_step = 0
        self._buffer = None
        self._chunk_buffer = None
        self._inference_ctx: Optional["InferenceContext"] = None
        self._debug = bool(debug or _env_flag("ILSTUDIO_ACTION_DEBUG"))
        
        # Lightweight statistics — always tracked, printed on reset() if debug=True
        self._stats = {
            "chunks_received": 0,
            "chunks_discarded": 0,
            "triggers_sent": 0,
            "waits": 0,
            "cumulative_wait_ms": 0.0,
            "total_remaining_on_replace": 0,   # sum of un-consumed steps when overwritten
            "replace_count": 0,                 # how many times a non-empty buffer was replaced
        }
        
        if self._debug:
            logger.trace(f"[ActionManager] {self.__class__.__name__} initialized (debug=True)")

    def _log_debug(self, message: str, *args):
        """Verbose tracing; off by default (loguru INFO). Enable with ILSTUDIO_ACTION_DEBUG=1 and LOGURU_LEVEL=TRACE."""
        if self._debug:
            logger.trace(f"[ActionManager:{self.__class__.__name__}] " + message, *args)

    def _buffer_status(self):
        with self._lock:
            chunk_len = len(self._chunk_buffer) if self._chunk_buffer is not None else 0
            return self.current_step, chunk_len

    def _summarize_action(self, action) -> str:
        if action is None:
            return "action=None"
        arr = action
        if isinstance(action, np.ndarray) and action.dtype == object and len(action) > 0:
            first = action[0]
            if isinstance(first, dict):
                arr = first.get("action", None)
        if arr is None:
            return "action=None"
        arr = np.asarray(arr)
        if arr.size == 0:
            return f"shape={arr.shape}, empty"
        flat = arr.reshape(-1)
        preview = ", ".join(f"{float(x):.4f}" for x in flat[:4])
        return f"shape={arr.shape}, preview=[{preview}]"

    def set_inference_context(self, inference_ctx: "InferenceContext"):
        """Set inference context for SHM-based inference."""
        self._inference_ctx = inference_ctx
        self._log_debug("bound inference context")

    def select_action(self):
        """
        Get single-step action, triggering inference if needed.
        
        Flow:
        1. Poll for new action chunks from inference subprocess (non-blocking)
        2. If buffer empty or should_infer(): send trigger to subprocess via ctrl_shm
        3. If buffer empty: block-wait for inference result
        4. Return single-step action via get()
        5. Auto-increment self.t
        """
        if self._inference_ctx is None:
            raise RuntimeError("InferenceContext not set. Call set_inference_context() first.")
        
        # Step 1: Poll for new action chunks (non-blocking)
        mact_list = self._inference_ctx.poll_action_chunks()
        if mact_list:
            self._stats["chunks_received"] += 1
            step_idx, chunk_len = self._buffer_status()
            self._log_debug(
                "received chunk_len={} while buffer_step={}/{}",
                len(mact_list),
                step_idx,
                chunk_len,
            )
            self.put(mact_list, timestamp=time.perf_counter())
        
        # Step 2: Check if we need to trigger inference
        buffer_empty = self.is_empty()
        need_infer = self.should_infer()
        if buffer_empty or need_infer:
            self._inference_ctx.send_trigger(self.t)
            self._stats["triggers_sent"] += 1
            self._log_debug(
                "sent trigger at t={} reason={}",
                self.t,
                "buffer_empty" if buffer_empty else "should_infer",
            )
        
        # Step 3: If buffer empty, block-wait for inference result
        if buffer_empty:
            self._wait_for_action_chunk(timeout=3600.0)
        
        # Step 4: Get single-step action
        action = self.get()
        if action is None:
            raise RuntimeError("ActionManager returned None - no action available")
        emit_step, emit_chunk_len = self._buffer_status()
        self._log_debug(
            "emit action t={} buffer_step={}/{} {}",
            self.t,
            emit_step,
            emit_chunk_len,
            self._summarize_action(action),
        )
        
        # Step 5: Auto-increment step counter
        self.t += 1
        
        return action

    def _wait_for_action_chunk(self, timeout: float = 30.0):
        """Block until a new action chunk is available."""
        t_start = time.perf_counter()
        self._stats["waits"] += 1
        
        while self.is_empty():
            mact_list = self._inference_ctx.poll_action_chunks()
            if mact_list:
                wait_ms = (time.perf_counter() - t_start) * 1000
                self._stats["cumulative_wait_ms"] += wait_ms
                self._stats["chunks_received"] += 1
                self._log_debug("waited {:.1f}ms for chunk_len={}", wait_ms, len(mact_list))
                self.put(mact_list, timestamp=time.perf_counter())
                break
            
            elapsed = time.perf_counter() - t_start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Timeout ({timeout}s) waiting for inference result. "
                    f"Check if inference process is alive."
                )
            
            if not self._inference_ctx.is_alive():
                raise RuntimeError("Inference process died unexpectedly")
            
            time.sleep(0.0001)  # 0.1ms

    def put(self, chunk, timestamp: float = None):
        with self._lock:
            # Track how many steps were wasted in the replaced buffer
            old_len = len(self._chunk_buffer) if self._chunk_buffer is not None else 0
            if self._chunk_buffer is not None:
                remained = max(0, old_len - self.current_step)
                if remained > 0:
                    self._stats["total_remaining_on_replace"] += remained
                    self._stats["replace_count"] += 1
            self._chunk_buffer = chunk
            self.current_step = 0
        self._log_debug(
            "stored chunk_len={} replaced_old_len={}",
            len(chunk) if chunk is not None else 0,
            old_len,
        )

    def get(self, timestamp: float = None):
        with self._lock:
            if self._chunk_buffer is None or self.current_step >= len(self._chunk_buffer):
                return None
            action = self._chunk_buffer[self.current_step]
            self.current_step += 1
            return action

    def is_empty(self) -> bool:
        with self._lock:
            if self._chunk_buffer is None:
                return True
            return self.current_step >= len(self._chunk_buffer)

    def should_infer(self) -> bool:
        """Default: always False → inference only when buffer is empty (synchronous)."""
        return False

    def reset(self):
        with self._lock:
            self._chunk_buffer = None
            self._buffer = None
            self.current_step = 0
            self.t = 0
        if self._inference_ctx is not None:
            self._inference_ctx.send_reset()
        
        if self._debug:
            s = self._stats
            avg_wait = s["cumulative_wait_ms"] / max(1, s["waits"])
            avg_wasted = (
                s["total_remaining_on_replace"] / max(1, s["replace_count"])
                if s["replace_count"] > 0 else 0.0
            )
            logger.info(
                f"[ActionManager] RESET | "
                f"chunks: recv={s['chunks_received']} discard={s['chunks_discarded']} | "
                f"triggers={s['triggers_sent']} waits={s['waits']} avg_wait={avg_wait:.1f}ms | "
                f"avg_unused_actions={avg_wasted:.1f}"
            )
        
        # Reset per-rollout stats
        for k in self._stats:
            self._stats[k] = 0 if isinstance(self._stats[k], int) else 0.0
