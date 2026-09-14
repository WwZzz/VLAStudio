"""
Strictly synchronous chunk manager: infer → play whole chunk → infer → ...

Unlike BasicActionManager, mid-chunk arrivals never replace the buffer, and we do
not prefetch. Matches "推理完再下发" open-loop ACT eval.
"""

from __future__ import annotations

import time

from loguru import logger

from deploy.action_manager.base import BasicActionManager


class SyncChunkManager(BasicActionManager):
    """
    Synchronous open-loop chunk execution.

    1. If buffer empty: discard any stale SHM chunks, trigger inference, block-wait
    2. Dispatch steps one-by-one at the control-loop publish rate
    3. Never accept a new chunk until the current one is fully consumed
    """

    def should_infer(self) -> bool:
        return False

    def put(self, chunk, timestamp: float = None):
        if self._chunk_buffer is None or self.is_empty():
            return super().put(chunk, timestamp=timestamp)
        with self._lock:
            self._stats["chunks_discarded"] += 1
        self._log_debug(
            "sync discard chunk_len={} while step={}/{}",
            len(chunk) if chunk is not None else 0,
            self.current_step,
            len(self._chunk_buffer) if self._chunk_buffer is not None else 0,
        )
        return

    def _drain_stale_chunks(self):
        """Drop leftover results so the next wait gets a fresh inference."""
        if self._inference_ctx is None:
            return
        n = 0
        while True:
            mact_list = self._inference_ctx.poll_action_chunks()
            if not mact_list:
                break
            n += 1
            self._stats["chunks_discarded"] += 1
        if n:
            logger.info(
                "[SyncChunk] drained {} stale chunk(s) before next inference",
                n,
            )

    def select_action(self):
        if self._inference_ctx is None:
            raise RuntimeError("InferenceContext not set. Call set_inference_context() first.")

        if self.is_empty():
            self._drain_stale_chunks()
            self._inference_ctx.send_trigger(self.t)
            self._stats["triggers_sent"] += 1
            self._log_debug("sync trigger at t={}", self.t)
            self._wait_for_action_chunk(timeout=3600.0)

        action = self.get()
        if action is None:
            raise RuntimeError("SyncChunkManager returned None - no action available")

        self.t += 1
        return action
