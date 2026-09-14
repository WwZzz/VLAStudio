"""
Async chunk-ratio action manager: prefetch inference when the current chunk
is consumed past a fraction of its length (e.g. 0.7 → after 70% of steps emitted).

When a new chunk arrives asynchronously, observations may be stale. ``put`` aligns
the incoming chunk by scanning left-to-right for the step whose action is closest
to the reference action (current execution point on the previous chunk), then
truncates the new chunk from that index and replaces the old buffer entirely.
"""

from __future__ import annotations

import numpy as np

from .base import BasicActionManager


def _extract_action_from_step(step) -> np.ndarray | None:
    """Same convention as MetaPolicy steps / deserialize_mact_list."""
    if isinstance(step, np.ndarray) and step.dtype == object and len(step) > 0:
        first = step[0]
        if isinstance(first, dict):
            act = first.get("action", None)
            if act is not None:
                return np.asarray(act, dtype=np.float64)
            return None
    if isinstance(step, np.ndarray):
        return np.asarray(step, dtype=np.float64)
    return None


def _action_distance(ref: np.ndarray, a: np.ndarray) -> float:
    r = np.asarray(ref, dtype=np.float64).ravel()
    b = np.asarray(a, dtype=np.float64).ravel()
    if r.size == 0 or b.size == 0:
        return float("inf")
    m = min(r.size, b.size)
    return float(np.linalg.norm(r[:m] - b[:m]))


class AsyncChunkRatioManager(BasicActionManager):
    """
    Trigger an extra inference once per chunk when execution progress reaches
    ``chunk_ratio`` of the current chunk (by emitted step index / chunk length).

    When ``put`` receives a new chunk while still executing a previous plan and
    at least one step of the old chunk has been emitted (``current_step > 0``),
    the new chunk is **truncated** from index ``j`` chosen to minimize the
    distance between the action at ``new_chunk[j]`` and the **last emitted**
    action (``old_chunk[current_step - 1]``). The old buffer is discarded and
    replaced by ``new_chunk[j:]``. If ``current_step == 0`` or there is no old
    chunk, the incoming chunk is stored unchanged.

    Args:
        chunk_ratio: Float in (0, 1], e.g. 0.7 for 70%.
    """

    def __init__(self, chunk_ratio: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        if not (0.0 < chunk_ratio <= 1.0):
            raise ValueError(
                f"chunk_ratio must be in (0, 1], got {chunk_ratio!r}"
            )
        self.chunk_ratio = float(chunk_ratio)
        self._ratio_prefetch_sent = False

    def _align_chunk_to_reference(self, chunk: list) -> tuple[list, int, float | None]:
        """
        Find start index ``j`` in ``chunk`` minimizing distance to reference action.
        Returns (chunk[j:], j, best_distance or None if skipped).
        """
        if chunk is None or len(chunk) == 0:
            return chunk, 0, None

        ref_step = None
        ref_idx = None
        with self._lock:
            obuf = self._chunk_buffer
            cstep = self.current_step
            if obuf is None or len(obuf) == 0:
                return chunk, 0, None
            if cstep <= 0:
                return chunk, 0, None
            ref_idx = cstep - 1
            ref_step = obuf[ref_idx]

        ref_action = _extract_action_from_step(ref_step)
        if ref_action is None or ref_action.size == 0:
            return chunk, 0, None

        best_j = 0
        best_d = float("inf")
        for j in range(len(chunk)):
            a = _extract_action_from_step(chunk[j])
            if a is None or a.size == 0:
                continue
            d = _action_distance(ref_action, a)
            if d < best_d:
                best_d = d
                best_j = j

        if best_d == float("inf"):
            return chunk, 0, None

        aligned = chunk[best_j:]
        return aligned, best_j, best_d

    def put(self, chunk, timestamp=None):
        if chunk is None or len(chunk) == 0:
            super().put(chunk, timestamp)
            with self._lock:
                self._ratio_prefetch_sent = False
            return

        aligned, start_j, best_d = self._align_chunk_to_reference(chunk)
        if best_d is not None and (start_j > 0 or len(aligned) != len(chunk)):
            self._log_debug(
                "async chunk align: start_j={}/{} best_dist={:.6f} len {} -> {}",
                start_j,
                len(chunk),
                best_d,
                len(chunk),
                len(aligned),
            )

        super().put(aligned, timestamp)
        with self._lock:
            self._ratio_prefetch_sent = False

    def should_infer(self) -> bool:
        with self._lock:
            if self._chunk_buffer is None:
                return False
            chunk_len = len(self._chunk_buffer)
            if chunk_len == 0:
                return False
            progress = self.current_step / chunk_len
            if progress < self.chunk_ratio:
                return False
            if self._ratio_prefetch_sent:
                return False
            self._ratio_prefetch_sent = True
            self._log_debug(
                "should_infer prefetch: progress={:.3f} >= chunk_ratio={:.3f} step={}/{}",
                progress,
                self.chunk_ratio,
                self.current_step,
                chunk_len,
            )
            return True

    def reset(self):
        super().reset()
        with self._lock:
            self._ratio_prefetch_sent = False
