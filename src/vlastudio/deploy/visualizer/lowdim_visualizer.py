"""
Low-dimensional time-series visualizer: one figure with one subplot per dimension.

Reads one or more device SHM dicts; extracts a 1D vector per stream.
``visualizer.args.array_key`` selects the field: top-level name (e.g. ``action``,
``qpos``) or dotted path for nested dicts (e.g. ``observation.qpos``). If omitted,
tries in order: qpos, action, lowdim, state, values. Teleop devices publish
``action``; many robots publish ``qpos``. Set ``debug: true`` or env
``ILSTUDIO_LOWDIM_DEBUG=1`` for a one-time SHM payload summary. Streams in the
same process must share the same vector length ``D`` (optional ``dim`` in YAML).
SHM names that never appear within ``shm_connect_timeout_per_stream`` (default 3s) are
skipped; the plot uses only connected streams.

- X axis: spans the **currently drawn samples** (tight) + padding; never wider than
  ``time_window_s`` (cap). Not a fixed long empty window.
- Y axis: value per dimension; limits only **expand** as new data arrives.

A **background thread** polls SHM at device rate. Redraw uses the canvas backend's
``new_timer()`` (Tk ``after`` / Qt timer) at ``fps``, plus **synchronous**
``canvas.draw()`` so frames are not merged into rare ``draw_idle`` bursts (a common
TkAgg issue with ``FuncAnimation``).

Requires ``matplotlib``. GUI backend: ``TkAgg`` / ``Qt5Agg`` when available.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from itertools import islice
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from vlastudio.deploy.shm_utils import SharedMemoryChannel
from vlastudio.deploy.visualizer.base import BaseVisualizer

# Prefer keys in this order when ``array_key_hint`` is None.
_DEFAULT_ARRAY_KEYS = ("qpos", "action", "lowdim", "state", "values")


def _get_by_path(root: dict, path: str) -> Optional[Any]:
    """Resolve ``a.b.c`` in nested dicts; single segment uses top-level key."""
    path = str(path).strip()
    if not path:
        return None
    cur: Any = root
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _as_float_vector(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        v = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    return v if v.size > 0 else None


def _extract_vector(
    data: dict,
    array_key_hint: Optional[str],
    default_keys: Tuple[str, ...] = _DEFAULT_ARRAY_KEYS,
) -> Optional[np.ndarray]:
    if not isinstance(data, dict):
        return None
    if array_key_hint:
        k = str(array_key_hint).strip()
        if k:
            raw = _get_by_path(data, k)
            vec = _as_float_vector(raw)
            if vec is not None:
                return vec
            return None
    for k in default_keys:
        if k not in data:
            continue
        vec = _as_float_vector(data[k])
        if vec is not None:
            return vec
    for k, v in data.items():
        if str(k).startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            vv = _as_float_vector(v)
            if vv is not None and vv.size <= 256:
                return vv
    return None


def _shm_payload_summary(data: dict) -> str:
    parts = []
    for k, v in sorted(data.items()):
        if str(k).startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            parts.append(f"{k}=ndarray{tuple(v.shape)}")
        else:
            parts.append(f"{k}={type(v).__name__}")
    return "{" + ", ".join(parts) + "}"


def _configure_matplotlib_backend(debug: bool = False) -> str:
    """Returns matplotlib backend name in use."""
    import matplotlib

    env_be = os.environ.get("MPLBACKEND", "").strip()
    if env_be:
        try:
            matplotlib.use(env_be, force=True)
            name = matplotlib.get_backend()
            if debug:
                print(f"[LowDimVisualizer][debug] MPLBACKEND env → backend={name!r}", flush=True)
            return name
        except Exception as e:
            print(
                f"[LowDimVisualizer] Invalid MPLBACKEND={env_be!r} ({e}); "
                f"falling back to TkAgg/Qt.",
                flush=True,
            )

    last_err = None
    for b in ("TkAgg", "Qt5Agg", "QtAgg"):
        try:
            matplotlib.use(b, force=True)
            name = matplotlib.get_backend()
            if debug:
                print(f"[LowDimVisualizer][debug] selected GUI backend={name!r}", flush=True)
            return name
        except Exception as e:
            last_err = e
    matplotlib.use("Agg", force=True)
    print(
        "[LowDimVisualizer] Warning: using Agg backend (no display). "
        f"Last GUI backend error: {last_err!r}. "
        "Install tk or PyQt5, or set MPLBACKEND=TkAgg.",
        flush=True,
    )
    return matplotlib.get_backend()


class LowDimVisualizer(BaseVisualizer):
    """
    Multiple SHM streams, same feature dim ``D``; one figure with ``D`` subplots (stacked).
    Each subplot overlays one curve per stream (legend = shm name).
    """

    def __init__(
        self,
        shm_entries: List[Tuple[str, Optional[str]]],
        *,
        group_id: str = "lowdim",
        expected_dim: Optional[int] = None,
        fps: float = 60.0,
        time_window_s: float = 12.0,
        max_points: int = 8000,
        max_plot_points: Optional[int] = None,
        debug: bool = False,
        shm_connect_timeout_per_stream: float = 3.0,
    ):
        first = shm_entries[0][0] if shm_entries else "lowdim"
        super().__init__(shm_name=first, fps=fps)
        self.shm_entries = list(shm_entries)
        self.group_id = str(group_id)
        self.expected_dim = int(expected_dim) if expected_dim is not None else None
        self.fps = float(fps)
        self._shm_connect_timeout_per_stream = max(0.1, float(shm_connect_timeout_per_stream))
        # Max visible time span (seconds); x-axis fits data but never exceeds this.
        self.time_window_s = max(0.5, float(time_window_s))
        self._hist_pad_s = 4.0  # keep deque a bit longer than max span for trimming
        self.max_points = int(max_points)
        cap = max_plot_points if max_plot_points is not None else min(self.max_points, 2000)
        self._plot_cap = max(200, int(cap))
        env_dbg = os.environ.get("ILSTUDIO_LOWDIM_DEBUG", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self._debug = bool(debug) or env_dbg

        self.shm_names = [e[0] for e in self.shm_entries]
        self.key_hints = [e[1] for e in self.shm_entries]
        n = len(self.shm_names)
        self.shm_channels: List[Optional[SharedMemoryChannel]] = [None] * n
        self._logged_payload_streams: set = set()

        self._t0: Optional[float] = None
        self._d: Optional[int] = None
        # history[s] = deque of (t_rel, vec copy)
        self._hist: List[Deque[Tuple[float, np.ndarray]]] = [
            deque(maxlen=self.max_points) for _ in range(n)
        ]
        self._y_min: Optional[np.ndarray] = None
        self._y_max: Optional[np.ndarray] = None

        self._fig = None
        self._axes: List = []
        self._lines: List[List] = []  # [dim][stream] Line2D
        self.is_running = False
        self._hist_lock = threading.Lock()
        self._sample_thread: Optional[threading.Thread] = None
        self._sample_thread_running = False
        self._draw_timer = None
        self._redraw_frame_i = 0

    def setup(self) -> bool:
        """Satisfies ``BaseVisualizer``; the real GUI setup runs in ``start()`` via ``_setup_plots``."""
        return True

    def visualize(self, data: dict) -> bool:
        """Satisfies ``BaseVisualizer``; drawing is timer-driven in ``start()``."""
        return True

    def connect(self, timeout: Optional[float] = None) -> bool:
        """
        Try each SHM for up to ``timeout`` seconds (default: ``shm_connect_timeout_per_stream``).
        Streams that never become available are skipped; the window runs on the rest only.
        """
        per_t = (
            float(timeout)
            if timeout is not None
            else self._shm_connect_timeout_per_stream
        )
        per_t = max(0.1, per_t)
        for i, name in enumerate(self.shm_names):
            t0 = time.time()
            connected = False
            while time.time() - t0 < per_t:
                try:
                    self.shm_channels[i] = SharedMemoryChannel(
                        name, is_writer=False, timeout=1.0
                    )
                    print(f"[LowDimVisualizer] Connected SHM: {name}", flush=True)
                    connected = True
                    break
                except Exception:
                    time.sleep(0.05)
            if not connected:
                self.shm_channels[i] = None
                print(
                    f"[LowDimVisualizer] SHM '{name}' not available within {per_t:.1f}s — skipping",
                    flush=True,
                )

        keep = [i for i, ch in enumerate(self.shm_channels) if ch is not None]
        if not keep:
            print(
                "[LowDimVisualizer] No SHM connected (all streams missing or timed out).",
                flush=True,
            )
            return False

        self.shm_names = [self.shm_names[i] for i in keep]
        self.key_hints = [self.key_hints[i] for i in keep]
        self.shm_channels = [self.shm_channels[i] for i in keep]
        self._hist = [self._hist[i] for i in keep]
        self._logged_payload_streams.clear()

        if self._debug:
            print(
                f"[LowDimVisualizer][debug] connected streams={self.shm_names!r} "
                f"array_key hints={self.key_hints!r} expected_dim={self.expected_dim}",
                flush=True,
            )
        return True

    def _log_payload_once(self, stream_i: int, data: dict, vec: Optional[np.ndarray]) -> None:
        if not self._debug or stream_i in self._logged_payload_streams:
            return
        self._logged_payload_streams.add(stream_i)
        hint = self.key_hints[stream_i]
        print(
            f"[LowDimVisualizer][debug] first payload shm={self.shm_names[stream_i]!r} "
            f"array_key={hint!r} summary={_shm_payload_summary(data)} "
            f"extract_ok={vec is not None} shape={None if vec is None else vec.shape}",
            flush=True,
        )
        if vec is None and hint:
            print(
                f"[LowDimVisualizer][debug] hint {hint!r} did not resolve to a numeric vector. "
                f"Teleop devices usually publish key 'action'; robots often 'qpos'.",
                flush=True,
            )

    def _bootstrap_dim(self) -> bool:
        """Read until we know D (from first valid vector)."""
        deadline = time.time() + 90.0
        last_wait_log = 0.0
        while time.time() < deadline:
            for i, ch in enumerate(self.shm_channels):
                if ch is None:
                    continue
                data = ch.read(blocking=False, skip_unchanged=False)
                if data is None:
                    continue
                vec = _extract_vector(data, self.key_hints[i])
                self._log_payload_once(i, data, vec)
                if vec is None:
                    continue
                d = int(vec.size)
                if self.expected_dim is not None and d != self.expected_dim:
                    print(
                        f"[LowDimVisualizer] Stream {self.shm_names[i]} dim={d} "
                        f"!= expected lowdim_dim={self.expected_dim}"
                    )
                    return False
                self._d = d
                self._t0 = time.perf_counter()
                t_rel = 0.0
                vcopy = vec.astype(np.float64, copy=True)
                self._hist[i].append((t_rel, vcopy))
                self._y_min = vcopy.copy()
                self._y_max = vcopy.copy()
                if self._debug:
                    print(
                        f"[LowDimVisualizer][debug] bootstrap ok stream={self.shm_names[i]!r} D={d}",
                        flush=True,
                    )
                return True
            now = time.time()
            if self._debug and now - last_wait_log > 5.0:
                last_wait_log = now
                print(
                    "[LowDimVisualizer][debug] still waiting for first numeric vector… "
                    "(wrong array_key / device not writing?)",
                    flush=True,
                )
            time.sleep(0.002)
        print(
            "[LowDimVisualizer] Timeout waiting for first lowdim sample. "
            "Check array_key matches SHM keys (see ILSTUDIO_LOWDIM_DEBUG=1).",
            flush=True,
        )
        return False

    def _setup_plots(self) -> bool:
        backend = _configure_matplotlib_backend(debug=self._debug)
        import matplotlib.pyplot as plt

        self._plt = plt
        if str(backend).lower() == "agg":
            print(
                "[LowDimVisualizer] Interactive backend unavailable; no window will appear.",
                flush=True,
            )
        d = int(self._d)
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(self.shm_names))))

        self._y_min = np.full(d, np.inf, dtype=np.float64)
        self._y_max = np.full(d, -np.inf, dtype=np.float64)
        for dq in self._hist:
            for _, vec in dq:
                self._y_min = np.minimum(self._y_min, vec)
                self._y_max = np.maximum(self._y_max, vec)

        h = max(3.5, min(2.4 * d, 24.0))
        fig, axes = plt.subplots(
            d,
            1,
            figsize=(9, h),
            sharex=True,
            constrained_layout=False,
        )
        if d == 1:
            self._axes = [axes]
        else:
            self._axes = list(axes)
        self._fig = fig
        try:
            fig.canvas.manager.set_window_title(f"LowDim [{self.group_id}]")
        except Exception:
            pass

        for di in range(d):
            ax = self._axes[di]
            ax.set_ylabel(f"d{di}")
            ax.grid(True, alpha=0.3)
            if di == d - 1:
                ax.set_xlabel("t (s)")
            lines_row = []
            for si, name in enumerate(self.shm_names):
                (line,) = ax.plot(
                    [],
                    [],
                    label=name,
                    color=colors[si % len(colors)],
                    linewidth=1.2,
                )
                lines_row.append(line)
            ax.legend(loc="upper right", fontsize=7)
            self._lines.append(lines_row)

        try:
            fig.tight_layout()
        except Exception:
            pass

        def _on_close(_evt):
            self.is_running = False
            self._sample_thread_running = False
            self._stop_draw_timer()

        fig.canvas.mpl_connect("close_event", _on_close)

        if self.fps <= 0:
            interval_ms = 5
        else:
            interval_ms = max(5, int(round(1000.0 / min(self.fps, 200.0))))

        try:
            self._draw_timer = fig.canvas.new_timer(interval=interval_ms)
            self._draw_timer.single_shot = False
            self._draw_timer.add_callback(self._timer_redraw)
            self._draw_timer.start()
        except Exception as e:
            print(
                f"[LowDimVisualizer] canvas.new_timer failed ({e!r}); "
                f"redraw may be broken on this backend.",
                flush=True,
            )
            self._draw_timer = None

        plt.ion()
        if self._debug:
            print(
                f"[LowDimVisualizer][debug] figure num={fig.number} backend={backend!r}",
                flush=True,
            )
        print(
            f"[LowDimVisualizer] 1 figure, {d} subplots, group={self.group_id!r}, "
            f"{len(self.shm_names)} stream(s). Close window or Ctrl+C to stop.",
            flush=True,
        )
        return True

    def _stop_draw_timer(self) -> None:
        if self._draw_timer is None:
            return
        try:
            self._draw_timer.stop()
        except Exception:
            pass
        self._draw_timer = None

    def _timer_redraw(self, *args) -> None:
        self._redraw_tick()

    def _redraw_tick(self) -> None:
        if not self.is_running or self._t0 is None:
            return
        self._redraw_frame_i += 1
        t_now = time.perf_counter() - self._t0
        if self._redraw_frame_i % 4 == 0:
            self._trim_time(t_now)
        if not self._update_plots(t_now):
            self.is_running = False
            self._sample_thread_running = False
            self._stop_draw_timer()

    def _sample_loop(self) -> None:
        """Drain every new SHM write (device rate); runs in a background thread."""
        assert self._t0 is not None and self._d is not None
        d = self._d
        while self._sample_thread_running:
            any_new = False
            for i, ch in enumerate(self.shm_channels):
                if ch is None:
                    continue
                while True:
                    data = ch.read(blocking=False, skip_unchanged=True)
                    if data is None:
                        break
                    vec = _extract_vector(data, self.key_hints[i])
                    if vec is None or vec.size != d:
                        break
                    t_rel = time.perf_counter() - self._t0
                    vcopy = vec.astype(np.float64, copy=True)
                    # Short lock per sample so the GUI thread can refresh between writes.
                    with self._hist_lock:
                        self._hist[i].append((t_rel, vcopy))
                        self._y_min = np.minimum(self._y_min, vcopy)
                        self._y_max = np.maximum(self._y_max, vcopy)
                    any_new = True
            if not any_new:
                time.sleep(0.00015)

    def _trim_time(self, t_now: float) -> None:
        keep_s = self.time_window_s + self._hist_pad_s
        t_cut = t_now - keep_s
        with self._hist_lock:
            for dq in self._hist:
                while dq and dq[0][0] < t_cut:
                    dq.popleft()

    def _xlim_for_tails(
        self, tails: List[List[Tuple[float, np.ndarray]]], t_now: float
    ) -> Tuple[float, float]:
        """Tight x-range from plotted points; cap width at ``time_window_s``."""
        all_ts: List[float] = []
        for seg in tails:
            for t, _ in seg:
                all_ts.append(t)
        if not all_ts:
            pad = 0.2
            lo = max(0.0, t_now - pad)
            return lo, max(lo + 1e-3, t_now + 1e-6)
        t_min = min(all_ts)
        t_max = max(all_ts)
        span = t_max - t_min
        xpad = max(1e-6, 0.04 * span if span > 1e-12 else 0.02)
        lo = t_min - xpad
        hi = t_max + xpad
        cap = self.time_window_s
        if hi - lo > cap:
            lo = hi - cap
        min_span = min(0.2, cap * 0.25)
        if hi - lo < min_span:
            mid = 0.5 * (hi + lo)
            lo = mid - 0.5 * min_span
            hi = mid + 0.5 * min_span
        hi = max(hi, lo + 1e-6)
        return lo, hi

    def _update_plots(self, t_now: float) -> bool:
        plt = self._plt
        fig = self._fig
        d = self._d
        assert d is not None

        with self._hist_lock:
            if self._y_min is None or self._y_max is None:
                return True
            y_min = self._y_min.copy()
            y_max = self._y_max.copy()
            tails: List[List[Tuple[float, np.ndarray]]] = []
            for dq in self._hist:
                n = len(dq)
                start = max(0, n - self._plot_cap)
                tails.append(list(islice(dq, start, None)))

        x_lo, x_hi = self._xlim_for_tails(tails, t_now)
        for di in range(d):
            ax = self._axes[di]
            for si, seg in enumerate(tails):
                if not seg:
                    self._lines[di][si].set_data([], [])
                    continue
                n = len(seg)
                ta = np.empty(n, dtype=np.float64)
                ya = np.empty(n, dtype=np.float64)
                for j, p in enumerate(seg):
                    ta[j] = p[0]
                    ya[j] = float(p[1][di])
                self._lines[di][si].set_data(ta, ya)
            ax.set_xlim(x_lo, x_hi)
            span = float(y_max[di] - y_min[di])
            pad = 0.05 * span if span > 1e-12 else 1.0
            ax.set_ylim(float(y_min[di]) - pad, float(y_max[di]) + pad)

        use_idle = os.environ.get("ILSTUDIO_LOWDIM_DRAW_IDLE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            if use_idle:
                fig.canvas.draw_idle()
            else:
                fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception:
            pass

        try:
            if self._fig is None or not plt.fignum_exists(self._fig.number):
                return False
        except Exception:
            return False
        return True

    def cleanup(self) -> None:
        self._sample_thread_running = False
        if self._sample_thread is not None:
            self._sample_thread.join(timeout=1.0)
            self._sample_thread = None
        self._stop_draw_timer()
        try:
            if hasattr(self, "_plt") and self._fig is not None:
                self._plt.close(self._fig)
        except Exception:
            pass
        for ch in self.shm_channels:
            if ch is not None:
                try:
                    ch.destroy()
                except Exception:
                    pass

    def start(self) -> None:
        t0_wall = time.perf_counter()
        if not self.connect():
            return
        if self._debug:
            print(
                f"[LowDimVisualizer][debug] connect elapsed {time.perf_counter() - t0_wall:.3f}s",
                flush=True,
            )
        t1 = time.perf_counter()
        if not self._bootstrap_dim():
            self.cleanup()
            return
        if self._debug:
            print(
                f"[LowDimVisualizer][debug] bootstrap elapsed {time.perf_counter() - t1:.3f}s",
                flush=True,
            )
        t2 = time.perf_counter()
        if not self._setup_plots():
            self.cleanup()
            return
        if self._debug:
            print(
                f"[LowDimVisualizer][debug] matplotlib setup {time.perf_counter() - t2:.3f}s; "
                f"total to first frame {time.perf_counter() - t0_wall:.3f}s",
                flush=True,
            )

        self.is_running = True
        self._redraw_frame_i = 0
        self._sample_thread_running = True
        self._sample_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sample_thread.start()
        try:
            self._plt.show(block=True)
        except KeyboardInterrupt:
            print("\n[LowDimVisualizer] Interrupted")
        finally:
            self.is_running = False
            self._sample_thread_running = False
            self.cleanup()
            print("[LowDimVisualizer] Stopped")


def start_lowdim_visualizer(
    shm_entries: List[Tuple[str, Optional[str]]],
    *,
    group_id: str = "lowdim",
    expected_dim: Optional[int] = None,
    fps: float = 60.0,
    time_window_s: float = 12.0,
    max_points: int = 8000,
    max_plot_points: Optional[int] = None,
    debug: bool = False,
    shm_connect_timeout_per_stream: float = 3.0,
) -> None:
    """
    Entry point for ``multiprocessing.Process``.

    Args:
        shm_entries: ``(shm_name, lowdim_array_key_or_None)`` per stream.
        group_id: Window title / grouping label.
        expected_dim: If set, first sample must match; all streams must match ``D``.
        fps: Plot redraw cap (``<=0`` = uncapped). SHM is sampled in a thread at device rate.
        time_window_s: Max x-axis span (seconds); actual window fits drawn data, up to this cap.
        max_points: Max points retained per stream (deque maxlen).
        max_plot_points: Max points drawn per line (default min(max_points, 2000)).
        shm_connect_timeout_per_stream: Seconds to wait per SHM before skipping a missing stream.
    """
    viz = LowDimVisualizer(
        shm_entries,
        group_id=group_id,
        expected_dim=expected_dim,
        fps=fps,
        time_window_s=time_window_s,
        max_points=max_points,
        max_plot_points=max_plot_points,
        debug=debug,
        shm_connect_timeout_per_stream=shm_connect_timeout_per_stream,
    )
    viz.start()


Visualizer = LowDimVisualizer
