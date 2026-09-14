"""
eval_real runtime: output manifest, process/SHM/recorder cleanup, session recorder subprocess,
cross-platform non-blocking keyboard input (:class:`KBHit`) for eval_real / collect_data,
and the eval_real main-loop pause menu (save / reset / resume / exit).

The session recorder runs in a separate process so the eval_real main loop never opens device SHM.
"""

from __future__ import annotations

# CRITICAL: patch resource_tracker before multiprocessing / SHM in this module (recorder spawn)
import deploy.shm_utils  # noqa: F401

import atexit
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional

if os.name == "nt":
    import msvcrt
else:
    import select
    import termios

import numpy as np
from loguru import logger


def _get_process_context():
    """Prefer fork on POSIX to reduce subprocess cold-start latency."""
    if os.name != "nt":
        try:
            return mp.get_context("fork")
        except ValueError:
            pass
    return mp.get_context("spawn")


# ---------------------------------------------------------------------------
# Session recorder (subprocess entry: eval_real_session_recorder_main)
# ---------------------------------------------------------------------------


def _to_rgb_hwc_u8(arr: np.ndarray) -> Optional[np.ndarray]:
    from deploy.utils import _as_uint8_image

    img = _as_uint8_image(arr)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = np.clip(img.astype(np.float32) / 257.0, 0.0, 255.0).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            img = img[..., :3]
        elif img.shape[2] != 3:
            return None
    else:
        return None
    return img


def _extract_images_for_mosaic(synced_data: dict) -> List[np.ndarray]:
    images: List[np.ndarray] = []

    def _visit(value) -> None:
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if str(sub_key).startswith("__"):
                    continue
                _visit(sub_val)
            return
        if isinstance(value, np.ndarray):
            img = _to_rgb_hwc_u8(value)
            if img is not None:
                images.append(img)

    for dev_name, dev_data in (synced_data or {}).items():
        if str(dev_name).startswith("__") or not isinstance(dev_data, dict):
            continue
        _visit(dev_data)

    return images


def synced_frame_to_rgb_mosaic(synced_data: dict) -> Optional[np.ndarray]:
    images = _extract_images_for_mosaic(synced_data)
    if not images:
        return None

    if len(images) == 1:
        return images[0]

    import cv2

    target_h = min(img.shape[0] for img in images)
    resized = []
    for img in images:
        h, w = img.shape[:2]
        if h != target_h:
            new_w = max(1, int(round(w * (target_h / float(h)))))
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(img)
    return np.concatenate(resized, axis=1)


def eval_real_session_recorder_main(
    device_shm_names: List[str],
    video_path: str,
    dataset_output_dir: str,
    dataset_format: Optional[str],
    task: str,
    episode_id: int,
    sensing_rate: float,
    stop_event,
    cmd_queue=None,
    ack_queue=None,
    recorder_ready_event=None,
    initial_recording_active: bool = True,
) -> None:
    """
    Blocking loop until ``stop_event`` is set.

    When ``dataset_format`` is set (e.g. ``lerobotv21``), records synchronized device data
    into ``dataset_output_dir`` via DataSaver. When ``dataset_format`` is omitted/None but
    ``dataset_output_dir`` and ``video_path`` are set, records mosaic MP4 only (no dataset).
    If ``video_path`` is e.g.
    ``.../video/real_eval.mp4``, files are ``.../video/real_eval_episode_{idx:04d}.mp4``.

    Episode lifecycle is controlled by the main process via ``cmd_queue``:

    - ``"pause_recording"``: while the eval main loop is paused, do not write dataset
      or mosaic video (SHM may still be read to stay in sync).
    - ``"save_episode"``: save the open episode only. Does **not** open/create the
      next episode file (avoids truncating an existing next MP4 when back-filling).
    - ``"discard_episode"`` / reset:
        * unsaved frames → discard buffer and reopen the **same** episode;
        * after a save (nothing open) → open the **next free** episode index.
    - ``"resume_recording"``: re-enable writing into the already-open episode.
      Does **not** create a new episode (press ``r`` after ``s`` to open next).
    - ``"exit_discard"``: discard the current in-progress episode (if any), drop
      deferred next-episode state, stop writing; used when the user exits from the
      pause menu.

    Acknowledgement tuples are sent back via ``ack_queue``:
    ``(status_str, frame_count_or_zero, episode_idx_or_none)`` — for
    ``save_episode``, third field is the **saved** episode index; for
    ``discard_episode``, the discarded index (or ``None`` on no-op). For
    ``resume_recording``, third field is the active episode index.

    ``stop_event`` must be a ``multiprocessing.Event`` (picklable for spawn).

    If ``recorder_ready_event`` is set, it is ``set()`` once the recorder has taken the first
    synchronized frame and (when a dataset saver exists) successfully passed it to
    :meth:`~data_utils.data_saver.BaseDataSaver.add_frame` so streaming is initialized — so the
    main process can wait and start the control loop in lockstep with episode recording.
    """
    import queue as _queue_mod

    from deploy.shm_utils import SharedMemoryChannel, SharedMemoryDataSynchronizer
    from deploy.utils import RateLimiter
    from data_utils.data_saver import create_data_saver

    import imageio

    shm_channels = []
    remaining_names = list(device_shm_names)
    connect_deadline = time.time() + 15.0
    while remaining_names and time.time() < connect_deadline and not stop_event.is_set():
        next_remaining = []
        for name in remaining_names:
            try:
                ch = SharedMemoryChannel(name, is_writer=False, timeout=0.2)
                if ch.shm is None:
                    next_remaining.append(name)
                    continue
                shm_channels.append((name, ch))
                logger.info("[SessionRecorder] connected to {}", name)
            except Exception:
                next_remaining.append(name)
        if not next_remaining:
            # All names in this round connected; clear so we do not falsely warn below.
            remaining_names = []
            break
        remaining_names = next_remaining
        time.sleep(0.1)

    for name in remaining_names:
        logger.warning("[SessionRecorder] skip {}: timed out waiting for SHM", name)

    valid = [(n, c) for n, c in shm_channels if c is not None and c.shm is not None]
    if not valid:
        logger.error("[SessionRecorder] no SHM channels; exiting recorder process")
        return

    sync = SharedMemoryDataSynchronizer(
        shm_channels=valid,
        buffer_maxlen=200,
        max_tolerance_s=0.05,
    )

    video_dir = ""
    video_basename = ""
    if video_path:
        video_dir = os.path.dirname(video_path) or "."
        video_basename = os.path.splitext(os.path.basename(video_path))[0] or "real_eval"
        os.makedirs(video_dir, exist_ok=True)

    def _episode_video_file(ep_idx: int) -> str:
        return os.path.join(video_dir, f"{video_basename}_episode_{int(ep_idx):04d}.mp4")

    fps = float(max(1.0, sensing_rate))
    rl = RateLimiter()
    writer = None
    current_video_path: Optional[str] = None
    video_mosaic_count = 0
    data_saver = None
    queued_frames = 0
    actual_episode_idx = None
    device_name_list = [name for name, _ in valid]
    episode_start_deferred = False
    # Video-only: True while a mosaic writer is open for the current episode.
    episode_open = False
    # After a successful save: waiting for ``r`` to open the next free episode.
    saved_awaiting_next = False
    # Video-only: after discarding an *open* episode, next start reuses that idx.
    video_only_reuse_same_episode_after_discard = False
    recording_active = bool(initial_recording_active)

    def _close_video_writer(*, delete_file: bool) -> None:
        nonlocal writer, current_video_path, video_mosaic_count
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            writer = None
        path = current_video_path
        current_video_path = None
        # Only delete on explicit discard, or empty file that we just created.
        # Never delete when closing a successful save (delete_file=False).
        if path and os.path.isfile(path):
            try:
                if delete_file:
                    os.remove(path)
                elif video_mosaic_count == 0:
                    # Empty open (never written): remove stub; do not touch
                    # pre-existing files we refused to open.
                    os.remove(path)
            except OSError:
                pass
        video_mosaic_count = 0

    def _episode_file_exists(ep_idx: int) -> bool:
        if not video_path:
            return False
        p = _episode_video_file(ep_idx)
        try:
            return os.path.isfile(p) and os.path.getsize(p) > 0
        except OSError:
            return False

    def _next_free_episode_idx(start_from: Optional[int] = None) -> int:
        """Smallest free index >= start_from that has no non-empty MP4."""
        if start_from is None:
            start_from = 0
        idx = max(0, int(start_from))
        # Cap scan so a full directory cannot hang forever.
        for _ in range(100000):
            if not _episode_file_exists(idx):
                return idx
            idx += 1
        return idx

    def _open_video_writer_for_episode(ep_idx: int) -> int:
        """Open mosaic writer for ``ep_idx``, skipping existing non-empty files.

        Returns the actual episode index opened (may be > ep_idx if skipped).
        """
        nonlocal writer, current_video_path, actual_episode_idx
        if not video_path:
            return int(ep_idx)
        _close_video_writer(delete_file=False)
        idx = int(ep_idx)
        if _episode_file_exists(idx):
            skipped = idx
            idx = _next_free_episode_idx(start_from=idx + 1)
            logger.warning(
                "[SessionRecorder] episode {} already exists — using free index {}",
                skipped,
                idx,
            )
        p = _episode_video_file(idx)
        writer = imageio.get_writer(p, fps=fps)
        current_video_path = p
        actual_episode_idx = idx
        logger.info("[SessionRecorder] episode {} video -> {} (max {:.1f} Hz)", idx, p, fps)
        return idx

    def _next_video_only_episode_idx() -> int:
        """Next free mosaic index under video_dir (max existing + 1, or 0)."""
        if not video_dir or not os.path.isdir(video_dir):
            return 0
        prefix = f"{video_basename}_episode_"
        suffix = ".mp4"
        max_idx = -1
        try:
            names = os.listdir(video_dir)
        except OSError:
            return 0
        for name in names:
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            mid = name[len(prefix) : -len(suffix)]
            try:
                max_idx = max(max_idx, int(mid))
            except ValueError:
                continue
        return max_idx + 1

    def _start_new_episode(requested_idx=None, *, allow_overwrite: bool = False):
        """Open a new episode for recording (still may be paused).

        - ``requested_idx`` + ``allow_overwrite``: open that exact slot (``-s N`` start,
          or reopen after discard of the same idx).
        - ``requested_idx is None``: advance to the next **free** index (never clobber
          an existing non-empty MP4 — safe for 补录).
        """
        nonlocal actual_episode_idx, queued_frames, video_mosaic_count, episode_open
        nonlocal video_only_reuse_same_episode_after_discard, saved_awaiting_next
        nonlocal writer, current_video_path
        queued_frames = 0
        video_mosaic_count = 0
        saved_awaiting_next = False
        reuse = False
        if data_saver is None:
            if requested_idx is not None:
                actual_episode_idx = int(requested_idx)
                video_only_reuse_same_episode_after_discard = False
                reuse = bool(allow_overwrite)
            else:
                if video_only_reuse_same_episode_after_discard:
                    video_only_reuse_same_episode_after_discard = False
                    reuse = True  # same idx; file was deleted on discard
                elif actual_episode_idx is None:
                    actual_episode_idx = _next_free_episode_idx(0)
                else:
                    actual_episode_idx = _next_free_episode_idx(
                        start_from=int(actual_episode_idx) + 1
                    )
            logger.info(
                "[SessionRecorder] started video-only episode {} (mosaic MP4)",
                actual_episode_idx,
            )
            if reuse or (
                requested_idx is not None and allow_overwrite
            ):
                if video_path:
                    _close_video_writer(delete_file=False)
                    p = _episode_video_file(actual_episode_idx)
                    writer = imageio.get_writer(p, fps=fps)
                    current_video_path = p
                    logger.info(
                        "[SessionRecorder] episode {} video -> {} (max {:.1f} Hz)",
                        actual_episode_idx,
                        p,
                        fps,
                    )
            else:
                _open_video_writer_for_episode(actual_episode_idx)
            episode_open = True
            return
        actual_episode_idx = data_saver.start_episode(
            episode_idx=requested_idx,
            device_names=device_name_list,
            teleop_device=None,
        )
        logger.info("[SessionRecorder] started episode {} in {}", actual_episode_idx, data_saver.dataset_path)
        if requested_idx is not None and allow_overwrite and video_path:
            _close_video_writer(delete_file=False)
            p = _episode_video_file(int(actual_episode_idx))
            writer = imageio.get_writer(p, fps=fps)
            current_video_path = p
        else:
            _open_video_writer_for_episode(actual_episode_idx)
        episode_open = True

    def _finish_current_episode(save: bool) -> int:
        nonlocal queued_frames, episode_open, video_only_reuse_same_episode_after_discard
        nonlocal saved_awaiting_next
        if data_saver is None:
            # Already finished: do not touch reuse/index.
            if not episode_open:
                return 0
            n = int(queued_frames)
            _close_video_writer(delete_file=not save)
            ep = actual_episode_idx
            queued_frames = 0
            episode_open = False
            # Reuse same idx only when discarding a non-empty open trial.
            video_only_reuse_same_episode_after_discard = (not save) and n > 0
            if save and n > 0:
                saved_awaiting_next = True
            if ep is not None:
                action = "saved" if save else "discarded"
                logger.info("[SessionRecorder] {} episode {} ({} video frames)", action, ep, n)
            return n
        # Dataset path: ignore save/discard if episode already closed.
        if not episode_open and not data_saver.is_recording:
            return 0
        n_buf = int(queued_frames)
        _close_video_writer(delete_file=not save)
        if not data_saver.is_recording:
            episode_open = False
            return 0
        n = data_saver.finish_episode(save=save and n_buf > 0)
        episode_open = False
        if save and n > 0:
            saved_awaiting_next = True
        action = "saved" if save else "discarded"
        logger.info("[SessionRecorder] {} episode {} ({} frames)", action, actual_episode_idx, n)
        return n

    def _process_cmd_queue():
        nonlocal episode_start_deferred, recording_active, queued_frames, saved_awaiting_next
        if cmd_queue is None:
            return
        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except _queue_mod.Empty:
                break

            if cmd == "pause_recording":
                recording_active = False
                if ack_queue is not None:
                    ack_queue.put(("paused", 0, None))
            elif cmd == "resume_recording":
                if episode_start_deferred:
                    episode_start_deferred = False
                # Do NOT auto-create next episode here — that wiped existing MP4s
                # when ``s`` used to open N+1. After save, user must ``r`` first.
                if not episode_open and saved_awaiting_next:
                    logger.warning(
                        "[SessionRecorder] resume with no open episode after save — "
                        "press r to open the next episode first"
                    )
                recording_active = bool(episode_open)
                if ack_queue is not None:
                    ack_queue.put(("resumed", 0, actual_episode_idx if episode_open else None))
            elif cmd == "save_episode":
                # Save only — never open/truncate the next episode file.
                ep_saved = actual_episode_idx if episode_open else None
                n = _finish_current_episode(save=True)
                episode_start_deferred = False
                if ack_queue is not None:
                    ack_queue.put(("saved", n, ep_saved))
            elif cmd == "discard_episode":
                n_buf = int(queued_frames) if episode_open else 0
                if episode_open and n_buf > 0:
                    # Unsaved trial: discard and reopen SAME index.
                    ep_disc = actual_episode_idx
                    n_disc = _finish_current_episode(save=False)
                    episode_start_deferred = False
                    _start_new_episode(
                        requested_idx=ep_disc,
                        allow_overwrite=True,
                    )
                    if ack_queue is not None:
                        ack_queue.put(("discarded", n_disc, ep_disc))
                elif saved_awaiting_next or not episode_open:
                    # After save (or nothing open): reset opens the next FREE episode.
                    episode_start_deferred = False
                    _start_new_episode(requested_idx=None)
                    if ack_queue is not None:
                        # n=0 means no discard; third field = newly opened episode.
                        ack_queue.put(("discarded", 0, actual_episode_idx))
                else:
                    # Open but empty: no-op.
                    if ack_queue is not None:
                        ack_queue.put(("discarded", 0, None))
            elif cmd == "exit_discard":
                recording_active = False
                episode_start_deferred = False
                saved_awaiting_next = False
                if episode_open:
                    _finish_current_episode(save=False)
                queued_frames = 0
                if ack_queue is not None:
                    ack_queue.put(("exit_ok", 0, None))
            else:
                logger.warning("[SessionRecorder] unknown command: {}", cmd)

    try:
        requested_episode_idx = None if episode_id is None or int(episode_id) < 0 else int(episode_id)

        if dataset_output_dir and dataset_format:
            data_saver = create_data_saver(
                format=dataset_format,
                output_dir=dataset_output_dir,
                fps=int(round(fps)),
                task=task,
            )
            # Explicit -s N may overwrite that slot; auto index never clobbers.
            _start_new_episode(
                requested_idx=requested_episode_idx,
                allow_overwrite=requested_episode_idx is not None,
            )
            logger.info(
                "[SessionRecorder] recording dataset to {} (episode={}, format={})",
                data_saver.dataset_path,
                actual_episode_idx,
                dataset_format,
            )
        elif dataset_output_dir and video_path:
            _start_new_episode(
                requested_idx=requested_episode_idx,
                allow_overwrite=requested_episode_idx is not None,
            )
            logger.info(
                "[SessionRecorder] video-only recording (no dataset); mosaic MP4 under {}",
                video_dir or os.path.dirname(video_path) or ".",
            )

        notified_ready = False

        while not stop_event.is_set():
            _process_cmd_queue()

            frame = sync.get_synced_frame_blocking(
                stop_check=lambda: stop_event.is_set(),
                poll_interval_s=0.001,
            )
            if frame is None:
                break
            frame["_sync_timestamp"] = time.perf_counter()

            wrote_dataset = False
            if (
                recording_active
                and data_saver is not None
                and data_saver.is_recording
                and data_saver.add_frame(frame)
            ):
                queued_frames += 1
                wrote_dataset = True

            if recording_active and writer is not None:
                mosaic = synced_frame_to_rgb_mosaic(frame)
                if mosaic is not None:
                    writer.append_data(mosaic)
                    video_mosaic_count += 1
                    if data_saver is None:
                        queued_frames += 1

            if recorder_ready_event is not None and not notified_ready:
                if (not recording_active) or data_saver is None or wrote_dataset:
                    notified_ready = True
                    recorder_ready_event.set()

            rl.sleep(sensing_rate)
    except Exception as e:
        logger.exception("[SessionRecorder] failed: {}", e)
    finally:
        if data_saver is not None:
            try:
                _finish_current_episode(save=queued_frames > 0)
            except Exception as e:
                logger.exception("[SessionRecorder] failed to finalize dataset recording: {}", e)
            finally:
                try:
                    data_saver.finalize()
                except Exception as e:
                    logger.exception("[SessionRecorder] failed to finalize dataset writer: {}", e)
        elif writer is not None:
            try:
                _finish_current_episode(save=queued_frames > 0)
            except Exception as e:
                logger.exception("[SessionRecorder] failed to finalize video-only episode: {}", e)
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        for _, ch in valid:
            try:
                ch.destroy()
            except Exception:
                pass
        logger.info(
            "[SessionRecorder] process exit (episode videos under {} if enabled)",
            video_dir or "(none)",
        )
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Main-process cleanup + manifest (eval_real.py)
# ---------------------------------------------------------------------------


def validate_eval_real_save_as_dataset(args: Any, parser: Any) -> None:
    """
    Normalize and validate ``--save-as-dataset`` on ``args`` (eval_real).
    Empty/whitespace becomes ``None``; invalid values call ``parser.error``.
    """
    _save = getattr(args, "save_as_dataset", None)
    if _save is not None:
        _save = _save.strip()
        if not _save:
            args.save_as_dataset = None
        else:
            _valid = {"lerobotv21", "lerobotv30", "hdf5"}
            if _save not in _valid:
                parser.error(
                    f"--save-as-dataset must be one of {sorted(_valid)}, got {_save!r}"
                )
            args.save_as_dataset = _save


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def setup_eval_real_output_dir(args) -> str:
    """
    If ``-o`` / ``output_dir`` is set: mkdir, write ``eval_real_manifest.json``, return absolute path.
    Otherwise return ``""``.
    """
    output_dir = (getattr(args, "output_dir", "") or "").strip()
    if not output_dir:
        return ""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Output directory (resolved): {}", output_dir)
    manifest = {
        "script": "eval_real.py",
        "output_dir": output_dir,
        "robot": getattr(args, "robot", ""),
        "model_name_or_path": getattr(args, "model_name_or_path", ""),
        "action_manager": getattr(args, "action_manager", ""),
        "publish_rate": getattr(args, "publish_rate", None),
        "sensing_rate": getattr(args, "sensing_rate", None),
        "save_as_dataset": getattr(args, "save_as_dataset", None),
        "task": getattr(args, "task", ""),
        "visualize": getattr(args, "visualize", False),
    }
    try:
        with open(os.path.join(output_dir, "eval_real_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.warning("Could not write eval_real_manifest.json: {}", e)
    return output_dir


def resolve_eval_real_output_dir(args) -> str:
    """
    Resolve ``-o`` / ``output_dir`` for eval_real.

    When unset, auto-creates ``data/eval_real/<ckpt>_<timestamp>/`` for mosaic MP4
    recording (pause-menu save). Set ``EVAL_REAL_NO_RECORD=1`` to disable.
    """
    output_dir = setup_eval_real_output_dir(args)
    if output_dir:
        return output_dir
    if _env_truthy("EVAL_REAL_NO_RECORD"):
        logger.info(
            "Session recording disabled (EVAL_REAL_NO_RECORD=1, no -o). "
            "Pause-menu save is unavailable."
        )
        return ""

    auto = (os.environ.get("EVAL_REAL_OUTPUT_DIR", "") or "").strip()
    if auto and auto.lower() not in ("0", "off", "false", "no"):
        args.output_dir = auto
        return setup_eval_real_output_dir(args)

    model = getattr(args, "model_name_or_path", "eval") or "eval"
    ckpt_name = os.path.basename(str(model).rstrip("/\\")) or "eval"
    # URLs like http://127.0.0.1:10002 → basename still has ':' which is illegal
    # on some filesystems (exFAT/NTFS mounts, etc.).
    ckpt_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in ckpt_name)
    ckpt_name = ckpt_name.strip("._") or "eval"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join("data", "eval_real", f"{ckpt_name}_{ts}")
    output_dir = setup_eval_real_output_dir(args)
    logger.info(
        "No -o given; auto-enabled session recording to {} "
        "(set EVAL_REAL_NO_RECORD=1 to disable, or pass -o explicitly).",
        output_dir,
    )
    return output_dir


class EvalRealRuntime:
    """
    Tracks device/inference/viz subprocesses, main-process SHM channels, and the optional
    session recorder (spawn via :meth:`start_session_recorder`). :meth:`cleanup_all` is idempotent
    (safe from atexit, signal, and finally).
    """

    def __init__(self) -> None:
        self.all_procs: List[Any] = []
        self.all_shm: List[Any] = []
        self._cleanup_done = False
        self.recorder_state: Dict[str, Any] = {
            "stop": None,
            "proc": None,
            "cmd_queue": None,
            "ack_queue": None,
        }

    @property
    def has_recorder(self) -> bool:
        return self.recorder_state.get("cmd_queue") is not None

    def start_session_recorder(
        self,
        *,
        output_dir: str,
        device_shm_names: List[str],
        dataset_format: Optional[str] = None,
        task: str = "",
        episode_id: int = -1,
        sensing_rate: float = 20.0,
        initial_recording_active: bool = True,
        wait_until_ready: bool = True,
        ready_timeout_s: float = 2.0,
    ) -> None:
        """
        Spawn :func:`eval_real_session_recorder_main` and fill :attr:`recorder_state`.
        No-op when ``output_dir`` or ``device_shm_names`` is empty.
        """
        if not output_dir or not device_shm_names:
            return

        ctx = _get_process_context()
        vid_path = os.path.join(output_dir, "video", "real_eval.mp4")
        os.makedirs(os.path.join(output_dir, "video"), exist_ok=True)
        rec_stop = ctx.Event()
        rec_ready = ctx.Event()
        rec_cmd_queue = ctx.Queue()
        rec_ack_queue = ctx.Queue()
        rec_proc = ctx.Process(
            target=eval_real_session_recorder_main,
            kwargs={
                "device_shm_names": list(device_shm_names),
                "video_path": vid_path,
                "dataset_output_dir": output_dir,
                "dataset_format": dataset_format,
                "task": task,
                "episode_id": episode_id,
                "sensing_rate": sensing_rate,
                "stop_event": rec_stop,
                "cmd_queue": rec_cmd_queue,
                "ack_queue": rec_ack_queue,
                "recorder_ready_event": rec_ready,
                "initial_recording_active": initial_recording_active,
            },
            name="eval_real_session_recorder",
            daemon=False,
        )
        rec_proc.start()
        self.recorder_state["stop"] = rec_stop
        self.recorder_state["proc"] = rec_proc
        self.recorder_state["cmd_queue"] = rec_cmd_queue
        self.recorder_state["ack_queue"] = rec_ack_queue
        logger.info(
            "Session recorder subprocess started (PID={}); main process does not open device SHM for it",
            rec_proc.pid,
        )
        if not wait_until_ready:
            return
        if not rec_ready.wait(timeout=ready_timeout_s):
            logger.warning(
                "Session recorder did not become ready within {}s; continuing without blocking.",
                ready_timeout_s,
            )
        else:
            logger.info(
                "Session recorder ready; startup prewarm complete.",
            )

    def stop_session_recorder(self) -> None:
        """Stop the optional session recorder subprocess before tearing down device SHM."""
        st = self.recorder_state
        p = st["proc"]
        if p is None:
            return
        ev = st["stop"]
        if ev is not None:
            ev.set()
        st["proc"] = None
        st["stop"] = None
        p.join(timeout=90.0)
        if p.is_alive():
            logger.warning("Session recorder did not exit; terminating")
            p.terminate()
            p.join(timeout=5.0)

    _RECORDER_ACK_STATUS = {
        "pause_recording": "paused",
        "resume_recording": "resumed",
        "save_episode": "saved",
        "discard_episode": "discarded",
        "exit_discard": "exit_ok",
    }

    def send_recorder_cmd(self, cmd_str: str, timeout: float = 30.0) -> Optional[tuple]:
        """Send a command to the recorder subprocess and wait for its ack.

        Drains any stale acks first (e.g. left by :meth:`post_recorder_cmd`) and, when
        the command has a known ack status, ignores mismatched acks until the matching
        one arrives or ``timeout`` elapses.
        """
        import queue as _queue_mod

        cq = self.recorder_state.get("cmd_queue")
        aq = self.recorder_state.get("ack_queue")
        if cq is None or aq is None:
            return None

        while True:
            try:
                stale = aq.get_nowait()
            except _queue_mod.Empty:
                break
            logger.warning("Dropping stale recorder ack before '{}': {}", cmd_str, stale)

        cq.put(cmd_str)
        expected = self._RECORDER_ACK_STATUS.get(cmd_str)
        deadline = time.perf_counter() + float(timeout)
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                ack = aq.get(timeout=remaining)
            except Exception:
                break
            if expected is None or (isinstance(ack, tuple) and ack and ack[0] == expected):
                return ack
            logger.warning(
                "Ignoring unexpected recorder ack for '{}' (want {!r}): {}",
                cmd_str,
                expected,
                ack,
            )
        logger.warning("Recorder did not acknowledge '{}' within {}s", cmd_str, timeout)
        return None

    def post_recorder_cmd(self, cmd_str: str) -> bool:
        """Send a recorder command without blocking for an acknowledgement.

        Prefer :meth:`send_recorder_cmd` when the caller needs the ack (frame counts,
        episode index). Fire-and-forget posts can leave stale acks that desync later
        ``send_recorder_cmd`` calls unless those drain the queue.
        """
        cq = self.recorder_state.get("cmd_queue")
        if cq is None:
            return False
        cq.put(cmd_str)
        return True

    def cleanup_all(self) -> None:
        if self._cleanup_done:
            return
        self._cleanup_done = True

        logger.info("Running cleanup...")
        self.stop_session_recorder()

        for p in self.all_procs:
            if p.is_alive():
                p.terminate()

        for p in self.all_procs:
            p.join(timeout=3.0)
            if p.is_alive():
                logger.warning("Process {} did not exit gracefully, killing...", p.pid)
                p.kill()
                p.join(timeout=1.0)

        for shm in self.all_shm:
            try:
                shm.destroy()
            except Exception:
                pass

        logger.info("Cleanup complete.")

    def register_exit_hooks(self) -> None:
        """Register SIGINT/SIGTERM and atexit to run :meth:`cleanup_all`."""

        def _on_signal(signum, frame):
            logger.info("Received signal {}, cleaning up...", signum)
            self.cleanup_all()
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        atexit.register(self.cleanup_all)


# ---------------------------------------------------------------------------
# Keyboard (eval_real / collect_data)
# ---------------------------------------------------------------------------


class KBHit:
    """Cross-platform keyboard input handler for non-blocking input."""

    def __init__(self):
        self.chars = ""
        self._line_accum = ""
        if os.name == "nt":
            pass
        else:
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)
            self.new_term[3] = self.new_term[3] & ~termios.ICANON & ~termios.ECHO
            self.set_normal_term()

    def set_normal_term(self):
        if os.name != "nt":
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def set_curses_term(self):
        if os.name != "nt":
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

    def check(self):
        if os.name == "nt":
            return msvcrt.kbhit()
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def getch(self):
        if os.name == "nt":
            return msvcrt.getwch()
        return sys.stdin.read(1)

    def getarrow(self):
        c1 = self.getch()
        if c1 == "\x1b":
            c2 = self.getch()
            c3 = self.getch()
            return c3
        return None

    def get_input(self):
        if os.name == "nt":
            while self.check():
                char = self.getch()
                if char in ("\r", "\n"):
                    res = self.chars.strip()
                    self.chars = ""
                    return res
                self.chars += char
                print("Current Input: ", self.chars)
            return None
        if self.check():
            while self.check():
                char = self.getch()
                if char in ("\n", "\r"):
                    res = self.chars.strip()
                    self.chars = ""
                    return res
                self.chars += char
                print("Current Input: ", self.chars)
        return None

    def clear_line_buffer(self) -> None:
        self._line_accum = ""

    def get_line(self, echo: bool = True):
        # Backspace (often ^H) and DEL (127); erase echoed char with \b \b
        _bs = ("\x08", "\x7f")

        def _apply_backspace() -> None:
            if not self._line_accum:
                return
            self._line_accum = self._line_accum[:-1]
            if echo:
                print("\b \b", end="", flush=True)

        if os.name == "nt":
            while self.check():
                char = self.getch()
                if char in ("\r", "\n"):
                    res = self._line_accum.strip()
                    self._line_accum = ""
                    return res
                if char in _bs:
                    _apply_backspace()
                    continue
                if echo:
                    print(char, end="", flush=True)
                self._line_accum += char
            return None
        while self.check():
            char = self.getch()
            if char in ("\n", "\r"):
                res = self._line_accum.strip()
                self._line_accum = ""
                return res
            if char in _bs:
                _apply_backspace()
                continue
            if echo:
                print(char, end="", flush=True)
            self._line_accum += char
        return None


# ---------------------------------------------------------------------------
# eval_real: pause menu (keyboard workflow in main control loop)
# ---------------------------------------------------------------------------

EvalRealPauseMenuStep = Literal["idle", "resumed", "exit", "still_paused"]

EVAL_REAL_PAUSE_MENU_WITH_RECORDER = (
    "\n"
    "========== PAUSED ==========\n"
    "  (Dataset / session video recording is paused.)\n"
    "  >>>  Type a command and Enter:\n"
    "  s / save+Enter — save current episode only (does NOT open/delete the next file)\n"
    "  r / reset+Enter — reset policy + go_home; then:\n"
    "                   • unsaved frames → discard & re-record SAME episode\n"
    "                   • after save → open next FREE episode (skip existing MP4s)\n"
    "  Enter only — resume into the currently open episode (after s: press r first)\n"
    "  quit / q+Enter — quit (discard unsaved episode if any, then exit)\n"
    "============================\n"
)

EVAL_REAL_PAUSE_MENU_NO_RECORDER = (
    "\n"
    "========== PAUSED ==========\n"
    "  (No session recorder — pass -o or unset EVAL_REAL_NO_RECORD to enable save.)\n"
    "  >>>  Type a command and Enter:\n"
    "  r / reset+Enter — reset policy and go home (stay paused)\n"
    "  Enter only — resume inference\n"
    "  quit / q+Enter — quit\n"
    "============================\n"
)


def eval_real_pause_menu_text(rt: EvalRealRuntime) -> str:
    return (
        EVAL_REAL_PAUSE_MENU_WITH_RECORDER
        if rt.has_recorder
        else EVAL_REAL_PAUSE_MENU_NO_RECORDER
    )


# Backward-compatible alias
EVAL_REAL_PAUSE_MENU = EVAL_REAL_PAUSE_MENU_WITH_RECORDER


def _normalize_eval_real_pause_command(raw: str) -> str:
    cmd = raw.strip().lower()
    if cmd == "save":
        return "s"
    if cmd == "reset":
        return "r"
    if cmd in ("q", "quit"):
        return "quit_menu"
    return cmd


def try_enter_eval_real_pause_menu(kb: KBHit, rt: EvalRealRuntime) -> bool:
    """Enter pause+menu if the user completed a line (e.g. Enter) while the loop was running."""
    if kb.get_input() is None:
        return False
    while kb.get_input() is not None:
        pass
    kb.clear_line_buffer()
    rt.send_recorder_cmd("pause_recording")
    print(eval_real_pause_menu_text(rt), end="", flush=True)
    print(">>> ", end="", flush=True)
    return True


def eval_real_pause_menu_step(
    kb: KBHit,
    rt: EvalRealRuntime,
    action_manager: Any,
    on_reset: Optional[Callable[[], None]] = None,
) -> EvalRealPauseMenuStep:
    """Handle one non-blocking read while paused; see :data:`EVAL_REAL_PAUSE_MENU`."""
    line = kb.get_line()
    if line is None:
        return "idle"

    cmd = _normalize_eval_real_pause_command(line)

    if cmd == "s":
        if not rt.has_recorder:
            print(
                "\nSave unavailable: no session recorder. "
                "Restart with -o OUTPUT_DIR (or unset EVAL_REAL_NO_RECORD).",
                flush=True,
            )
            print(">>> ", end="", flush=True)
            return "still_paused"
        print("\nSaving...", flush=True)
        ack = rt.send_recorder_cmd("save_episode")
        if ack and isinstance(ack, tuple) and ack and ack[0] == "saved":
            n_frames = int(ack[1]) if len(ack) > 1 else 0
            ep_idx = ack[2] if len(ack) > 2 else None
            ep_txt = f" episode {ep_idx}" if ep_idx is not None else ""
            if n_frames > 0:
                print(
                    f"Successfully Saved{ep_txt} ({n_frames} frames). "
                    "Press r to open the next episode (will not touch existing MP4s).",
                    flush=True,
                )
            else:
                print(
                    f"Saved{ep_txt} (0 frames — episode was empty).",
                    flush=True,
                )
            logger.info(
                "[Pause] Saved episode {}: {} frames; waiting for r to open next.",
                ep_idx,
                n_frames,
            )
        else:
            print("Save failed (recorder did not respond).", flush=True)
        print(">>> ", end="", flush=True)
        return "still_paused"

    if cmd == "r":
        print("\nResetting...", flush=True)
        ack = rt.send_recorder_cmd("discard_episode")
        try:
            action_manager.reset()
            if on_reset is not None:
                on_reset()
        except Exception as e:
            logger.exception("[Pause] Reset failed: {}", e)
            print(f"Reset failed: {e}", flush=True)
            print(">>> ", end="", flush=True)
            return "still_paused"
        n_disc = 0
        ep_idx = None
        if ack and isinstance(ack, tuple) and ack and ack[0] == "discarded":
            n_disc = int(ack[1]) if len(ack) > 1 else 0
            ep_idx = ack[2] if len(ack) > 2 else None
        if n_disc > 0:
            ep_txt = f" episode {ep_idx}" if ep_idx is not None else ""
            print(
                f"Successfully reset; discarded unsaved{ep_txt} "
                f"({n_disc} frames) — re-record the same episode. Enter=resume.",
                flush=True,
            )
            logger.info(
                "[Pause] Reset + discarded episode {} ({} frames); same index reopened.",
                ep_idx,
                n_disc,
            )
        else:
            ep_txt = f" episode {ep_idx}" if ep_idx is not None else ""
            print(
                f"Successfully reset (policy + go_home). "
                f"Next open{ep_txt}. Enter=resume.",
                flush=True,
            )
            logger.info(
                "[Pause] Reset; opened next episode {} (no discard).",
                ep_idx,
            )
        print(">>> ", end="", flush=True)
        return "still_paused"

    if cmd == "":
        ack = rt.send_recorder_cmd("resume_recording")
        ep_open = None
        if ack and isinstance(ack, tuple) and len(ack) > 2:
            ep_open = ack[2]
        if ep_open is None and rt.has_recorder:
            print(
                "[Control loop] No episode open after save — press r to open next, "
                "then Enter to resume.\n",
                flush=True,
            )
            print(">>> ", end="", flush=True)
            return "still_paused"
        print("[Control loop] Resumed.\n", flush=True)
        kb.clear_line_buffer()
        return "resumed"

    if cmd == "quit_menu":
        print("\nExiting (discarding any unsaved episode)...", flush=True)
        rt.send_recorder_cmd("exit_discard")
        logger.info("[Pause] exit from menu; unsaved episode discarded if present.")
        return "exit"

    print(
        f"\nUnknown command: {cmd!r}. Use s/save, r/reset, quit/q, or Enter alone.\n"
        ">>> ",
        end="",
        flush=True,
    )
    return "still_paused"
