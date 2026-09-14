"""
Helpers to build replay trajectories from a task YAML + LeRobot v2.1 dataset (first dataset entry).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from vlastudio import configs  # noqa: F401

from vlastudio.configs.loader import ConfigLoader
from loguru import logger


def parse_replay_key(replay_key: str) -> Tuple[str, int]:
    """
    Map config string to (source_name, row_index).

    source_name is currently only ``action`` (row within sample['action'] of shape (T, D)).

    Examples:
      - ``action_first``, ``first``, ``action:0``, ``action_0`` -> (action, 0)
      - ``action:3``, ``action_3`` -> (action, 3)
    """
    rk = (replay_key or "action_first").strip().lower()
    if rk in ("action_first", "first", "action:0", "action_0"):
        return "action", 0
    if rk.startswith("action:"):
        return "action", int(rk.split(":", 1)[1])
    if rk.startswith("action_") and rk != "action_first":
        return "action", int(rk.split("_", 1)[1])
    raise ValueError(
        f"Unknown replay_key={replay_key!r}. Use e.g. action_first, action:0, action:3."
    )


def episode_frame_indices(ds, target_episode_id: int) -> List[int]:
    pairs: List[Tuple[int, int]] = []
    n = len(ds)
    for i in range(n):
        di, ei, fo = ds.index_to_sample_map[i]
        egid = int(ds.per_dataset_episode_start[di]) + ds.per_dataset_episodes[di].index(ei)
        if egid == int(target_episode_id):
            pairs.append((fo, i))
    pairs.sort(key=lambda x: x[0])
    return [idx for _, idx in pairs]


def extract_per_frame_values(
    ds,
    frame_indices: Sequence[int],
    *,
    source: str,
    row: int,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for i in frame_indices:
        sample = ds[int(i)]
        if source != "action":
            raise ValueError(f"Unsupported replay source {source!r}")
        act = sample["action"]
        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()
        act = np.asarray(act, dtype=np.float32)
        if act.ndim != 2:
            raise ValueError(f"Expected sample['action'] (T,D), got {act.shape} at index {i}")
        r = int(row)
        if r < 0 or r >= act.shape[0]:
            raise ValueError(f"replay row={r} out of range for action shape {act.shape}")
        rows.append(act[r].copy())
    return np.stack(rows, axis=0)


def episode_action_trajectory(
    ds,
    target_episode_id: int,
    *,
    source: str,
    row: int,
) -> np.ndarray:
    """Per-frame action row for one episode, shape (T, D)."""
    frame_indices = episode_frame_indices(ds, target_episode_id)
    if not frame_indices:
        raise ValueError(
            f"No frames for episode_id={target_episode_id} "
            f"(num_episodes={getattr(ds, 'num_episodes', '?')})."
        )
    return extract_per_frame_values(ds, frame_indices, source=source, row=row)


def insert_linf_transition_bridge(
    a_end: np.ndarray,
    b_start: np.ndarray,
    *,
    linf_thresh: float,
    min_insert: int,
    max_insert: int,
) -> np.ndarray:
    """
    Linearly interpolate between the last action of episode A and the first of B.

    If L∞ distance is at most ``linf_thresh``, returns shape (0, D). When ``linf_thresh <= 0``,
    bridging is disabled (always empty).
    """
    if linf_thresh <= 0:
        return np.zeros((0, a_end.shape[0]), dtype=np.float32)
    a = np.asarray(a_end, dtype=np.float64).ravel()
    b = np.asarray(b_start, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"a_end/b_start dim mismatch {a.shape} vs {b.shape}")
    gap = float(np.max(np.abs(b - a)))
    if gap <= linf_thresh:
        return np.zeros((0, a_end.shape[0]), dtype=np.float32)
    n = int(np.ceil(gap / linf_thresh)) - 1
    n = max(int(min_insert), min(int(max_insert), max(1, n)))
    t = np.linspace(0.0, 1.0, num=n + 2, dtype=np.float64)[1:-1]
    out = (1.0 - t[:, None]) * a[None, :] + t[:, None] * b[None, :]
    return out.astype(np.float32, copy=False)


def merge_episode_action_trajectories(
    ds,
    episode_ids: Sequence[int],
    *,
    source: str,
    row: int,
    linf_thresh: float,
    min_insert: int,
    max_insert: int,
) -> np.ndarray:
    """Concatenate episodes in order; optional L∞-threshold linear bridges at boundaries."""
    eids = [int(x) for x in episode_ids]
    if not eids:
        raise ValueError("episode_ids is empty")
    parts: List[np.ndarray] = []
    for i, eid in enumerate(eids):
        traj = episode_action_trajectory(ds, eid, source=source, row=row)
        if i == 0:
            parts.append(traj)
            continue
        bridge = insert_linf_transition_bridge(
            parts[-1][-1],
            traj[0],
            linf_thresh=linf_thresh,
            min_insert=min_insert,
            max_insert=max_insert,
        )
        if bridge.shape[0] > 0:
            parts.append(bridge)
        parts.append(traj)
    return np.concatenate(parts, axis=0)


def append_loop_closure_bridge(
    actions_hd: np.ndarray,
    *,
    enabled: bool,
    linf_thresh: float,
    min_insert: int,
    max_insert: int,
) -> np.ndarray:
    """
    Append linear bridge from last frame toward first frame so chunk playback can ``loop``
    without a discrete jump from trajectory end back to chunk 0.

    If ``linf_thresh <= 0``, uses an effective threshold of ``gap / max_insert`` so that
    ``--loop`` still smooths the wrap when episode-transition bridging was disabled.
    """
    if not enabled or actions_hd.shape[0] < 2:
        return actions_hd
    a_end = actions_hd[-1]
    b_start = actions_hd[0]
    gap = float(
        np.max(np.abs(b_start.astype(np.float64) - a_end.astype(np.float64)))
    )
    if gap <= 1e-7:
        return actions_hd
    eff = float(linf_thresh)
    if eff <= 0:
        eff = max(gap / float(max_insert), 1e-12)
    bridge = insert_linf_transition_bridge(
        a_end,
        b_start,
        linf_thresh=eff,
        min_insert=min_insert,
        max_insert=max_insert,
    )
    if bridge.shape[0] == 0:
        return actions_hd
    return np.concatenate([actions_hd, bridge], axis=0)


def split_fixed_chunks(actions_hd: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    t, d = actions_hd.shape
    chunks: List[np.ndarray] = []
    for start in range(0, t, chunk_size):
        seg = actions_hd[start : start + chunk_size]
        if seg.shape[0] < chunk_size:
            pad = chunk_size - seg.shape[0]
            last = seg[-1:] if seg.shape[0] > 0 else np.zeros((1, d), dtype=np.float32)
            seg = np.concatenate([seg, np.repeat(last, pad, axis=0)], axis=0)
        chunks.append(seg.astype(np.float32, copy=False))
    return chunks


def load_raw_dataset_for_task(task: str, unknown_args: Optional[List[str]] = None):
    cfg_loader = ConfigLoader(args=SimpleNamespace(), unknown_args=unknown_args or [])
    task_config, _ = cfg_loader.load_task(str(task))
    if not task_config.get("datasets"):
        raise ValueError(f"Task {task!r} has no datasets[]")
    data_cfg = task_config["datasets"][0]
    ds_args = SimpleNamespace(is_training=False, device="cpu", dataset_id="", chunk_size=-1)
    from vlastudio.data_utils.utils import _create_dataset_from_config

    return _create_dataset_from_config(data_cfg, ds_args), task_config


def build_chunk_arrays_from_episodes(
    task: str,
    episode_ids: Sequence[int],
    chunk_size: int,
    replay_key: str,
    *,
    max_chunks: Optional[int] = None,
    unknown_args: Optional[List[str]] = None,
    transition_linf_thresh: float = 0.0,
    transition_min_insert: int = 4,
    transition_max_insert: int = 48,
    loop_closure: bool = False,
) -> Tuple[List[np.ndarray], str, str]:
    """
    Merge multiple episodes (in order) into one timeline, optionally inserting linear
    bridge actions between episodes when L∞ jump exceeds ``transition_linf_thresh``.

    If ``loop_closure`` is True, appends a last→first bridge before chunking so
    ``ReplayPolicy`` with ``loop`` does not jump from the demo end back to the start.

    Returns (list of (chunk_size, D) float32 arrays, ctrl_space, ctrl_type).
    """
    logger.info(
        "Replay: building trajectories from **raw** dataset samples; "
        "task YAML ``action_normalize`` / ``state_normalize`` are not applied here. "
        "Pseudo-ckpt uses identity normalizers at inference."
    )
    raw_ds, _task_cfg = load_raw_dataset_for_task(task, unknown_args=unknown_args)
    src, row = parse_replay_key(replay_key)
    ctrl_space = getattr(raw_ds, "ctrl_space", "joint")
    ctrl_type = getattr(raw_ds, "ctrl_type", "abs")

    actions_hd = merge_episode_action_trajectories(
        raw_ds,
        episode_ids,
        source=src,
        row=row,
        linf_thresh=float(transition_linf_thresh),
        min_insert=int(transition_min_insert),
        max_insert=int(transition_max_insert),
    )
    actions_hd = append_loop_closure_bridge(
        actions_hd,
        enabled=bool(loop_closure),
        linf_thresh=float(transition_linf_thresh),
        min_insert=int(transition_min_insert),
        max_insert=int(transition_max_insert),
    )
    chunks = split_fixed_chunks(actions_hd, chunk_size)
    if max_chunks is not None:
        chunks = chunks[: int(max_chunks)]
    return chunks, str(ctrl_space), str(ctrl_type)


def build_chunk_arrays_from_task(
    task: str,
    episode_id: int,
    chunk_size: int,
    replay_key: str,
    *,
    max_chunks: Optional[int] = None,
    unknown_args: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], str, str]:
    """Single-episode convenience wrapper around :func:`build_chunk_arrays_from_episodes`."""
    return build_chunk_arrays_from_episodes(
        task,
        [int(episode_id)],
        chunk_size,
        replay_key,
        max_chunks=max_chunks,
        unknown_args=unknown_args,
        transition_linf_thresh=0.0,
        loop_closure=False,
    )


def list_episode_ids(task: str, unknown_args: Optional[List[str]] = None) -> List[int]:
    raw_ds, _ = load_raw_dataset_for_task(task, unknown_args=unknown_args)
    seen = set()
    out: List[int] = []
    for i in range(len(raw_ds)):
        di, ei, _ = raw_ds.index_to_sample_map[i]
        egid = int(raw_ds.per_dataset_episode_start[di]) + raw_ds.per_dataset_episodes[di].index(ei)
        if egid not in seen:
            seen.add(egid)
            out.append(egid)
    out.sort()
    return out
