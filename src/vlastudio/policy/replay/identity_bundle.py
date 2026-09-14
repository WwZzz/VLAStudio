"""
Identity normalization metadata + minimal ``config.json`` for replay pseudo-checkpoints.

Replay stores **raw** actions (same units as the dataset). Inference uses
``Identity`` state/action normalizers so no training statistics are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

# Single synthetic dataset id in normalize.json (load_normalizers uses first entry by default)
REPLAY_NORM_DATASET_ID = "replay"


def replay_normalize_meta(
    ctrl_space: str,
    ctrl_type: str,
    *,
    dataset_id: str = REPLAY_NORM_DATASET_ID,
) -> Dict[str, Any]:
    """Metadata dict suitable for ``normalize.json`` (identity state + action)."""
    return {
        "datasets": [
            {
                "dataset_id": dataset_id,
                "ctrl_space": str(ctrl_space),
                "ctrl_type": str(ctrl_type),
            }
        ],
        "state": {dataset_id: "identity"},
        "action": {dataset_id: "identity"},
    }


def write_replay_normalize_json(
    out_dir: Path,
    ctrl_space: str,
    ctrl_type: str,
    *,
    dataset_id: str = REPLAY_NORM_DATASET_ID,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = replay_normalize_meta(ctrl_space, ctrl_type, dataset_id=dataset_id)
    with open(out_dir / "normalize.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def write_replay_config_json(
    out_dir: Path,
    *,
    chunk_size: int,
    action_dim: int,
    state_dim: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Minimal ``config.json`` for ``direct_loader`` (chunk_size, dims). Users may edit
    ``chunk_size`` here for tooling that reads args from ckpt; it must match the
    time dimension of arrays in ``replay_chunks.npz`` after any regeneration.
    """
    out_dir = Path(out_dir)
    cfg: Dict[str, Any] = {
        "chunk_size": int(chunk_size),
        "action_dim": int(action_dim),
        "policy_kind": "replay",
    }
    if state_dim is not None:
        cfg["state_dim"] = int(state_dim)
    if extra:
        cfg.update(extra)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
