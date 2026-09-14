"""
Replay policy: ``select_action`` ignores observations and returns pre-baked chunks from disk.

运行方式、Task 命名（``local.t0325`` 点号形式）见同目录 **README.md**。

伪目录由 ``scripts/generate_replay_checkpoint.py`` 生成，包含：

- ``policy_metadata.json`` (``policy_module``: ``policy.replay``)
- ``normalize.json`` + stats ``.pkl`` (copied from a reference trained ckpt)
- ``config.json`` (copied from reference, for ``direct_loader`` / chunk_size)
- ``replay_spec.json`` (loop, on_exhausted, ctrl_space/type)
- ``replay_chunks.npz`` — ``chunks`` shape ``(num_chunks, chunk_size, action_dim)`` in **raw**
  dataset action space; ``normalize.json`` uses **identity** normalizers (no training stats).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

import configs  # noqa: F401


def _checkpoint_root(model_name_or_path: str) -> Path:
    p = Path(model_name_or_path).expanduser().resolve()
    if p.name.startswith("checkpoint-"):
        return p.parent
    return p


def _load_ctrl_from_normalize_json(ckpt_dir: Path) -> tuple[str, str]:
    path = ckpt_dir / "normalize.json"
    if not path.is_file():
        return "joint", "abs"
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ds = (meta.get("datasets") or [{}])[0]
    return str(ds.get("ctrl_space", "joint")), str(ds.get("ctrl_type", "abs"))


class ReplayPolicy(nn.Module):
    """
    Each ``select_action`` emits one chunk ``(B, chunk_size, action_dim)`` in **normalized**
    action space (matches reference policy before denorm).
    """

    def __init__(
        self,
        chunks: List[np.ndarray],
        *,
        ctrl_space: str,
        ctrl_type: str,
        loop: bool,
        on_exhausted: str,
    ):
        super().__init__()
        if not chunks:
            raise ValueError("ReplayPolicy: empty chunks")
        self._chunks = [c.astype(np.float32, copy=False) for c in chunks]
        self.ctrl_space = str(ctrl_space)
        self.ctrl_type = str(ctrl_type)
        self.chunk_size = int(self._chunks[0].shape[0])
        self.action_dim = int(self._chunks[0].shape[1])
        for i, c in enumerate(self._chunks):
            if c.shape != (self.chunk_size, self.action_dim):
                raise ValueError(f"Chunk {i} has shape {c.shape}, expected ({self.chunk_size}, {self.action_dim})")

        self._loop = bool(loop)
        self._on_exhausted = str(on_exhausted or "repeat_last").lower()
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0
        logger.debug("ReplayPolicy: reset chunk index → 0")

    def _emit_chunk_array(self) -> np.ndarray:
        if self._idx < len(self._chunks):
            out = self._chunks[self._idx]
            self._idx += 1
            return out
        if self._loop:
            self._idx = 0
            return self._emit_chunk_array()
        if self._on_exhausted == "repeat_last":
            return self._chunks[-1]
        if self._on_exhausted in ("hold", "hold_last"):
            # Pad with the final frame only — avoids replaying a moving chunk and
            # re-solving IK from a drifted pose (major source of EE-replay stutter).
            last = np.asarray(self._chunks[-1][-1], dtype=np.float32).reshape(1, -1)
            return np.repeat(last, self.chunk_size, axis=0)
        if self._on_exhausted == "zero":
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        raise RuntimeError(
            f"ReplayPolicy exhausted ({len(self._chunks)} chunks); loop=false on_exhausted={self._on_exhausted!r}"
        )

    def select_action(self, obs: Any) -> torch.Tensor:
        dev = torch.device("cpu")
        if isinstance(obs, dict):
            for k in ("state", "image"):
                v = obs.get(k)
                if isinstance(v, torch.Tensor):
                    dev = v.device
                    break
        arr = self._emit_chunk_array()
        return torch.from_numpy(arr).unsqueeze(0).float().to(dev)


def load_model(args) -> Dict[str, Any]:
    """
    Load replay chunks from ``<ckpt>/replay_chunks.npz`` + ``replay_spec.json``.
    """
    ckpt = _checkpoint_root(str(args.model_name_or_path))
    spec_path = ckpt / "replay_spec.json"
    npz_path = ckpt / "replay_chunks.npz"
    if not spec_path.is_file():
        raise FileNotFoundError(f"Replay checkpoint missing replay_spec.json: {spec_path}")
    if not npz_path.is_file():
        raise FileNotFoundError(f"Replay checkpoint missing replay_chunks.npz: {npz_path}")

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    z = np.load(npz_path)
    if "chunks" not in z.files:
        raise KeyError("replay_chunks.npz must contain array 'chunks'")
    stacked = np.asarray(z["chunks"], dtype=np.float32)
    if stacked.ndim != 3:
        raise ValueError(f"chunks must be (N, H, D), got shape {stacked.shape}")
    chunks = [stacked[i].copy() for i in range(stacked.shape[0])]

    d_cs, d_ct = _load_ctrl_from_normalize_json(ckpt)
    ctrl_space = str(spec.get("ctrl_space", d_cs))
    ctrl_type = str(spec.get("ctrl_type", d_ct))
    # Default loop=True: if loop is false and chunks exhaust, on_exhausted repeat_last
    # would replay only the *last* chunk forever (not full demo), which is usually wrong
    # for dataset replay on hardware.
    loop = bool(spec["loop"]) if "loop" in spec else True
    on_exhausted = str(spec.get("on_exhausted", "repeat_last"))

    model = ReplayPolicy(
        chunks,
        ctrl_space=ctrl_space,
        ctrl_type=ctrl_type,
        loop=loop,
        on_exhausted=on_exhausted,
    )
    logger.info(
        "ReplayPolicy from ckpt {} | {} chunks × ({}, {}) loop={}",
        ckpt,
        len(chunks),
        model.chunk_size,
        model.action_dim,
        loop,
    )
    model.data_processor = None
    model.data_collator = None
    return {"model": model, "config": None}
