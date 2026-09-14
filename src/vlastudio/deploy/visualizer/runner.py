"""
Subprocess entry for visualizer YAML rows (``type`` → class or callable).

``type`` is a full class path (``module.Class``) with ``start()`` (typical for
``BaseVisualizer`` subclasses), or a module-level callable invoked as ``callable(**kwargs)``.
"""

from __future__ import annotations

import importlib
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# kwargs preparation (YAML list-of-lists → tuples for shm_entries)
# ---------------------------------------------------------------------------


def _prepare_init_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(kwargs)
    if (out.get("expected_dim") is None) and (out.get("dim") is not None):
        try:
            out["expected_dim"] = int(float(out.pop("dim")))
        except (TypeError, ValueError):
            pass
    se = out.get("shm_entries")
    if se is not None:
        parsed: List[Tuple[str, Optional[str]]] = []
        for item in se:
            if isinstance(item, (list, tuple)):
                shm = str(item[0])
                key = None
                if len(item) > 1 and item[1] is not None and str(item[1]).strip() != "":
                    key = str(item[1])
                parsed.append((shm, key))
            else:
                parsed.append((str(item), None))
        out["shm_entries"] = parsed
    return out


def build_visualizer_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    args = dict(cfg.get("args") or {})
    name = cfg.get("name")
    vtype = str(cfg.get("type", ""))
    lt = vtype.lower()
    if name is not None and str(name).strip() != "":
        n = str(name)
        if "cameravisualizer" in lt or "camera_visualizer" in lt or "start_camera_visualizer" in lt:
            args.setdefault("window_name", n)
        elif "lowdim" in lt:
            args.setdefault("group_id", n)
        else:
            args.setdefault("shm_name", n)
    return _prepare_init_kwargs(args)


def run_visualizer(type_str: str, kwargs: Dict[str, Any]) -> None:
    """``multiprocessing.Process`` target (spawn-safe)."""
    kwargs = _prepare_init_kwargs(dict(kwargs))
    parts = str(type_str).rsplit(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        print(f"[run_visualizer] Invalid type {type_str!r}", flush=True)
        return
    mod_name, attr_name = parts
    try:
        mod = importlib.import_module(mod_name)
        target = getattr(mod, attr_name)
    except Exception as e:
        print(f"[run_visualizer] Import {type_str!r} failed: {e}", flush=True)
        traceback.print_exc()
        return
    try:
        if isinstance(target, type):
            inst = target(**kwargs)
            start = getattr(inst, "start", None)
            if callable(start):
                start()
            else:
                print(f"[run_visualizer] {type_str} has no start()", flush=True)
        elif callable(target):
            target(**kwargs)
        else:
            print(f"[run_visualizer] {type_str} is not a class or callable", flush=True)
    except Exception as e:
        print(f"[run_visualizer] Run failed: {e}", flush=True)
        traceback.print_exc()


def start_visualizer_processes(configs: List[Dict[str, Any]]) -> List[Any]:
    """Spawn one subprocess per visualizer config."""
    import multiprocessing as mp

    from loguru import logger

    if os.name != "nt":
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")
    else:
        ctx = mp.get_context("spawn")
    procs = []
    for cfg in configs:
        vtype = cfg.get("type")
        if not vtype:
            logger.error("Visualizer entry missing 'type': {}", cfg)
            continue
        kwargs = build_visualizer_kwargs(cfg)
        logger.info(
            "Starting visualizer type={} name={!r}",
            vtype,
            cfg.get("name"),
        )
        p = ctx.Process(
            target=run_visualizer,
            args=(str(vtype), kwargs),
            daemon=False,
        )
        p.start()
        procs.append(p)
    return procs
