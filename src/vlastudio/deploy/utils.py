import importlib
import multiprocessing as mp
import os
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import yaml
from loguru import logger


def _safe_name(s: str) -> str:
    """Sanitize a string for filesystem paths."""
    if s is None:
        return "none"
    s = str(s)
    for ch in ["/", "\\", " ", ":", ";", "|", "\t", "\n", "\r"]:
        s = s.replace(ch, "_")
    return s


def _looks_image_array(v: np.ndarray) -> bool:
    if not isinstance(v, np.ndarray):
        return False
    if v.ndim == 2:
        return True
    if v.ndim == 3 and (v.shape[-1] in (1, 3, 4) or v.shape[0] in (1, 3, 4)):
        return True
    if v.ndim == 4 and (v.shape[-1] in (1, 3, 4) or v.shape[1] in (1, 3, 4)):
        return True
    return False


def _as_uint8_image(arr: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert a numpy array to an image suitable for cv2.imwrite.
    Returns HxW, HxWx1, HxWx3, or HxWx4 numpy array, or None if not image-like.
    """
    if not isinstance(arr, np.ndarray):
        return None

    x = arr

    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = np.transpose(x, (1, 2, 0))
    elif x.ndim == 4:
        return None

    if x.ndim == 2:
        pass
    elif x.ndim == 3 and x.shape[2] in (1, 3, 4):
        pass
    else:
        return None

    if np.issubdtype(x.dtype, np.floating):
        x_max = float(np.nanmax(x)) if x.size else 0.0
        if x_max <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    elif x.dtype == np.uint8:
        pass
    elif np.issubdtype(x.dtype, np.integer):
        if x.dtype == np.uint16:
            return x
        x = np.clip(x, 0, 255).astype(np.uint8)
    else:
        x = x.astype(np.uint8, copy=False)

    return x


def _save_images_from_value(
    *,
    value,
    out_dir: str,
    frame_idx: int,
    device_name: str,
    key_path: str,
    color_order: str = "rgb",
) -> int:
    """
    Save image(s) from ``value`` (np.ndarray or nested dict with ``image`` / image-like arrays).
    Recurses into dict values so arbitrary nesting (e.g. ``top_camera`` -> ``image``) works
    without hardcoded camera names.
    """
    import cv2

    saved = 0
    dev = _safe_name(device_name)
    key = _safe_name(key_path)

    if isinstance(value, dict):
        if "image" in value and isinstance(value["image"], np.ndarray):
            return _save_images_from_value(
                value=value["image"],
                out_dir=out_dir,
                frame_idx=frame_idx,
                device_name=device_name,
                key_path=f"{key_path}.image" if key_path else "image",
                color_order=color_order,
            )
        skip = frozenset({"timestamp", "__timestamp__", "t", "time"})
        for sk, sv in value.items():
            if sk in skip:
                continue
            sub_path = f"{key_path}.{sk}" if key_path else str(sk)
            if isinstance(sv, dict):
                saved += _save_images_from_value(
                    value=sv,
                    out_dir=out_dir,
                    frame_idx=frame_idx,
                    device_name=device_name,
                    key_path=sub_path,
                    color_order=color_order,
                )
            elif isinstance(sv, np.ndarray) and _looks_image_array(sv):
                saved += _save_images_from_value(
                    value=sv,
                    out_dir=out_dir,
                    frame_idx=frame_idx,
                    device_name=device_name,
                    key_path=sub_path,
                    color_order=color_order,
                )
        return saved

    if not isinstance(value, np.ndarray):
        return 0

    if value.ndim == 4:
        if value.shape[-1] in (1, 3, 4):
            for b in range(value.shape[0]):
                img = _as_uint8_image(value[b])
                if img is None:
                    continue
                if color_order == "rgb" and img.ndim == 3 and img.shape[2] in (3, 4):
                    if img.shape[2] == 3:
                        img = img[..., ::-1]
                    else:
                        img = img[..., [2, 1, 0, 3]]
                fp = os.path.join(out_dir, f"{frame_idx:06d}_{dev}_{key}_b{b:02d}.png")
                if cv2.imwrite(fp, img):
                    saved += 1
            return saved
        if value.shape[1] in (1, 3, 4):
            for b in range(value.shape[0]):
                img = _as_uint8_image(value[b])
                if img is None:
                    continue
                if color_order == "rgb" and img.ndim == 3 and img.shape[2] in (3, 4):
                    if img.shape[2] == 3:
                        img = img[..., ::-1]
                    else:
                        img = img[..., [2, 1, 0, 3]]
                fp = os.path.join(out_dir, f"{frame_idx:06d}_{dev}_{key}_b{b:02d}.png")
                if cv2.imwrite(fp, img):
                    saved += 1
            return saved
        return 0

    img = _as_uint8_image(value)
    if img is None:
        return 0

    if color_order == "rgb" and img.ndim == 3 and img.shape[2] in (3, 4):
        if img.shape[2] == 3:
            img = img[..., ::-1]
        else:
            img = img[..., [2, 1, 0, 3]]

    fp = os.path.join(out_dir, f"{frame_idx:06d}_{dev}_{key}.png")
    if cv2.imwrite(fp, img):
        saved += 1
    return saved


def save_synced_frame_images(
    *,
    synced_data: dict,
    out_dir: str,
    frame_idx: int,
    color_order: str = "rgb",
) -> int:
    """
    Save all image-like arrays inside a synced_data frame to ``out_dir``.
    Skips ``__timestamp__`` / metadata-only top-level keys. Uses heuristics on key names
    and array shapes; nested camera dicts are handled by recursion inside
    :func:`_save_images_from_value`.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = 0

    for dev_name, dev_data in (synced_data or {}).items():
        if dev_name.startswith("__"):
            continue
        if not isinstance(dev_data, dict):
            continue

        for k, v in dev_data.items():
            if str(k).startswith("__"):
                continue
            k_lower = str(k).lower()
            likely_image = any(
                token in k_lower
                for token in ("image", "rgb", "color", "frame", "depth", "camera")
            )
            if isinstance(v, dict):
                saved += _save_images_from_value(
                    value=v,
                    out_dir=out_dir,
                    frame_idx=frame_idx,
                    device_name=str(dev_name),
                    key_path=str(k),
                    color_order=color_order,
                )
            elif isinstance(v, np.ndarray):
                if likely_image or _looks_image_array(v):
                    saved += _save_images_from_value(
                        value=v,
                        out_dir=out_dir,
                        frame_idx=frame_idx,
                        device_name=str(dev_name),
                        key_path=str(k),
                        color_order=color_order,
                    )

    return saved


def finalize_deploy_device_dict(item: dict) -> dict:
    """Normalize one deploy YAML entry (robot/teleop/viz row) and flatten ``args`` like ConfigLoader.load_robot."""
    from vlastudio.configs.loader import ConfigLoader

    if not isinstance(item, dict):
        return item
    cfg = ConfigLoader.normalize_config(item)
    if "args" in cfg and isinstance(cfg["args"], dict):
        for key, value in cfg["args"].items():
            if key not in cfg:
                cfg[key] = value
    if "type" in cfg:
        cfg["target"] = cfg["type"]
    return cfg


def coerce_finalize_deploy_list(cfg_or_list: Any) -> List:
    """If YAML root is a list, finalize each dict; if a single dict, wrap as one-element list."""
    if isinstance(cfg_or_list, list):
        return [finalize_deploy_device_dict(x) if isinstance(x, dict) else x for x in cfg_or_list]
    if isinstance(cfg_or_list, dict):
        return [finalize_deploy_device_dict(cfg_or_list)]
    return []


def is_visualizer_config(cfg: dict) -> bool:
    """True if ``type`` resolves to a class that subclasses ``BaseVisualizer`` (and is not the base itself)."""
    if not isinstance(cfg, dict):
        return False
    t = cfg.get("type")
    if not isinstance(t, str):
        return False
    t = t.strip()
    if "." not in t:
        return False
    mod_name, _, attr_name = t.rpartition(".")
    if not mod_name or not attr_name:
        return False
    try:
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr_name, None)
    except Exception:
        return False
    if not isinstance(obj, type):
        return False
    from vlastudio.deploy.visualizer.base import BaseVisualizer

    try:
        return issubclass(obj, BaseVisualizer) and obj is not BaseVisualizer
    except TypeError:
        return False


def partition_visualizer_configs(
    configs: List[dict],
) -> Tuple[List[dict], List[dict]]:
    """Split ``configs`` into (device entries, visualizer entries)."""
    devices: List[dict] = []
    viz: List[dict] = []
    for c in configs:
        if is_visualizer_config(c):
            viz.append(c)
        else:
            devices.append(c)
    return devices, viz


def load_config(path: str) -> List[dict]:
    """Load config from yaml. Returns list of device configs."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if isinstance(raw, list):
        return raw
    return [raw]


def _device_spawns_children(cfg: dict) -> bool:
    """True if this device type spawns child processes (e.g. GUI), so it must not run in a daemon process."""
    device_type = cfg.get("type", "")
    return "keyboard" in device_type or "tk_slider" in device_type


def start_devices(configs: List[dict], daemon: bool = True) -> List[mp.Process]:
    """Start each device config in a separate process.
    
    Args:
        configs: List of device configurations
        daemon: If True, processes will be terminated when main process exits
                (including crashes/segfaults). Default True for safety.
                Devices that spawn children (e.g. Keyboard, TkSlider GUI) are always
                started with daemon=False so they can create subprocesses.
    """
    from vlastudio.deploy.base import start_device

    procs = []
    for cfg in configs:
        p = mp.Process(target=start_device, args=(cfg,))
        # Daemon processes cannot have children; Keyboard/TkSlider start GUI subprocesses
        p.daemon = False if _device_spawns_children(cfg) else daemon
        p.start()
        procs.append(p)
    return procs


def get_shm_name(cfg: dict) -> str:
    """Get SHM name from device config."""
    args = cfg.get("args", {})
    return args.get("name") or args.get("robot_id") or cfg.get("name", "unknown")


def load_device_configs(
    args,
    unknown_args,
    *,
    robot_attr: str = "robot",
    force_robot_control_shm: Optional[str] = None,
    cfg_loader: Any = None,
) -> Tuple[List[dict], List[str], List[dict], List[dict]]:
    """
    Load device configs from args.

    Visualizer rows: ``type`` is a class subclassing ``BaseVisualizer`` (other than
    ``BaseVisualizer`` itself). Those rows are removed from the device list.

    Args:
        args: Namespace with robot path on attribute ``robot_attr`` (default ``robot``).
            Optional ``teleop`` string for collect-style runs (missing or empty → no teleop).
        unknown_args: Passed to :class:`configs.loader.ConfigLoader`.
        robot_attr: Attribute on ``args`` holding robot config name or YAML path
            (default ``robot``, used by collect_data and eval_real).
        force_robot_control_shm: If set, every robot device gets this ``control_shm_name``
            (e.g. ``policy_control_shm`` for policy eval). Applied after optional teleop
            auto-fill, so it overrides teleop-derived control SHM when both apply.
        cfg_loader: Optional pre-built ``ConfigLoader``; if None, one is created.

    Returns:
        Tuple of (all_device_configs, all_shm_names, teleop_device_configs, visualizer_configs)
    """
    from vlastudio.configs.loader import ConfigLoader

    from vlastudio.deploy.base import is_robot_config

    if cfg_loader is None:
        cfg_loader = ConfigLoader(args=args, unknown_args=unknown_args)

    robot_path_arg = getattr(args, robot_attr, None)
    if robot_path_arg is None:
        raise ValueError(f"args.{robot_attr} is required for load_device_configs")

    robot_configs: List[dict] = []
    teleop_configs: List[dict] = []

    # Load robot config
    try:
        robot_cfg, robot_path = cfg_loader.load_robot(robot_path_arg)
        robot_configs = coerce_finalize_deploy_list(robot_cfg)
        logger.info("Loaded robot config from: {}", robot_path)
    except FileNotFoundError as e:
        logger.warning("ConfigLoader failed, falling back to direct YAML: {}", e)
        robot_configs = coerce_finalize_deploy_list(load_config(robot_path_arg))

    # Load teleop config
    teleop_s = str(getattr(args, "teleop", "") or "").strip()
    if teleop_s:
        try:
            teleop_cfg, teleop_path = cfg_loader.load_teleop(teleop_s)
            teleop_configs = coerce_finalize_deploy_list(teleop_cfg)
            logger.info("Loaded teleop config from: {}", teleop_path)
        except FileNotFoundError as e:
            logger.warning("ConfigLoader failed, falling back to direct YAML: {}", e)
            teleop_configs = coerce_finalize_deploy_list(load_config(teleop_s))

    robot_devices, robot_viz = partition_visualizer_configs(robot_configs)
    teleop_devices, teleop_viz = partition_visualizer_configs(teleop_configs)
    visualizer_configs = teleop_viz + robot_viz

    # Auto-fill control_shm_name for robot configs if teleop is specified
    if teleop_devices:
        teleop_shm_name = get_shm_name(teleop_devices[0])
        for cfg in robot_devices:
            try:
                if is_robot_config(cfg):
                    args_dict = cfg.get("args", {})
                    if args_dict.get("control_shm_name") is None:
                        if "args" not in cfg:
                            cfg["args"] = {}
                        cfg["args"]["control_shm_name"] = teleop_shm_name
                        logger.info(
                            "Auto-set control_shm_name='{}' for robot '{}'",
                            teleop_shm_name,
                            get_shm_name(cfg),
                        )
            except Exception:
                pass

    if force_robot_control_shm:
        for cfg in robot_devices:
            try:
                if is_robot_config(cfg):
                    if "args" not in cfg:
                        cfg["args"] = {}
                    cfg["args"]["control_shm_name"] = force_robot_control_shm
                    logger.info(
                        "Set control_shm_name='{}' for robot '{}'",
                        force_robot_control_shm,
                        get_shm_name(cfg),
                    )
            except Exception:
                pass

    if teleop_devices:
        logger.info("Teleop devices: {}", [get_shm_name(c) for c in teleop_devices])
    logger.info("Robot devices: {}", [get_shm_name(c) for c in robot_devices])
    if visualizer_configs:
        logger.info(
            "Visualizer entries: {}",
            [v.get("name") or v.get("type") for v in visualizer_configs],
        )
    all_configs = teleop_devices + robot_devices
    all_shm_names = [get_shm_name(c) for c in all_configs]
    return all_configs, all_shm_names, teleop_devices, visualizer_configs


class RateLimiter:
    """
    A class to manage the rate for a single thread.
    Each thread should have its own instance.
    """

    def __init__(self):
        self._last_sleep_time = time.perf_counter()

    def sleep(self, rate: float):
        """
        Sleeps for a duration that maintains the desired loop rate.

        Args:
            rate (float): The desired loop frequency in Hz.
        """
        if rate <= 0:
            return

        target_period = 1.0 / rate
        current_time = time.perf_counter()
        elapsed_time = current_time - self._last_sleep_time
        sleep_duration = target_period - elapsed_time

        if sleep_duration > 0:
            time.sleep(sleep_duration)

        # Update the timestamp for the next iteration
        self._last_sleep_time = time.perf_counter()