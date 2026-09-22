"""
Base Visualizer for Robot Data Visualization

This module provides the base class for visualizers that can display robot data
from shared memory. Each robot module can provide its own Visualizer class that
inherits from BaseVisualizer.

Usage:
    1. Create a Visualizer class in your robot module (e.g., deploy/robot/so101_sim/visualizer.py)
    2. Export it in the module's __init__.py
    3. vlastudio collect will automatically detect and start the visualizer as a subprocess
"""

import time
import importlib
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from vlastudio.deploy.shm_utils import SharedMemoryChannel


class BaseVisualizer(ABC):
    """
    Base class for robot visualizers.
    
    Visualizers read robot data from shared memory and display it.
    Each visualizer runs as a separate subprocess to avoid blocking the main process.
    """
    
    def __init__(self, shm_name: str, fps: float = 60.0, **kwargs):
        """
        Initialize the visualizer.
        
        Args:
            shm_name: Name of the shared memory to read robot data from
            fps: Target visualization frame rate (default: 60)
        """
        self.shm_name = shm_name
        self.fps = fps
        self.shm: Optional[SharedMemoryChannel] = None
        self.is_running = False

    def connect(self, timeout: float = 30.0) -> bool:
        """
        Connect to the robot's shared memory.
        
        Args:
            timeout: Maximum time to wait for SHM to be available
            
        Returns:
            True if connected successfully
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.shm = SharedMemoryChannel(self.shm_name, is_writer=False, timeout=1.0)
                print(f"[{self.__class__.__name__}] Connected to SHM: {self.shm_name}")
                return True
            except Exception as e:
                print(f"[{self.__class__.__name__}] Waiting for SHM '{self.shm_name}'... ({e})")
                time.sleep(0.5)
        
        print(f"[{self.__class__.__name__}] Failed to connect to SHM: {self.shm_name}")
        return False
    
    @abstractmethod
    def setup(self) -> bool:
        """
        Setup the visualization environment (e.g., create window, load model).
        Called once after connecting to SHM.
        
        Returns:
            True if setup successful
        """
        pass
    
    @abstractmethod
    def visualize(self, data: dict) -> bool:
        """
        Visualize one frame of data.
        
        Args:
            data: Data dictionary from shared memory (typically contains 'qpos', 'gpos', etc.)
            
        Returns:
            True to continue, False to stop
        """
        pass
    
    def cleanup(self):
        """
        Cleanup resources (e.g., close windows).
        Called when stopping.
        """
        pass

    def start(self):
        """
        Main loop: connect to SHM, setup, and continuously visualize data.
        """
        if not self.connect():
            return
        
        if not self.setup():
            print(f"[{self.__class__.__name__}] Setup failed")
            return
        
        self.is_running = True
        frame_interval = 1.0 / self.fps
        last_frame_time = 0.0
        
        print(f"[{self.__class__.__name__}] Visualization started at {self.fps} FPS")
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Rate limiting
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                
                last_frame_time = current_time
                
                # Read data from SHM
                data = self.shm.read(blocking=False, skip_unchanged=False)
                if data is not None:
                    if not self.visualize(data):
                        break
                        
        except KeyboardInterrupt:
            print(f"\n[{self.__class__.__name__}] Interrupted")
        finally:
            self.cleanup()
            if self.shm:
                try:
                    self.shm.destroy()
                except Exception:
                    pass
            print(f"[{self.__class__.__name__}] Stopped")
    
    def stop(self):
        """Stop the visualization loop."""
        self.is_running = False


def start_visualizer(visualizer_type: str, shm_name: str, **kwargs):
    """
    Start a visualizer by its type string.
    
    This function is called by vlastudio collect to start visualizers in subprocesses.
    
    Args:
        visualizer_type: Full module path to the Visualizer class
                        (e.g., "deploy.robot.so101_sim.Visualizer")
        shm_name: Name of the shared memory to visualize
        **kwargs: Additional arguments passed to the Visualizer constructor
    """
    try:
        # Import the visualizer class
        parts = visualizer_type.rsplit('.', 1)
        module_path = parts[0]
        class_name = parts[1] if len(parts) > 1 else "Visualizer"
        
        module = importlib.import_module(module_path)
        visualizer_class = getattr(module, class_name)
        
        # Create and start
        visualizer = visualizer_class(shm_name=shm_name, **kwargs)
        visualizer.start()
        
    except Exception as e:
        print(f"[start_visualizer] Error: {e}")
        import traceback
        traceback.print_exc()


def get_visualizer_class(device_type: str) -> Optional[type]:
    """
    Check if a device module has a Visualizer class.
    
    Args:
        device_type: Full module path to the device class
                    (e.g., "deploy.robot.so101_sim.So101SimRobot")
    
    Returns:
        The Visualizer class if found, None otherwise
    """
    try:
        # Get the module path (without the class name)
        parts = device_type.rsplit('.', 1)
        module_path = parts[0]
        
        # Try to import the module and check for Visualizer
        module = importlib.import_module(module_path)
        
        if hasattr(module, 'Visualizer'):
            visualizer_class = getattr(module, 'Visualizer')
            # Check if it's a class (could be BaseVisualizer subclass or CameraVisualizer)
            if isinstance(visualizer_class, type):
                return visualizer_class
        
        return None
        
    except Exception:
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _first_non_null_float(specs: List[Dict[str, Any]], field: str, default: float) -> float:
    for s in specs:
        v = s.get(field)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


def _first_non_null_int(specs: List[Dict[str, Any]], field: str, default: int) -> int:
    for s in specs:
        v = s.get(field)
        if v is not None:
            try:
                return int(float(v))
            except (TypeError, ValueError):
                pass
    return default


def get_visualizer_type_string(device_type: str) -> Optional[str]:
    """
    Get the full type string for a device's Visualizer class.
    
    Args:
        device_type: Full module path to the device class
        
    Returns:
        Full module path to Visualizer class, or None if not found
    """
    viz_class = get_visualizer_class(device_type)
    if viz_class is not None:
        parts = device_type.rsplit('.', 1)
        return parts[0] + ".Visualizer"
    return None


def _lowdim_share_group_id(device_shm_name: str, share_id: int) -> str:
    """
    ``share_id >= 0``: same id → one LowDimVisualizer (e.g. teleop + robot both ``id: 0``).
    ``share_id < 0`` (default when ``id`` omitted): solo, one process per device SHM.
    """
    if share_id >= 0:
        return f"lowdim_share_{share_id}"
    return f"lowdim_solo_{device_shm_name}"


def _parse_lowdim_visualizer_entry(cfg: dict) -> Optional[Tuple[int, Dict[str, Any]]]:
    """
    Top-level ``visualizer: {type: lowdim, id: ..., args: {...}}``.

    Returns:
        (share_id, merged_args_dict) or None if this device does not request lowdim viz.
    """
    v = cfg.get("visualizer")
    if not isinstance(v, dict):
        return None
    if str(v.get("type", "")).strip().lower() != "lowdim":
        return None
    raw_id = v.get("id", -1)
    try:
        share_id = int(raw_id)
    except (TypeError, ValueError):
        share_id = -1
    sub = v.get("args")
    if not isinstance(sub, dict):
        sub = {}
    return share_id, sub


def start_all_visualizers(
    device_configs: list,
    get_shm_name_func,
    is_camera_config_func,
    visualizer_configs: Optional[list] = None,
    camera_fps: float = 30.0,
    camera_scale: float = 0.5,
    lowdim_fps: float = 60.0,
    lowdim_time_window_s: float = 12.0,
) -> list:
    """
    Start visualizers for all devices that support visualization.

    If ``visualizer_configs`` is non-empty, only those YAML entries are started
    (each ``type`` must resolve to a class subclassing ``BaseVisualizer``, other than
    the base class itself), each in its own subprocess; embedded device ``visualizer:``
    blocks and auto camera batching are skipped.

    Otherwise (legacy):
    1. Collects all camera devices and starts a single CameraVisualizer for them
    2. Collects devices with top-level ``visualizer: {type: lowdim, id, args}``;
       same ``id`` when ``id >= 0`` share one LowDimVisualizer; ``id`` omitted
       (default -1) or any negative ``id`` uses a solo group per device SHM
    3. Starts individual visualizers for other devices that have a Visualizer class

    Args:
        device_configs: List of device configuration dicts
        get_shm_name_func: Function to get SHM name from config
        is_camera_config_func: Function to check if config is a camera
        visualizer_configs: Optional list of visualizer dicts from YAML
        camera_fps: FPS for camera visualizer
        camera_scale: Scale factor for camera display
        lowdim_fps: Plot redraw cap for lowdim (override ``visualizer.args.fps``; ``0`` = uncapped).
            SHM sampling runs in a thread at device rate, independent of this.
        lowdim_time_window_s: Default max x-axis span for lowdim (override:
            ``visualizer.args.time_window_s``); actual window fits data up to this cap

    Returns:
        List of started Process objects
    """
    import multiprocessing as mp
    from loguru import logger

    if visualizer_configs:
        from vlastudio.deploy.visualizer.runner import start_visualizer_processes

        logger.info(
            "Starting {} visualizer YAML entries (legacy embedded viz skipped)",
            len(visualizer_configs),
        )
        return start_visualizer_processes(visualizer_configs)

    viz_procs = []
    camera_shm_names = []
    lowdim_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for cfg in device_configs:
        device_type = cfg.get("type", "")
        device_shm_name = get_shm_name_func(cfg)

        try:
            if is_camera_config_func(cfg):
                camera_shm_names.append(device_shm_name)
                continue
        except Exception:
            pass

        lowdim_entry = _parse_lowdim_visualizer_entry(cfg)
        if lowdim_entry is not None:
            share_id, sub = lowdim_entry
            gid = _lowdim_share_group_id(device_shm_name, share_id)
            lowdim_groups[gid].append(
                {
                    "shm": device_shm_name,
                    "dim": _optional_int(sub.get("dim") or sub.get("lowdim_dim")),
                    "key": sub.get("array_key")
                    or sub.get("lowdim_array_key")
                    or sub.get("lowdim_key"),
                    "viz_fps": sub.get("fps") or sub.get("lowdim_viz_fps"),
                    "viz_tw": sub.get("time_window_s")
                    or sub.get("lowdim_viz_time_window_s"),
                    "viz_max_pts": sub.get("max_points")
                    or sub.get("lowdim_viz_max_points"),
                    "viz_plot_cap": sub.get("max_plot_points")
                    or sub.get("plot_point_cap"),
                    "viz_debug": bool(sub.get("debug")),
                }
            )
            continue

        viz_class = get_visualizer_class(device_type)
        if viz_class is not None:
            parts = device_type.rsplit(".", 1)
            viz_type = parts[0] + ".Visualizer"

            logger.info("Starting visualizer for {}: {}", device_shm_name, viz_type)
            viz_kwargs = {
                key: device_args[key]
                for key in ("xml_path", "scene_name", "scene_xml_path")
                if device_args.get(key) is not None
            }
            viz_proc = mp.Process(
                target=start_visualizer,
                args=(viz_type, device_shm_name),
                daemon=True,
            )
            viz_proc.start()
            viz_procs.append(viz_proc)
            logger.info("Visualizer subprocess started for SHM: {}", device_shm_name)

    # Lowdim first: starting it must not wait on importing OpenCV (camera_visualizer).
    if lowdim_groups:
        from vlastudio.deploy.visualizer.lowdim_visualizer import start_lowdim_visualizer

        lowdim_ctx = mp.get_context("spawn")
        for gid, specs in lowdim_groups.items():
            declared_dims = {s["dim"] for s in specs if s["dim"] is not None}
            if len(declared_dims) > 1:
                logger.error(
                    "lowdim group={!r}: inconsistent dim declarations {}; skip",
                    gid,
                    specs,
                )
                continue
            expected_dim = next(iter(declared_dims)) if declared_dims else None
            shm_entries: List[Tuple[str, Optional[str]]] = [
                (s["shm"], s["key"] if s["key"] else None) for s in specs
            ]
            fps = _first_non_null_float(specs, "viz_fps", lowdim_fps)
            tw = _first_non_null_float(specs, "viz_tw", lowdim_time_window_s)
            max_pts = _first_non_null_int(specs, "viz_max_pts", 8000)
            viz_plot_cap: Optional[int] = None
            for s in specs:
                v = s.get("viz_plot_cap")
                if v is not None:
                    try:
                        viz_plot_cap = int(float(v))
                        break
                    except (TypeError, ValueError):
                        pass
            viz_debug = any(bool(s.get("viz_debug")) for s in specs)

            logger.info(
                "Starting lowdim visualizer group={!r} shms={} expected_dim={}",
                gid,
                [e[0] for e in shm_entries],
                expected_dim,
            )
            viz_proc = lowdim_ctx.Process(
                target=start_lowdim_visualizer,
                args=(shm_entries,),
                kwargs={
                    "group_id": gid,
                    "expected_dim": expected_dim,
                    "fps": fps,
                    "time_window_s": tw,
                    "max_points": max_pts,
                    "max_plot_points": viz_plot_cap,
                    "debug": viz_debug,
                },
                daemon=False,
            )
            viz_proc.start()
            viz_procs.append(viz_proc)
            logger.info("Lowdim visualizer subprocess started for group {!r}", gid)

    if camera_shm_names:
        from vlastudio.deploy.visualizer.camera_visualizer import start_camera_visualizer

        logger.info("Starting camera visualizer for: {}", camera_shm_names)
        viz_proc = mp.Process(
            target=start_camera_visualizer,
            args=(camera_shm_names,),
            kwargs={"fps": camera_fps, "scale": camera_scale},
            daemon=True,
        )
        viz_proc.start()
        viz_procs.append(viz_proc)
        logger.info("Camera visualizer started for {} cameras", len(camera_shm_names))

    return viz_procs


def stop_all_visualizers(viz_procs: list):
    """
    Stop all visualizer processes.
    
    Args:
        viz_procs: List of Process objects to stop
    """
    for viz_proc in viz_procs:
        if viz_proc is not None:
            viz_proc.terminate()
            viz_proc.join(timeout=1.0)
            if viz_proc.is_alive():
                viz_proc.kill()