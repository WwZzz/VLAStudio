"""
Visualizer module for displaying device data from shared memory.

Available visualizers:
- BaseVisualizer: Base class for custom visualizers
- CameraVisualizer: Display camera images in a grid
- LowDimVisualizer: Time-series plots for low-dimensional SHM vectors (see lowdim_visualizer.py)
"""

from vlastudio.deploy.visualizer.base import (
    BaseVisualizer,
    start_visualizer,
    get_visualizer_class,
    get_visualizer_type_string,
    start_all_visualizers,
    stop_all_visualizers,
)
from vlastudio.deploy.visualizer.camera_visualizer import (
    CameraVisualizer,
    start_camera_visualizer,
)
from vlastudio.deploy.visualizer.mujoco_camera_visualizer import (
    MujocoCameraVisualizer,
    start_mujoco_camera_visualizer,
)
from vlastudio.deploy.visualizer.mujoco_proxy_visualizer import (
    MujocoProxyVisualizer,
    start_mujoco_proxy_visualizer,
)

__all__ = [
    "BaseVisualizer",
    "CameraVisualizer",
    "LowDimVisualizer",
    "start_visualizer",
    "start_camera_visualizer",
    "start_lowdim_visualizer",
    "get_visualizer_class",
    "get_visualizer_type_string",
    "start_all_visualizers",
    "stop_all_visualizers",
]
