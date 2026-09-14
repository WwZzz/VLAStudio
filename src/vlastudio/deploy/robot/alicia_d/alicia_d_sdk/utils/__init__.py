# Copyright (c) 2025 Synria Robotics Co., Ltd.
# GPL-3.0 License
#
# Modified for ILStudio: trajectory_utils (robocore-dependent) is lazy-loaded

# Core utilities (no robocore dependency)
from alicia_d_sdk.utils.logger import *
from alicia_d_sdk.utils.fps_utils import precise_sleep

# Trajectory utils are lazy-loaded since they depend on robocore
_trajectory_utils_names = [
    "record_waypoints_manual",
    "load_joint_waypoints_from_file",
    "save_joint_waypoints_to_file",
    "record_joint_waypoints_manual",
    "load_cartesian_waypoints_from_file",
    "save_cartesian_waypoints_to_file",
    "record_cartesian_waypoints_manual",
    "handle_waypoint_recording",
    "load_or_generate_joint_waypoints",
    "load_or_generate_cartesian_waypoints",
    "display_joint_waypoints",
    "display_cartesian_waypoints",
    "display_joint_trajectory_stats",
    "display_cartesian_trajectory_stats",
    "verify_cartesian_waypoints",
    "display_ik_results",
    "plot_trajectory",
]

_traj_cache = {}

def __getattr__(name):
    """Lazy load trajectory_utils (requires robocore)."""
    if name in _trajectory_utils_names:
        if name not in _traj_cache:
            from alicia_d_sdk.utils import trajectory_utils
            _traj_cache[name] = getattr(trajectory_utils, name)
        return _traj_cache[name]
    raise AttributeError(f"module 'alicia_d_sdk.utils' has no attribute '{name}'")

