"""SO101 Plus robot package."""

from .config import SO101PlusConfig
from .so101_plus import SO101Plus, JOINT_NAMES

# Heavy imports (robot wrapper / IK) are lazy so `python -m deploy.robot.so101_plus`
# and config imports stay light.

def __getattr__(name: str):
    if name == "So101Plus":
        from .robot import So101Plus as _So101Plus

        return _So101Plus
    if name in {
        "DEFAULT_END_FRAME",
        "default_urdf_path",
        "hw_arm_to_ik",
        "ik_to_hw_arm",
        "make_piper_ik_solver",
        "make_ik_solver",
        "ScipyUrdfIKSolver",
        "So101PlusClikIKSolver",
    }:
        from . import ik as _ik

        return getattr(_ik, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SO101PlusConfig",
    "SO101Plus",
    "So101Plus",
    "JOINT_NAMES",
    "DEFAULT_END_FRAME",
    "default_urdf_path",
    "hw_arm_to_ik",
    "ik_to_hw_arm",
    "make_piper_ik_solver",
    "make_ik_solver",
    "ScipyUrdfIKSolver",
    "So101PlusClikIKSolver",
]
