# Copyright (c) 2025 Synria Robotics Co., Ltd.
# GPL-3.0 License - see LICENSE file for details
#
# Modified for ILStudio integration:
# - Core hardware functions (ServoDriver) work without robocore/synriard
# - RoboCore features (SynriaRobotAPI, FK/IK, create_robot) are lazy-loaded
"""
Alicia-D SDK v6.1.0 - Hardware Layer

Core functionality (NO external deps beyond pyserial/numpy):
- ServoDriver: Low-level hardware control
- SerialComm: Serial communication
- DataParser: Protocol parsing

Optional functionality (requires robocore/synriard):
- SynriaRobotAPI: High-level robot API
- create_robot: Factory function
- RoboCore kinematics (FK/IK/Jacobian)
"""

__version__ = "6.1.0"
__author__ = "Synria Robotics"
__description__ = "Alicia-D Robot Arm SDK v6.1.0"

# Core hardware layer - always available, no robocore dependency
from alicia_d_sdk.hardware import ServoDriver, SerialComm, DataParser, JointState

__all__ = [
    # Hardware Layer (always available)
    "ServoDriver",
    "SerialComm",
    "DataParser",
    "JointState",
    # Optional (requires robocore/synriard)
    "SynriaRobotAPI",
    "create_robot",
    "RobotModel",
    "forward_kinematics",
    "inverse_kinematics",
    "jacobian",
]

# Lazy loading for robocore-dependent features
_robocore_cache = {}

def __getattr__(name):
    """Lazy load robocore-dependent components."""
    if name in _robocore_cache:
        return _robocore_cache[name]
    
    if name == "SynriaRobotAPI":
        from alicia_d_sdk.api import SynriaRobotAPI
        _robocore_cache[name] = SynriaRobotAPI
        return SynriaRobotAPI
    
    if name == "create_robot":
        _robocore_cache[name] = _create_robot_impl
        return _create_robot_impl
    
    if name == "RobotModel":
        from robocore.modeling import RobotModel
        _robocore_cache[name] = RobotModel
        return RobotModel
    
    if name in ("forward_kinematics", "inverse_kinematics", "jacobian"):
        import robocore.kinematics as kin
        func = getattr(kin, name)
        _robocore_cache[name] = func
        return func
    
    raise AttributeError(f"module 'alicia_d_sdk' has no attribute '{name}'")


def _get_gripper_type_from_json() -> str:
    """Read gripper type from JSON file, return default if not found."""
    import json
    from pathlib import Path
    json_path = Path(__file__).parent / "api" / "gripper_type.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_type = data.get("type_name")
            if isinstance(cached_type, str) and cached_type:
                return cached_type
        except Exception:
            pass
    return "50mm"


def _create_robot_impl(
    port: str = "",
    version: str = "v5_6",
    variant: str = None,
    model_format: str = "urdf",
    debug_mode: bool = False,
    auto_connect: bool = True,
    base_link: str = "base_link",
    end_link: str = "tool0",
    backend = None,
    device: str = "cpu",
    model_path: str = None,
    gripper_type: str = None,
):
    """
    Create robot instance (requires robocore/synriard).
    
    For ILStudio integration, use deploy.robot.alicia_d.AliciaD instead.
    """
    from synriard import get_model_path
    from robocore.modeling import RobotModel
    from alicia_d_sdk.api import SynriaRobotAPI
    
    servo_driver = ServoDriver(port=port, debug_mode=debug_mode)
    
    effective_gripper_type = gripper_type if gripper_type is not None else _get_gripper_type_from_json()
    variant = variant if variant is not None else f"gripper_{effective_gripper_type}"
    
    if model_path is None:
        model_path = get_model_path(
            "Alicia_D",
            version=version,
            variant=variant,
            model_format=model_format
        )
    robot_model = RobotModel(str(model_path), base_link=base_link, end_link=end_link)
    
    return SynriaRobotAPI(
        servo_driver=servo_driver,
        robot_model=robot_model,
        auto_connect=auto_connect,
        backend=backend,
        device=device
    )
