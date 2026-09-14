"""
SO101 Plus Leader teleoperator wrapper for ILStudio.
"""

import os
from typing import Optional
from pathlib import Path

import numpy as np

from deploy.teleoperator.base import BaseTeleopDevice
from .config import SO101PlusLeaderConfig
from .so101_plus_leader import SO101PlusLeader


def _default_robots_calib_dir() -> Path:
    """Same directory used by `python -m deploy.robot.so101_plus --calibrate`."""
    hf = os.environ.get("HF_LEROBOT_HOME")
    if hf:
        base = Path(hf).expanduser()
    else:
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        base = Path(hf_home).expanduser() / "lerobot"
    return base / "calibration" / "robots" / "so101_plus"


class So101PlusLeader(BaseTeleopDevice):
    """ILStudio teleop device that reads SO101 Plus leader joint positions."""

    def __init__(
        self,
        name: str = "so101_plus_leader",
        max_size_mb: int = 1,
        fps: float = 100.0,
        com: str = "/dev/ttyACM0",
        robot_id: str = "so101_plus_leader",
        calibration_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps)

        self.com = com
        self.robot_id = robot_id

        robot_config = SO101PlusLeaderConfig(port=com, id=robot_id)
        # Leader JSON is saved under robots/so101_plus/ (CLI), not teleoperators/.
        calib_dir = Path(calibration_dir).expanduser() if calibration_dir else _default_robots_calib_dir()
        robot_config.calibration_dir = calib_dir
        self._teleop_device = SO101PlusLeader(robot_config)
        self._teleop_device.connect()
        self._motors = list(self._teleop_device.bus.motors)

    def get_data(self) -> Optional[dict]:
        return self._teleop_device.get_action()

    def convert_data_to_action(self, data: dict) -> np.ndarray:
        return np.array([data[mname + ".pos"] for mname in self._motors], dtype=np.float64)

    def close(self):
        super().close()
        if self._teleop_device.is_connected:
            self._teleop_device.disconnect()
