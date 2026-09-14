"""SO101++ Leader teleoperator wrapper for ILStudio."""

from typing import Optional
from pathlib import Path

import numpy as np

from deploy.teleoperator.base import BaseTeleopDevice
from .config import SO101PPLeaderConfig
from .so101_pp_leader import SO101PPLeader


class So101PPLeader(BaseTeleopDevice):
    def __init__(
        self,
        name: str = "so101_pp_leader",
        max_size_mb: int = 1,
        fps: float = 100.0,
        com: str = "/dev/ttyACM0",
        robot_id: str = "so101_pp_leader",
        calibration_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps)

        self.com = com
        self.robot_id = robot_id

        robot_config = SO101PPLeaderConfig(port=com, id=robot_id)
        if calibration_dir:
            robot_config.calibration_dir = Path(calibration_dir).expanduser()
        self._teleop_device = SO101PPLeader(robot_config)
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
