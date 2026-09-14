"""
Koch Leader Teleoperator
"""

import numpy as np
from typing import Optional
from pathlib import Path

from lerobot.teleoperators.koch_leader import KochLeaderConfig as RobotKochLeaderConfig, KochLeader as RobotKochLeader
from vlastudio.deploy.teleoperator.base import BaseTeleopDevice


class KochLeader(BaseTeleopDevice):
    """
    Concrete implementation of teleoperation using lerobot's KochLeader
    """
    def __init__(self, 
                 name: str = "koch_leader",
                 max_size_mb: int = 1,
                 fps: float = 100.0,
                 com: str = "/dev/ttyACM0",
                 robot_id: str = "koch_leader_arm",
                 elbow_drive_mode: int = 1,
                 calibration_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Koch Leader teleoperation device
        
        Args:
            name: Name of the shared memory segment
            max_size_mb: Maximum size of shared memory in MB
            fps: Control frequency in Hz
            com: Communication port for the leader device
            robot_id: Identifier for the robot
            elbow_drive_mode: Elbow drive mode setting
            calibration_dir: Optional calibration directory path
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps)
        
        self.com = com
        self.robot_id = robot_id
        self.elbow_drive_mode = elbow_drive_mode
        
        # Use the official lerobot support:
        robot_config = RobotKochLeaderConfig(port=com, id=robot_id)
        if calibration_dir:
            robot_config.calibration_dir = Path(calibration_dir)
        self._teleop_device = RobotKochLeader(robot_config)
        self._teleop_device.connect()
        self._teleop_device.bus.write("Drive_Mode", "elbow_flex", self.elbow_drive_mode)
        self._motors = list(self._teleop_device.bus.motors)
        
    def get_data(self) -> Optional[dict]:
        """Get the observation data for the Leader device"""
        return self._teleop_device.get_action()
    
    def convert_data_to_action(self, data: dict) -> np.ndarray:
        """Convert the observation data to the standardized robot action"""
        return np.array([data[mname + '.pos'] for mname in self._motors])

    def close(self):
        """Close the teleoperation device"""
        super().close()
        if self._teleop_device.is_connected:
            self._teleop_device.disconnect()


# ==============================================================================
# Test (run Leader in main process so calibration input() works, then read SHM)
# ==============================================================================

if __name__ == "__main__":
    import importlib
    import threading
    import time
    import yaml
    from pathlib import Path

    from vlastudio.deploy.shm_utils import SharedMemoryChannel

    # Load config
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "teleop" / "koch_leader.yaml"
    with open(cfg_path, "r") as f:
        device_config = yaml.safe_load(f)

    # Create device in main process so calibration (input()) works
    device_type = device_config["type"]
    module_name, class_name = device_type.rsplit(".", 1)
    module = importlib.import_module(module_name)
    device_class = getattr(module, class_name)
    device = device_class(**device_config["args"])
    # Calibration already done in __init__ -> connect()

    shm_name = device_config["args"]["name"]
    # Run device.start() in a thread (blocking loop that writes to SHM)
    start_thread = threading.Thread(target=device.start, daemon=True)
    start_thread.start()

    time.sleep(0.5)
    print("Reading from SHM (Ctrl+C to stop)...")
    print("Move the leader arm to see action updates.\n")

    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=10.0)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None and "action" in data:
                arr = data["action"]
                print(f"  action: {arr}", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        device.close()
        start_thread.join(timeout=2.0)
        print("Done.")
