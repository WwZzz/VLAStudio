"""
So101 Leader 
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from .bi_so101_leader import BiSO101LeaderConfig as RobotBiSO101LeaderConfig, BiSO101Leader as RobotBiSO101Leader
from vlastudio.deploy.robot.base import BaseRobot
import numpy as np
import traceback
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
from pynput import keyboard
from vlastudio.deploy.teleoperator.base import BaseTeleopDevice
from typing import Optional
from pathlib import Path
    

class BiSO101Leader(BaseTeleopDevice):
    """
    Concrete implementation of teleoperation using So101 Leader
    参照 Koch Leader 的实现方式
    """
    def __init__(self, 
                left_arm_port: str="/dev/ttyACM1",    
                right_arm_port: str="/dev/ttyACM2",    
                robot_id: str="biso101_leader",
                calibration_dir: Optional[str]=None,
                **kwargs):
        """
        Initialize the BiSO101 Leader teleoperation device
        
        Args:
            shm_name: Name of the shared memory segment
            shm_shape: Shape of the shared memory array
            shm_dtype: Data type of the shared memory array
            action_dim: The dim of the flattened action
            frequency: Control frequency in Hz
            left_arm_port: Communication port for the left arm
            right_arm_port: Communication port for the right arm
            robot_id: Identifier for the robot
        """
        super().__init__(name=robot_id, max_size_mb=1, fps=1000)
        
        self.left_arm_port = left_arm_port
        self.right_arm_port = right_arm_port
        self.robot_id = robot_id
        
        # Use the official lerobot support:
        robot_config = RobotBiSO101LeaderConfig(left_arm_port=left_arm_port, right_arm_port=right_arm_port, id=robot_id)
        if calibration_dir:
            robot_config.calibration_dir = Path(calibration_dir)
        self._teleop_device = RobotBiSO101Leader(robot_config)
        self._teleop_device.connect()
        self._left_motors = list(self._teleop_device.left_arm.bus.motors)
        self._right_motors = list(self._teleop_device.right_arm.bus.motors)
        
    def get_data(self):
        """Get the observation data for the Leader device"""
        return self._teleop_device.get_action()
    
    def convert_data_to_action(self, data: dict):
        """Convert the observation data to the standardized robot action"""
        left_qpos = np.array([data['left_'+mname+'.pos'] for mname in self._left_motors])
        right_qpos = np.array([data['right_'+mname+'.pos'] for mname in self._right_motors])
        return np.concatenate([left_qpos, right_qpos])

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
    cfg_path = Path(__file__).resolve().parents[3] / "configs" / "teleop" / "biso101_leader.yaml"
    with open(cfg_path, "r") as f:
        device_config = yaml.safe_load(f)

    # Create device in main process so calibration (input()) works
    device_type = device_config["type"]
    module_name, class_name = device_type.rsplit(".", 1)
    module = importlib.import_module(module_name)
    device_class = getattr(module, class_name)
    device = device_class(**device_config["args"])
    # Calibration already done in __init__ -> connect()

    shm_name = device_config["args"]["robot_id"]
    # Run device.start() in a thread (blocking loop that writes to SHM)
    start_thread = threading.Thread(target=device.start, daemon=True)
    start_thread.start()

    time.sleep(0.5)
    print("Reading from SHM (Ctrl+C to stop)...")
    print("Move the leader arms to see action updates (left 6 + right 6).\n")

    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=10.0)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None and "action" in data:
                arr = data["action"]
                n = len(arr)
                if n >= 12:
                    left, right = arr[:6], arr[6:12]
                    print(
                        f"  left: [{left[0]:.3f}, {left[1]:.3f}, {left[2]:.3f}, "
                        f"{left[3]:.3f}, {left[4]:.3f}, {left[5]:.3f}]  "
                        f"right: [{right[0]:.3f}, {right[1]:.3f}, {right[2]:.3f}, "
                        f"{right[3]:.3f}, {right[4]:.3f}, {right[5]:.3f}]",
                        end="\r",
                        flush=True,
                    )
                else:
                    print(f"  action: {arr}", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        device.close()
        start_thread.join(timeout=2.0)
        print("Done.")