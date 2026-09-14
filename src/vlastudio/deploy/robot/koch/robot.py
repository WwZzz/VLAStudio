"""
Koch Follower Robot with Camera Integration
Integrates camera directly into lerobot's default KochFollower
"""

import numpy as np
import traceback
import time
from typing import Optional
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.koch_follower import KochFollowerConfig, KochFollower
try:
    from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
except ImportError:
    from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from vlastudio.deploy.robot.base import BaseRobot
from vlastudio.benchmark.base import MetaObs


class KochFollowerWithCamera(BaseRobot):
    """
    Integrate camera directly into lerobot's default KochFollower
    So that the camera becomes part of the robot, rather than an external component
    """
    def __init__(self,
                 name: str = "koch_follower",
                 max_size_mb: int = 64,
                 fps: float = 100.0,
                 control_shm_name: Optional[str] = None,
                 com: str = "/dev/ttyACM0",
                 robot_id: str = "koch_follower_arm",
                 camera_configs: dict = {},
                 calibration_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Koch Follower robot with camera
        
        Args:
            name: Name for the shared memory segment
            max_size_mb: Maximum size of shared memory in MB
            fps: Control frequency in Hz
            control_shm_name: Name of the control shared memory (for receiving actions)
            com: Communication port for the robot
            robot_id: Identifier for the robot
            camera_configs: Dictionary of camera configurations
            calibration_dir: Optional calibration directory path
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        
        self.com = com
        self.robot_id = robot_id
        
        # Create camera configurations
        camera_configs_dict = {}
        for cam_name, cam_config in camera_configs.items():
            camera_configs_dict[cam_name] = OpenCVCameraConfig(**cam_config)
        
        # Robot arm part - pass camera configurations to KochFollower
        robot_config = KochFollowerConfig(
            port=com,
            id=robot_id,
            cameras=camera_configs_dict
        )
        if calibration_dir:
            robot_config.calibration_dir = Path(calibration_dir)
        self._robot = KochFollower(robot_config)
        self._motors = list(self._robot.bus.motors)
        
        # Connect to robot
        retry_counts = 0
        max_connect_retry = 10
        while not self.connect():
            print(f"Retrying for {retry_counts} time...")
            retry_counts += 1
            if retry_counts > max_connect_retry:
                raise RuntimeError("Failed to connect to robot after max retries")
            time.sleep(1)
    
    def connect(self):
        """Connect robot and cameras"""
        try:
            if not self._robot.is_connected:
                self._robot.connect()
        except DeviceAlreadyConnectedError as e:
            print(f"Robot already connected: {e}")
            pass
        except Exception as e:
            print(f"Failed to connect to robot due to {e}")
            traceback.print_exc()
            return False
        print("Robot connected")
        return True
    
    def get_action_dim(self):
        """Get action dimension"""
        return len(self._motors)

    def get_observation(self):
        """Get complete observation data (including camera images)"""
        try:
            obs = self._robot.get_observation()
            qpos = np.array([obs[mname + '.pos'] for mname in self._motors], dtype=np.float64)
            return {'qpos': qpos, **obs}
        except Exception as e:
            print(f"Error getting observation: {e}")
            return None

    def obs2meta(self, device_data):
        """Extract robot state from this device's SHM data."""
        if device_data is None:
            return {}
        qpos = device_data.get('qpos')
        if qpos is None:
            qpos = np.array([device_data[m + '.pos'] for m in self._motors], dtype=np.float32)
        return {'state': np.asarray(qpos, dtype=np.float32)}
    
    def shutdown(self):
        """Shutdown robot and cameras"""
        if self._robot.is_connected:
            self._robot.disconnect()

    def close(self):
        """Close the robot"""
        super().close()
        if self._robot.is_connected:
            self._robot.disconnect()
        
    def publish_action(self, action: np.ndarray):
        """Publish action to robot"""
        try:
            action_dict = {mname + '.pos': action[i] for i, mname in enumerate(self._motors)}
            self._robot.send_action(action_dict)
        except Exception as e:
            pass
    
    def is_running(self):
        """Check if robot is running"""
        return self._robot.is_connected

    def save_episode(self, file_path: str, observations: list, actions: list):
        """Save episode data to HDF5 file"""
        import h5py
        import os
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        def write_group(group, data_list, key_prefix=None):
            if isinstance(data_list[0], dict):
                for key in data_list[0].keys():
                    sub_list = [obs[key] for obs in data_list]
                    if isinstance(sub_list[0], dict):
                        sub_group = group.create_group(key)
                        write_group(sub_group, sub_list)
                    else:
                        try:
                            group.create_dataset(key, data=np.stack(sub_list))
                        except (TypeError, ValueError) as e:
                            print(f"Warning: Could not stack data for key '{key}'. Skipping. Error: {e}")
            else:
                try:
                    if key_prefix is None:
                        group.create_dataset('data', data=np.stack(data_list))
                    else:
                        group.create_dataset(key_prefix, data=np.stack(data_list))
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not stack data for key '{key_prefix}'. Skipping. Error: {e}")

        with h5py.File(file_path, 'w') as f:
            f.create_dataset('actions', data=np.array(actions, dtype=np.float32))
            obs_group = f.create_group('observations')
            if observations:
                write_group(obs_group, observations)


# ==============================================================================
# Test (run Follower in main process so connect/retry works, then read SHM)
# ==============================================================================

if __name__ == "__main__":
    import importlib
    import threading
    import yaml
    from pathlib import Path

    from vlastudio.deploy.shm_utils import SharedMemoryChannel

    # Load config
    cfg_path = Path(__file__).resolve().parents[3] / "configs" / "robot" / "koch_follower.yaml"
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
    device_config = raw[0] if isinstance(raw, list) else raw

    # Create device in main process
    device_type = device_config["type"]
    module_name, class_name = device_type.rsplit(".", 1)
    module = importlib.import_module(module_name)
    device_class = getattr(module, class_name)
    device = device_class(**device_config["args"])

    shm_name = device_config["args"]["name"]
    start_thread = threading.Thread(target=device.start, daemon=True)
    start_thread.start()

    time.sleep(0.5)
    print("Reading from Koch Follower SHM (Ctrl+C to stop)...")

    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=15.0)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None:
                arr = data.get("qpos")
                if arr is not None:
                    print(f"  qpos: {arr}", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        device.close()
        start_thread.join(timeout=2.0)
        print("Done.")
