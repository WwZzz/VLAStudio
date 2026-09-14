"""
Dummy Robot for Testing
A virtual robot interface for testing purposes that generates mock observation data
"""

import numpy as np
import time
import threading
from typing import Dict, Any, Optional
import os

from vlastudio.deploy.robot.base import BaseRobot


class DummyRobot(BaseRobot):
    """
    A virtual robot interface for testing purposes.
    It dynamically generates mock observation data and validates actions
    based on a YAML configuration file.
    """

    def __init__(self,
                 name: str = "dummy_robot",
                 max_size_mb: int = 64,
                 fps: float = 30.0,
                 control_shm_name: Optional[str] = None,
                 obs_config: dict = {},
                 action_config: dict = {},
                 ctrl_type: str = 'delta',
                 ctrl_space: str = 'ee',
                 **kwargs):
        """
        Initialize the virtual robot according to the configuration
        
        Args:
            name: Name for the shared memory segment
            max_size_mb: Maximum size of shared memory in MB
            fps: Control frequency in Hz
            control_shm_name: Name of the control shared memory
            obs_config: Observation configuration dictionary
            action_config: Action configuration dictionary
            ctrl_type: Control type ('delta' or 'abs')
            ctrl_space: Control space ('ee' or 'joint')
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        
        print("[DummyRobot] Initializing...")
        self.shutdown_event = threading.Event()

        self.dtypes = {
            'uint8': np.uint8,
            'float32': np.float32,
            'float64': np.float64,
            'int': np.int32,
            'int32': np.int32,
            'int64': np.int64,
        }

        self.obs_config = obs_config
        self.action_config = action_config
        self.ctrl_type = ctrl_type
        self.ctrl_space = ctrl_space

        print("[DummyRobot] Initialization complete, ready.")
        print(f"[DummyRobot] Observation format: {self.obs_config}")
        print(f"[DummyRobot] Action format: {self.action_config}")

    def connect(self):
        """
        Simulates connecting to the robot SDK.
        """
        print("[DummyRobot] Simulating connection...")
        time.sleep(0.5)  # Simulate connection delay
        print("[DummyRobot] Connection successful.")
        return True

    def _generate_data_from_config(self, config_dict: Dict) -> Dict[str, Any]:
        """
        Recursive function to generate data based on a nested dictionary configuration.

        - If the dictionary contains 'shape' and 'dtype' keys, a NumPy array is generated.
        - Otherwise, it recursively processes each value in the dictionary.
        """
        generated_data = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                if 'shape' in value and 'dtype' in value:
                    # Found a leaf node: the definition of an array.
                    shape = tuple(value['shape'] if isinstance(value['shape'][0], int) else eval(value['shape']))

                    dtype = self.dtypes.get(value['dtype'], np.float32)

                    if dtype == np.uint8:
                        # For image data, generates integers in the 0-255 range.
                        data = (np.random.rand(*shape) * 255).astype(dtype)
                    else:
                        # For other data, generates floats in the 0-1 range.
                        data = np.random.rand(*shape).astype(dtype)

                    generated_data[key] = data
                else:
                    # Recursive call to process nested dictionaries.
                    generated_data[key] = self._generate_data_from_config(value)
            else:
                # Exception case, non-dictionary type.
                print(f"[Warning] Non-dictionary value found in configuration, skipping: {key}: {value}")
        return generated_data

    def get_observation(self) -> Dict[str, Any]:
        """
        Dynamically generates mock observation data based on the configuration at initialization.

        - Conceptually, this is a blocking operation, as it waits for all sensor data to arrive.
        - The data shape and type strictly follow the definitions in the YAML file.
        """
        # The actual blocking is controlled by robot.rate_sleep in the main program.
        return self._generate_data_from_config(self.obs_config)
    
    def get_action_dim(self):
        expected_shape = tuple(self.action_config['shape'] if isinstance(self.action_config['shape'][0], int) else eval(self.action_config['shape']))
        return np.prod(expected_shape).item()
        
    def publish_action(self, action: np.ndarray):
        """
        Simulates publishing an action command to the robot and validates the action format.
        """
        # Validate the action's shape and data type.
        expected_shape = tuple(self.action_config['shape'] if isinstance(self.action_config['shape'][0], int) else eval(
            self.action_config['shape']))
        expected_dtype = self.dtypes.get(self.action_config['dtype'], np.float32)

        # Simulates publishing the action; here it just prints.
        print(f"[DummyRobot] Received action, publishing...")
        print(f"  Action content: {np.round(action, 3)}")

    def is_running(self) -> bool:
        """
        Checks if the robot system is still running.
        A threading event is used to simulate system shutdown.
        """
        return not self.shutdown_event.is_set()

    def rate_sleep(self, hz: int):
        """
        Performs a sleep to control the frequency.
        """
        time.sleep(1.0 / hz)

    def shutdown(self):
        """Shutdown the robot"""
        self.shutdown_event.set()

    def close(self):
        """Close the robot"""
        super().close()
        self.shutdown_event.set()


# ==============================================================================
# Test
# ==============================================================================

if __name__ == '__main__':
    import yaml
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "robot" / "dummy.yaml"
    with open(cfg_path, 'r') as f:
        raw = yaml.safe_load(f)
    device_config = raw[0] if isinstance(raw, list) else raw

    robot = DummyRobot(**device_config.get('args', {}))
    robot.connect()
    print("Robot is Running: ", robot.is_running())
    obs = robot.get_observation()
    print("Observation:")

    def show_array_info(data, path=""):
        """Recursively shows the shape and dtype of all arrays in a nested dictionary."""
        if isinstance(data, dict):
            for k, v in data.items():
                new_path = f"{path}/{k}" if path else k
                show_array_info(v, new_path)
        elif isinstance(data, np.ndarray):
            shape = tuple(data.shape)
            dtype = str(data.dtype)
            print(f"{path}: shape={shape}, dtype={dtype}")

    show_array_info(obs)
    action = np.random.rand(14)
    robot.publish_action(action)
    robot.shutdown()
    print("After set shutdown, Robot is Running: ", robot.is_running())