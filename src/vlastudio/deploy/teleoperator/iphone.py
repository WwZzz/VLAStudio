"""
iPhone Phyphox IMU Teleoperator
Uses iPhone's IMU data via Phyphox app for teleoperation
"""

import numpy as np
import requests
import time
import sys
from typing import Optional

from scipy.spatial.transform import Rotation as R
from vlastudio.deploy.teleoperator.base import BaseTeleopDevice
from vlastudio.deploy.utils import RateLimiter

# Cross-platform input detection
if sys.platform == "win32":
    import msvcrt
else:
    import select


class IMUProcessor:
    """
    Processes IMU data and calculates per-frame relative pose transformations.
    Does not perform state accumulation, only calculates instantaneous changes.
    - acc_unit : m/s² (linear acceleration)
    - gyro_unit: rad/s
    """

    def __init__(self, calib_samples: int = 150):
        self.N = calib_samples
        self.ACC_THRESHOLD = 0.25  # m/s^2
        self.GYRO_THRESHOLD = 0.25  # rad/s
        self.reset()

    def reset(self):
        """Reset calibration data"""
        self.cnt = 0
        self.lin_acc_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.calibrated = False
        self._lin_acc_samples = []
        self._gyro_samples = []
        print("\nIMU Processor has been reset. Please keep phone stationary for recalibration.")

    def calibrate_step(self, lin_acc_raw, gyro_raw):
        """Accumulate data during calibration phase"""
        if self.cnt < self.N:
            self._lin_acc_samples.append(lin_acc_raw)
            self._gyro_samples.append(gyro_raw)
            self.cnt += 1
            if self.cnt >= self.N:
                self._finalize_calibration()
        return self.calibrated

    def _finalize_calibration(self):
        """Calculate sensor biases"""
        self.lin_acc_bias = np.mean(self._lin_acc_samples, axis=0)
        self.gyro_bias = np.mean(self._gyro_samples, axis=0)
        self.calibrated = True
        print("\nIMU calibration complete.")
        print(f"  - Linear acceleration bias: {self.lin_acc_bias}")
        print(f"  - Gyroscope bias: {self.gyro_bias}")

    def calculate_delta_pose(self, lin_acc_raw, gyro_raw, dt):
        """
        Calculate per-frame relative pose transformation.
        Returns: (delta_translation, delta_rotation_euler)
        """
        lin_acc_corr = lin_acc_raw - self.lin_acc_bias
        gyro_corr = gyro_raw - self.gyro_bias

        # Apply noise threshold
        if np.linalg.norm(lin_acc_corr) < self.ACC_THRESHOLD:
            lin_acc_corr[:] = 0.0

        if np.linalg.norm(gyro_corr) < self.GYRO_THRESHOLD:
            gyro_corr[:] = 0.0

        # Calculate instantaneous displacement (physically represents delta_v = a * dt)
        delta_translation = lin_acc_corr * dt

        # Calculate instantaneous rotation (angular velocity * time -> Euler angles)
        delta_rotation_euler = gyro_corr * dt

        return delta_translation, delta_rotation_euler


class IPhonePhyphox(BaseTeleopDevice):
    """
    iPhone Phyphox IMU Teleoperator
    Uses iPhone's IMU data via Phyphox app for teleoperation
    """
    
    def __init__(self,
                 name: str = "iphone_teleop",
                 max_size_mb: int = 1,
                 fps: float = 30.0,
                 phyphox_ip: str = "192.168.1.5",
                 phyphox_port: int = 80,
                 calib_samples: int = 150,
                 dt: Optional[float] = None,
                 **kwargs):
        """
        Initialize the iPhone Phyphox teleoperation device
        
        Args:
            name: Name of the shared memory segment
            max_size_mb: Maximum size of shared memory in MB
            fps: Control frequency in Hz
            phyphox_ip: IP address of the Phyphox app
            phyphox_port: Port of the Phyphox app
            calib_samples: Number of samples for calibration
            dt: Time delta for calculations (overrides fps if provided)
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps)

        if dt is not None:
            self.dt = dt
            self.fps = 1.0 / dt
        else:
            self.dt = 1.0 / fps

        self.url = f"http://{phyphox_ip}:{phyphox_port}/get?lin_accX&lin_accY&lin_accZ&gyroX&gyroY&gyroZ"
        self.processor = IMUProcessor(calib_samples=calib_samples)

    def get_data(self) -> Optional[dict]:
        """Get linear acceleration and angular velocity"""
        try:
            response = requests.get(self.url, timeout=0.5)
            response.raise_for_status()
            data = response.json()["buffer"]
            ax = data["lin_accX"]["buffer"][-1]
            ay = data["lin_accY"]["buffer"][-1]
            az = data["lin_accZ"]["buffer"][-1]
            gx = data["gyroX"]["buffer"][-1]
            gy = data["gyroY"]["buffer"][-1]
            gz = data["gyroZ"]["buffer"][-1]

            if any(v is None for v in [ax, ay, az, gx, gy, gz]):
                return None
            return {"imu": np.array([ax, ay, az, gx, gy, gz])}
        except Exception:
            return None

    def convert_data_to_action(self, data: dict) -> np.ndarray:
        """Convert sensor observations to relative pose transformation"""
        imu = data.get("imu")
        if imu is None:
            return np.zeros(6)

        lin_acc_raw = imu[:3]
        gyro_raw = imu[3:]

        # Handle calibration
        if not self.processor.calibrated:
            print(f"Calibrating... {self.processor.cnt}/{self.processor.N}", end='\r')
            self.processor.calibrate_step(lin_acc_raw, gyro_raw)
            return np.zeros(6)

        delta_translation, delta_rotation_euler = self.processor.calculate_delta_pose(
            lin_acc_raw, gyro_raw, self.dt
        )

        return np.concatenate([delta_translation, delta_rotation_euler])


# ==============================================================================
# Test (start iPhone teleop in subprocess, read from SHM and print)
# ==============================================================================

if __name__ == "__main__":
    import multiprocessing as mp
    import yaml
    from pathlib import Path

    from vlastudio.deploy.base import start_device
    from vlastudio.deploy.shm_utils import SharedMemoryChannel

    # Load config
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "teleop" / "iphone.yaml"
    with open(cfg_path, "r") as f:
        device_config = yaml.safe_load(f)

    shm_name = device_config["args"]["name"]
    proc = mp.Process(target=start_device, args=(device_config,))
    proc.start()

    time.sleep(0.5)
    print("Reading from SHM (Ctrl+C to stop)...")
    print("Move the iPhone to see action updates.\n")

    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=10.0)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None and "action" in data:
                arr = data["action"]
                if len(arr) >= 6:
                    print(
                        f"  trans: [{arr[0]:.3f}, {arr[1]:.3f}, {arr[2]:.3f}]  "
                        f"rot: [{arr[3]:.3f}, {arr[4]:.3f}, {arr[5]:.3f}]",
                        end="\r",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
        print("Done.")
