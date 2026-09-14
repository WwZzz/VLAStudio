import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
from vlastudio.deploy.utils import RateLimiter
from vlastudio.deploy.base import BaseDevice
# --- 1. Base Teleoperation Device Class ---

class BaseTeleopDevice(BaseDevice):
    """
    Abstract base class for teleoperation devices
    """
    def __init__(self, name:str, max_size_mb: int=1, fps: float=1000):
        super().__init__(name, max_size_mb=1, fps=fps)

    @abstractmethod
    def convert_data_to_action(self, data: dict) -> np.ndarray:
        """Convert observation data to standardized robot action"""
        pass

    def start(self):
        # create shared memory
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)
        self.is_running = True
        rate_limiter = RateLimiter()
        while self.is_running:
            data = self.get_data()
            if data is not None:
                action = self.convert_data_to_action(data)
                self.write_data_to_shm({"action": action})
            rate_limiter.sleep(self.fps)

    # def run(self):
    #     """
    #     Main loop: get observation, convert to action, and write to buffer at specified frequency
    #     """
    #     self.connect_to_buffer()
    #     rate_limiter = RateLimiter()
    #     rate = 1.0 / self.frequency
    #     try:
    #         while not self.stop_event.is_set():
    #             start_time = time.time()
    #             # Core three steps
    #             observation = self.get_observation()
    #             action = self.observation_to_action(observation)
    #             # print(f"Teleop device: action = {action}")
    #             self.put_action_to_buffer(action)
    #             rate_limiter.sleep(self.frequency)
    #             # elapsed_time = time.time() - start_time
    #             # sleep_time = rate - elapsed_time
    #             # if sleep_time > 0:
    #             #     time.sleep(sleep_time)
    #     finally:
    #         print("Teleop device: Shutting down...")
    #         if self.shm:
    #             self.shm.close()

