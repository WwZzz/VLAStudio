"""
MuJoCo camera visualizer - display camera images embedded in a robot SHM stream.

This mirrors `camera_visualizer.py`, but instead of reading multiple camera SHMs,
it reads one robot SHM and extracts image arrays by top-level camera keys.
"""

import importlib.util
import os


def _ensure_qt_font_dir():
    spec = importlib.util.find_spec("cv2")
    if not spec or not spec.origin:
        return
    cv2_dir = os.path.dirname(spec.origin)
    fonts_dir = os.path.join(cv2_dir, "qt", "fonts")
    if os.path.isdir(fonts_dir):
        return
    try:
        os.makedirs(fonts_dir, exist_ok=True)
    except OSError:
        return
    for sysfont in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF", "/usr/share/fonts"):
        if not os.path.isdir(sysfont):
            continue
        for f in os.listdir(sysfont):
            if not (f.endswith(".ttf") or f.endswith(".otf")):
                continue
            src = os.path.join(sysfont, f)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(fonts_dir, f)
            if os.path.lexists(dst):
                continue
            try:
                os.symlink(src, dst)
            except OSError:
                pass
        if os.listdir(fonts_dir):
            break


_ensure_qt_font_dir()

import cv2
import numpy as np
import time
from typing import List, Optional
from loguru import logger

from vlastudio.deploy.shm_utils import SharedMemoryChannel


class MujocoCameraVisualizer:
    """
    Visualize named camera images stored in a robot SHM observation dict.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        robot_shm_name: Optional[str] = None,
        camera_keys: Optional[List[str]] = None,
        window_name: str = "MuJoCo Camera View",
        fps: float = 30.0,
        grid_cols: Optional[int] = None,
        scale: float = 1.0,
        bgr_to_rgb: bool = False,
        **kwargs,
    ):
        if robot_shm_name is None:
            robot_shm_name = kwargs.pop("shm_name", None)
        if robot_shm_name is None:
            raise ValueError("robot_shm_name is required")
        if camera_keys is None:
            camera_keys = kwargs.pop("camera_keys", None)
        if camera_keys is None:
            raise ValueError("camera_keys is required")

        self.name = name or f"{robot_shm_name}_mujoco_camera_visualizer"
        self.robot_shm_name = robot_shm_name
        self.camera_keys = camera_keys
        self.window_name = window_name
        self.fps = fps
        self.scale = scale
        self.bgr_to_rgb = bgr_to_rgb
        self.is_running = False
        self.robot_shm: Optional[SharedMemoryChannel] = None

        n_cameras = len(camera_keys)
        if grid_cols is None:
            if n_cameras <= 2:
                grid_cols = max(n_cameras, 1)
            elif n_cameras <= 4:
                grid_cols = 2
            else:
                grid_cols = 3
        self.grid_cols = grid_cols
        self.grid_rows = (max(n_cameras, 1) + grid_cols - 1) // grid_cols

    def connect(self, timeout: float = 30.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.robot_shm = SharedMemoryChannel(self.robot_shm_name, is_writer=False, timeout=1.0)
                logger.info(f"[MujocoCameraVisualizer] Connected to robot SHM: {self.robot_shm_name}")
                return True
            except Exception:
                logger.info(f"[MujocoCameraVisualizer] Waiting for robot SHM '{self.robot_shm_name}'...")
                time.sleep(0.5)
        print(f"[MujocoCameraVisualizer] Failed to connect to robot SHM: {self.robot_shm_name}")
        return False

    def setup(self) -> bool:
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            return True
        except cv2.error as e:
            logger.error(f"[MujocoCameraVisualizer] OpenCV GUI unavailable: {e}")
            return False

    def _create_grid_image(self, images: List[Optional[np.ndarray]]) -> np.ndarray:
        max_h, max_w = 0, 0
        for img in images:
            if img is not None:
                h, w = img.shape[:2]
                max_h = max(max_h, h)
                max_w = max(max_w, w)

        if max_h == 0 or max_w == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        target_h = int(max_h * self.scale)
        target_w = int(max_w * self.scale)
        grid_h = self.grid_rows * target_h
        grid_w = self.grid_cols * target_w
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for i, img in enumerate(images):
            row = i // self.grid_cols
            col = i % self.grid_cols
            y_start = row * target_h
            x_start = col * target_w

            if img is not None:
                if img.shape[:2] != (target_h, target_w):
                    img = cv2.resize(img, (target_w, target_h))
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                grid[y_start:y_start + target_h, x_start:x_start + target_w] = img
            else:
                cv2.putText(
                    grid,
                    f"{self.camera_keys[i]}: No Data",
                    (x_start + 10, y_start + target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (128, 128, 128),
                    2,
                )

        for i, name in enumerate(self.camera_keys):
            row = i // self.grid_cols
            col = i % self.grid_cols
            y_start = row * target_h
            x_start = col * target_w
            cv2.rectangle(
                grid,
                (x_start, y_start),
                (x_start + len(name) * 10 + 10, y_start + 25),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                grid,
                name,
                (x_start + 5, y_start + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return grid

    def _read_camera_images(self) -> List[Optional[np.ndarray]]:
        if self.robot_shm is None:
            return [None] * len(self.camera_keys)

        try:
            data = self.robot_shm.read(blocking=False, skip_unchanged=False)
        except Exception:
            data = None

        images: List[Optional[np.ndarray]] = []
        if not isinstance(data, dict):
            return [None] * len(self.camera_keys)

        for key in self.camera_keys:
            img = data.get(key, None)
            if isinstance(img, np.ndarray):
                if len(img.shape) == 3 and img.shape[2] == 3:
                    convert_code = cv2.COLOR_BGR2RGB if self.bgr_to_rgb else cv2.COLOR_RGB2BGR
                    img = cv2.cvtColor(img, convert_code)
                images.append(img)
            else:
                images.append(None)
        return images

    def visualize(self) -> bool:
        images = self._read_camera_images()
        grid = self._create_grid_image(images)
        cv2.imshow(self.window_name, grid)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            return False
        return True

    def cleanup(self):
        cv2.destroyAllWindows()

    def start(self):
        if not self.connect():
            print("[MujocoCameraVisualizer] No robot SHM connected, exiting")
            return
        if not self.setup():
            logger.error("[MujocoCameraVisualizer] Setup failed")
            return

        self.is_running = True
        frame_interval = 1.0 / self.fps
        last_frame_time = 0.0

        print(
            f"[MujocoCameraVisualizer] Started for {self.robot_shm_name} "
            f"with {len(self.camera_keys)} cameras at {self.fps} FPS"
        )
        print("[MujocoCameraVisualizer] Press 'q' or ESC to close")

        try:
            while self.is_running:
                current_time = time.time()
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                last_frame_time = current_time
                if not self.visualize():
                    break
        except KeyboardInterrupt:
            logger.info("[MujocoCameraVisualizer] Interrupted")
        finally:
            self.cleanup()
            if self.robot_shm is not None:
                try:
                    self.robot_shm.destroy()
                except Exception:
                    pass
            logger.info("[MujocoCameraVisualizer] Stopped")

    def stop(self):
        self.is_running = False

    def close(self):
        self.stop()
        if self.robot_shm is not None:
            try:
                self.robot_shm.destroy()
            except Exception:
                pass
            self.robot_shm = None
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def start_mujoco_camera_visualizer(robot_shm_name: str, camera_keys: List[str], **kwargs):
    visualizer = MujocoCameraVisualizer(
        robot_shm_name=robot_shm_name,
        camera_keys=camera_keys,
        **kwargs,
    )
    visualizer.start()


Visualizer = MujocoCameraVisualizer
