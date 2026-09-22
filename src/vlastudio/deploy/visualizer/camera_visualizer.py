"""
Camera Visualizer - Display camera images from shared memory

This visualizer can display multiple camera feeds in a single window,
reading image data from multiple shared memory channels.
"""

import importlib.util
import os
import sys

# Qt in OpenCV writes to fd 2 directly, so we must create the font dir it expects
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
from typing import List, Optional, Tuple
from loguru import logger

from vlastudio.deploy.shm_utils import SharedMemoryChannel

from vlastudio.deploy.visualizer.base import BaseVisualizer


class CameraVisualizer(BaseVisualizer):
    """
    Visualizer for camera images from shared memory.
    
    Supports multiple cameras displayed in a grid layout.
    Unlike BaseVisualizer, this class manages multiple SHM channels.
    """
    
    def __init__(
        self,
        shm_names: List[str],
        window_name: str = "Camera View",
        fps: float = 30.0,
        grid_cols: Optional[int] = None,
        scale: float = 1.0,
        reconnect_interval_s: float = 1.0,
        initial_connect_timeout_s: float = 5.0,
        per_camera_connect_timeout_s: float = 0.2,
        **kwargs
    ):
        """
        Initialize the camera visualizer.
        
        Args:
            shm_names: List of shared memory names for cameras
            window_name: OpenCV window title
            fps: Target display frame rate
            grid_cols: Number of columns in grid layout (auto if None)
            scale: Scale factor for display (0.5 = half size)
        """
        first = shm_names[0] if shm_names else ""
        super().__init__(shm_name=first, fps=fps, **kwargs)
        self.shm_names = list(shm_names)
        self.window_name = window_name
        self.scale = scale
        
        # Auto-calculate grid layout
        n_cameras = len(shm_names)
        if grid_cols is None:
            if n_cameras <= 2:
                grid_cols = n_cameras
            elif n_cameras <= 4:
                grid_cols = 2
            else:
                grid_cols = 3
        self.grid_cols = grid_cols
        self.grid_rows = (n_cameras + grid_cols - 1) // grid_cols
        
        # SHM channels
        self.shm_channels: List[Optional[SharedMemoryChannel]] = [None] * n_cameras
        self.reconnect_interval_s = float(reconnect_interval_s)
        self.initial_connect_timeout_s = float(initial_connect_timeout_s)
        self.per_camera_connect_timeout_s = float(per_camera_connect_timeout_s)
        self._last_reconnect_ts = 0.0
        self._missing_once: set[str] = set()

    def _try_connect_missing_channels(self) -> int:
        """Try to connect any missing camera SHM without blocking startup for all cameras."""
        connected_now = 0
        for i, name in enumerate(self.shm_names):
            if self.shm_channels[i] is not None:
                continue
            try:
                self.shm_channels[i] = SharedMemoryChannel(
                    name,
                    is_writer=False,
                    timeout=self.per_camera_connect_timeout_s,
                )
                logger.info(f"[CameraVisualizer] Connected to camera SHM: {name}")
                connected_now += 1
            except Exception:
                if name not in self._missing_once:
                    logger.info(f"[CameraVisualizer] Waiting for camera SHM '{name}'...")
                    self._missing_once.add(name)
        return connected_now
        
    def connect(self, timeout: float = 30.0) -> bool:
        """
        Connect to all camera shared memory channels.
        
        Args:
            timeout: Maximum time to wait for each SHM
            
        Returns:
            True if at least one camera connected
        """
        deadline = time.time() + timeout
        connected_count = 0
        while time.time() < deadline:
            connected_count += self._try_connect_missing_channels()
            if any(ch is not None for ch in self.shm_channels):
                return True
            time.sleep(0.1)

        for i, name in enumerate(self.shm_names):
            if self.shm_channels[i] is None:
                logger.warning(f"[CameraVisualizer] Failed to connect to camera SHM: {name}")
        return False
    
    def setup(self) -> bool:
        """Setup OpenCV window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        return True
    
    def _create_grid_image(self, images: List[Optional[np.ndarray]]) -> np.ndarray:
        """
        Create a grid image from multiple camera images.
        
        Args:
            images: List of images (None for missing cameras)
            
        Returns:
            Combined grid image
        """
        # Find the max dimensions
        max_h, max_w = 0, 0
        for img in images:
            if img is not None:
                h, w = img.shape[:2]
                max_h = max(max_h, h)
                max_w = max(max_w, w)
        
        if max_h == 0 or max_w == 0:
            # No valid images, return placeholder
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Apply scale
        target_h = int(max_h * self.scale)
        target_w = int(max_w * self.scale)
        
        # Create grid
        grid_h = self.grid_rows * target_h
        grid_w = self.grid_cols * target_w
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for i, img in enumerate(images):
            row = i // self.grid_cols
            col = i % self.grid_cols
            y_start = row * target_h
            x_start = col * target_w
            
            if img is not None:
                # Resize if needed
                if img.shape[:2] != (target_h, target_w):
                    img = cv2.resize(img, (target_w, target_h))
                
                # Handle grayscale
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
                grid[y_start:y_start+target_h, x_start:x_start+target_w] = img
            else:
                # Draw placeholder with camera name
                cv2.putText(
                    grid,
                    f"Camera {i}: No Data",
                    (x_start + 10, y_start + target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (128, 128, 128),
                    2
                )
        
        # Add camera labels
        for i, name in enumerate(self.shm_names):
            row = i // self.grid_cols
            col = i % self.grid_cols
            y_start = row * target_h
            x_start = col * target_w
            
            # Draw label background
            cv2.rectangle(
                grid,
                (x_start, y_start),
                (x_start + len(name) * 10 + 10, y_start + 25),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                grid,
                name,
                (x_start + 5, y_start + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return grid
    
    def visualize(self, data: dict) -> bool:
        """Satisfies ``BaseVisualizer``; ``data`` is unused (multi-SHM read inside)."""
        return self._visualize_frame()

    def _visualize_frame(self) -> bool:
        """
        Read from all cameras and display.
        
        Returns:
            True to continue, False to stop
        """
        if time.time() - self._last_reconnect_ts >= self.reconnect_interval_s:
            self._last_reconnect_ts = time.time()
            self._try_connect_missing_channels()

        images = []
        
        for i, shm in enumerate(self.shm_channels):
            if shm is None:
                images.append(None)
                continue
            
            try:
                data = shm.read(blocking=False, skip_unchanged=False)
                if data is not None and 'image' in data:
                    img = data['image']
                    # Convert from RGB to BGR for OpenCV if needed
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    images.append(img)
                else:
                    images.append(None)
            except Exception:
                images.append(None)
        
        # Create and display grid
        grid = self._create_grid_image(images)
        cv2.imshow(self.window_name, grid)
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            return False
        
        return True
    
    def cleanup(self):
        """Cleanup OpenCV resources."""
        cv2.destroyAllWindows()
    
    def start(self):
        """Main loop: connect, setup, and continuously display camera images."""
        if not self.connect(timeout=self.initial_connect_timeout_s):
            logger.error("[CameraVisualizer] No cameras connected, exiting")
            return
        
        if not self.setup():
            logger.error("[CameraVisualizer] Setup failed")
            return
        
        self.is_running = True
        frame_interval = 1.0 / self.fps
        last_frame_time = 0.0
        
        logger.info(
            f"[CameraVisualizer] Started with {len(self.shm_names)} cameras at {self.fps} FPS; "
            "press 'q' or ESC to close"
        )
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Rate limiting
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                
                last_frame_time = current_time
                
                if not self._visualize_frame():
                    break
                    
        except KeyboardInterrupt:
            logger.info("[CameraVisualizer] Interrupted")
        finally:
            self.cleanup()
            for shm in self.shm_channels:
                if shm is not None:
                    try:
                        shm.destroy()
                    except Exception:
                        pass
            logger.info("[CameraVisualizer] Stopped")
    
    def stop(self):
        """Stop the visualization loop."""
        self.is_running = False


def start_camera_visualizer(shm_names: List[str], **kwargs):
    """
    Start the camera visualizer.
    
    This function is called by vlastudio collect to start the camera visualizer.
    
    Args:
        shm_names: List of camera shared memory names
        **kwargs: Additional arguments passed to CameraVisualizer
    """
    visualizer = CameraVisualizer(shm_names=shm_names, **kwargs)
    visualizer.start()


# Alias for consistency with device module pattern
Visualizer = CameraVisualizer


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    import multiprocessing as mp
    
    # Test with dummy camera names
    logger.info("Testing CameraVisualizer (will fail to connect if no cameras are running).")
    
    visualizer = CameraVisualizer(
        shm_names=["test_camera_1", "test_camera_2"],
        fps=30.0,
        scale=0.5,
    )
    
    # This will timeout since no cameras are running
    visualizer.start()
