"""Camera-feed visualizer for ``vlastudio collect``.

A ``BaseVisualizer`` that reads a robot's observation stream from its shared
memory and shows the camera images in a window with OpenCV. Add it as a
visualizer row in a robot YAML and ``vlastudio collect --visualize`` starts it as
a subprocess that lives as long as the collect session.

Cross-platform / headless policy (shared with ``LiveCameraViewer``):
- Windows: a window opens.
- POSIX: a window opens only when ``DISPLAY`` is set (physical X server or
  Xvfb). Headless sessions read frames without a window (``show=False`` forces
  headless even with a display).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from vlastudio.deploy.camera_viewer import _to_bgr, display_available
from vlastudio.deploy.visualizer.base import BaseVisualizer


class CameraViewVisualizer(BaseVisualizer):
    """Display one robot's camera frames read from its shared memory channel."""

    def __init__(
        self,
        shm_name: str,
        camera_names: Optional[Sequence[str]] = None,
        fps: float = 30.0,
        window_name: str = "Camera",
        window_scale: float = 1.0,
        show: bool = True,
        **kwargs,
    ):
        super().__init__(shm_name=shm_name, fps=fps)
        self.camera_names = list(camera_names) if camera_names else None
        self.window_name = str(window_name)
        self.window_scale = float(window_scale)
        self.show = bool(show) and display_available()
        self._waiting_logged = False

    def setup(self) -> bool:
        if self.show:
            import cv2

            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            except cv2.error:
                pass
        print(
            f"[CameraViewVisualizer] window={'on' if self.show else 'off (headless)'} "
            f"fps={self.fps} shm={self.shm_name}",
            flush=True,
        )
        return True

    def _extract(self, data: Dict) -> List[np.ndarray]:
        import cv2

        frames: List[np.ndarray] = []
        names = self.camera_names or [k for k in data.keys()]
        for name in names:
            image = data.get(name)
            if image is None or not isinstance(image, np.ndarray):
                continue
            if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
                continue
            bgr = _to_bgr(image)
            if self.window_scale != 1.0:
                bgr = cv2.resize(
                    bgr,
                    (int(bgr.shape[1] * self.window_scale), int(bgr.shape[0] * self.window_scale)),
                )
            frames.append(bgr)
        return frames

    def visualize(self, data: dict) -> bool:
        import cv2

        frames = self._extract(data)
        if not frames:
            if not self._waiting_logged:
                print(
                    f"[CameraViewVisualizer] waiting for camera frames (keys={sorted(data.keys())[:8]})",
                    flush=True,
                )
                self._waiting_logged = True
            return True
        self._waiting_logged = False

        if self.show:
            if len(frames) == 1:
                view = frames[0]
            else:
                view = np.hstack(frames)
            cv2.imshow(self.window_name, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return False
        return True

    def cleanup(self):
        if self.show:
            import cv2

            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
