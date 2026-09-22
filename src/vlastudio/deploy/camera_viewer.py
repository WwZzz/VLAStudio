"""Cross-platform live camera feed viewer for robot/sim devices.

Reads the observation stream a robot or simulated robot publishes to its shared
memory and displays the camera images in real time with OpenCV (``cv2.imshow``).

Display policy (platform- and headless-aware):

- Windows: a windowing desktop is assumed available, so the window opens.
- Linux/macOS: the window opens only when a display is present (``DISPLAY``
  is set). In a headless session without ``DISPLAY`` the viewer logs each frame
  and keeps reading without opening a window.
- To render into a virtual screen, run under Xvfb (e.g. ``xvfb-run`` or set
  ``DISPLAY=:99`` with an Xvfb server); ``cv2.imshow`` then draws into the
  virtual display and the API behaves exactly as with a real display.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Sequence

import numpy as np

from vlastudio.deploy.shm_utils import SharedMemoryChannel


def display_available() -> bool:
    """Best-effort check that ``cv2.imshow`` can open a window.

    Windows always has a desktop; POSIX requires a live ``DISPLAY`` (a physical
    X server, Wayland via XWayland, or Xvfb).
    """
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY"))


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """Mujoco/OpenCV convention: sim renders RGB; cv2.imshow expects BGR."""
    if image.ndim == 3 and image.shape[2] == 3:
        return image[..., ::-1]
    return image


class LiveCameraViewer:
    """Read robot observations from a SHM channel and show camera frames live.

    Args:
        shm_name: Name of the robot device's shared memory channel.
        camera_names: Which camera keys to show; default all present per frame.
        fps: Target display frame rate.
        title: OpenCV window title.
        window_scale: Scale factor for the displayed frame (0.5, 1.0, 2.0...).
        show: If False, never opens a window (headless mode even with display).
    """

    def __init__(
        self,
        shm_name: str,
        camera_names: Optional[Sequence[str]] = None,
        fps: float = 30.0,
        title: str = "Camera",
        window_scale: float = 1.0,
        show: bool = True,
    ):
        self.shm_name = shm_name
        self.camera_names = list(camera_names) if camera_names else None
        self.fps = float(fps)
        self.title = title
        self.window_scale = float(window_scale)
        self.show_requested = bool(show)
        self.display_ok = display_available()
        self.show = self.show_requested and self.display_ok

    def _make_window_name(self, camera_name: str) -> str:
        if len(self.camera_names) == 1:
            return self.title
        return f"{self.title} - {camera_name}"

    def run(self, max_frames: Optional[int] = None) -> int:
        """Read and display frames until stopped or ``max_frames`` reached.

        Returns the number of frames processed.
        """
        import cv2

        shm = SharedMemoryChannel(self.shm_name, is_writer=False, timeout=30.0)
        print(f"[LiveCameraViewer] Connected to SHM '{self.shm_name}'", flush=True)
        if not self.show:
            if self.show_requested and not self.display_ok:
                print(
                    "[LiveCameraViewer] Headless: no windowing display detected "
                    "(set DISPLAY to a real X server or Xvfb to enable).",
                    flush=True,
                )
            else:
                print(
                    "[LiveCameraViewer] Window disabled (show=False).",
                    flush=True,
                )

        self._prepare_windows()
        frame_interval = 1.0 / max(self.fps, 1.0)
        last = 0.0
        count = 0
        try:
            while True:
                now = time.monotonic()
                if now - last < frame_interval:
                    time.sleep(0.001)
                    continue
                last = now

                data = shm.read(blocking=False, skip_unchanged=False)
                if data is None:
                    continue

                frames = self._extract_frames(data)
                if not frames and count % 30 == 0:
                    print(
                        f"[LiveCameraViewer] frame {count}: no camera keys in obs "
                        f"(have {sorted(data.keys())[:10]})",
                        flush=True,
                    )

                for name, image in frames.items():
                    if self.show:
                        cv2.imshow(self._make_window_name(name), image)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            raise KeyboardInterrupt
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
        except KeyboardInterrupt:
            print("\n[LiveCameraViewer] Stopped by user", flush=True)
        finally:
            if self.show:
                cv2.destroyAllWindows()
            try:
                shm.destroy()
            except Exception:
                pass
        return count

    def _prepare_windows(self) -> None:
        if not self.show:
            return
        import cv2

        try:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        except cv2.error:
            pass

    def _extract_frames(self, data: Dict) -> Dict[str, np.ndarray]:
        """Pull camera images (keyed by camera name) out of an observation dict."""
        import cv2

        frames: Dict[str, np.ndarray] = {}
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
            frames[name] = bgr
        return frames
