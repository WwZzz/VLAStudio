"""
OpenCV camera device: init and capture logic aligned with lerobot-style flow.
"""

import math
import platform
import time
from typing import Optional, Union
import cv2
from deploy.base import BaseDevice
import numpy as np
from loguru import logger

def _get_cv2_backend() -> int:
    """Choose OpenCV VideoCapture backend by platform."""
    if platform.system() == "Windows":
        return int(cv2.CAP_MSMF)
    return int(cv2.CAP_V4L2)

class OpenCVCamera(BaseDevice):
    """
    Camera device using OpenCV VideoCapture.
    Supports device index (e.g. 0) or path (e.g. '/dev/video0').
    """

    def __init__(
        self,
        name: str,
        index_or_path: Union[int, str],
        max_size_mb: int = 64,
        fps: float = 30.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        warmup_s: float = 1.0,
        fourcc: Optional[str] = None,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        bgr_to_rgb: bool = False,
    ):
        super().__init__(name, max_size_mb, fps)
        self.index_or_path = index_or_path
        self.width = width
        self.height = height
        self.warmup_s = warmup_s
        self.fourcc = fourcc
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.bgr_to_rgb = bgr_to_rgb
        self._backend = _get_cv2_backend()
        self.camera: Optional[cv2.VideoCapture] = None
        self._capture_width: Optional[int] = None
        self._capture_height: Optional[int] = None
        self._read_fail_count = 0
        self._read_ok_count = 0
        self._last_fail_log_t = 0.0

        self._open_and_configure()

    def _open_and_configure(self) -> None:
        """Open camera, apply FPS/resolution/fourcc, then optional warmup."""
        self.camera = cv2.VideoCapture(self.index_or_path, self._backend)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            raise ConnectionError(
                f"OpencvCamera: failed to open {self.index_or_path}. "
                "Check device index or path (e.g. /dev/video0)."
            )

        self._configure_capture_settings()
        if self.warmup_s > 0:
            self._warmup()

    def _configure_capture_settings(self) -> None:
        """Set FOURCC (if given), then width/height, then FPS."""
        cap = self.camera
        if cap is None:
            return

        if self.fourcc is not None:
            if len(self.fourcc) != 4:
                raise ValueError(f"fourcc must be 4 characters, got {self.fourcc!r}")
            fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)

        default_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        default_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if self.width is None or self.height is None:
            self._capture_width = default_w
            self._capture_height = default_h
            self.width = default_w
            self.height = default_h
        else:
            self._capture_width = self.width
            self._capture_height = self.height
            ok_w = cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            ok_h = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            actual_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            actual_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if not ok_w or actual_w != self.width:
                raise RuntimeError(
                    f"OpencvCamera: failed to set width={self.width} (actual={actual_w})"
                )
            if not ok_h or actual_h != self.height:
                raise RuntimeError(
                    f"OpencvCamera: failed to set height={self.height} (actual={actual_h})"
                )

        if self.fps is not None:
            ok = cap.set(cv2.CAP_PROP_FPS, float(self.fps))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            if not ok or not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
                raise RuntimeError(
                    f"OpencvCamera: failed to set fps={self.fps} (actual={actual_fps})"
                )

    def _warmup(self) -> None:
        """Read frames for warmup_s seconds to stabilize camera."""
        start = time.time()
        while time.time() - start < self.warmup_s:
            self.camera.read()
            time.sleep(0.1)

    @property
    def is_connected(self) -> bool:
        return (
            self.camera is not None
            and isinstance(self.camera, cv2.VideoCapture)
            and self.camera.isOpened()
        )

    def get_data(self) -> Optional[dict]:
        if self.camera is None:
            return None
        ret, image = self.camera.read()
        if not ret or image is None:
            self._read_fail_count += 1
            now = time.time()
            if (
                self._read_fail_count in (1, 10, 30)
                or (now - self._last_fail_log_t) >= 5.0
            ):
                self._last_fail_log_t = now
                hint = (
                    "stream lost after earlier frames — check cable/USB power, "
                    "or dmesg 'Not enough bandwidth' if sharing a hub"
                    if self._read_ok_count > 0
                    else (
                        "no frames yet — if another camera works on the same hub, "
                        "check dmesg for 'Not enough bandwidth' and use a different USB port"
                    )
                )
                logger.warning(
                    f"[{self.name}] Camera read failed "
                    f"(fails={self._read_fail_count}, oks={self._read_ok_count}, "
                    f"path={self.index_or_path}). {hint}."
                )
            return None
        # Recovered from failures.
        if self._read_fail_count > 0:
            logger.info(
                f"[{self.name}] Camera read recovered after {self._read_fail_count} fails"
            )
            self._read_fail_count = 0
        self._read_ok_count += 1
        if self.horizontal_flip and self.vertical_flip:
            image = cv2.flip(image, -1)  # -1 = both horizontal and vertical flip
        elif self.horizontal_flip:
            image = cv2.flip(image, 1)  # 1 = horizontal flip
        elif self.vertical_flip:
            image = cv2.flip(image, 0)  # 0 = vertical flip
        if self.bgr_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {"image": image, "timestamp": time.perf_counter()}

    def _try_reopen(self) -> bool:
        """Release and reopen capture (USB unplug/replug or stream stall)."""
        try:
            if self.camera is not None:
                self.camera.release()
        except Exception:
            pass
        self.camera = None
        try:
            self._open_and_configure()
            logger.info(f"[{self.name}] Camera reopened: {self.index_or_path}")
            return True
        except Exception as e:
            logger.warning(f"[{self.name}] Camera reopen failed: {e}")
            self.camera = None
            return False

    @staticmethod
    def obs2meta(device_data: dict) -> dict:
        """Return image in CHW uint8 format for MetaObs assembly."""
        if device_data is None:
            return {}
        img = device_data.get('image')
        if img is None or not isinstance(img, np.ndarray):
            return {}
        if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
        elif img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.ascontiguousarray(img)
        else:
            return {}
        return {'image': img[np.newaxis]}  # (1, C, H, W)

    def close(self) -> None:
        super().close()
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None

    def start(self):
        """Override: camera.read() is already blocking, no rate limiter needed."""
        from deploy.shm_utils import SharedMemoryChannel
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)
        self.is_running = True
        # Print actual camera settings for debugging
        if self.camera is not None:
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            fourcc_int = int(self.camera.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])
            w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(
                f"[{self.name}] Camera opened: {w}x{h} @ {actual_fps:.1f}fps, fourcc={fourcc_str}"
            )
        consecutive_fail = 0
        last_reopen_t = 0.0
        while self.is_running:
            data = self.get_data()
            if data is not None:
                consecutive_fail = 0
                self.write_data_to_shm(data)
                continue
            # Failed read: do not busy-spin (can hit millions of fails/sec).
            consecutive_fail += 1
            now = time.time()
            if consecutive_fail >= 30 and (now - last_reopen_t) >= 2.0:
                last_reopen_t = now
                self._try_reopen()
                consecutive_fail = 0
            time.sleep(0.05)

# ==============================================================================
# Test (Multi-camera: read config, start one process per camera, display 3 views)
# ==============================================================================

if __name__ == "__main__":
    import multiprocessing as mp
    import time
    import yaml
    from pathlib import Path

    import numpy as np
    import cv2

    from deploy.base import start_device
    from deploy.shm_utils import SharedMemoryChannel

    # Load config (list of camera configs)
    cfg_path = Path(__file__).resolve().parents[3] / "configs" / "robot" / "camera_multiple_views.yaml"
    with open(cfg_path, "r") as f:
        camera_configs = yaml.safe_load(f)

    # Start one process per camera
    procs = []
    shm_names = []
    for cfg in camera_configs:
        shm_names.append(cfg["args"]["name"])
        p = mp.Process(target=start_device, args=(cfg,))
        p.start()
        procs.append(p)

    time.sleep(1.5)
    print("Connecting to camera SHM channels...\n")

    # Connect to each camera's SHM
    shm_channels = []
    placeholder_h, placeholder_w = 480, 640
    for name in shm_names:
        try:
            ch = SharedMemoryChannel(name, is_writer=False, timeout=5.0)
            shm_channels.append((name, ch))
        except Exception:
            shm_channels.append((name, None))

    placeholders = {name: np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8) for name in shm_names}
    for name in shm_names:
        cv2.putText(
            placeholders[name], f"{name} (no data)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )

    win_name = "Camera Views (head | left_wrist | right_wrist)"
    try:
        while True:
            frames = []
            for name, ch in shm_channels:
                if ch is None:
                    frames.append(placeholders[name].copy())
                    continue
                data = ch.read(blocking=False)
                if data is not None and "image" in data:
                    img = data["image"]
                    if isinstance(img, np.ndarray):
                        # Add label
                        labeled = img.copy()
                        cv2.putText(labeled, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        frames.append(labeled)
                    else:
                        frames.append(placeholders[name].copy())
                else:
                    frames.append(placeholders[name].copy())

            if frames:
                stitched = np.hstack(frames)
                cv2.imshow(win_name, stitched)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        for p in procs:
            p.terminate()
            p.join(timeout=2.0)
            if p.is_alive():
                p.kill()
        print("Done.")
 