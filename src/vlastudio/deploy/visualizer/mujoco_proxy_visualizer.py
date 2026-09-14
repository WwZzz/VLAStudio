"""
Generic proxy visualizer for single-context MuJoCo robots.

This process does not create any MuJoCo model/data/viewer. It only:
1. reads viewer state from the robot-owned SHM channel
2. writes viewer commands to the robot-owned command SHM channel
3. optionally shows a small status window for keyboard control
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 may be unavailable in headless envs
    cv2 = None

from vlastudio.deploy.shm_utils import SharedMemoryChannel
from vlastudio.deploy.simulation.mujoco import (
    get_mujoco_viewer_command_shm_name,
    get_mujoco_viewer_state_shm_name,
)
from vlastudio.deploy.visualizer.base import BaseVisualizer


class MujocoProxyVisualizer(BaseVisualizer):
    def __init__(
        self,
        shm_name: str,
        robot_shm_name: Optional[str] = None,
        viewer_command_shm_name: Optional[str] = None,
        viewer_state_shm_name: Optional[str] = None,
        default_camera_name: Optional[str] = None,
        auto_open: bool = True,
        show_tcp_frame: bool = True,
        fps: float = 15.0,
        window_name: str = "MuJoCo Viewer Proxy",
        show_window: bool = False,
        **kwargs,
    ):
        del kwargs
        robot_shm_name = robot_shm_name or shm_name
        self.robot_shm_name = robot_shm_name
        self.viewer_command_shm_name = viewer_command_shm_name or get_mujoco_viewer_command_shm_name(robot_shm_name)
        self.viewer_state_shm_name = viewer_state_shm_name or get_mujoco_viewer_state_shm_name(robot_shm_name)
        super().__init__(shm_name=self.viewer_state_shm_name, fps=fps)

        self.default_camera_name = default_camera_name
        self.auto_open = bool(auto_open)
        self.show_tcp_frame = bool(show_tcp_frame)
        self.window_name = window_name
        self.show_window = bool(show_window)

        self.command_shm: Optional[SharedMemoryChannel] = None
        self.gui_available = cv2 is not None and self.show_window
        self._last_status_line = None
        self._last_open_request_time = 0.0

    def connect(self, timeout: float = 30.0) -> bool:
        try:
            self.command_shm = SharedMemoryChannel(
                name=self.viewer_command_shm_name,
                max_size_mb=1,
                is_writer=True,
            )
            print(f"[MujocoProxyVisualizer] Command SHM ready: {self.viewer_command_shm_name}")
        except Exception as e:
            print(f"[MujocoProxyVisualizer] Failed to create command SHM '{self.viewer_command_shm_name}': {e}")
            return False
        if not super().connect(timeout=timeout):
            return False
        print(f"[MujocoProxyVisualizer] Connected to viewer state SHM: {self.viewer_state_shm_name}")
        return True

    def _send_command(self, cmd: str, **payload) -> None:
        if self.command_shm is None:
            return
        message = {"cmd": cmd}
        message.update(payload)
        self.command_shm.write(message)

    def _ensure_desired_remote_state(self, state: dict) -> None:
        now = time.time()
        viewer_running = bool(state.get("viewer_running", False))

        # Command SHM is last-write-wins, so issue at most one command per frame.
        if self.auto_open and not viewer_running:
            if now - self._last_open_request_time > 0.5:
                self._send_command("open")
                self._last_open_request_time = now
            return

        if state.get("show_tcp_frame") != self.show_tcp_frame:
            self._send_command("toggle_tcp_frame", value=self.show_tcp_frame)
            return

        desired_camera = self.default_camera_name
        if desired_camera and state.get("selected_camera") != desired_camera:
            available = state.get("available_cameras", []) or []
            if desired_camera in available:
                self._send_command("set_camera", camera_name=desired_camera)
                return

    def setup(self) -> bool:
        if not self.gui_available:
            print("[MujocoProxyVisualizer] Running in headless proxy mode")
            return True
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            return True
        except cv2.error as e:
            print(f"[MujocoProxyVisualizer] OpenCV GUI unavailable: {e}")
            self.gui_available = False
            return True

    def _render_status_image(self, state: dict) -> np.ndarray:
        canvas = np.zeros((360, 920, 3), dtype=np.uint8)
        fg = (240, 240, 240)
        dim = (160, 160, 160)
        y = 28

        def put(line: str, color=fg, scale: float = 0.6):
            nonlocal y
            cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += 28

        put(f"Robot SHM: {self.robot_shm_name}")
        put(f"Viewer status: {state.get('viewer_status', 'unknown')}")
        put(f"Viewer running: {state.get('viewer_running', False)}")
        put(f"Selected camera: {state.get('selected_camera', '<none>')}")
        put(f"TCP frame: {state.get('show_tcp_frame', False)}")
        put(f"Scene: {state.get('scene_name') or '<none>'}")
        put(f"Last error: {state.get('last_error') or '<none>'}", color=(120, 200, 255))
        y += 10
        put("Controls:", color=(120, 255, 120))
        put("o=open viewer, c=close viewer, t=toggle tcp frame", color=dim, scale=0.55)
        put("1-9=switch camera, q/ESC=quit proxy", color=dim, scale=0.55)
        y += 10
        put("Available cameras:", color=(120, 255, 120))

        cameras = state.get("available_cameras", []) or []
        if cameras:
            for idx, camera_name in enumerate(cameras[:9], start=1):
                put(f"{idx}. {camera_name}", color=dim, scale=0.55)
        else:
            put("<none>", color=dim, scale=0.55)
        return canvas

    def _handle_key(self, key: int, state: dict) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord("o"):
            self._send_command("open")
        elif key == ord("c"):
            self._send_command("close")
        elif key == ord("t"):
            self._send_command("toggle_tcp_frame")
        elif ord("1") <= key <= ord("9"):
            cameras = state.get("available_cameras", []) or []
            index = key - ord("1")
            if index < len(cameras):
                self._send_command("set_camera", camera_name=cameras[index])
        return True

    def visualize(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return True
        self._ensure_desired_remote_state(data)

        if not self.gui_available:
            status_line = (
                data.get("viewer_status"),
                data.get("viewer_running"),
                data.get("selected_camera"),
                data.get("last_error"),
            )
            if status_line != self._last_status_line:
                print(
                    "[MujocoProxyVisualizer] status={} running={} camera={} error={}".format(
                        data.get("viewer_status"),
                        data.get("viewer_running"),
                        data.get("selected_camera"),
                        data.get("last_error") or "",
                    )
                )
                self._last_status_line = status_line
            return True

        frame = self._render_status_image(data)
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return self._handle_key(key, data)

    def cleanup(self):
        try:
            self._send_command("close")
        except Exception:
            pass
        if self.command_shm is not None:
            try:
                self.command_shm.destroy()
            except Exception:
                pass
            self.command_shm = None
        if self.gui_available:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass


def start_mujoco_proxy_visualizer(shm_name: str, **kwargs):
    MujocoProxyVisualizer(shm_name=shm_name, **kwargs).start()
