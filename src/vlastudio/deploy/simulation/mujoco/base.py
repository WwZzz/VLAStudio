from __future__ import annotations

import importlib
import inspect
import os
import tempfile
import threading
import time
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple
from xml.sax.saxutils import escape
import sys

import mujoco
import numpy as np

from vlastudio.benchmark.base import MetaObs
from vlastudio.deploy.robot.base import BaseRobot
from vlastudio.deploy.shm_utils import SharedMemoryChannel


SCENES_DIR = Path(__file__).resolve().parent / "scenes"
_MUJOCO_WARNING_HANDLER_INSTALLED = False
_MUJOCO_WARNING_CALLBACK = None
VIEWER_COMMAND_SHM_SUFFIX = "__viewer_cmd"
VIEWER_STATE_SHM_SUFFIX = "__viewer_state"

# Shared `table_block.xml` static geoms — used to detect arm/base penetration into scene props.
TABLE_BLOCK_STATIC_GEOM_NAMES: frozenset = frozenset(
    {
        "floor",
        "table_top",
        "table_leg_fl",
        "table_leg_fr",
        "table_leg_rl",
        "table_leg_rr",
    }
)
# Contacts involving these non-static geoms are ignored (e.g. free-moving block).
TABLE_BLOCK_PAIR_IGNORE_OTHER_GEOMS: frozenset = frozenset({"block_geom"})
DEFAULT_STATIC_COLLISION_PENETRATION_SLOP: float = 0.0015


def _cleanup_mujoco_log_file() -> None:
    log_path = Path.cwd() / "MUJOCO_LOG.TXT"
    if log_path.exists():
        try:
            log_path.unlink()
        except OSError:
            pass


def _install_mujoco_warning_handler() -> None:
    """
    Install a process-local MuJoCo warning handler.

    Without a user warning callback MuJoCo writes warnings to `MUJOCO_LOG.TXT`.
    We redirect them to stderr instead so normal runs stay clean.
    """
    global _MUJOCO_WARNING_HANDLER_INSTALLED, _MUJOCO_WARNING_CALLBACK
    if _MUJOCO_WARNING_HANDLER_INSTALLED:
        return

    _cleanup_mujoco_log_file()

    def _warning_callback(message):
        text = message.decode("utf-8", errors="replace") if isinstance(message, bytes) else str(message)
        print(f"[MuJoCo Warning] {text}", file=sys.stderr)

    if hasattr(mujoco, "set_mju_user_warning"):
        mujoco.set_mju_user_warning(_warning_callback)
        _MUJOCO_WARNING_CALLBACK = _warning_callback
        _MUJOCO_WARNING_HANDLER_INSTALLED = True
        return

    if hasattr(mujoco, "set_mjcb_warning"):
        mujoco.set_mjcb_warning(_warning_callback)
        _MUJOCO_WARNING_CALLBACK = _warning_callback
        _MUJOCO_WARNING_HANDLER_INSTALLED = True
        return

    _MUJOCO_WARNING_HANDLER_INSTALLED = True


def get_scene_directory() -> Path:
    return SCENES_DIR


def get_mujoco_viewer_command_shm_name(robot_name: str) -> str:
    return f"{robot_name}{VIEWER_COMMAND_SHM_SUFFIX}"


def get_mujoco_viewer_state_shm_name(robot_name: str) -> str:
    return f"{robot_name}{VIEWER_STATE_SHM_SUFFIX}"


def _abspath(path: str | os.PathLike[str]) -> str:
    return str(Path(path).expanduser().resolve())


def resolve_scene_xml_path(
    scene_name: Optional[str] = None,
    scene_xml_path: Optional[str] = None,
) -> Optional[str]:
    if scene_xml_path:
        return _abspath(scene_xml_path)
    if scene_name:
        candidate = SCENES_DIR / f"{scene_name}.xml"
        if not candidate.exists():
            raise FileNotFoundError(f"MuJoCo scene '{scene_name}' not found at {candidate}")
        return str(candidate.resolve())
    return None


def prepare_mujoco_xml_path(
    robot_xml_path: str,
    *,
    xml_path: Optional[str] = None,
    scene_name: Optional[str] = None,
    scene_xml_path: Optional[str] = None,
    generated_name: str = "mujoco_device",
) -> Tuple[str, Optional[str]]:
    """
    Resolve the XML file to load.

    Priority:
    1. xml_path: exact root XML, loaded unchanged
    2. robot_xml_path + scene_xml_path
    3. robot_xml_path + scene_name
    4. robot_xml_path only
    """
    if xml_path:
        return _abspath(xml_path), None

    robot_xml_path = _abspath(robot_xml_path)
    resolved_scene_path = resolve_scene_xml_path(scene_name=scene_name, scene_xml_path=scene_xml_path)
    if resolved_scene_path is None:
        return robot_xml_path, None

    robot_xml_parent = Path(robot_xml_path).parent
    robot_include_path = Path(robot_xml_path).name
    scene_include_path = os.path.relpath(resolved_scene_path, start=robot_xml_parent)

    root_xml = (
        f'<mujoco model="{escape(generated_name)}">\n'
        f'  <size nconmax="512" njmax="1024"/>\n'
        f'  <include file="{escape(robot_include_path)}"/>\n'
        f'  <include file="{escape(scene_include_path)}"/>\n'
        f"</mujoco>\n"
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        prefix=f"{generated_name}_",
        dir=robot_xml_parent,
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(root_xml)
        temp_path = f.name
    return temp_path, temp_path


class MujocoDeviceBase(BaseRobot):
    """
    Shared base class for MuJoCo simulation robots.

    Subclasses own robot-specific qpos/action semantics while this base handles:
    - XML and optional scene composition
    - `mjmodel` / `mjdata` lifecycle
    - named XML camera discovery and rendering
    - deterministic image packing into robot observations
    - optional in-process passive viewer + proxy SHM (`enable_viewer`, default True)
    - optional static-scene penetration guard for `scene_name=table_block` (see
      `enable_static_scene_contact_guard`, `_interpolate_pose_avoiding_static_contact_locked`)
    """

    def __init__(
        self,
        name: str,
        max_size_mb: int = 64,
        fps: float = 50.0,
        control_shm_name: Optional[str] = None,
        xml_path: Optional[str] = None,
        scene_name: Optional[str] = None,
        scene_xml_path: Optional[str] = None,
        camera_names: Optional[Sequence[str]] = None,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_render_fps: Optional[float] = None,
        enable_viewer: bool = True,
        viewer_command_shm_name: Optional[str] = None,
        viewer_state_shm_name: Optional[str] = None,
        viewer_default_camera: Optional[str] = None,
        viewer_auto_open: bool = False,
        viewer_show_tcp_frame: bool = True,
        viewer_show_left_ui: bool = True,
        viewer_show_right_ui: bool = True,
        viewer_update_fps: float = 60.0,
        enable_static_scene_contact_guard: Optional[bool] = None,
        static_scene_geom_names: Optional[Sequence[str]] = None,
        static_collision_penetration_slop: float = DEFAULT_STATIC_COLLISION_PENETRATION_SLOP,
        **kwargs,
    ):
        _install_mujoco_warning_handler()
        if "enable_viewer_proxy" in kwargs:
            enable_viewer = bool(kwargs.pop("enable_viewer_proxy"))
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        self.xml_path = xml_path
        self.scene_name = scene_name
        self.scene_xml_path = scene_xml_path
        if enable_static_scene_contact_guard is None:
            enable_static_scene_contact_guard = scene_name == "table_block"
        self._enable_static_scene_contact_guard = bool(enable_static_scene_contact_guard)
        self.static_collision_penetration_slop = float(static_collision_penetration_slop)
        self._static_scene_geom_names = (
            frozenset(static_scene_geom_names)
            if static_scene_geom_names is not None
            else TABLE_BLOCK_STATIC_GEOM_NAMES
        )
        self.camera_names = list(camera_names) if camera_names is not None else None
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_render_fps = float(camera_render_fps if camera_render_fps is not None else fps)
        self.enable_viewer = bool(enable_viewer)
        self.viewer_command_shm_name = viewer_command_shm_name or get_mujoco_viewer_command_shm_name(name)
        self.viewer_state_shm_name = viewer_state_shm_name or get_mujoco_viewer_state_shm_name(name)
        self.viewer_auto_open = bool(viewer_auto_open)
        self.viewer_default_camera = viewer_default_camera
        self.viewer_show_left_ui = bool(viewer_show_left_ui)
        self.viewer_show_right_ui = bool(viewer_show_right_ui)
        self.viewer_update_fps = float(viewer_update_fps)

        self._lock = threading.RLock()
        self.mjmodel = None
        self.mjdata = None
        self._renderer = None
        self._viewer = None
        self._viewer_command_shm: Optional[SharedMemoryChannel] = None
        self._viewer_state_shm: Optional[SharedMemoryChannel] = None
        self._viewer_status = "disabled" if not self.enable_viewer else "closed"
        self._viewer_last_error = ""
        self._viewer_selected_camera: Optional[str] = None
        self._viewer_show_tcp_frame = bool(viewer_show_tcp_frame)
        self._viewer_available_camera_ids: "OrderedDict[str, int]" = OrderedDict()
        self._viewer_thread: Optional[threading.Thread] = None
        self._viewer_thread_stop = threading.Event()
        self._camera_thread: Optional[threading.Thread] = None
        self._camera_thread_stop = threading.Event()
        self._camera_image_lock = threading.Lock()
        self._camera_render_data = None
        self._latest_camera_images: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._last_camera_render_time = 0.0
        self._generated_xml_path: Optional[str] = None
        self._loaded_xml_path: Optional[str] = None
        self._camera_ids: "OrderedDict[str, int]" = OrderedDict()

        robot_xml_path = self._default_robot_xml_path()
        loaded_xml_path, generated_xml_path = prepare_mujoco_xml_path(
            robot_xml_path,
            xml_path=self.xml_path,
            scene_name=self.scene_name,
            scene_xml_path=self.scene_xml_path,
            generated_name=self.__class__.__name__.lower(),
        )
        self._loaded_xml_path = loaded_xml_path
        self._generated_xml_path = generated_xml_path

        print(f"[{self.__class__.__name__}] Loading MuJoCo model from {loaded_xml_path}")
        self.mjmodel = mujoco.MjModel.from_xml_path(loaded_xml_path)
        self._configure_loaded_model()
        self.mjdata = mujoco.MjData(self.mjmodel)

        self._build_joint_indices()
        self._apply_initial_state()
        self._setup_cameras()
        self._setup_viewer()
        self._setup_viewer_overlay_state()

    @abstractmethod
    def _default_robot_xml_path(self) -> str:
        """Return the absolute path to the robot-only MJCF file."""

    @abstractmethod
    def _build_joint_indices(self) -> None:
        """Resolve robot-specific joint/actuator indices from `self.mjmodel`."""

    def _configure_loaded_model(self) -> None:
        """Optional hook to adjust the loaded MuJoCo model before creating `MjData`."""
        return None

    def _has_static_scene_penetration_locked(self) -> bool:
        """True if any contact penetrates between a static scene geom and a dynamic geom."""
        if not self._enable_static_scene_contact_guard:
            return False
        slop = self.static_collision_penetration_slop
        static_names = self._static_scene_geom_names
        for i in range(self.mjdata.ncon):
            contact = self.mjdata.contact[i]
            if contact.dist >= -slop:
                continue
            geom1 = contact.geom1
            geom2 = contact.geom2
            name1 = mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_GEOM, geom1)
            name2 = mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_GEOM, geom2)
            is_static_1 = name1 in static_names
            is_static_2 = name2 in static_names
            if is_static_1 == is_static_2:
                continue
            other_name = name2 if is_static_1 else name1
            if other_name in TABLE_BLOCK_PAIR_IGNORE_OTHER_GEOMS:
                continue
            return True
        return False

    def _interpolate_pose_avoiding_static_contact_locked(
        self,
        start_vec: np.ndarray,
        target_vec: np.ndarray,
        apply_vec: Callable[[np.ndarray], None],
        *,
        max_delta_per_step: float = 0.05,
        max_steps: int = 40,
    ) -> np.ndarray:
        """
        Move from start_vec toward target_vec in steps; stop before first static-scene penetration.
        Subclasses supply apply_vec(cand) to write cand into mjdata and mj_forward.
        """
        if not self._enable_static_scene_contact_guard:
            return np.asarray(target_vec, dtype=np.float64).copy()
        start_vec = np.asarray(start_vec, dtype=np.float64).copy()
        target_vec = np.asarray(target_vec, dtype=np.float64).copy()
        if start_vec.shape != target_vec.shape:
            raise ValueError("start_vec and target_vec must have the same shape")
        delta = float(np.max(np.abs(target_vec - start_vec)))
        steps = max(1, min(max_steps, int(np.ceil(delta / max_delta_per_step))))
        accepted = start_vec.copy()
        for idx in range(1, steps + 1):
            alpha = idx / steps
            cand = start_vec + (target_vec - start_vec) * alpha
            apply_vec(cand)
            if self._has_static_scene_penetration_locked():
                break
            accepted = cand.copy()
        return accepted

    @abstractmethod
    def _apply_initial_state(self) -> None:
        """Write initial robot state into `self.mjdata` and call `mj_forward` if needed."""

    @abstractmethod
    def _process_robot_action(self, action_dict: dict) -> dict:
        """Robot-specific action preprocessing."""

    @abstractmethod
    def _publish_robot_action(self, action: np.ndarray) -> None:
        """Robot-specific MuJoCo stepping / state updates."""

    @abstractmethod
    def _get_robot_observation_core(self) -> Dict[str, np.ndarray]:
        """Return robot state observation without camera images."""

    def connect(self) -> bool:
        return True

    def process_action(self, action_dict: dict) -> dict:
        return self._process_robot_action(action_dict)

    def publish_action(self, action: np.ndarray):
        self._publish_robot_action(action)

    def _setup_cameras(self) -> None:
        if self.mjmodel is None or self.mjmodel.ncam <= 0:
            return

        available = {
            name: cam_id
            for cam_id in range(self.mjmodel.ncam)
            if (name := mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_CAMERA, cam_id))
        }
        if not available:
            return

        self._viewer_available_camera_ids = OrderedDict((name, available[name]) for name in sorted(available.keys()))
        if self.camera_names is None:
            selected_names = sorted(available.keys())
        else:
            missing = [name for name in self.camera_names if name not in available]
            if missing:
                raise ValueError(
                    f"[{self.__class__.__name__}] camera_names not found in model: {missing}. "
                    f"available={sorted(available.keys())}"
                )
            selected_names = list(self.camera_names)

        self._camera_ids = OrderedDict((name, available[name]) for name in selected_names)
        if self._camera_ids:
            self._camera_render_data = mujoco.MjData(self.mjmodel)
            self._latest_camera_images = OrderedDict()
            self._last_camera_render_time = 0.0

    def _setup_viewer(self) -> None:
        if not self.enable_viewer:
            return
        if self.viewer_default_camera and self.viewer_default_camera in self._viewer_available_camera_ids:
            self._viewer_selected_camera = self.viewer_default_camera

    def _setup_viewer_overlay_state(self) -> None:
        """Resolve any robot-specific viewer overlay state."""
        pass

    def _draw_viewer_overlay_locked(self, viewer) -> None:
        """Draw robot-specific overlays into an already-locked passive viewer."""
        del viewer

    def _ensure_viewer_state_channel(self) -> None:
        if not self.enable_viewer or self._viewer_state_shm is not None:
            return
        self._viewer_state_shm = self.create_shm(name=self.viewer_state_shm_name, max_size_mb=1, is_writer=True)

    def _ensure_viewer_command_reader(self) -> None:
        if not self.enable_viewer or self._viewer_command_shm is not None:
            return
        try:
            self._viewer_command_shm = SharedMemoryChannel(
                self.viewer_command_shm_name,
                is_writer=False,
                timeout=0.01,
            )
        except Exception:
            self._viewer_command_shm = None

    def _apply_viewer_camera_locked(self) -> None:
        if self._viewer is None or self._viewer_selected_camera is None:
            return
        camera_id = self._viewer_available_camera_ids.get(self._viewer_selected_camera)
        if camera_id is None:
            return
        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._viewer.cam.fixedcamid = camera_id

    def _copy_state_to_camera_render_data_locked(self) -> None:
        if self._camera_render_data is None or self.mjdata is None:
            return
        self._camera_render_data.time = self.mjdata.time
        self._camera_render_data.qpos[:] = self.mjdata.qpos
        self._camera_render_data.qvel[:] = self.mjdata.qvel
        self._camera_render_data.act[:] = self.mjdata.act
        self._camera_render_data.ctrl[:] = self.mjdata.ctrl
        if self.mjmodel.nmocap > 0:
            self._camera_render_data.mocap_pos[:] = self.mjdata.mocap_pos
            self._camera_render_data.mocap_quat[:] = self.mjdata.mocap_quat

    def _render_images_from_data_locked(self, render_data) -> "OrderedDict[str, np.ndarray]":
        images: "OrderedDict[str, np.ndarray]" = OrderedDict()
        if self._renderer is None:
            return images
        for camera_name, camera_id in self._camera_ids.items():
            self._renderer.update_scene(render_data, camera=camera_id)
            images[camera_name] = np.array(self._renderer.render(), copy=True)
        return images

    def _configure_camera_renderer(self) -> None:
        if self._renderer is None:
            return
        # Camera streams favor throughput over visual effects so the robot loop and
        # live viewer stay responsive while offscreen cameras keep updating.
        for flag_name in (
            "mjRND_SHADOW",
            "mjRND_REFLECTION",
            "mjRND_SKYBOX",
            "mjRND_HAZE",
            "mjRND_FOG",
        ):
            try:
                self._renderer.scene.flags[getattr(mujoco.mjtRndFlag, flag_name)] = 0
            except Exception:
                continue

    def _open_viewer_locked(self) -> None:
        if not self.enable_viewer:
            return
        if self._viewer is not None:
            try:
                if self._viewer.is_running():
                    self._viewer_status = "running"
                    self._apply_viewer_camera_locked()
                    return
            except Exception:
                pass
            self._close_viewer_locked()

        try:
            viewer_module = importlib.import_module("mujoco.viewer")
            launch_passive = viewer_module.launch_passive
            launch_kwargs = {}
            try:
                signature = inspect.signature(launch_passive)
                if "show_left_ui" in signature.parameters:
                    launch_kwargs["show_left_ui"] = self.viewer_show_left_ui
                if "show_right_ui" in signature.parameters:
                    launch_kwargs["show_right_ui"] = self.viewer_show_right_ui
            except Exception:
                pass
            self._viewer = launch_passive(
                self.mjmodel,
                self.mjdata,
                **launch_kwargs,
            )
            self._viewer_status = "running"
            self._viewer_last_error = ""
            self._apply_viewer_camera_locked()
        except Exception as e:
            self._viewer = None
            self._viewer_status = "error"
            self._viewer_last_error = str(e)

    def _close_viewer_locked(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
        self._viewer = None
        if self.enable_viewer:
            self._viewer_status = "closed"

    def _sync_viewer_locked(self) -> None:
        if not self.enable_viewer:
            return
        if self._viewer is None:
            return
        try:
            if not self._viewer.is_running():
                self._close_viewer_locked()
                return
            if hasattr(self._viewer, "lock"):
                with self._viewer.lock():
                    if hasattr(self._viewer, "user_scn"):
                        self._viewer.user_scn.ngeom = 0
                        if self._viewer_show_tcp_frame:
                            self._draw_viewer_overlay_locked(self._viewer)
            self._viewer.sync()
            self._viewer_status = "running"
            self._viewer_last_error = ""
        except Exception as e:
            self._close_viewer_locked()
            self._viewer_last_error = str(e)
            self._viewer_status = "error"

    def _handle_viewer_command_locked(self, command: dict) -> None:
        cmd = command.get("cmd")
        if not cmd:
            return
        if cmd == "open":
            self._open_viewer_locked()
            return
        if cmd == "close":
            self._close_viewer_locked()
            return
        if cmd == "set_camera":
            camera_name = command.get("camera_name")
            if camera_name in self._viewer_available_camera_ids:
                self._viewer_selected_camera = camera_name
                self._apply_viewer_camera_locked()
                self._viewer_last_error = ""
            else:
                self._viewer_last_error = f"Unknown camera '{camera_name}'"
            return
        if cmd == "toggle_tcp_frame":
            value = command.get("value")
            self._viewer_show_tcp_frame = (not self._viewer_show_tcp_frame) if value is None else bool(value)

    def _poll_viewer_commands_locked(self) -> None:
        if not self.enable_viewer:
            return
        self._ensure_viewer_command_reader()
        if self._viewer_command_shm is None:
            return
        try:
            command = self._viewer_command_shm.read(blocking=False, skip_unchanged=True)
        except Exception:
            self._viewer_command_shm = None
            command = None
        if isinstance(command, dict):
            self._handle_viewer_command_locked(command)

    def _publish_viewer_state_locked(self) -> None:
        if not self.enable_viewer:
            return
        self._ensure_viewer_state_channel()
        if self._viewer_state_shm is None:
            return
        viewer_running = False
        if self._viewer is not None:
            try:
                viewer_running = bool(self._viewer.is_running())
            except Exception:
                viewer_running = False
        state = {
            "robot_shm_name": self.name,
            "viewer_running": viewer_running,
            "viewer_status": self._viewer_status,
            "selected_camera": self._viewer_selected_camera,
            "available_cameras": list(self._viewer_available_camera_ids.keys()),
            "show_tcp_frame": self._viewer_show_tcp_frame,
            "scene_name": self.scene_name,
            "scene_xml_path": self.scene_xml_path,
            "xml_path": self.xml_path,
            "loaded_xml_path": self._loaded_xml_path,
            "last_error": self._viewer_last_error,
            "heartbeat": time.perf_counter(),
        }
        self._viewer_state_shm.write(state)

    def _viewer_loop(self) -> None:
        interval = 1.0 / max(self.viewer_update_fps, 1.0)
        while not self._viewer_thread_stop.is_set():
            with self._lock:
                if self.mjmodel is not None:
                    self._poll_viewer_commands_locked()
                    self._sync_viewer_locked()
                    self._publish_viewer_state_locked()
            self._viewer_thread_stop.wait(interval)

    def _camera_loop(self) -> None:
        interval = 1.0 / max(self.camera_render_fps, 1.0)
        while not self._camera_thread_stop.is_set():
            if self._renderer is None and self._camera_ids:
                self._renderer = mujoco.Renderer(
                    self.mjmodel,
                    width=self.camera_width,
                    height=self.camera_height,
                )
                self._configure_camera_renderer()
            with self._lock:
                if self.mjmodel is None or self._renderer is None or self._camera_render_data is None:
                    snapshot_ready = False
                else:
                    self._copy_state_to_camera_render_data_locked()
                    snapshot_ready = True
            if snapshot_ready:
                mujoco.mj_forward(self.mjmodel, self._camera_render_data)
                images = self._render_images_from_data_locked(self._camera_render_data)
                with self._camera_image_lock:
                    self._latest_camera_images = images
                    self._last_camera_render_time = time.perf_counter()
            self._camera_thread_stop.wait(interval)

    def get_observation(self) -> Dict[str, np.ndarray]:
        with self._lock:
            obs = dict(self._get_robot_observation_core())
        with self._camera_image_lock:
            images = OrderedDict(
                (camera_name, np.array(image, copy=True))
                for camera_name, image in self._latest_camera_images.items()
            )
        for camera_name, image in images.items():
            obs[camera_name] = image
        return obs

    def start(self):
        self._ensure_viewer_state_channel()
        with self._lock:
            if self.enable_viewer and self.viewer_auto_open:
                self._open_viewer_locked()
            self._publish_viewer_state_locked()
        if self.enable_viewer and self._viewer_thread is None:
            self._viewer_thread_stop.clear()
            self._viewer_thread = threading.Thread(
                target=self._viewer_loop,
                name=f"{self.name}_viewer_loop",
                daemon=True,
            )
            self._viewer_thread.start()
        if self._camera_ids and self._camera_thread is None:
            self._camera_thread_stop.clear()
            self._camera_thread = threading.Thread(
                target=self._camera_loop,
                name=f"{self.name}_camera_loop",
                daemon=True,
            )
            self._camera_thread.start()
        try:
            super().start()
        finally:
            self._viewer_thread_stop.set()
            if self._viewer_thread is not None:
                self._viewer_thread.join(timeout=1.0)
                self._viewer_thread = None
            self._camera_thread_stop.set()
            if self._camera_thread is not None:
                self._camera_thread.join(timeout=1.0)
                self._camera_thread = None
            with self._lock:
                self._close_viewer_locked()
                if self._viewer_command_shm is not None:
                    try:
                        self._viewer_command_shm.destroy()
                    except Exception:
                        pass
                    self._viewer_command_shm = None
                if self._viewer_state_shm is not None:
                    try:
                        self._viewer_state_shm.destroy()
                    except Exception:
                        pass
                    self._viewer_state_shm = None

    def obs2meta(self, obs):
        qpos = np.asarray(obs.get("qpos", np.zeros(self.get_action_dim())), dtype=np.float32)
        frames = []
        for camera_name in self._camera_ids:
            if camera_name in obs:
                frames.append(np.asarray(obs[camera_name]))

        if "image" in obs and isinstance(obs["image"], np.ndarray):
            img = np.asarray(obs["image"])
            if img.ndim == 3:
                frames = [img]
            elif img.ndim == 4:
                frames = [img[i] for i in range(img.shape[0])]

        if frames:
            image = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
        else:
            image = np.zeros((1, 3, self.camera_height, self.camera_width), dtype=np.uint8)

        return MetaObs(state=qpos, state_joint=qpos, image=image)

    def is_running(self) -> bool:
        return self.mjmodel is not None

    def shutdown(self):
        self._viewer_thread_stop.set()
        self._camera_thread_stop.set()
        self._close_viewer_locked()
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        self._camera_render_data = None
        self._latest_camera_images = OrderedDict()
        self._last_camera_render_time = 0.0
        self.mjmodel = None
        self.mjdata = None
        if self._generated_xml_path is not None:
            try:
                os.remove(self._generated_xml_path)
            except FileNotFoundError:
                pass
            except Exception:
                pass
            self._generated_xml_path = None

    def close(self):
        super().close()
        self.shutdown()
