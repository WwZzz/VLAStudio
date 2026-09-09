"""
SO101 Plus MuJoCo simulation robot.

Same action interface as the real ``So101Plus`` (qpos / delta_ee / rel_ee) so
Quest VR teleop configs can be reused against this sim.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import mujoco
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R

from deploy.simulation.mujoco.base import MujocoDeviceBase

# Load IK helpers without importing so101_plus package __init__ (needs lerobot).
import importlib.util as _ilu
import sys as _sys

_ik_path = Path(__file__).resolve().parents[1] / "so101_plus" / "ik.py"
_spec = _ilu.spec_from_file_location("so101_plus_ik_standalone", _ik_path)
_ik = _ilu.module_from_spec(_spec)
_sys.modules[_spec.name] = _ik
assert _spec.loader is not None
_spec.loader.exec_module(_ik)
DEFAULT_END_FRAME = _ik.DEFAULT_END_FRAME
default_urdf_path = _ik.default_urdf_path
hw_arm_to_ik = _ik.hw_arm_to_ik
ik_to_hw_arm = _ik.ik_to_hw_arm
make_piper_ik_solver = _ik.make_piper_ik_solver
transform_matrix_from_xyz_rpy = _ik.transform_matrix_from_xyz_rpy

# MuJoCo / URDF order (gripper last among revolute joints)
MJ_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
# Hardware / ILStudio observation order
HW_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "wrist_yaw",
    "gripper",
]

N_ARM = 6
N_JOINTS = 7

# URDF limits, hardware order (roll/yaw swapped vs URDF joint5/6)
DEFAULT_QLIMIT_MIN = np.array(
    [-1.91986, -1.74533, -1.68948, -1.46608, -2.96856, -1.53589, -0.174109], dtype=np.float64
)
DEFAULT_QLIMIT_MAX = np.array(
    [1.91986, 1.74533, 1.5708, 1.64061, 2.86084, 1.53589, 1.74575], dtype=np.float64
)
DEFAULT_INIT_QPOS_HW = np.zeros(N_JOINTS, dtype=np.float64)


def _T_to_xyz_rpy(T: np.ndarray) -> np.ndarray:
    xyz = T[:3, 3]
    rpy = R.from_matrix(T[:3, :3]).as_euler("xyz")
    return np.concatenate([xyz, rpy]).astype(np.float64)


def _draw_viewer_frame(viewer, pos: np.ndarray, axis_length: float = 0.05, axis_radius: float = 0.002):
    axes = [
        (np.array([1.0, 0.0, 0.0]), (1.0, 0.0, 0.0, 0.7)),
        (np.array([0.0, 1.0, 0.0]), (0.0, 1.0, 0.0, 0.7)),
        (np.array([0.0, 0.0, 1.0]), (0.0, 0.0, 1.0, 0.7)),
    ]
    for axis, color in axes:
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).flatten(),
            np.array(color, dtype=np.float32),
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, axis_radius, pos, pos + axis * axis_length)
        viewer.user_scn.ngeom += 1


class So101PlusSimRobot(MujocoDeviceBase):
    """SO101 Plus sim: 7-DoF arm, flange EE at ``gripper_body``."""

    CONTROL_MODES = ("qpos", "delta_ee", "rel_ee")
    QPOS_INPUT_UNITS = ("normalized", "radians")

    def __init__(
        self,
        name: str = "so101_plus_sim",
        max_size_mb: int = 64,
        fps: float = 50.0,
        control_shm_name: Optional[str] = None,
        xml_path: Optional[str] = None,
        scene_name: Optional[str] = None,
        scene_xml_path: Optional[str] = None,
        camera_names: Optional[List[str]] = None,
        camera_width: int = 640,
        camera_height: int = 480,
        urdf_path: Optional[str] = None,
        control_mode: str = "rel_ee",
        qpos_input_unit: str = "radians",
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_scale: float = 1.0,
        gripper_absolute: bool = True,
        ik_end_frame: str = DEFAULT_END_FRAME,
        ik_ee_offset_xyz: Optional[List[float]] = None,
        ik_ee_offset_rpy: Optional[List[float]] = None,
        ik_eps: float = 0.05,
        ik_max_iter: int = 60,
        ik_max_solve_time_s: float = 0.05,
        ik_rotation_weight: float = 0.1,
        max_joint_delta_deg: float = 30.0,
        clip_joint_delta: bool = True,
        ik_joint_reg_weights: Optional[List[float]] = None,
        ik_regularize_to_warmstart: bool = True,
        init_qpos: Optional[List[float]] = None,
        enable_viewer: bool = True,
        viewer_auto_open: bool = True,
        diag_log: bool = False,
        diag_log_path: Optional[str] = None,
        diag_print_hz: float = 5.0,
        **kwargs,
    ):
        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Invalid control_mode '{control_mode}'")
        if qpos_input_unit not in self.QPOS_INPUT_UNITS:
            raise ValueError(f"Invalid qpos_input_unit '{qpos_input_unit}'")

        self.control_mode = control_mode
        self.qpos_input_unit = qpos_input_unit
        self.position_scale = float(position_scale)
        self.rotation_scale = float(rotation_scale)
        self.gripper_scale = float(gripper_scale)
        self.gripper_absolute = bool(gripper_absolute)
        self.max_joint_delta_rad = np.deg2rad(float(max_joint_delta_deg))
        self.clip_joint_delta = bool(clip_joint_delta)
        self.qlimit_min = DEFAULT_QLIMIT_MIN.copy()
        self.qlimit_max = DEFAULT_QLIMIT_MAX.copy()
        self.init_qpos_hw = np.array(
            init_qpos if init_qpos is not None else DEFAULT_INIT_QPOS_HW, dtype=np.float64
        )
        self.diag_log = bool(diag_log)
        self.diag_print_hz = float(diag_print_hz)
        self._diag_last_print = 0.0
        self._diag_fp = None
        if self.diag_log:
            path = Path(diag_log_path) if diag_log_path else Path("data/so101_plus_sim_diag.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._diag_fp = open(path, "a", encoding="utf-8")
            print(f"[So101PlusSim] diag_log → {path.resolve()}")

        self.urdf_path = str(urdf_path) if urdf_path else default_urdf_path()
        ee_offset = transform_matrix_from_xyz_rpy(
            ik_ee_offset_xyz or [0.0, 0.0, 0.0],
            ik_ee_offset_rpy or [0.0, 0.0, 0.0],
        )
        self._ik = make_piper_ik_solver(
            urdf_path=self.urdf_path,
            end_frame=ik_end_frame,
            ee_offset_matrix=ee_offset,
            rotation_weight=float(ik_rotation_weight),
            max_iterations=int(ik_max_iter),
            tol=float(ik_eps),
            max_solve_time_s=float(ik_max_solve_time_s) if ik_max_solve_time_s else None,
            position_only=False,
            jump_reset_threshold_deg=float(max_joint_delta_deg),
            joint_regularization_weights=ik_joint_reg_weights,
            regularize_to_warmstart=bool(ik_regularize_to_warmstart),
        )

        self._lock = threading.RLock()
        self.current_q_hw = self.init_qpos_hw.copy()
        self.current_q_ik = hw_arm_to_ik(self.current_q_hw[:N_ARM])
        self.current_gripper = float(self.current_q_hw[6])
        self._target_q_ik = self.current_q_ik.copy()
        self._target_gripper = self.current_gripper
        self.current_gpos = np.zeros(6, dtype=np.float64)

        self._rel_ee_anchor_pose: Optional[np.ndarray] = None
        self._rel_ee_vr_unsqueeze_active = False
        self._rel_ee_pos_offset = np.zeros(3)
        self._rel_ee_rot_offset = np.zeros(3)
        self._rel_ee_last_raw_pos: Optional[np.ndarray] = None
        self._rel_ee_last_raw_euler: Optional[np.ndarray] = None

        if camera_names is None:
            camera_names = ["frontview"]

        print(f"[So101PlusSim] mode={control_mode} EE={ik_end_frame} urdf={self.urdf_path}")
        print(f"[So101PlusSim] IK backend={getattr(self._ik, '_backend', type(self._ik).__name__)}")

        super().__init__(
            name=name,
            max_size_mb=max_size_mb,
            fps=fps,
            control_shm_name=control_shm_name,
            xml_path=xml_path,
            scene_name=scene_name,
            scene_xml_path=scene_xml_path,
            camera_names=camera_names,
            camera_width=camera_width,
            camera_height=camera_height,
            enable_viewer=enable_viewer,
            viewer_auto_open=viewer_auto_open,
            **kwargs,
        )
        print("[So101PlusSim] Initialized")

    def _default_robot_xml_path(self) -> str:
        return str(Path(__file__).parent / "mujoco_model" / "so101_plus.xml")

    def handle_control_message(self, message: Optional[dict]) -> bool:
        if super().handle_control_message(message):
            return True
        if isinstance(message, dict) and message.get("go_home", False):
            logger.info("[So101PlusSim] Go-home")
            self.set_home()
            return True
        return False

    def reset(self):
        self.set_home()

    def _build_joint_indices(self) -> None:
        self.qpos_indices = np.array(
            [self.mjmodel.jnt_qposadr[self.mjmodel.joint(n).id] for n in MJ_JOINT_NAMES],
            dtype=np.int32,
        )
        self.qvel_indices = np.array(
            [self.mjmodel.jnt_dofadr[self.mjmodel.joint(n).id] for n in MJ_JOINT_NAMES],
            dtype=np.int32,
        )
        self.actuator_indices = np.array(
            [self.mjmodel.actuator(n).id for n in MJ_JOINT_NAMES],
            dtype=np.int32,
        )
        self._physics_substeps = 10

    def _hw7_to_mj7(self, q_hw: np.ndarray) -> np.ndarray:
        q = np.asarray(q_hw, dtype=np.float64).reshape(-1)
        out = np.zeros(N_JOINTS, dtype=np.float64)
        out[:N_ARM] = hw_arm_to_ik(q[:N_ARM])
        out[6] = float(q[6])
        return out

    def _mj7_to_hw7(self, q_mj: np.ndarray) -> np.ndarray:
        q = np.asarray(q_mj, dtype=np.float64).reshape(-1)
        out = np.zeros(N_JOINTS, dtype=np.float64)
        out[:N_ARM] = ik_to_hw_arm(q[:N_ARM])
        out[6] = float(q[6])
        return out

    def _apply_initial_state(self) -> None:
        q_mj = self._hw7_to_mj7(self.init_qpos_hw)
        self.mjdata.qpos[self.qpos_indices] = q_mj
        self.mjdata.ctrl[self.actuator_indices] = q_mj
        self.mjdata.qvel[:] = 0.0
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        self.current_q_hw = self.init_qpos_hw.copy()
        self.current_q_ik = hw_arm_to_ik(self.current_q_hw[:N_ARM])
        self.current_gripper = float(self.current_q_hw[6])
        self._target_q_ik = self.current_q_ik.copy()
        self._target_gripper = self.current_gripper
        self._ik.reset(self.current_q_ik)
        self._ik.set_reference(self.current_q_ik)
        T = self._ik.compute_fk(self.current_q_ik)
        self.current_gpos = _T_to_xyz_rpy(T)

    def _setup_viewer_overlay_state(self) -> None:
        try:
            self._viewer_tcp_site_id = self.mjmodel.site("tcp").id
        except KeyError:
            self._viewer_tcp_site_id = None

    def _draw_viewer_overlay_locked(self, viewer) -> None:
        if getattr(self, "_viewer_tcp_site_id", None) is None:
            return
        _draw_viewer_frame(viewer, self.mjdata.site_xpos[self._viewer_tcp_site_id].copy(), 0.07, 0.003)

    def get_action_dim(self) -> int:
        return N_JOINTS

    def _get_robot_observation_core(self) -> Dict[str, np.ndarray]:
        return {
            "qpos": self.current_q_hw.copy(),
            "gpos": self.current_gpos.copy(),
        }

    def _norm_to_rad_hw(self, norm: np.ndarray) -> np.ndarray:
        n = np.asarray(norm, dtype=np.float64).reshape(-1)
        out = np.zeros(N_JOINTS, dtype=np.float64)
        for i in range(N_ARM):
            out[i] = self.qlimit_min[i] + (np.clip(n[i], -100, 100) + 100) / 200.0 * (
                self.qlimit_max[i] - self.qlimit_min[i]
            )
        out[6] = self.qlimit_min[6] + np.clip(n[6], 0, 100) / 100.0 * (
            self.qlimit_max[6] - self.qlimit_min[6]
        )
        return out

    def _process_robot_action(self, action_dict: dict) -> dict:
        action = action_dict.get("action", None)
        if action is None:
            return action_dict
        action = np.asarray(action, dtype=np.float64).reshape(-1)

        if self.control_mode == "qpos":
            return self._process_qpos(action_dict, action)
        if self.control_mode == "delta_ee":
            return self._process_delta_ee(action_dict, action)
        return self._process_rel_ee(action_dict, action)

    def _process_qpos(self, action_dict: dict, action: np.ndarray) -> dict:
        if len(action) != N_JOINTS:
            print(f"[So101PlusSim] qpos expects {N_JOINTS}D, got {len(action)}")
            return action_dict
        if self.qpos_input_unit == "normalized":
            q_hw = self._norm_to_rad_hw(action)
        else:
            q_hw = action.copy()
        q_hw = np.clip(q_hw, self.qlimit_min, self.qlimit_max)
        action_dict["action"] = self._hw7_to_mj7(q_hw)
        return action_dict

    def _process_delta_ee(self, action_dict: dict, action: np.ndarray) -> dict:
        if len(action) != 7:
            print(f"[So101PlusSim] delta_ee expects 7D, got {len(action)}")
            return action_dict
        pos = action[:3] * self.position_scale
        eul = action[3:6] * self.rotation_scale
        g = action[6]
        with self._lock:
            T = self._ik.get_current_pose()
            Tt = T.copy()
            Tt[:3, 3] = Tt[:3, 3] + pos
            if np.linalg.norm(eul) > 1e-10:
                Tt[:3, :3] = R.from_euler("xyz", eul).as_matrix() @ T[:3, :3]
            ok, q_arm, _, _ = self._ik.solve(Tt, self._target_q_ik)
            if not ok:
                action_dict["action"] = None
                return action_dict
            if self.gripper_absolute:
                grip = float(np.clip(g * self.gripper_scale, 0.0, 1.0)) * (
                    self.qlimit_max[6] - self.qlimit_min[6]
                ) + self.qlimit_min[6]
            else:
                grip = float(np.clip(self._target_gripper + g * self.gripper_scale, self.qlimit_min[6], self.qlimit_max[6]))
            self._target_q_ik = q_arm.copy()
            self._target_gripper = grip
            q_mj = np.zeros(N_JOINTS)
            q_mj[:N_ARM] = q_arm
            q_mj[6] = grip
            action_dict["action"] = q_mj
        return action_dict

    def _process_rel_ee(self, action_dict: dict, action: np.ndarray) -> dict:
        if len(action) != 7:
            print(f"[So101PlusSim] rel_ee expects 7D, got {len(action)}")
            return action_dict

        pos = action[:3] * self.position_scale
        eul = action[3:6] * self.rotation_scale
        g_raw = float(action[6])

        vr_unsq = bool(action_dict.get("unsqueeze_active", True))
        anchor_just_set = bool(action_dict.get("anchor_just_set", False))

        if np.allclose(action, 0.0, atol=1e-6):
            falling = self._rel_ee_vr_unsqueeze_active and not vr_unsq
            if not (anchor_just_set or falling):
                action_dict["action"] = None
                return action_dict

        with self._lock:
            was = self._rel_ee_vr_unsqueeze_active
            self._rel_ee_vr_unsqueeze_active = vr_unsq
            if not vr_unsq:
                action_dict["action"] = None
                return action_dict

            edge = anchor_just_set or (vr_unsq and not was)
            if edge or self._rel_ee_anchor_pose is None:
                self._ik.reset(self._target_q_ik)
                fk = self._ik.compute_fk(self._target_q_ik)
                self._rel_ee_anchor_pose = fk.copy()
                self._ik.set_reference(self._target_q_ik, fk)
                self._rel_ee_pos_offset[:] = 0.0
                self._rel_ee_rot_offset[:] = 0.0
                if (
                    anchor_just_set
                    and np.allclose(pos, 0, atol=1e-8)
                    and np.allclose(eul, 0, atol=1e-8)
                ):
                    action_dict["action"] = None
                    return action_dict

            cp = pos - self._rel_ee_pos_offset
            ce = eul - self._rel_ee_rot_offset
            Tt = self._rel_ee_anchor_pose.copy()
            Tt[:3, 3] = Tt[:3, 3] + cp
            if np.linalg.norm(ce) > 1e-10:
                Tt[:3, :3] = R.from_euler("xyz", ce).as_matrix() @ self._rel_ee_anchor_pose[:3, :3]

            self._rel_ee_last_raw_pos = pos.copy()
            self._rel_ee_last_raw_euler = eul.copy()

            ok, q_arm, iters, err = self._ik.solve(Tt, self._target_q_ik)
            timed_out = bool(getattr(self._ik, "_last_solve_timed_out", False))
            q_before = self._target_q_ik.copy()

            if not ok and not timed_out:
                self._diag_emit(
                    stage="ik_fail",
                    vr_pos=pos,
                    vr_eul=eul,
                    vr_raw=action,
                    ok=False,
                    iters=iters,
                    err=err,
                    timed_out=timed_out,
                )
                action_dict["action"] = None
                return action_dict

            dq = (q_arm - self._target_q_ik + np.pi) % (2 * np.pi) - np.pi
            ik_jump = float(np.max(np.abs(dq)))
            clipped = False
            if ik_jump > self.max_joint_delta_rad:
                if self.clip_joint_delta and ik_jump > 1e-9:
                    q_arm = self._target_q_ik + dq * (self.max_joint_delta_rad / ik_jump)
                    try:
                        self._ik.reset(q_arm)
                    except Exception:
                        pass
                    clipped = True
                else:
                    if self._rel_ee_last_raw_pos is not None:
                        self._rel_ee_pos_offset = self._rel_ee_last_raw_pos.copy()
                        self._rel_ee_rot_offset = self._rel_ee_last_raw_euler.copy()
                        self._rel_ee_anchor_pose = self._ik.compute_fk(self._target_q_ik)
                    self._diag_emit(
                        stage="ik_jump",
                        vr_pos=pos,
                        vr_eul=eul,
                        vr_raw=action,
                        ok=ok,
                        iters=iters,
                        err=err,
                        timed_out=timed_out,
                        dq_ik_deg=np.rad2deg(dq),
                    )
                    action_dict["action"] = None
                    return action_dict

            if self.gripper_absolute:
                grip = float(np.clip(g_raw * self.gripper_scale, 0.0, 1.0))
                grip = self.qlimit_min[6] + grip * (self.qlimit_max[6] - self.qlimit_min[6])
            else:
                grip = float(
                    np.clip(
                        self._target_gripper + g_raw * self.gripper_scale,
                        self.qlimit_min[6],
                        self.qlimit_max[6],
                    )
                )

            self._target_q_ik = q_arm.copy()
            self._target_gripper = grip
            q_mj = np.zeros(N_JOINTS, dtype=np.float64)
            q_mj[:N_ARM] = q_arm
            q_mj[6] = grip
            action_dict["action"] = q_mj

            dq_ik = q_arm - q_before
            q_hw = self._mj7_to_hw7(q_mj)
            self._diag_emit(
                stage="ok_clip" if clipped else ("ok_timeout" if timed_out else "ok"),
                vr_pos=pos,
                vr_eul=eul,
                vr_raw=action,
                ok=ok,
                iters=iters,
                err=err,
                timed_out=timed_out,
                clipped=clipped,
                dq_ik_deg=np.rad2deg(dq_ik),
                q_hw_deg=np.rad2deg(q_hw),
                gpos=_T_to_xyz_rpy(self._ik.compute_fk(q_arm)),
            )
        return action_dict

    def _diag_emit(self, stage: str, **kwargs) -> None:
        if not self.diag_log:
            return
        now = time.time()
        row = {"t": now, "stage": stage}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                row[k] = [round(float(x), 5) for x in v.reshape(-1)]
            elif isinstance(v, (float, np.floating)):
                row[k] = round(float(v), 5)
            else:
                row[k] = v
        if self._diag_fp is not None:
            self._diag_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._diag_fp.flush()

        if now - self._diag_last_print < (1.0 / max(self.diag_print_hz, 0.1)):
            return
        self._diag_last_print = now
        vr_pos = kwargs.get("vr_pos")
        vr_eul = kwargs.get("vr_eul")
        q_hw = kwargs.get("q_hw_deg")
        dq = kwargs.get("dq_ik_deg")
        # URDF IK order dq: [pan,lift,elbow,flex,yaw,roll]
        # HW order:         [pan,lift,elbow,flex,roll,yaw,grip]
        def _a(x):
            return None if x is None else np.asarray(x, dtype=np.float64).reshape(-1)

        dq_a, qhw_a = _a(dq), _a(q_hw)
        yaw_dq = float(dq_a[4]) if dq_a is not None and dq_a.size >= 6 else 0.0
        roll_dq = float(dq_a[5]) if dq_a is not None and dq_a.size >= 6 else 0.0
        roll_hw = float(qhw_a[4]) if qhw_a is not None and qhw_a.size >= 7 else float("nan")
        yaw_hw = float(qhw_a[5]) if qhw_a is not None and qhw_a.size >= 7 else float("nan")
        eul = _a(vr_eul) if vr_eul is not None else np.zeros(3)
        pos = _a(vr_pos) if vr_pos is not None else np.zeros(3)
        print(
            f"[DIAG {stage}] VR pos={np.round(pos,3)} eul_deg={np.round(np.rad2deg(eul),1)} | "
            f"IK ok={kwargs.get('ok')} iters={kwargs.get('iters')} err={kwargs.get('err')} "
            f"timeout={kwargs.get('timed_out')} | "
            f"dYaw={yaw_dq:+.1f}° dRoll={roll_dq:+.1f}° → HW roll={roll_hw:.1f}° yaw={yaw_hw:.1f}°"
        )

    def _publish_robot_action(self, action: np.ndarray) -> None:
        q_mj = np.asarray(action, dtype=np.float64).reshape(-1)
        if len(q_mj) != N_JOINTS:
            return
        with self._lock:
            self.mjdata.ctrl[self.actuator_indices] = q_mj
            for _ in range(self._physics_substeps):
                mujoco.mj_step(self.mjmodel, self.mjdata)
            # Snap to commanded pose for deterministic teleop preview
            self.mjdata.qpos[self.qpos_indices] = q_mj
            self.mjdata.qvel[self.qvel_indices] = 0.0
            mujoco.mj_forward(self.mjmodel, self.mjdata)

            self.current_q_hw = self._mj7_to_hw7(q_mj)
            self.current_q_ik = q_mj[:N_ARM].copy()
            self.current_gripper = float(q_mj[6])
            self.current_gpos = _T_to_xyz_rpy(self._ik.compute_fk(self.current_q_ik))

    def set_home(self):
        """Reset to init pose (go_home from Quest B)."""
        q_mj = self._hw7_to_mj7(self.init_qpos_hw)
        self.publish_action(q_mj)
        self._target_q_ik = q_mj[:N_ARM].copy()
        self._target_gripper = float(q_mj[6])
        self._rel_ee_anchor_pose = None
        self._rel_ee_vr_unsqueeze_active = False
        try:
            self._ik.reset(self._target_q_ik)
            self._ik.set_reference(self._target_q_ik)
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="rel_ee", choices=So101PlusSimRobot.CONTROL_MODES)
    parser.add_argument("--visualize", action="store_true", default=True)
    args = parser.parse_args()

    robot = So101PlusSimRobot(
        name="so101_plus_sim_test",
        control_mode=args.mode,
        enable_viewer=True,
        viewer_auto_open=True,
        scene_name=None,
        xml_path=str(Path(__file__).parent / "mujoco_model" / "scene.xml"),
    )
    print("obs", {k: v.shape for k, v in robot.get_observation().items() if hasattr(v, "shape")})
    print("Running viewer — Ctrl+C to exit")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
