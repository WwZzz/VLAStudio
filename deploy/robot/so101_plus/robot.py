"""
SO101 Plus Follower Robot (single arm) with optional camera devices via YAML.

Adds wrist_yaw (servo ID 7) on top of the stock 6-motor SO101 follower.

Control modes:
  1. qpos    — 7D normalized joint targets
  2. delta_ee — 7D delta EE + Piper global IK (flange: gripper_body)
  3. rel_ee   — 7D relative EE (Quest VR anchor) + Piper global IK
"""

from __future__ import annotations

import json
import time
import threading
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

try:
    from lerobot.utils.errors import DeviceAlreadyConnectedError
except ImportError:
    from lerobot.errors import DeviceAlreadyConnectedError

from deploy.robot.base import BaseRobot
from deploy.utils import RateLimiter
from .config import SO101PlusConfig
from .so101_plus import SO101Plus, JOINT_NAMES
from .ik import (
    DEFAULT_END_FRAME,
    default_urdf_path,
    hw_arm_to_ik,
    ik_to_hw_arm,
    make_piper_ik_solver,
    transform_matrix_from_xyz_rpy,
)

# Default control limits in radians (hardware order, gripper last).
# Aligned with so101_plus.urdf joint1..joint6 / joint7 (with roll/yaw swapped into HW order).
DEFAULT_QLIMIT_MIN = [-1.91986, -1.74533, -1.68948, -1.46608, -2.96856, -1.53589, -0.174109]
DEFAULT_QLIMIT_MAX = [1.91986, 1.74533, 1.5708, 1.64061, 2.86084, 1.53589, 1.74575]
DEFAULT_JOINT_SIGNS = [1, 1, 1, 1, 1, 1, 1]

N_BODY_JOINTS = 6  # all joints except gripper use RANGE_M100_100
N_JOINTS = 7
N_ARM = 6


class So101Plus(BaseRobot):
    """
    ILStudio wrapper around SO101Plus.

    Observation / action qpos order (7D, normalized):
      [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, wrist_yaw, gripper]

    EE modes map Quest/teleop 7D pose commands onto the wrist flange frame
    ``gripper_body`` (not gripper tip ``gripper_finger``).
    """

    ROBOT_CONFIG_CLS = SO101PlusConfig
    ROBOT_BACKEND_CLS = SO101Plus

    CONTROL_MODES = ("qpos", "delta_ee", "rel_ee")

    def __init__(
        self,
        name: str = "so101_plus",
        max_size_mb: int = 64,
        fps: float = 100.0,
        control_shm_name: Optional[str] = None,
        com: str = "/dev/ttyACM0",
        robot_id: str = "so101_plus_arm",
        camera_configs: dict = {},
        calibration_dir: Optional[str] = None,
        control_mode: str = "qpos",
        qlimit_min: Optional[List[float]] = None,
        qlimit_max: Optional[List[float]] = None,
        joint_signs: Optional[List[int]] = None,
        urdf_path: Optional[str] = None,
        ik_end_frame: str = DEFAULT_END_FRAME,
        ik_ee_offset_xyz: Optional[List[float]] = None,
        ik_ee_offset_rpy: Optional[List[float]] = None,
        ik_ee_offset_matrix: Optional[List[float]] = None,
        ik_eps: float = 0.05,
        ik_max_iter: int = 60,
        ik_max_solve_time_s: float = 0.022,
        ik_rotation_weight: float = 1.0,
        # False = Alicia-style 6D pose (needed for wrist_roll / wrist_yaw). True = XYZ only.
        ik_position_only: bool = False,
        # URDF arm order [pan,lift,elbow,flex,yaw,roll]. Low yaw/roll → prefer wrist servos.
        ik_joint_reg_weights: Optional[List[float]] = None,
        ik_regularize_to_warmstart: bool = True,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_scale: float = 1.0,
        gripper_absolute: bool = False,
        max_joint_delta_deg: float = 30.0,
        clip_joint_delta: bool = True,
        joint_smooth_alpha: float = 1.0,
        joint_stop_hold: bool = False,
        joint_still_deg: float = 0.12,
        joint_still_frames: int = 5,
        joint_move_deg: float = 0.40,
        joint_teleop_brake_deg: float = 2.0,
        joint_teleop_brake_speed_deg_s: float = 80.0,
        joint_cmd_deadband_deg: float = 0.0,
        joint_creep_deg: float = 0.0,
        # go_home: interpolate to captured/current home (not calibration zeros)
        home_duration_s: float = 1.5,
        home_step_sleep: float = 0.03,
        # Optional 7D normalized home. None → capture pose right after connect.
        home_qpos: Optional[List[float]] = None,
        debug: bool = False,
        gripper_trace: bool = False,
        diag_log: bool = False,
        diag_log_path: Optional[str] = None,
        diag_print_hz: float = 5.0,
        **kwargs,
    ):
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)

        self.com = com
        self.robot_id = robot_id
        self.debug = bool(debug)
        self.gripper_trace = bool(gripper_trace)
        self.diag_log = bool(diag_log)
        self.diag_print_hz = float(diag_print_hz)
        self._diag_last_print = 0.0
        self._diag_fp = None
        if self.diag_log:
            path = Path(diag_log_path) if diag_log_path else Path("data/so101_plus_diag.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._diag_fp = open(path, "a", encoding="utf-8")
            logger.info(f"[So101Plus] diag_log → {path.resolve()}")

        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Invalid control_mode '{control_mode}'. Must be one of {self.CONTROL_MODES}")
        self.control_mode = control_mode

        self.qlimit_min = np.array(qlimit_min if qlimit_min is not None else DEFAULT_QLIMIT_MIN, dtype=np.float64)
        self.qlimit_max = np.array(qlimit_max if qlimit_max is not None else DEFAULT_QLIMIT_MAX, dtype=np.float64)
        self.joint_signs = np.array(joint_signs if joint_signs is not None else DEFAULT_JOINT_SIGNS, dtype=np.float64)

        if len(self.qlimit_min) != N_JOINTS or len(self.qlimit_max) != N_JOINTS:
            raise ValueError(f"qlimit_min/max must be length {N_JOINTS}")
        if len(self.joint_signs) != N_JOINTS:
            raise ValueError(f"joint_signs must be length {N_JOINTS}")

        self.position_scale = float(position_scale)
        self.rotation_scale = float(rotation_scale)
        self.gripper_scale = float(gripper_scale)
        self.gripper_absolute = bool(gripper_absolute)

        self.max_joint_delta_rad = np.deg2rad(float(max_joint_delta_deg))
        self.clip_joint_delta = bool(clip_joint_delta)
        self.ik_position_only = bool(ik_position_only)
        self.joint_smooth_alpha = float(np.clip(joint_smooth_alpha, 0.05, 1.0))
        self.joint_stop_hold = bool(joint_stop_hold)
        self.joint_still_rad = np.deg2rad(float(joint_still_deg))
        self.joint_still_frames = max(1, int(joint_still_frames))
        self.joint_move_rad = np.deg2rad(float(joint_move_deg))
        self.joint_teleop_brake_rad = np.deg2rad(float(joint_teleop_brake_deg))
        self.joint_teleop_brake_speed_deg_s = float(joint_teleop_brake_speed_deg_s)
        self.home_duration_s = max(0.5, float(home_duration_s))
        self.home_step_sleep = max(0.01, float(home_step_sleep))
        self._home_qpos: Optional[np.ndarray] = None
        self._home_qpos_from_yaml = False
        if home_qpos is not None:
            hq = np.asarray(home_qpos, dtype=np.float64).reshape(-1)
            if len(hq) != N_JOINTS:
                raise ValueError(f"home_qpos must be length {N_JOINTS}, got {len(hq)}")
            self._home_qpos = hq.copy()
            self._home_qpos_from_yaml = True
        # YAML compat (unused for so101_plus teleop feel knobs above).
        _ = (joint_cmd_deadband_deg, joint_creep_deg, camera_configs)

        self.urdf_path = str(urdf_path) if urdf_path else default_urdf_path()
        if ik_ee_offset_matrix is not None:
            ee_offset_matrix = np.asarray(ik_ee_offset_matrix, dtype=np.float64)
        else:
            ee_offset_matrix = transform_matrix_from_xyz_rpy(
                ik_ee_offset_xyz or [0.0, 0.0, 0.0],
                ik_ee_offset_rpy or [0.0, 0.0, 0.0],
            )
        self._ik_settings = {
            "end_frame": ik_end_frame,
            "ee_offset_matrix": ee_offset_matrix,
            "eps": float(ik_eps),
            "max_iter": int(ik_max_iter),
            "rotation_weight": float(ik_rotation_weight),
            "max_solve_time_s": float(ik_max_solve_time_s) if ik_max_solve_time_s else None,
            "position_only": bool(ik_position_only),
            "joint_regularization_weights": (
                list(ik_joint_reg_weights)
                if ik_joint_reg_weights is not None
                else None
            ),
            "regularize_to_warmstart": bool(ik_regularize_to_warmstart),
        }
        self._ik_solver = None

        # State (arm in IK/URDF order radians; gripper normalized 0-100)
        self._current_q_ik: Optional[np.ndarray] = None
        self._current_gripper: float = 0.0
        self._target_q_ik: Optional[np.ndarray] = None
        self._smooth_q_ik: Optional[np.ndarray] = None
        self._target_gripper: float = 0.0
        self._prev_ik_qpos: Optional[np.ndarray] = None
        self._still_count: int = 0
        self._teleop_holding_still: bool = False
        self._ik_initialized: bool = False
        self._norm_offset = np.zeros(N_JOINTS, dtype=np.float64)
        self._norm_calibrated = False
        self._lock = threading.RLock()

        self._rel_ee_anchor_pose: Optional[np.ndarray] = None
        self._rel_ee_vr_unsqueeze_active: bool = False
        self._rel_ee_pos_offset: np.ndarray = np.zeros(3)
        self._rel_ee_rot_offset: np.ndarray = np.zeros(3)
        self._rel_ee_last_raw_pos: Optional[np.ndarray] = None
        self._rel_ee_last_raw_euler: Optional[np.ndarray] = None
        self._last_ik_warn_time: float = 0.0
        self._ik_warn_suppressed: int = 0

        print(f"[So101Plus] Joints: {list(JOINT_NAMES)}")
        print(f"[So101Plus] Control mode: {control_mode}")
        if control_mode in ("delta_ee", "rel_ee"):
            print(
                f"[So101Plus] IK end_frame={ik_end_frame} urdf={self.urdf_path} "
                f"(flange, not gripper tip); position_only={self.ik_position_only}"
            )

        robot_config = self.ROBOT_CONFIG_CLS(
            port=com,
            id=robot_id,
            cameras={},
        )
        if calibration_dir:
            robot_config.calibration_dir = Path(calibration_dir).expanduser()
        self._robot = self.ROBOT_BACKEND_CLS(robot_config)
        self._motors = list(self._robot.bus.motors)

        if len(self._motors) != N_JOINTS:
            raise RuntimeError(f"Expected {N_JOINTS} motors, got {len(self._motors)}: {self._motors}")

        retry_counts = 0
        max_connect_retry = 10
        while not self.connect():
            print(f"Retrying for {retry_counts} time...")
            retry_counts += 1
            if retry_counts > max_connect_retry:
                raise RuntimeError("Failed to connect to robot after max retries")
            time.sleep(1)

        if self.control_mode in ("delta_ee", "rel_ee"):
            # Force-load IK early so pinocchio errors surface before teleop starts.
            _ = self.ik_solver
            obs = self.get_observation()
            if obs is not None and self._current_q_ik is not None:
                self.ik_solver.reset(self._current_q_ik)
                self.ik_solver.set_reference(self._current_q_ik)
                self._target_q_ik = self._current_q_ik.copy()
                self._smooth_q_ik = self._current_q_ik.copy()
                self._ik_initialized = True
            if self._home_qpos_from_yaml:
                logger.info(f"[So101Plus] using YAML home_qpos: {np.round(self._home_qpos, 2)}")
            else:
                self.capture_home_from_current(obs=obs)
        else:
            if self._home_qpos_from_yaml:
                logger.info(f"[So101Plus] using YAML home_qpos: {np.round(self._home_qpos, 2)}")
            else:
                self.capture_home_from_current()

    def capture_home_from_current(self, obs: Optional[dict] = None, force: bool = False) -> np.ndarray:
        """Record current normalized qpos as go_home target (unless YAML home_qpos set)."""
        if self._home_qpos is not None and self._home_qpos_from_yaml and not force:
            return self._home_qpos
        if obs is None:
            try:
                obs = self.get_observation()
            except Exception:
                obs = None
        if obs is not None and obs.get("qpos") is not None:
            self._home_qpos = np.asarray(obs["qpos"], dtype=np.float64).reshape(-1)[:N_JOINTS].copy()
        elif self._home_qpos is None:
            self._home_qpos = np.zeros(N_JOINTS, dtype=np.float64)
        self._home_qpos_from_yaml = False
        logger.info(f"[So101Plus] home_qpos set to current: {np.round(self._home_qpos, 2)}")
        return self._home_qpos

    def _sync_state_to_qpos(self, qpos_norm: np.ndarray) -> None:
        """Align gripper + IK warmstart with a normalized joint command."""
        qpos_norm = np.asarray(qpos_norm, dtype=np.float64).reshape(-1)[:N_JOINTS]
        self._target_gripper = float(qpos_norm[6])
        self._current_gripper = float(qpos_norm[6])
        if self.control_mode not in ("delta_ee", "rel_ee"):
            return
        try:
            q_rad = self._normalized_to_radians(qpos_norm)
            q_ik = hw_arm_to_ik(q_rad[:N_ARM])
            self._target_q_ik = q_ik.copy()
            self._smooth_q_ik = q_ik.copy()
            self._current_q_ik = q_ik.copy()
            self._prev_ik_qpos = None
            self._still_count = 0
            self._teleop_holding_still = False
            self._ik_initialized = True
            self.ik_solver.reset(q_ik)
            self.ik_solver.set_reference(q_ik)
        except Exception as e:
            logger.warning(f"[So101Plus] _sync_state_to_qpos IK sync failed: {e}")

    def set_home(self):
        """Slow joint-space interpolate to captured home_qpos."""
        try:
            obs = self.get_observation()
            start = (
                np.asarray(obs["qpos"], dtype=np.float64).reshape(-1)[:N_JOINTS].copy()
                if obs is not None and obs.get("qpos") is not None
                else np.zeros(N_JOINTS, dtype=np.float64)
            )
        except Exception:
            start = np.zeros(N_JOINTS, dtype=np.float64)

        if self._home_qpos is None:
            self.capture_home_from_current(obs={"qpos": start})
        home = self._home_qpos.copy()

        n_steps = max(1, int(np.ceil(self.home_duration_s / self.home_step_sleep)))
        logger.info(
            f"[So101Plus] go_home interpolate {n_steps} steps "
            f"(~{self.home_duration_s:.1f}s) → home={np.round(home, 2)}"
        )
        for alpha in np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)[1:]:
            target = start + (home - start) * alpha
            self.publish_action(target)
            time.sleep(self.home_step_sleep)

        self.publish_action(home)
        self._sync_state_to_qpos(home)
        self._rel_ee_anchor_pose = None
        self._rel_ee_vr_unsqueeze_active = False
        self._rel_ee_pos_offset = np.zeros(3)
        self._rel_ee_rot_offset = np.zeros(3)
        time.sleep(min(0.3, self.home_step_sleep * 4))
        logger.info("[So101Plus] go_home finished")

    # ------------------------------------------------------------------ IK
    @property
    def ik_solver(self):
        if self._ik_solver is None:
            s = self._ik_settings
            self._ik_solver = make_piper_ik_solver(
                urdf_path=self.urdf_path,
                end_frame=s["end_frame"],
                ee_offset_matrix=s["ee_offset_matrix"],
                rotation_weight=s["rotation_weight"],
                max_iterations=s["max_iter"],
                tol=s["eps"],
                max_solve_time_s=s["max_solve_time_s"],
                jump_reset_threshold_deg=np.rad2deg(self.max_joint_delta_rad),
                position_only=s["position_only"],
                joint_regularization_weights=s.get("joint_regularization_weights"),
                regularize_to_warmstart=s.get("regularize_to_warmstart", True),
            )
            backend = getattr(self._ik_solver, "_backend", "unknown")
            logger.info(
                f"[So101Plus] Teleop IK ready ({backend}): end_frame={s['end_frame']}, "
                f"eps={s['eps']}, max_iter={s['max_iter']}, "
                f"max_solve_time_s={s['max_solve_time_s']}, "
                f"position_only={s['position_only']}"
            )
        return self._ik_solver

    def connect(self):
        """Connect robot (triggers hardware calibration if needed)."""
        try:
            if not self._robot.is_connected:
                self._robot.connect()
        except DeviceAlreadyConnectedError as e:
            print(f"Robot already connected: {e}")
        except Exception as e:
            print(f"Failed to connect to robot due to {e}")
            traceback.print_exc()
            return False
        print("Robot connected")
        print(f"  motors: {[(m, self._robot.bus.motors[m].id) for m in self._motors]}")
        print(f"  calibration file: {self._robot.calibration_fpath}")
        return True

    def get_action_dim(self):
        return len(self._motors)

    # ---------------------------------------------------- normalized ↔ rad
    def _normalized_to_radians(self, normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized joint values to radians (hardware order).

        - Joints 0-5 (body + wrist_yaw): RANGE_M100_100 [-100, 100]
        - Joint 6 (gripper): RANGE_0_100 [0, 100]
        """
        radians = np.zeros(N_JOINTS, dtype=np.float64)
        for i in range(N_BODY_JOINTS):
            qmin, qmax = self.qlimit_min[i], self.qlimit_max[i]
            norm_clamped = np.clip(normalized[i] * self.joint_signs[i], -100.0, 100.0)
            radians[i] = (norm_clamped + 100.0) / 200.0 * (qmax - qmin) + qmin

        qmin, qmax = self.qlimit_min[6], self.qlimit_max[6]
        norm_clamped = np.clip(normalized[6] * self.joint_signs[6], 0.0, 100.0)
        radians[6] = norm_clamped / 100.0 * (qmax - qmin) + qmin
        return radians

    def _radians_to_normalized(self, radians: np.ndarray) -> np.ndarray:
        """Inverse of _normalized_to_radians (hardware order)."""
        normalized = np.zeros(N_JOINTS, dtype=np.float64)
        for i in range(N_BODY_JOINTS):
            qmin, qmax = self.qlimit_min[i], self.qlimit_max[i]
            rad_clamped = np.clip(radians[i], qmin, qmax)
            normalized[i] = ((rad_clamped - qmin) / (qmax - qmin)) * 200.0 - 100.0
            normalized[i] *= self.joint_signs[i]

        qmin, qmax = self.qlimit_min[6], self.qlimit_max[6]
        rad_clamped = np.clip(radians[6], qmin, qmax)
        normalized[6] = ((rad_clamped - qmin) / (qmax - qmin)) * 100.0
        normalized[6] *= self.joint_signs[6]
        return normalized + self._norm_offset

    def _calibrate_normalization(self, qpos_normalized: np.ndarray, qpos_rad: np.ndarray):
        self._norm_offset = np.zeros(N_JOINTS, dtype=np.float64)
        recovered = self._radians_to_normalized(qpos_rad)
        self._norm_offset = np.asarray(qpos_normalized, dtype=np.float64) - recovered
        self._norm_calibrated = True
        if self.debug:
            print(f"[So101Plus] norm_offset={self._norm_offset}")

    def _hw7_norm_to_ik6(self, q_norm: np.ndarray) -> np.ndarray:
        q_hw = self._normalized_to_radians(q_norm)
        return hw_arm_to_ik(q_hw[:N_ARM])

    def _ik6_gripper_to_hw7_norm(self, q_ik: np.ndarray, gripper_norm: float) -> np.ndarray:
        q_hw = np.zeros(N_JOINTS, dtype=np.float64)
        q_hw[:N_ARM] = ik_to_hw_arm(q_ik)
        # Gripper rad slot unused for publish; set from normalized directly below.
        q_hw[6] = self._normalized_to_radians(
            np.array([0, 0, 0, 0, 0, 0, gripper_norm], dtype=np.float64)
        )[6]
        out = self._radians_to_normalized(q_hw)
        out[6] = float(np.clip(gripper_norm, 0.0, 100.0))
        out[:N_ARM] = np.clip(out[:N_ARM], -100.0, 100.0)
        return out

    # ----------------------------------------------------------- observation
    def get_observation(self):
        """Get complete observation data."""
        try:
            obs = self._robot.get_observation()
            qpos = np.array([obs[mname + ".pos"] for mname in self._motors], dtype=np.float64)
            if self.control_mode in ("delta_ee", "rel_ee"):
                q_rad = self._normalized_to_radians(qpos)
                if not self._norm_calibrated:
                    self._calibrate_normalization(qpos, q_rad)
                self._current_q_ik = hw_arm_to_ik(q_rad[:N_ARM])
                self._current_gripper = float(qpos[6])
            return {"qpos": qpos, **obs}
        except Exception as e:
            print(f"Error getting observation: {e}")
            return None

    # -------------------------------------------------------- rel_ee helpers
    def _warning_throttled_ik(self, msg: str):
        now = time.time()
        if now - self._last_ik_warn_time >= 1.0:
            if self._ik_warn_suppressed > 0:
                msg += f" (suppressed {self._ik_warn_suppressed} repeats)"
            logger.warning(msg)
            self._last_ik_warn_time = now
            self._ik_warn_suppressed = 0
        else:
            self._ik_warn_suppressed += 1

    def _reanchor_to_last_good_pose(self):
        joints = self._target_q_ik
        if joints is not None:
            fk_pose = self.ik_solver.compute_fk(joints)
            self._rel_ee_anchor_pose = fk_pose.copy()
            self.ik_solver.reset(joints)
            self.ik_solver.set_reference(joints, fk_pose)

    def refresh_rel_ee_anchor(self, use_actual_joints: bool = True) -> bool:
        if self.control_mode != "rel_ee":
            return False
        with self._lock:
            self.get_observation()
            joints_to_use = None
            if self._target_q_ik is not None:
                joints_to_use = self._target_q_ik
            elif self._current_q_ik is not None:
                joints_to_use = self._current_q_ik
            if joints_to_use is None:
                return False
            fk_pose = self.ik_solver.compute_fk(joints_to_use)
            self._rel_ee_anchor_pose = fk_pose.copy()
            self.ik_solver.reset(joints_to_use)
            self.ik_solver.set_reference(joints_to_use, fk_pose)
            return True

    # -------------------------------------------------------- process_action
    def process_action(self, action_dict: dict) -> dict:
        """qpos: passthrough. delta_ee/rel_ee: IK → normalized 7D."""
        if self.control_mode == "qpos":
            return action_dict

        action = action_dict.get("action", None)
        if action is None:
            return action_dict

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if len(action) != 7:
            logger.warning(f"[So101Plus] {self.control_mode} expects 7D action, got {len(action)}D")
            return action_dict

        pos_component = action[:3] * self.position_scale
        euler_component = action[3:6] * self.rotation_scale

        if np.allclose(action, 0, atol=1e-6):
            if self.control_mode == "rel_ee":
                input_unsqueeze = action_dict.get("unsqueeze_active", True)
                anchor_set = action_dict.get("anchor_just_set", False)
                internal_unsqueeze = self._rel_ee_vr_unsqueeze_active
                if anchor_set or (internal_unsqueeze and not input_unsqueeze):
                    pass
                else:
                    action_dict["action"] = None
                    return action_dict
            else:
                action_dict["action"] = None
                return action_dict

        with self._lock:
            if self.control_mode == "delta_ee":
                if not self._ik_initialized:
                    if self._current_q_ik is not None:
                        self.ik_solver.reset(self._current_q_ik)
                        self._target_q_ik = self._current_q_ik.copy()
                        self._smooth_q_ik = self._current_q_ik.copy()
                        self._ik_initialized = True
                    else:
                        action_dict["action"] = None
                        return action_dict
                current_T = self.ik_solver.get_current_pose()
                target_T = current_T.copy()
                target_T[:3, 3] += pos_component
                if (not self.ik_position_only) and np.linalg.norm(euler_component) > 1e-10:
                    from scipy.spatial.transform import Rotation as R

                    R_delta = R.from_euler("xyz", euler_component).as_matrix()
                    target_T[:3, :3] = R_delta @ current_T[:3, :3]
                # else: keep current orientation (float) while applying position delta
                new_gripper = float(np.clip(self._target_gripper, 0.0, 100.0))

            elif self.control_mode == "rel_ee":
                try:
                    vr_unsqueeze_active = bool(action_dict.get("unsqueeze_active", True))
                    anchor_just_set = bool(action_dict.get("anchor_just_set", False))
                    vr_was_active = self._rel_ee_vr_unsqueeze_active
                    self._rel_ee_vr_unsqueeze_active = vr_unsqueeze_active

                    if not vr_unsqueeze_active:
                        if self._target_q_ik is not None:
                            self._smooth_q_ik = self._target_q_ik.copy()
                        self._teleop_holding_still = False
                        self._still_count = 0
                        self._prev_ik_qpos = None
                        action_dict["action"] = None
                        return action_dict

                    trigger_reason = ""
                    if anchor_just_set:
                        trigger_reason = "anchor_just_set"
                    elif vr_unsqueeze_active and not vr_was_active:
                        trigger_reason = "unsqueeze_edge"

                    if trigger_reason:
                        if not self.refresh_rel_ee_anchor(use_actual_joints=True):
                            action_dict["action"] = None
                            return action_dict
                        self._rel_ee_pos_offset = np.zeros(3)
                        self._rel_ee_rot_offset = np.zeros(3)
                        if (
                            anchor_just_set
                            and np.allclose(pos_component, 0, atol=1e-8)
                            and np.allclose(euler_component, 0, atol=1e-8)
                        ):
                            action_dict["action"] = None
                            return action_dict

                    if self._rel_ee_anchor_pose is None:
                        if not self.refresh_rel_ee_anchor(use_actual_joints=True):
                            action_dict["action"] = None
                            return action_dict
                        self._rel_ee_pos_offset = np.zeros(3)
                        self._rel_ee_rot_offset = np.zeros(3)

                    corrected_pos = pos_component - self._rel_ee_pos_offset
                    corrected_euler = euler_component - self._rel_ee_rot_offset
                    if self.ik_position_only:
                        # Chase XYZ only; let flange orientation float with current FK so
                        # the arm can reconfigure (holding anchor R drives singularities).
                        target_T = self.ik_solver.get_current_pose()
                        target_T[:3, 3] = self._rel_ee_anchor_pose[:3, 3] + corrected_pos
                    else:
                        target_T = self._rel_ee_anchor_pose.copy()
                        target_T[:3, 3] += corrected_pos
                        if np.linalg.norm(corrected_euler) > 1e-10:
                            from scipy.spatial.transform import Rotation as R

                            R_rel = R.from_euler("xyz", corrected_euler).as_matrix()
                            target_T[:3, :3] = R_rel @ self._rel_ee_anchor_pose[:3, :3]

                    self._rel_ee_last_raw_pos = pos_component.copy()
                    self._rel_ee_last_raw_euler = euler_component.copy()
                    new_gripper = float(np.clip(self._target_gripper, 0.0, 100.0))
                except Exception as e:
                    logger.error(f"[So101Plus] rel_ee process error: {e}")
                    traceback.print_exc()
                    action_dict["action"] = None
                    return action_dict
            else:
                return action_dict

            q_before = (
                self._target_q_ik.copy()
                if self._target_q_ik is not None
                else (self._current_q_ik.copy() if self._current_q_ik is not None else None)
            )
            success, q_arm, iters, err = self.ik_solver.solve(target_T)
            timed_out = bool(getattr(self.ik_solver, "_last_solve_timed_out", False))

            if not success and not timed_out:
                recovery = self._target_q_ik if self._target_q_ik is not None else self._current_q_ik
                if recovery is not None:
                    self.ik_solver.reset(recovery)
                    success, q_arm, iters, err = self.ik_solver.solve(target_T)
                    timed_out = bool(getattr(self.ik_solver, "_last_solve_timed_out", False))

            if not success:
                if self.control_mode == "rel_ee" and self._rel_ee_last_raw_pos is not None:
                    self._rel_ee_pos_offset = self._rel_ee_last_raw_pos.copy()
                    self._rel_ee_rot_offset = self._rel_ee_last_raw_euler.copy()
                    self._reanchor_to_last_good_pose()
                    self._warning_throttled_ik(
                        f"[So101Plus] IK failed, re-anchored: err={err:.4e}, pos={target_T[:3, 3]}"
                    )
                else:
                    self._warning_throttled_ik(
                        f"[So101Plus] IK failed: err={err:.4e}, iters={iters}, pos={target_T[:3, 3]}"
                    )
                self._diag_emit(
                    stage="ik_fail",
                    vr_pos=pos_component,
                    vr_eul=euler_component,
                    ok=False,
                    iters=iters,
                    err=err,
                    timed_out=timed_out,
                )
                action_dict["action"] = None
                return action_dict

            reference_q = self._target_q_ik if self._target_q_ik is not None else self._current_q_ik
            clipped = False
            if reference_q is not None:
                dq = (q_arm - reference_q + np.pi) % (2.0 * np.pi) - np.pi
                ik_jump = float(np.max(np.abs(dq)))
                if ik_jump > self.max_joint_delta_rad:
                    if self.clip_joint_delta and ik_jump > 1e-9:
                        # Limit per-step joint change instead of freeze+reanchor (teleop feel).
                        q_arm = reference_q + dq * (self.max_joint_delta_rad / ik_jump)
                        self.ik_solver.reset(q_arm)
                        clipped = True
                    else:
                        if self.control_mode == "rel_ee" and self._rel_ee_last_raw_pos is not None:
                            self._rel_ee_pos_offset = self._rel_ee_last_raw_pos.copy()
                            self._rel_ee_rot_offset = self._rel_ee_last_raw_euler.copy()
                            self._reanchor_to_last_good_pose()
                        else:
                            self.ik_solver.reset(reference_q)
                        self._warning_throttled_ik(
                            f"[So101Plus] IK jump rejected: {np.rad2deg(ik_jump):.1f}° "
                            f"> {np.rad2deg(self.max_joint_delta_rad):.1f}°"
                        )
                        self._diag_emit(
                            stage="ik_jump",
                            vr_pos=pos_component,
                            vr_eul=euler_component,
                            ok=success,
                            iters=iters,
                            err=err,
                            timed_out=timed_out,
                            dq_ik_deg=np.rad2deg(dq),
                        )
                        action_dict["action"] = None
                        return action_dict

            self._target_q_ik = q_arm.copy()
            self._target_gripper = new_gripper

            ik_step = float("inf")
            if self._prev_ik_qpos is not None:
                d_ik = (q_arm - self._prev_ik_qpos + np.pi) % (2.0 * np.pi) - np.pi
                ik_step = float(np.max(np.abs(d_ik)))
            self._prev_ik_qpos = q_arm.copy()

            if self.joint_stop_hold:
                if ik_step >= self.joint_move_rad:
                    self._teleop_holding_still = False
                    self._still_count = 0
                elif ik_step < self.joint_still_rad:
                    self._still_count += 1
                    if (not self._teleop_holding_still) and self._still_count >= self.joint_still_frames:
                        self._teleop_holding_still = True
                        if self._smooth_q_ik is not None:
                            self._smooth_q_ik = self._smooth_q_ik.copy()

            if self.joint_stop_hold and self._teleop_holding_still and self._smooth_q_ik is not None:
                q_out = self._smooth_q_ik.copy()
            else:
                a = self.joint_smooth_alpha
                if a < 1.0 - 1e-9 and self._smooth_q_ik is not None:
                    prev = self._smooth_q_ik
                    err_q = (q_arm - prev + np.pi) % (2.0 * np.pi) - np.pi
                    q_out = prev + a * err_q
                else:
                    q_out = q_arm
                self._smooth_q_ik = np.asarray(q_out, dtype=np.float64).copy()

            action_dict["action"] = self._ik6_gripper_to_hw7_norm(q_out, new_gripper)
            dq_ik = q_arm - q_before if q_before is not None else np.zeros(N_ARM)
            # IK order: [pan,lift,elbow,flex,yaw,roll] → HW: roll=idx4, yaw=idx5 after remap
            q_hw_rad = ik_to_hw_arm(q_out)
            self._diag_emit(
                stage="ok_clip" if clipped else ("ok_timeout" if timed_out else "ok"),
                vr_pos=pos_component,
                vr_eul=euler_component,
                ok=success,
                iters=iters,
                err=err,
                timed_out=timed_out,
                clipped=clipped,
                holding=bool(self.joint_stop_hold and self._teleop_holding_still),
                dq_ik_deg=np.rad2deg(dq_ik),
                q_hw_deg=np.rad2deg(q_hw_rad),
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

        def _arr(x):
            if x is None:
                return None
            return np.asarray(x, dtype=np.float64).reshape(-1)

        dq_a = _arr(dq)
        qhw_a = _arr(q_hw)
        # URDF IK dq: [pan,lift,elbow,flex,yaw,roll]
        yaw_dq = float(dq_a[4]) if dq_a is not None and dq_a.size >= 6 else 0.0
        roll_dq = float(dq_a[5]) if dq_a is not None and dq_a.size >= 6 else 0.0
        # HW arm after ik_to_hw_arm: [pan,lift,elbow,flex,roll,yaw]
        # HW arm 6D: [pan,lift,elbow,flex,roll,yaw]
        roll_hw = float(qhw_a[4]) if qhw_a is not None and qhw_a.size >= 6 else float("nan")
        yaw_hw = float(qhw_a[5]) if qhw_a is not None and qhw_a.size >= 6 else float("nan")
        eul = _arr(vr_eul) if vr_eul is not None else np.zeros(3)
        pos = _arr(vr_pos) if vr_pos is not None else np.zeros(3)
        print(
            f"[DIAG {stage}] VR pos={np.round(pos,3)} eul_deg={np.round(np.rad2deg(eul),1)} | "
            f"IK ok={kwargs.get('ok')} iters={kwargs.get('iters')} err={kwargs.get('err')} "
            f"timeout={kwargs.get('timed_out')} hold={kwargs.get('holding')} | "
            f"dYaw={yaw_dq:+.1f}° dRoll={roll_dq:+.1f}° → HW roll={roll_hw:.1f}° yaw={yaw_hw:.1f}°"
        )

    # ---------------------------------------------------------------- start
    def start(self):
        """Control loop: for EE modes, update gripper before IK (Alicia-style)."""
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)

        if self.control_shm_name is not None and self.control_shm is None:
            for i in range(10):
                try:
                    self.control_shm = self.connect_to_existing_shm(self.control_shm_name)
                    logger.info(f"Connected to control SHM: {self.control_shm_name}")
                    break
                except ValueError as e:
                    if i < 9:
                        logger.warning(f"Waiting for control SHM '{self.control_shm_name}'... ({i + 1}/10)")
                        time.sleep(0.5)
                    else:
                        logger.error(f"Failed to connect to control SHM: {e}")

        if self.control_mode == "qpos":
            return super().start()

        self.is_running = True
        rate_limiter = RateLimiter()
        data = self.get_data()
        if data is not None:
            self.write_data(data)

        logger.info(f"[So101Plus] Starting EE loop (mode={self.control_mode}, fps={self.fps})")
        idle_count = 0

        while self.is_running:
            action = self.read_action()
            if action is not None:
                idle_count = 0

                if action.get("cmd") == "reset" or action.get("go_home", False):
                    logger.info("[So101Plus] Go-home command received")
                    self.set_home()
                    self._ik_initialized = False
                    self._rel_ee_anchor_pose = None
                    self._rel_ee_vr_unsqueeze_active = False
                    self._rel_ee_pos_offset = np.zeros(3)
                    self._rel_ee_rot_offset = np.zeros(3)
                    continue

                raw_action = action.get("action", None)
                tg_before = float(self._target_gripper)
                unsqz_now = bool(action.get("unsqueeze_active", False))
                if (
                    raw_action is not None
                    and len(raw_action) >= 7
                ):
                    g_cmd = float(raw_action[6]) * self.gripper_scale
                    if self.gripper_absolute:
                        if unsqz_now:
                            # teleop open amount [0,1] → normalized [0,100]
                            self._target_gripper = float(np.clip(g_cmd * 100.0, 0.0, 100.0))
                    else:
                        self._target_gripper = float(
                            np.clip(self._target_gripper + g_cmd * 100.0, 0.0, 100.0)
                        )

                gripper_changed = abs(self._target_gripper - tg_before) > 0.5
                if gripper_changed and self._target_q_ik is not None:
                    q_hold = self._smooth_q_ik if self._smooth_q_ik is not None else self._target_q_ik
                    self.publish_action(self._ik6_gripper_to_hw7_norm(q_hold, self._target_gripper))

                t0 = time.perf_counter()
                action = self.process_action(action)
                ik_ms = (time.perf_counter() - t0) * 1000.0
                if ik_ms > 12.0:
                    if not hasattr(self, "_last_ik_slow_log") or (time.time() - self._last_ik_slow_log) >= 1.0:
                        to = bool(getattr(self.ik_solver, "_last_solve_timed_out", False))
                        nit = getattr(self.ik_solver, "_last_solve_iterations", "?")
                        logger.warning(
                            f"[So101Plus] Slow IK: {ik_ms:.1f}ms (iters={nit}, timed_out={to})"
                        )
                        self._last_ik_slow_log = time.time()

                action_array = action.get("action", None)
                if action_array is not None:
                    out = np.asarray(action_array, dtype=np.float64).copy()
                    out[6] = self._target_gripper
                    self.publish_action(out)
            else:
                idle_count += 1
                if idle_count > 20 and self._rel_ee_vr_unsqueeze_active:
                    self._rel_ee_vr_unsqueeze_active = False

            data = self.get_data()
            if data is not None:
                self.write_data(data)
            rate_limiter.sleep(self.fps)

    def obs2meta(self, device_data):
        if device_data is None:
            return {}
        qpos = device_data.get("qpos")
        if qpos is None:
            qpos = np.array([device_data[m + ".pos"] for m in self._motors], dtype=np.float32)
        return {"state": np.asarray(qpos, dtype=np.float32)}

    def shutdown(self):
        if self._robot.is_connected:
            self._robot.disconnect()

    def close(self):
        super().close()
        if self._robot.is_connected:
            self._robot.disconnect()

    def publish_action(self, action: np.ndarray):
        try:
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            if len(action) != N_JOINTS:
                print(f"[So101Plus] Warning: expected {N_JOINTS}D action, got {len(action)}D")
                return
            action_dict = {mname + ".pos": action[i] for i, mname in enumerate(self._motors)}
            self._robot.send_action(action_dict)
        except Exception:
            pass

    def is_running(self):
        return self._robot.is_connected

    def save_episode(self, file_path: str, observations: list, actions: list):
        import h5py
        import os

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        def write_group(group, data_list, key_prefix=None):
            if isinstance(data_list[0], dict):
                for key in data_list[0].keys():
                    sub_list = [obs[key] for obs in data_list]
                    if isinstance(sub_list[0], dict):
                        sub_group = group.create_group(key)
                        write_group(sub_group, sub_list)
                    else:
                        try:
                            group.create_dataset(key, data=np.stack(sub_list))
                        except (TypeError, ValueError) as e:
                            print(f"Warning: Could not stack data for key '{key}'. Skipping. Error: {e}")
            else:
                try:
                    if key_prefix is None:
                        group.create_dataset("data", data=np.stack(data_list))
                    else:
                        group.create_dataset(key_prefix, data=np.stack(data_list))
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not stack data for key '{key_prefix}'. Skipping. Error: {e}")

        with h5py.File(file_path, "w") as f:
            f.create_dataset("actions", data=np.array(actions, dtype=np.float32))
            obs_group = f.create_group("observations")
            if observations:
                write_group(obs_group, observations)
