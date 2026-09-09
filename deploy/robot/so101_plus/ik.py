"""
SO101 Plus IK helpers.

Preferred backend: local ``PiperStyleGlobalIKSolver`` in ``piper_global_ik.py``
(standalone copy for so101_plus — does not import Alicia-D).

Fallback: reBot B601–style damped least-squares CLIK (Pinocchio or NumPy).

EE default: wrist flange ``gripper_body`` (not tip ``gripper_finger``).

Joint-order note
----------------
Hardware / ILStudio qpos (gripper last):
  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, wrist_yaw, gripper]

URDF revolute chain (joint1..joint7):
  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper]

IK solves the 6 arm joints with gripper locked. Callers must convert via
``hw_arm_to_ik`` / ``ik_to_hw_arm``.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

DEFAULT_URDF_PATH = Path(__file__).resolve().parent / "so101_plus_model" / "so101_plus.urdf"
DEFAULT_END_FRAME = "gripper_body"
DEFAULT_LOCK_JOINTS = ("joint7",)

# Prefer distal wrist DOFs for orientation: high cost on proximal joints,
# very low on URDF joint5 (yaw) / joint6 (roll). Order = IK/URDF arm:
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll]
DEFAULT_JOINT_REG_WEIGHTS = (20.0, 20.0, 10.0, 5.0, 0.02, 0.02)
DEFAULT_IK_REGULARIZATION_WEIGHT = 0.1

_HW_TO_IK = np.array([0, 1, 2, 3, 5, 4], dtype=np.int64)


def default_urdf_path() -> str:
    return str(DEFAULT_URDF_PATH)


def hw_arm_to_ik(q_hw_arm: np.ndarray) -> np.ndarray:
    """6D hardware arm radians → 6D URDF/IK radians."""
    q = np.asarray(q_hw_arm, dtype=np.float64).reshape(-1)
    if q.size < 6:
        raise ValueError(f"Expected >=6 arm joints, got {q.size}")
    return q[_HW_TO_IK].copy()


def ik_to_hw_arm(q_ik_arm: np.ndarray) -> np.ndarray:
    """6D URDF/IK radians → 6D hardware arm radians (same permutation)."""
    return hw_arm_to_ik(q_ik_arm)


def transform_matrix_from_xyz_rpy(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    """Build a 4x4 transform from xyz + extrinsic XYZ rpy."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    rot_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rot_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot_z @ rot_y @ rot_x
    T[:3, 3] = xyz
    return T


# ---------------------------------------------------------------------------
# URDF chain helpers (Pinocchio-free FK / numerical Jacobian CLIK)
# ---------------------------------------------------------------------------


def _parse_xyz_rpy(origin_el) -> Tuple[np.ndarray, np.ndarray]:
    if origin_el is None:
        return np.zeros(3), np.zeros(3)
    xyz = np.fromstring(origin_el.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(origin_el.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    if xyz.size != 3:
        xyz = np.zeros(3)
    if rpy.size != 3:
        rpy = np.zeros(3)
    return xyz, rpy


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec(axis / n * angle).as_matrix()


@dataclass
class _UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4
    axis: np.ndarray
    lower: float
    upper: float


def _load_urdf_joints(urdf_path: str) -> dict:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = {}
    for j_el in root.findall("joint"):
        name = j_el.get("name")
        jtype = j_el.get("type", "fixed")
        parent = j_el.find("parent").get("link")
        child = j_el.find("child").get("link")
        xyz, rpy = _parse_xyz_rpy(j_el.find("origin"))
        origin = transform_matrix_from_xyz_rpy(xyz, rpy)
        axis_el = j_el.find("axis")
        if axis_el is not None:
            axis = np.fromstring(axis_el.get("xyz", "0 0 1"), sep=" ", dtype=np.float64)
        else:
            axis = np.array([0.0, 0.0, 1.0])
        lim = j_el.find("limit")
        lower = float(lim.get("lower", "-3.14159")) if lim is not None else -np.pi
        upper = float(lim.get("upper", "3.14159")) if lim is not None else np.pi
        joints[name] = _UrdfJoint(name, jtype, parent, child, origin, axis, lower, upper)
    return joints


def _chain_to_frame(joints: dict, end_frame: str, lock_joint_names: Sequence[str]) -> List[_UrdfJoint]:
    child_to_joint = {j.child: j for j in joints.values()}
    chain_rev: List[_UrdfJoint] = []
    link = end_frame
    seen = set()
    while link in child_to_joint:
        if link in seen:
            raise RuntimeError(f"Cycle while walking URDF to '{end_frame}'")
        seen.add(link)
        j = child_to_joint[link]
        chain_rev.append(j)
        link = j.parent
    chain = list(reversed(chain_rev))
    lock = set(lock_joint_names)
    return [j for j in chain if j.name not in lock]


def _se3_log_err(T_cur: np.ndarray, T_tgt: np.ndarray) -> np.ndarray:
    """6D twist error in current LOCAL frame: log(T_cur^{-1} * T_tgt)."""
    T_err = np.linalg.inv(T_cur) @ T_tgt
    t = T_err[:3, 3]
    rotvec = Rotation.from_matrix(T_err[:3, :3]).as_rotvec()
    return np.concatenate([t, rotvec])


def _weight_twist(err: np.ndarray, J: Optional[np.ndarray], pw: float, rw: float):
    err_w = err.copy()
    err_w[:3] *= pw
    err_w[3:] *= rw
    if J is None:
        return err_w, None
    Jw = J.copy()
    Jw[:3, :] *= pw
    Jw[3:, :] *= rw
    return err_w, Jw


# ---------------------------------------------------------------------------
# Shared CLIK iteration (B601 algorithm)
# ---------------------------------------------------------------------------


def _clik_step(
    J: np.ndarray,
    err: np.ndarray,
    *,
    damping: float,
    step_size: float,
    prev_err_norm: float,
) -> np.ndarray:
    """One B601 DLS step: dq = step * J^T (J J^T + λ I)^{-1} err."""
    lam = damping * max(1.0, prev_err_norm * 10.0)
    jjt = J @ J.T
    jjt.flat[:: jjt.shape[0] + 1] += lam
    return step_size * (J.T @ np.linalg.solve(jjt, err))


# ---------------------------------------------------------------------------
# Pinocchio CLIK solver (preferred)
# ---------------------------------------------------------------------------


class So101PlusClikIKSolver:
    """
    reBot B601–style CLIK for so101_plus.

    Public surface matches the previous Piper/SciPy solvers used by ``robot.py``.
    """

    def __init__(
        self,
        urdf_path: str,
        end_frame: str = DEFAULT_END_FRAME,
        ee_offset_matrix: Optional[np.ndarray] = None,
        lock_joint_names: Sequence[str] = DEFAULT_LOCK_JOINTS,
        position_weight: float = 1.0,
        rotation_weight: float = 0.1,
        max_iterations: int = 200,
        tol: float = 1e-3,
        jump_reset_threshold_deg: float = 30.0,
        max_solve_time_s: Optional[float] = 0.022,
        step_size: float = 0.8,
        damping: float = 1e-6,
        position_only: bool = False,
        # Kept for YAML / factory compatibility (unused by CLIK):
        total_cost_scale: float = 20.0,
        regularization_weight: float = 0.01,
        regularize_to_warmstart: bool = False,
        exact_when_converged: bool = False,
    ):
        _ = (total_cost_scale, regularization_weight, regularize_to_warmstart)

        import pinocchio as pin

        self._pin = pin
        full = pin.buildModelFromUrdf(str(urdf_path))
        lock_ids = []
        for name in lock_joint_names:
            jid = full.getJointId(name)
            if jid >= full.njoints:
                raise ValueError(f"lock joint '{name}' not in URDF")
            lock_ids.append(jid)
        q0 = pin.neutral(full)
        self.model = pin.buildReducedModel(full, lock_ids, q0)
        self.data = self.model.createData()

        if end_frame not in [f.name for f in self.model.frames]:
            raise ValueError(f"end_frame '{end_frame}' not found in reduced model")
        self.end_frame_id = self.model.getFrameId(end_frame)
        self.end_frame = end_frame

        if ee_offset_matrix is None:
            ee_offset_matrix = np.eye(4, dtype=np.float64)
        self.ee_offset_matrix = np.asarray(ee_offset_matrix, dtype=np.float64)
        self._ee_offset_se3 = pin.SE3(
            self.ee_offset_matrix[:3, :3], self.ee_offset_matrix[:3, 3]
        )

        self.nq = int(self.model.nq)
        self.position_weight = float(position_weight)
        self.rotation_weight = float(max(rotation_weight, 1e-6))
        self.position_only = bool(position_only)
        self.max_iterations = int(max_iterations)
        # B601 default is 1e-4 on ||log6||; keep caller eps but clamp to a sane CLIK scale.
        self.tol = float(max(min(tol, 5e-2), 1e-5))
        self.jump_reset_threshold_rad = np.deg2rad(float(jump_reset_threshold_deg))
        self.max_solve_time_s = (
            None if max_solve_time_s is None or float(max_solve_time_s) <= 0 else float(max_solve_time_s)
        )
        self.step_size = float(step_size)
        self.damping = float(damping)
        self.exact_when_converged = bool(exact_when_converged)

        lo = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
        hi = np.asarray(self.model.upperPositionLimit, dtype=np.float64)
        self._lo = np.where(np.isfinite(lo), lo, -np.pi)
        self._hi = np.where(np.isfinite(hi), hi, np.pi)

        self.init_data = np.zeros(self.nq, dtype=np.float64)
        self.history_data = np.zeros(self.nq, dtype=np.float64)
        self._q_ref = np.zeros(self.nq, dtype=np.float64)
        self._reference_pose: Optional[np.ndarray] = None
        self._last_solve_iterations = 0
        self._last_solve_timed_out = False
        self._backend = "b601_clik_pinocchio"

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self._lo, self._hi)

    def _fk_se3(self, q: np.ndarray):
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.end_frame_id] * self._ee_offset_se3

    def _error(self, q: np.ndarray, target_se3) -> Tuple[float, np.ndarray]:
        pin = self._pin
        T_cur = self._fk_se3(q)
        if self.position_only:
            # Position error in WORLD, mapped to a 3-vector used with WORLD position Jacobian.
            err = (target_se3.translation - T_cur.translation).copy()
            err_w = self.position_weight * err
            return float(np.linalg.norm(err_w)), err
        err = pin.log6(T_cur.inverse() * target_se3).vector.copy()
        err_w, _ = _weight_twist(err, None, self.position_weight, self.rotation_weight)
        return float(np.linalg.norm(err_w)), err

    def _jacobian_local(self, q: np.ndarray) -> np.ndarray:
        """Jacobian for CLIK: LOCAL 6D, or WORLD 3D position when position_only."""
        pin = self._pin
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        if self.position_only:
            # WORLD-frame translational Jacobian of flange origin.
            J_world = pin.getFrameJacobian(
                self.model, self.data, self.end_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            # Offset: for pure rotation offset, origin of ee coincides with frame origin.
            return J_world[:3, :]
        J_frame = pin.getFrameJacobian(
            self.model, self.data, self.end_frame_id, pin.ReferenceFrame.LOCAL
        )
        if np.allclose(self.ee_offset_matrix, np.eye(4)):
            return J_frame
        return self._ee_offset_se3.inverse().action @ J_frame

    def reset(self, q: Optional[np.ndarray] = None):
        if q is None:
            q = np.zeros(self.nq, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64).reshape(-1)[: self.nq]
        self.init_data = self._clamp(q).copy()
        self.history_data = self.init_data.copy()

    def get_current_q(self) -> np.ndarray:
        return self.init_data.copy()

    def set_reference(self, q_ref: np.ndarray, pose_ref: Optional[np.ndarray] = None):
        q_ref = np.asarray(q_ref, dtype=np.float64).reshape(-1)[: self.nq]
        self._q_ref = self._clamp(q_ref).copy()
        if pose_ref is None:
            pose_ref = self.compute_fk(self._q_ref)
        self._reference_pose = np.asarray(pose_ref, dtype=np.float64).copy()

    def compute_fk(self, q: np.ndarray) -> np.ndarray:
        q = self._clamp(np.asarray(q, dtype=np.float64).reshape(-1)[: self.nq])
        return self._fk_se3(q).homogeneous.copy()

    def get_current_pose(self) -> np.ndarray:
        return self.compute_fk(self.init_data)

    def solve(
        self, target_pose: np.ndarray, q_init: Optional[np.ndarray] = None
    ) -> Tuple[bool, np.ndarray, int, float]:
        pin = self._pin
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (4, 4):
            raise ValueError(f"target_pose must have shape (4, 4), got {target_pose.shape}")
        target_se3 = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

        if q_init is None:
            q = self.init_data.copy()
        else:
            q = self._clamp(np.asarray(q_init, dtype=np.float64).reshape(-1)[: self.nq])

        prev_err, err = self._error(q, target_se3)
        if self.exact_when_converged and prev_err < self.tol:
            self.init_data = q.copy()
            self.history_data = q.copy()
            self._last_solve_iterations = 0
            self._last_solve_timed_out = False
            return True, q.copy(), 0, prev_err

        self._last_solve_timed_out = False
        t0 = time.perf_counter()
        best_q = q.copy()
        best_err = prev_err
        iters = 0

        for iteration in range(self.max_iterations):
            iters = iteration + 1
            if self.max_solve_time_s is not None and (time.perf_counter() - t0) > self.max_solve_time_s:
                self._last_solve_timed_out = True
                break
            if prev_err < self.tol:
                break

            J = self._jacobian_local(q)
            if self.position_only:
                err_w = self.position_weight * err
                Jw = self.position_weight * J
            else:
                err_w, Jw = _weight_twist(err, J, self.position_weight, self.rotation_weight)
            dq = _clik_step(
                Jw, err_w, damping=self.damping, step_size=self.step_size, prev_err_norm=prev_err
            )
            # Redundant arm: pull toward reference posture in nullspace (avoids collapse).
            if self.position_only and Jw.shape[0] == 3:
                lam = self.damping * max(1.0, prev_err * 10.0)
                jjt = Jw @ Jw.T
                jjt.flat[:: jjt.shape[0] + 1] += lam
                J_pinv = Jw.T @ np.linalg.solve(jjt, np.eye(Jw.shape[0]))
                N = np.eye(self.nq) - J_pinv @ Jw
                dq = dq + 0.15 * (N @ (self._q_ref - q))

            # B601: backtracking line search (up to 4 halvings); if all fail, keep q and continue.
            alpha = 1.0
            for _ in range(4):
                q_new = self._clamp(pin.integrate(self.model, q, alpha * dq))
                new_err, err_new = self._error(q_new, target_se3)
                if new_err < prev_err:
                    q = q_new
                    err = err_new
                    prev_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_q = q.copy()
                    break
                alpha *= 0.5

        # Teleop: always publish best intermediate for warm-start continuity.
        success = True
        sol_q = best_q
        self.init_data = sol_q.copy()
        self.history_data = sol_q.copy()
        self._last_solve_iterations = iters
        return success, sol_q.copy(), iters, best_err


# ---------------------------------------------------------------------------
# NumPy CLIK fallback (no Pinocchio)
# ---------------------------------------------------------------------------


class ScipyUrdfIKSolver:
    """
    Pinocchio-free CLIK with the same B601 DLS + line-search update.

    Uses analytic geometric Jacobian from the URDF chain (finite-diff free).
    Kept name ``ScipyUrdfIKSolver`` for import compatibility.
    """

    def __init__(
        self,
        urdf_path: str,
        end_frame: str = DEFAULT_END_FRAME,
        ee_offset_matrix: Optional[np.ndarray] = None,
        lock_joint_names: Sequence[str] = DEFAULT_LOCK_JOINTS,
        position_weight: float = 1.0,
        rotation_weight: float = 0.1,
        max_iterations: int = 200,
        tol: float = 1e-3,
        jump_reset_threshold_deg: float = 30.0,
        max_solve_time_s: Optional[float] = 0.022,
        step_size: float = 0.8,
        damping: float = 1e-6,
        position_only: bool = False,
        total_cost_scale: float = 20.0,
        regularization_weight: float = 0.01,
        regularize_to_warmstart: bool = False,
        exact_when_converged: bool = False,
    ):
        _ = (total_cost_scale, regularization_weight, regularize_to_warmstart)
        all_joints = _load_urdf_joints(urdf_path)
        chain = _chain_to_frame(all_joints, end_frame, lock_joint_names)
        self._chain = chain
        self._actuated = [j for j in chain if j.joint_type == "revolute"]
        if not self._actuated:
            raise ValueError(f"No revolute joints on path to '{end_frame}'")

        self.nq = len(self._actuated)
        self.position_weight = float(position_weight)
        self.rotation_weight = float(max(rotation_weight, 1e-6))
        self.position_only = bool(position_only)
        self.max_iterations = int(max_iterations)
        self.tol = float(max(min(tol, 5e-2), 1e-5))
        self.jump_reset_threshold_rad = np.deg2rad(float(jump_reset_threshold_deg))
        self.max_solve_time_s = (
            None if max_solve_time_s is None or float(max_solve_time_s) <= 0 else float(max_solve_time_s)
        )
        self.step_size = float(step_size)
        self.damping = float(damping)
        self.exact_when_converged = bool(exact_when_converged)

        if ee_offset_matrix is None:
            ee_offset_matrix = np.eye(4, dtype=np.float64)
        self.ee_offset_matrix = np.asarray(ee_offset_matrix, dtype=np.float64)

        self._lo = np.array([j.lower for j in self._actuated], dtype=np.float64)
        self._hi = np.array([j.upper for j in self._actuated], dtype=np.float64)

        self.init_data = np.zeros(self.nq, dtype=np.float64)
        self.history_data = np.zeros(self.nq, dtype=np.float64)
        self._q_ref = np.zeros(self.nq, dtype=np.float64)
        self._reference_pose: Optional[np.ndarray] = None
        self._last_solve_iterations = 0
        self._last_solve_timed_out = False
        self._backend = "b601_clik_numpy"

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self._lo, self._hi)

    def reset(self, q: Optional[np.ndarray] = None):
        if q is None:
            q = np.zeros(self.nq, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64).reshape(-1)[: self.nq]
        self.init_data = self._clamp(q).copy()
        self.history_data = self.init_data.copy()

    def get_current_q(self) -> np.ndarray:
        return self.init_data.copy()

    def set_reference(self, q_ref: np.ndarray, pose_ref: Optional[np.ndarray] = None):
        q_ref = np.asarray(q_ref, dtype=np.float64).reshape(-1)[: self.nq]
        self._q_ref = self._clamp(q_ref).copy()
        if pose_ref is None:
            pose_ref = self.compute_fk(self._q_ref)
        self._reference_pose = np.asarray(pose_ref, dtype=np.float64).copy()

    def compute_fk(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)[: self.nq]
        T = np.eye(4, dtype=np.float64)
        qi = 0
        for j in self._chain:
            T = T @ j.origin
            if j.joint_type == "revolute":
                R = _axis_angle_matrix(j.axis, float(q[qi]))
                Tj = np.eye(4, dtype=np.float64)
                Tj[:3, :3] = R
                T = T @ Tj
                qi += 1
        return T @ self.ee_offset_matrix

    def get_current_pose(self) -> np.ndarray:
        return self.compute_fk(self.init_data)

    def _fk_and_jacobian_world(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """FK + geometric Jacobian in WORLD, then convert twist to LOCAL."""
        T = np.eye(4, dtype=np.float64)
        joint_origins_world = []
        axes_world = []
        qi = 0
        for j in self._chain:
            T = T @ j.origin
            if j.joint_type == "revolute":
                axis_local = j.axis / (np.linalg.norm(j.axis) + 1e-15)
                axis_w = T[:3, :3] @ axis_local
                origin_w = T[:3, 3].copy()
                joint_origins_world.append(origin_w)
                axes_world.append(axis_w)
                R = _axis_angle_matrix(j.axis, float(q[qi]))
                Tj = np.eye(4, dtype=np.float64)
                Tj[:3, :3] = R
                T = T @ Tj
                qi += 1
        T_ee = T @ self.ee_offset_matrix
        p_ee = T_ee[:3, 3]
        J_w = np.zeros((6, self.nq), dtype=np.float64)
        for i in range(self.nq):
            w = axes_world[i]
            J_w[:3, i] = np.cross(w, p_ee - joint_origins_world[i])
            J_w[3:, i] = w
        # WORLD → LOCAL: v_local = R^T v_world, ω_local = R^T ω_world
        R = T_ee[:3, :3]
        Ad = np.zeros((6, 6), dtype=np.float64)
        Ad[:3, :3] = R.T
        Ad[3:, 3:] = R.T
        J_local = Ad @ J_w
        return T_ee, J_local

    def solve(
        self, target_pose: np.ndarray, q_init: Optional[np.ndarray] = None
    ) -> Tuple[bool, np.ndarray, int, float]:
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (4, 4):
            raise ValueError(f"target_pose must have shape (4, 4), got {target_pose.shape}")

        if q_init is None:
            q = self.init_data.copy()
        else:
            q = self._clamp(np.asarray(q_init, dtype=np.float64).reshape(-1)[: self.nq])

        T_cur, J6 = self._fk_and_jacobian_world(q)
        if self.position_only:
            err = target_pose[:3, 3] - T_cur[:3, 3]
            prev_err = float(np.linalg.norm(self.position_weight * err))
        else:
            err = _se3_log_err(T_cur, target_pose)
            err_w, _ = _weight_twist(err, None, self.position_weight, self.rotation_weight)
            prev_err = float(np.linalg.norm(err_w))

        if self.exact_when_converged and prev_err < self.tol:
            self.init_data = q.copy()
            self.history_data = q.copy()
            self._last_solve_iterations = 0
            self._last_solve_timed_out = False
            return True, q.copy(), 0, prev_err

        self._last_solve_timed_out = False
        t0 = time.perf_counter()
        best_q = q.copy()
        best_err = prev_err
        iters = 0

        for iteration in range(self.max_iterations):
            iters = iteration + 1
            if self.max_solve_time_s is not None and (time.perf_counter() - t0) > self.max_solve_time_s:
                self._last_solve_timed_out = True
                break
            if prev_err < self.tol:
                break

            T_cur, J6 = self._fk_and_jacobian_world(q)
            if self.position_only:
                err = target_pose[:3, 3] - T_cur[:3, 3]
                err_w = self.position_weight * err
                Jw = self.position_weight * J6[:3, :]
            else:
                err = _se3_log_err(T_cur, target_pose)
                err_w, Jw = _weight_twist(err, J6, self.position_weight, self.rotation_weight)
            dq = _clik_step(
                Jw, err_w, damping=self.damping, step_size=self.step_size, prev_err_norm=prev_err
            )

            alpha = 1.0
            for _ in range(4):
                q_new = self._clamp(q + alpha * dq)
                T_new = self.compute_fk(q_new)
                if self.position_only:
                    err_new = target_pose[:3, 3] - T_new[:3, 3]
                    new_err = float(np.linalg.norm(self.position_weight * err_new))
                else:
                    err_new = _se3_log_err(T_new, target_pose)
                    err_new_w, _ = _weight_twist(err_new, None, self.position_weight, self.rotation_weight)
                    new_err = float(np.linalg.norm(err_new_w))
                if new_err < prev_err:
                    q = q_new
                    prev_err = new_err
                    if self.position_only:
                        err = err_new
                    if new_err < best_err:
                        best_err = new_err
                        best_q = q.copy()
                    break
                alpha *= 0.5

        self.init_data = best_q.copy()
        self.history_data = best_q.copy()
        self._last_solve_iterations = iters
        return True, best_q.copy(), iters, best_err


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _pinocchio_importable() -> bool:
    try:
        import pinocchio  # noqa: F401

        return True
    except Exception:
        return False


def make_ik_solver(
    urdf_path: Optional[str] = None,
    end_frame: str = DEFAULT_END_FRAME,
    ee_offset_matrix: Optional[np.ndarray] = None,
    lock_joint_names: Sequence[str] = DEFAULT_LOCK_JOINTS,
    *,
    position_weight: float = 1.0,
    rotation_weight: float = 0.1,
    max_iterations: int = 200,
    tol: float = 1e-3,
    max_solve_time_s: Optional[float] = 0.022,
    jump_reset_threshold_deg: float = 30.0,
    step_size: float = 0.8,
    damping: float = 1e-6,
    position_only: bool = False,
    regularize_to_warmstart: bool = True,
    exact_when_converged: bool = False,
    regularization_weight: Optional[float] = None,
    joint_regularization_weights: Optional[Sequence[float]] = None,
):
    """
    Build so101_plus IK.

    Prefer local ``PiperStyleGlobalIKSolver`` so wrist_roll / wrist_yaw track flange
    orientation. Fall back to B601 CLIK only when Piper cannot be constructed, or
    when ``position_only=True``.
    """
    path = urdf_path or default_urdf_path()
    if not Path(path).exists():
        raise FileNotFoundError(f"so101_plus URDF not found: {path}")

    rw = max(float(rotation_weight), 1e-6)
    jw = (
        DEFAULT_JOINT_REG_WEIGHTS
        if joint_regularization_weights is None
        else tuple(float(x) for x in joint_regularization_weights)
    )
    reg_w = (
        DEFAULT_IK_REGULARIZATION_WEIGHT
        if regularization_weight is None
        else float(regularization_weight)
    )

    # Local Piper-style 6D global IK (required for wrist_yaw / wrist_roll tracking).
    if not position_only and _pinocchio_importable():
        try:
            from .piper_global_ik import PiperStyleGlobalIKSolver

            solver = PiperStyleGlobalIKSolver(
                urdf_path=path,
                end_frame=end_frame,
                ee_offset_matrix=ee_offset_matrix,
                lock_joint_names=tuple(lock_joint_names),
                position_weight=float(position_weight),
                rotation_weight=rw,
                max_iterations=int(max_iterations),
                tol=float(tol),
                jump_reset_threshold_deg=float(jump_reset_threshold_deg),
                max_solve_time_s=max_solve_time_s,
                regularize_to_warmstart=bool(regularize_to_warmstart),
                exact_when_converged=bool(exact_when_converged),
                regularization_weight=reg_w,
                joint_regularization_weights=jw,
            )
            # Match CLIK surface used by robot.py logging / warm-start.
            if not hasattr(solver, "nq"):
                solver.nq = int(solver.reduced_model.nq)  # type: ignore[attr-defined]
            if not hasattr(solver, "position_only"):
                solver.position_only = False  # type: ignore[attr-defined]
            logger.info(
                "[so101_plus.ik] Using local PiperStyleGlobalIKSolver "
                "(end_frame=%r, nq=%d, backend=%s, joint_reg=%s, reg_w=%.3f)",
                end_frame,
                solver.nq,
                getattr(solver, "_backend", "?"),
                list(jw),
                reg_w,
            )
            return solver
        except Exception as e:
            logger.warning(
                "[so101_plus.ik] Piper IK unavailable (%s); falling back to B601 CLIK.", e
            )

    kwargs = dict(
        urdf_path=path,
        end_frame=end_frame,
        ee_offset_matrix=ee_offset_matrix,
        lock_joint_names=tuple(lock_joint_names),
        position_weight=position_weight,
        rotation_weight=rw,
        max_iterations=int(max_iterations),
        tol=float(tol),
        jump_reset_threshold_deg=float(jump_reset_threshold_deg),
        max_solve_time_s=max_solve_time_s,
        step_size=float(step_size),
        damping=float(damping),
        position_only=bool(position_only),
        regularize_to_warmstart=regularize_to_warmstart,
        exact_when_converged=exact_when_converged,
        regularization_weight=reg_w,
    )

    if _pinocchio_importable():
        try:
            solver = So101PlusClikIKSolver(**kwargs)
            logger.info(
                "[so101_plus.ik] Using reBot B601-style Pinocchio CLIK (end_frame=%r, nq=%d)",
                end_frame,
                solver.nq,
            )
            return solver
        except Exception as e:
            logger.warning("[so101_plus.ik] Pinocchio CLIK init failed (%s); NumPy fallback.", e)

    logger.warning(
        "[so101_plus.ik] Using NumPy B601-style CLIK fallback for end_frame=%r.",
        end_frame,
    )
    return ScipyUrdfIKSolver(**kwargs)


# Backward-compatible alias used by robot.py / __init__.py
make_piper_ik_solver = make_ik_solver
