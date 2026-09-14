"""
Bessica-D bimanual kinematics (Pinocchio).

Loads the bundled URDF and provides FK / per-arm damped least-squares IK
for the 7-DOF arms (frames on *arm_link7). MuJoCo joint order is
right_arm_joint1..7 then left_arm_joint1..7; configuration vectors are
mapped by joint name so Pinocchio q stays consistent.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import pinocchio as pin


def _atan2(a, b):
    return round(math.atan2(a, b), 6)


def _cos(x):
    return round(math.cos(x), 6)


def _sqrt(x):
    return round(math.sqrt(max(0.0, x)), 6)


def T_to_pose_xyz_rpy(T: np.ndarray) -> np.ndarray:
    """End-effector pose [x, y, z, roll, pitch, yaw] (extrinsic XYZ Euler)."""
    R = T[:3, :3]
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    beta = _atan2(-R[2, 0], _sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if abs(_cos(beta)) > 1e-6:
        alpha = _atan2(R[1, 0] / _cos(beta), R[0, 0] / _cos(beta))
        gamma = _atan2(R[2, 1] / _cos(beta), R[2, 2] / _cos(beta))
    else:
        alpha = 0.0
        gamma = _atan2(R[0, 1], R[1, 1])
    return np.array([x, y, z, gamma, beta, alpha], dtype=np.float64)


def apply_delta_pose_world(T: np.ndarray, dpos: np.ndarray, drot_xyz: np.ndarray) -> np.ndarray:
    """Apply translation in world frame and rotation delta in local frame (axis-angle)."""
    out = T.copy()
    out[:3, 3] = T[:3, 3] + dpos
    if np.linalg.norm(drot_xyz) > 1e-12:
        dR = pin.exp3(drot_xyz)
        out[:3, :3] = T[:3, :3] @ dR
    return out


class BessicaBimanualKinematics:
    """Pinocchio model for Bessica-D (14 revolute arm joints, fixed base + grippers)."""

    RIGHT_JOINTS: List[str] = [f"right_arm_joint{i}" for i in range(1, 8)]
    LEFT_JOINTS: List[str] = [f"left_arm_joint{i}" for i in range(1, 8)]

    def __init__(self, urdf_path: Optional[str] = None):
        if urdf_path is None:
            urdf_path = str(Path(__file__).resolve().parent / "assets" / "Bessica-D_Covered.urdf")
        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        for jn in self.RIGHT_JOINTS + self.LEFT_JOINTS:
            self.model.getJointId(jn)

        self._frame_r = self._resolve_frame("right")
        self._frame_l = self._resolve_frame("left")
        self._right_v_cols = self._velocity_cols(self.model, self.RIGHT_JOINTS)
        self._left_v_cols = self._velocity_cols(self.model, self.LEFT_JOINTS)

        self._w_err = np.array([0.45, 0.45, 0.45, 1.0, 1.0, 1.0], dtype=np.float64)

    def _resolve_frame(self, side: str) -> int:
        for name in (f"{side}_arm_link7", f"{side}_arm_joint7"):
            try:
                return self.model.getFrameId(name)
            except ValueError:
                continue
        raise ValueError(f"No frame for {side} arm link7 / joint7 in URDF")

    @staticmethod
    def _velocity_cols(model: pin.Model, names: List[str]) -> List[int]:
        cols: List[int] = []
        for n in names:
            jid = model.getJointId(n)
            j = model.joints[jid]
            for k in range(j.nv):
                cols.append(j.idx_v + k)
        return cols

    def q_mujoco_to_pin(self, q_mj: np.ndarray) -> np.ndarray:
        """Map MuJoCo qpos (right 7 then left 7) to Pinocchio q (name-ordered)."""
        names = self.RIGHT_JOINTS + self.LEFT_JOINTS
        if len(q_mj) != 14:
            raise ValueError(f"Expected 14 joint values, got {len(q_mj)}")
        q = pin.neutral(self.model)
        for i, name in enumerate(names):
            jid = self.model.getJointId(name)
            iq = self.model.joints[jid].idx_q
            for k in range(self.model.joints[jid].nq):
                q[iq + k] = q_mj[i]
        return q

    def q_pin_to_mujoco(self, q: np.ndarray) -> np.ndarray:
        names = self.RIGHT_JOINTS + self.LEFT_JOINTS
        out = np.zeros(14, dtype=np.float64)
        for i, name in enumerate(names):
            jid = self.model.getJointId(name)
            iq = self.model.joints[jid].idx_q
            out[i] = q[iq]
        return out

    def fk_T(self, q_pin: np.ndarray, side: str) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q_pin)
        pin.updateFramePlacements(self.model, self.data)
        fid = self._frame_r if side == "right" else self._frame_l
        oMf = self.data.oMf[fid]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = oMf.rotation
        T[:3, 3] = oMf.translation
        return T

    def fk_pose(self, q_pin: np.ndarray, side: str) -> np.ndarray:
        return T_to_pose_xyz_rpy(self.fk_T(q_pin, side))

    def solve_ik_arm(
        self,
        q_pin: np.ndarray,
        side: str,
        target_T: np.ndarray,
        max_iter: int = 30,
        eps: float = 2e-3,
        dt: float = 0.35,
        damp: float = 2e-3,
    ) -> Tuple[bool, np.ndarray]:
        """
        Damped least-squares IK for one arm; other joints in q_pin are held as
        initial guess and only arm columns of the Jacobian are used.
        """
        cols = self._right_v_cols if side == "right" else self._left_v_cols
        nv_a = len(cols)
        fid = self._frame_r if side == "right" else self._frame_l
        target = pin.SE3(target_T[:3, :3], target_T[:3, 3])
        q = q_pin.copy()
        reg = damp * np.eye(nv_a)

        best_err = float("inf")
        best_q = q.copy()

        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            dMi = self.data.oMf[fid].actInv(target)
            err = pin.log(dMi).vector
            err_w = self._w_err * err
            nerr = float(np.linalg.norm(err_w))
            if nerr < best_err:
                best_err = nerr
                best_q = q.copy()
            if nerr < eps:
                return True, q

            J = pin.computeFrameJacobian(
                self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL
            )
            J_a = J[:, cols]
            J_w = self._w_err[:, None] * J_a
            try:
                dq_a = np.linalg.solve(J_w.T @ J_w + reg, J_w.T @ err_w)
            except np.linalg.LinAlgError:
                dq_a = np.linalg.lstsq(J_w, err_w, rcond=None)[0]

            v = np.zeros(self.model.nv, dtype=np.float64)
            for j, c in enumerate(cols):
                v[c] = dq_a[j]
            q = pin.integrate(self.model, q, v * dt)
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)

        if best_err < eps * 8.0:
            return True, best_q
        return False, best_q

    def solve_delta_bimanual(
        self,
        q_pin: np.ndarray,
        d_right: np.ndarray,
        d_left: np.ndarray,
        position_scale: float,
        rotation_scale: float,
        fast: bool = True,
    ) -> Tuple[bool, np.ndarray]:
        """
        Apply EE deltas for both arms (each 6D: dx,dy,dz, dr,dp,dy).
        Gripper channels are ignored at kinematics level.
        """
        max_iter = 18 if fast else 35
        # Keyboard delta_ee sends ~2mm translation steps. The previous fast epsilon
        # (3e-3) treated those as already converged, so translational commands were
        # effectively ignored even though rotations still moved.
        eps = 3e-4 if fast else 1e-4
        dt = 0.45 if fast else 0.35

        q = q_pin.copy()
        ok_all = True

        for side, d in ("right", d_right), ("left", d_left):
            T = self.fk_T(q, side)
            dpos = d[:3] * position_scale
            drot = d[3:6] * rotation_scale
            if np.sum(np.abs(dpos)) + np.sum(np.abs(drot)) < 1e-12:
                continue
            tgt = apply_delta_pose_world(T, dpos, drot)
            ok, q = self.solve_ik_arm(q, side, tgt, max_iter=max_iter, eps=eps, dt=dt)
            ok_all = ok_all and ok

        return ok_all, q

    def solve_targets_bimanual(
        self,
        q_pin: np.ndarray,
        T_right: np.ndarray,
        T_left: np.ndarray,
        fast: bool = True,
    ) -> Tuple[bool, np.ndarray]:
        """
        IK to absolute SE(3) targets for right then left arm (sequential, same as delta stack).
        """
        max_iter = 18 if fast else 35
        eps = 3e-4 if fast else 1e-4
        dt = 0.45 if fast else 0.35
        q = q_pin.copy()
        ok_r, q = self.solve_ik_arm(q, "right", T_right, max_iter=max_iter, eps=eps, dt=dt)
        ok_l, q = self.solve_ik_arm(q, "left", T_left, max_iter=max_iter, eps=eps, dt=dt)
        return ok_r and ok_l, q


def default_urdf_path() -> str:
    return str(Path(__file__).resolve().parent / "assets" / "Bessica-D_Covered.urdf")
