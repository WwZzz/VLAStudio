"""
Piper-style global IK solver for Alicia-D.

This solver is intentionally independent from the existing realtime DLS solver.
It uses a Piper-style global formulation:

- optimize joint positions directly with `casadi + ipopt`
- minimize weighted SE(3) pose error
- add a small global joint regularization term
- warm start from the previous solution

Unlike the Piper reference, the end-effector frame is configurable:
- the endpoint selection is provided by `end_frame`
- the end-effector pose offset is provided by `ee_offset_matrix`
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


class PiperStyleGlobalIKSolver:
    """Global IK solver modeled after the Piper teleoperation implementation."""

    def __init__(
        self,
        urdf_path: str,
        end_frame: str = "link6",
        ee_offset_matrix: Optional[np.ndarray] = None,
        lock_joint_names: Sequence[str] = ("left_finger", "right_finger"),
        ee_frame_name: str = "ee",
        position_weight: float = 1.0,
        rotation_weight: float = 0.1,
        total_cost_scale: float = 20.0,
        regularization_weight: float = 0.01,
        max_iterations: int = 50,
        tol: float = 1e-4,
        jump_reset_threshold_deg: float = 30.0,
        # Teleop keeps defaults (reg→0, no exact early-exit).
        # Recording / inference EE control: regularize_to_warmstart + exact_when_converged.
        regularize_to_warmstart: bool = False,
        exact_when_converged: bool = False,
        # Soft wall-clock budget for scipy teleop (None = disabled). On exceed, hold warm-start.
        max_solve_time_s: Optional[float] = None,
    ):
        import pinocchio as pin
        from scipy import optimize

        self.pin = pin
        self.optimize = optimize

        self.urdf_path = str(urdf_path)
        self.end_frame = end_frame
        self.ee_frame_name = ee_frame_name
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
        self.total_cost_scale = total_cost_scale
        self.regularization_weight = regularization_weight
        self.max_iterations = max_iterations
        self.tol = tol
        self.jump_reset_threshold_rad = np.deg2rad(jump_reset_threshold_deg)
        self.regularize_to_warmstart = bool(regularize_to_warmstart)
        self.exact_when_converged = bool(exact_when_converged)
        self.max_solve_time_s = (
            None if max_solve_time_s is None or float(max_solve_time_s) <= 0
            else float(max_solve_time_s)
        )
        self._last_solve_iterations = max_iterations
        self._last_solve_timed_out = False

        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        reference_configuration = np.zeros(self.model.nq)
        lock_joint_ids = [
            self.model.getJointId(joint_name)
            for joint_name in lock_joint_names
            if self.model.existJointName(joint_name)
        ]
        self.reduced_model = pin.buildReducedModel(
            self.model,
            lock_joint_ids,
            reference_configuration,
        )
        # Data must be created AFTER all frames exist on the model; addFrame below
        # increases nframes, so oMf length is wrong if createData runs too early.

        if ee_offset_matrix is None:
            ee_offset_matrix = np.eye(4, dtype=np.float64)
        ee_offset_matrix = np.asarray(ee_offset_matrix, dtype=np.float64)
        if ee_offset_matrix.shape != (4, 4):
            raise ValueError(f"ee_offset_matrix must have shape (4, 4), got {ee_offset_matrix.shape}")
        self.ee_offset_matrix = ee_offset_matrix

        end_frame_id = self.reduced_model.getFrameId(end_frame)
        if end_frame_id >= len(self.reduced_model.frames):
            raise ValueError(f"Frame '{end_frame}' not found in reduced robot model")

        parent_frame = self.reduced_model.frames[end_frame_id]
        offset_se3 = pin.SE3(ee_offset_matrix[:3, :3], ee_offset_matrix[:3, 3])
        ee_placement = parent_frame.placement * offset_se3
        self.reduced_model.addFrame(
            pin.Frame(
                ee_frame_name,
                parent_frame.parent,
                parent_frame.previousFrame,
                ee_placement,
                pin.FrameType.OP_FRAME,
            )
        )
        self.ee_frame_id = self.reduced_model.getFrameId(ee_frame_name)
        self.reduced_data = self.reduced_model.createData()

        self._backend = "scipy"
        self._bounds = list(
            zip(
                self.reduced_model.lowerPositionLimit.tolist(),
                self.reduced_model.upperPositionLimit.tolist(),
            )
        )

        try:
            import casadi
            from pinocchio import casadi as cpin

            self.casadi = casadi
            self.cpin = cpin
            self.cmodel = cpin.Model(self.reduced_model)
            self.cdata = self.cmodel.createData()

            self.cq = casadi.SX.sym("q", self.reduced_model.nq, 1)
            self.cTf = casadi.SX.sym("tf", 4, 4)
            cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

            self.error = casadi.Function(
                "error",
                [self.cq, self.cTf],
                [
                    casadi.vertcat(
                        cpin.log6(
                            self.cdata.oMf[self.ee_frame_id].inverse() * cpin.SE3(self.cTf)
                        ).vector
                    )
                ],
            )

            self.opti = casadi.Opti()
            self.var_q = self.opti.variable(self.reduced_model.nq)
            self.param_tf = self.opti.parameter(4, 4)
            self.param_q_reg = self.opti.parameter(self.reduced_model.nq)

            error_vec = self.error(self.var_q, self.param_tf)
            pos_error = error_vec[:3]
            ori_error = error_vec[3:]
            total_cost = (
                casadi.sumsqr(self.position_weight * pos_error)
                + casadi.sumsqr(self.rotation_weight * ori_error)
            )
            if self.regularize_to_warmstart:
                regularization = casadi.sumsqr(self.var_q - self.param_q_reg)
            else:
                regularization = casadi.sumsqr(self.var_q)

            self.opti.subject_to(
                self.opti.bounded(
                    self.reduced_model.lowerPositionLimit,
                    self.var_q,
                    self.reduced_model.upperPositionLimit,
                )
            )
            self.opti.minimize(
                self.total_cost_scale * total_cost
                + self.regularization_weight * regularization
            )
            self.opti.solver(
                "ipopt",
                {
                    "ipopt": {
                        "print_level": 0,
                        "max_iter": max_iterations,
                        "tol": tol,
                    },
                    "print_time": False,
                },
            )
            self._backend = "casadi"
        except Exception:
            self.casadi = None
            self.cpin = None
            self.cmodel = None
            self.cdata = None
            self.cq = None
            self.cTf = None
            self.error = None
            self.opti = None
            self.var_q = None
            self.param_tf = None
            self.param_q_reg = None

        self.init_data = np.zeros(self.reduced_model.nq, dtype=np.float64)
        self.history_data = np.zeros(self.reduced_model.nq, dtype=np.float64)
        self._q_ref = np.zeros(self.reduced_model.nq, dtype=np.float64)
        self._reference_pose: Optional[np.ndarray] = None

    def reset(self, q: Optional[np.ndarray] = None):
        """Reset warm-start state."""
        if q is None:
            q = np.zeros(self.reduced_model.nq, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64).reshape(-1)[: self.reduced_model.nq]
        self.init_data = q.copy()
        self.history_data = q.copy()

    def get_current_q(self) -> np.ndarray:
        """Return the current warm-start configuration."""
        return self.init_data.copy()

    def set_reference(self, q_ref: np.ndarray, pose_ref: Optional[np.ndarray] = None):
        """Compatibility hook for AliciaD anchor updates."""
        q_ref = np.asarray(q_ref, dtype=np.float64).reshape(-1)[: self.reduced_model.nq]
        self._q_ref = q_ref.copy()
        if pose_ref is None:
            pose_ref = self.compute_fk(q_ref)
        self._reference_pose = np.asarray(pose_ref, dtype=np.float64).copy()

    def _solve_impl(self, target_pose: np.ndarray, q_init: np.ndarray) -> Tuple[bool, np.ndarray]:
        if self._backend == "casadi":
            self.opti.set_initial(self.var_q, q_init)
            self.opti.set_value(self.param_tf, target_pose)
            self.opti.set_value(self.param_q_reg, q_init)
            try:
                self.opti.solve_limited()
                sol_q = np.asarray(self.opti.value(self.var_q), dtype=np.float64).reshape(-1)
                self._last_solve_iterations = self.max_iterations
                return True, sol_q
            except Exception:
                return False, q_init.copy()

        q_reg = np.asarray(q_init, dtype=np.float64)
        self._last_solve_timed_out = False
        t0 = time.perf_counter()
        n_cb = [0]
        # Track best pose-error iterate so a time-budget abort can still move toward target.
        best_q = q_init.copy()
        best_err = self._compute_error_norm(q_init, target_pose)

        class _BudgetExceeded(Exception):
            pass

        def _maybe_update_best(q: np.ndarray) -> None:
            nonlocal best_q, best_err
            err = self._compute_error_norm(q, target_pose)
            if err < best_err:
                best_err = err
                best_q = np.asarray(q, dtype=np.float64).copy()

        def _budget_pick() -> np.ndarray:
            """Prefer best intermediate if it does not jump too far from warm-start."""
            dq = float(np.max(np.abs(best_q - q_init)))
            if np.all(np.isfinite(best_q)) and dq <= self.jump_reset_threshold_rad:
                return best_q.copy()
            return q_init.copy()

        def objective(q: np.ndarray) -> float:
            nonlocal best_q, best_err
            if (
                self.max_solve_time_s is not None
                and (time.perf_counter() - t0) > self.max_solve_time_s
            ):
                self._last_solve_timed_out = True
                raise _BudgetExceeded()
            q = np.asarray(q, dtype=np.float64)
            self.pin.forwardKinematics(self.reduced_model, self.reduced_data, q)
            self.pin.updateFramePlacements(self.reduced_model, self.reduced_data)
            current_pose = self.reduced_data.oMf[self.ee_frame_id]
            target = self.pin.SE3(target_pose[:3, :3], target_pose[:3, 3])
            err = self.pin.log(current_pose.actInv(target)).vector
            pos_error = err[:3]
            ori_error = err[3:]
            # Track every evaluate — timeout often hits before L-BFGS callback (nit=0),
            # so callback-only best tracking used to freeze at warmstart.
            err_norm = float(np.linalg.norm(np.concatenate([
                self.position_weight * pos_error,
                self.rotation_weight * ori_error,
            ])))
            if np.all(np.isfinite(q)) and err_norm < best_err:
                best_err = err_norm
                best_q = q.copy()
            total_cost = (
                np.sum((self.position_weight * pos_error) ** 2)
                + np.sum((self.rotation_weight * ori_error) ** 2)
            )
            if self.regularize_to_warmstart:
                regularization = np.sum((q - q_reg) ** 2)
            else:
                regularization = np.sum(q**2)
            return float(self.total_cost_scale * total_cost + self.regularization_weight * regularization)

        def _time_callback(xk: np.ndarray):
            n_cb[0] += 1
            xk = np.asarray(xk, dtype=np.float64)
            if np.all(np.isfinite(xk)):
                _maybe_update_best(xk)
            if self.max_solve_time_s is not None and (time.perf_counter() - t0) > self.max_solve_time_s:
                self._last_solve_timed_out = True
                return True
            return False

        try:
            result = self.optimize.minimize(
                objective,
                q_init,
                method="L-BFGS-B",
                bounds=self._bounds,
                callback=_time_callback if self.max_solve_time_s is not None else None,
                options={
                    "maxiter": self.max_iterations,
                    "maxfun": max(self.max_iterations * 4, 40),
                    "ftol": self.tol,
                },
            )
        except _BudgetExceeded:
            self._last_solve_iterations = int(n_cb[0])
            self._last_solve_timed_out = True
            return True, _budget_pick()

        self._last_solve_iterations = int(getattr(result, "nit", n_cb[0] or self.max_iterations))
        if self._last_solve_timed_out:
            if np.all(np.isfinite(result.x)):
                _maybe_update_best(result.x)
            return True, _budget_pick()

        if result.success:
            return True, np.asarray(result.x, dtype=np.float64)
        if np.all(np.isfinite(result.x)):
            return False, np.asarray(result.x, dtype=np.float64)
        return False, q_init.copy()

    def _compute_error_norm(self, q: np.ndarray, target_pose: np.ndarray) -> float:
        """Compute weighted pose error norm for diagnostics."""
        self.pin.forwardKinematics(self.reduced_model, self.reduced_data, q)
        self.pin.updateFramePlacements(self.reduced_model, self.reduced_data)
        current_pose = self.reduced_data.oMf[self.ee_frame_id]
        target = self.pin.SE3(target_pose[:3, :3], target_pose[:3, 3])
        err = self.pin.log(current_pose.actInv(target)).vector
        err_w = np.concatenate([
            self.position_weight * err[:3],
            self.rotation_weight * err[3:],
        ])
        return float(np.linalg.norm(err_w))

    def solve(
        self, target_pose: np.ndarray, q_init: Optional[np.ndarray] = None
    ) -> Tuple[bool, np.ndarray, int, float]:
        """
        Solve global IK for a target 4x4 pose.

        Returns:
            (success, q, iterations, error_norm)
        """
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (4, 4):
            raise ValueError(f"target_pose must have shape (4, 4), got {target_pose.shape}")

        if q_init is None:
            q_init = self.init_data
        else:
            q_init = np.asarray(q_init, dtype=np.float64).reshape(-1)[: self.reduced_model.nq]

        if self.exact_when_converged:
            err_init = self._compute_error_norm(q_init, target_pose)
            if err_init <= self.tol:
                self.init_data = q_init.copy()
                self.history_data = q_init.copy()
                self._last_solve_iterations = 0
                return True, q_init.copy(), 0, err_init

        success, sol_q = self._solve_impl(target_pose, q_init)
        if not success:
            err_norm = self._compute_error_norm(q_init, target_pose)
            return False, q_init.copy(), self._last_solve_iterations, err_norm

        if self.init_data is not None:
            max_diff = float(np.max(np.abs(self.history_data - sol_q)))
            self.init_data = sol_q.copy()
            if max_diff > self.jump_reset_threshold_rad:
                self.init_data = np.zeros(self.reduced_model.nq, dtype=np.float64)
        else:
            self.init_data = sol_q.copy()

        self.history_data = sol_q.copy()
        err_norm = self._compute_error_norm(sol_q, target_pose)
        return True, sol_q, self._last_solve_iterations, err_norm

    def compute_fk(self, q: np.ndarray) -> np.ndarray:
        """Compute FK of the configured end-effector frame."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)[: self.reduced_model.nq]
        self.pin.forwardKinematics(self.reduced_model, self.reduced_data, q)
        self.pin.updateFramePlacements(self.reduced_model, self.reduced_data)
        pose = self.reduced_data.oMf[self.ee_frame_id]
        T = np.eye(4)
        T[:3, :3] = pose.rotation
        T[:3, 3] = pose.translation
        return T

    def get_current_pose(self) -> np.ndarray:
        """Return FK of the current warm-start state."""
        return self.compute_fk(self.init_data)

    @staticmethod
    def create_transform_matrix(
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> np.ndarray:
        """Create a 4x4 transform matrix from xyz + rpy."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        rot_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        rot_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rot_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        rotation = rot_z @ rot_y @ rot_x

        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = np.array([x, y, z], dtype=np.float64)
        return T
