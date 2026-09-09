"""
Sequence-level EE → qpos IK for Alicia-D (reusable redundancy resolution).

Anchors the first frame to the current joint configuration, then solves the
remaining targets with temporal warm-starts so consecutive frames stay on the
same IK branch. Optional polish pass re-solves each frame from its own solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class SequenceIKStats:
    """Diagnostics for one sequence solve."""

    num_frames: int
    fails: int
    max_dq_deg: float
    mean_dq_deg: float
    max_fk_err: float
    mean_fk_err: float
    max_hint_err_deg: Optional[float] = None
    mean_hint_err_deg: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_frames": self.num_frames,
            "fails": self.fails,
            "max_dq_deg": self.max_dq_deg,
            "mean_dq_deg": self.mean_dq_deg,
            "max_fk_err": self.max_fk_err,
            "mean_fk_err": self.mean_fk_err,
            "max_hint_err_deg": self.max_hint_err_deg,
            "mean_hint_err_deg": self.mean_hint_err_deg,
        }


def _ee_row_to_pose(ee: np.ndarray, transform_fn) -> np.ndarray:
    ee = np.asarray(ee, dtype=np.float64).reshape(-1)
    if ee.shape[0] < 6:
        raise ValueError(f"EE row must be >=6D, got {ee.shape}")
    return transform_fn(ee[:3], ee[3:6])


def solve_sequence(
    solver,
    target_ee: np.ndarray,
    q_current: np.ndarray,
    *,
    transform_fn=None,
    q_hint: Optional[np.ndarray] = None,
    use_hint_warmstart: bool = False,
    polish: bool = True,
    max_step_deg: float = 15.0,
    log: bool = True,
) -> Tuple[np.ndarray, SequenceIKStats]:
    """
    Solve abs EE trajectory to joint trajectory.

    Args:
        solver: PiperStyleGlobalIKSolver (preferably reversible / warmstart-reg).
        target_ee: (T, 6|7) xyz+rpy[+gripper]
        q_current: (6,) current arm joints (anchors redundancy)
        transform_fn: xyz,rpy -> 4x4; default uses AliciaD helper
        q_hint: optional (T, 6|7) for diagnostics; optionally warmstart if
            ``use_hint_warmstart`` (default False so offline == runtime algorithm)
        use_hint_warmstart: if True and q_hint given, warmstart each frame from hint
        polish: second pass with warmstart=q[t]
        max_step_deg: hard-reject IK solutions that jump more than this from
            warmstart (branch-switch / singularity guard); always keep warmstart
        log: emit summary via loguru

    Returns:
        qpos (T, 7) float32, stats
    """
    if transform_fn is None:
        from deploy.robot.alicia_d.robot import _transform_matrix_from_xyz_rpy

        transform_fn = _transform_matrix_from_xyz_rpy

    ee = np.asarray(target_ee, dtype=np.float64)
    if ee.ndim != 2 or ee.shape[1] < 6:
        raise ValueError(f"target_ee must be (T, >=6), got {ee.shape}")
    t_len = ee.shape[0]
    q0 = np.asarray(q_current, dtype=np.float64).reshape(-1)[:6]
    if q0.shape[0] < 6:
        raise ValueError(f"q_current must be 6D, got {q0.shape}")

    hint = None
    if q_hint is not None:
        hint = np.asarray(q_hint, dtype=np.float64)
        if hint.ndim != 2 or hint.shape[0] != t_len or hint.shape[1] < 6:
            raise ValueError(f"q_hint must be ({t_len}, >=6), got {hint.shape}")

    out = np.zeros((t_len, 7), dtype=np.float64)
    fails = 0
    fk_errs: list = []
    max_step = np.deg2rad(float(max_step_deg))

    def _warm(t: int, fallback: np.ndarray) -> np.ndarray:
        if use_hint_warmstart and hint is not None:
            return hint[t, :6].copy()
        return np.asarray(fallback, dtype=np.float64).copy()

    def _accept(q_init: np.ndarray, q_arm: np.ndarray, pose: np.ndarray, ok: bool):
        """Hard-reject joint jumps > max_step (branch-switch / singularity thrash)."""
        nonlocal fails
        q_arm = np.asarray(q_arm, dtype=np.float64).reshape(-1)[:6]
        q_init = np.asarray(q_init, dtype=np.float64).reshape(-1)[:6]
        # Shortest-path joint delta (avoid ±2π false jumps).
        d = (q_arm - q_init + np.pi) % (2 * np.pi) - np.pi
        jump = float(np.max(np.abs(d)))
        err_sol = float(solver._compute_error_norm(q_arm, pose))
        if jump > max_step:
            # Always keep warmstart — never accept a large IK jump even if FK looks better.
            # Near-home singularity often finds another branch with lower FK err but wild Δq.
            err_ws = float(solver._compute_error_norm(q_init, pose))
            fails += 1
            return q_init, err_ws, False
        if not ok:
            fails += 1
        return q_arm, err_sol, True

    # Critical: reversible solvers use exact_when_converged with tol~0.05.
    # That early-exits with q_prev for many frames whose EE still "fits", producing
    # long joint freezes then catch-up jumps ("一步卡一下"). Disable during sequence.
    prev_exact = bool(getattr(solver, "exact_when_converged", False))
    solver.exact_when_converged = False
    try:
        # Pass 1: current-anchor + temporal chain
        solver.reset(q0)
        q_prev = q0.copy()
        for t in range(t_len):
            pose = _ee_row_to_pose(ee[t], transform_fn)
            q_init = _warm(t, q_prev if t > 0 else q0)
            ok, q_arm, _iters, _err = solver.solve(pose, q_init=q_init)
            q_arm, err, _ = _accept(q_init, q_arm, pose, ok)
            out[t, :6] = q_arm
            out[t, 6] = float(ee[t, 6]) if ee.shape[1] >= 7 else 0.5
            fk_errs.append(float(err))
            q_prev = np.asarray(q_arm, dtype=np.float64).copy()

        # Pass 2: polish from own solution (suppress spikes)
        if polish and t_len > 0:
            for t in range(t_len):
                pose = _ee_row_to_pose(ee[t], transform_fn)
                q_init = out[t, :6].copy()
                ok, q_arm, _iters, _err = solver.solve(pose, q_init=q_init)
                q_arm, err, _ = _accept(q_init, q_arm, pose, ok)
                out[t, :6] = q_arm
                fk_errs[t] = float(err)
    finally:
        solver.exact_when_converged = prev_exact

    # Smoothness diagnostics
    frozen = 0
    if t_len >= 2:
        dq = np.rad2deg(np.abs(np.diff(out[:, :6], axis=0)))
        max_dq = float(np.max(dq))
        mean_dq = float(np.mean(dq))
        frozen = int(np.sum(np.max(dq, axis=1) < 1e-4))
    else:
        max_dq, mean_dq = 0.0, 0.0

    max_fk = float(np.max(fk_errs)) if fk_errs else 0.0
    mean_fk = float(np.mean(fk_errs)) if fk_errs else 0.0

    max_hint = mean_hint = None
    if hint is not None:
        herr = np.rad2deg(np.abs(out[:, :6] - hint[:, :6]))
        max_hint = float(np.max(herr))
        mean_hint = float(np.mean(herr))

    stats = SequenceIKStats(
        num_frames=t_len,
        fails=fails,
        max_dq_deg=max_dq,
        mean_dq_deg=mean_dq,
        max_fk_err=max_fk,
        mean_fk_err=mean_fk,
        max_hint_err_deg=max_hint,
        mean_hint_err_deg=mean_hint,
    )
    if log:
        msg = (
            f"SequenceIK: T={t_len} fails={fails} "
            f"Δq(max/mean)={max_dq:.2f}/{mean_dq:.2f}° "
            f"frozen_frames={frozen}/{max(t_len - 1, 0)} "
            f"fk_err(max/mean)={max_fk:.4f}/{mean_fk:.4f}"
        )
        if max_hint is not None:
            msg += f" vs_hint(max/mean)={max_hint:.2f}/{mean_hint:.2f}°"
        logger.info(msg)

    return out.astype(np.float32), stats


def solve_sequence_with_probe(
    target_ee: np.ndarray,
    q_current: np.ndarray,
    *,
    q_hint: Optional[np.ndarray] = None,
    use_hint_warmstart: bool = False,
    polish: bool = True,
    urdf_path: Optional[str] = None,
    end_frame: str = "link6",
    ik_eps: float = 0.05,
    ik_max_iter: int = 200,
    ik_rotation_weight: float = 0.1,
    regularization_weight: float = 1.0,
    max_step_deg: float = 15.0,
    log: bool = True,
) -> Tuple[np.ndarray, SequenceIKStats]:
    """Convenience: build a reversible Piper IK solver and run solve_sequence."""
    from deploy.robot.alicia_d.robot import AliciaD

    probe = AliciaD(
        name="sequence_ik_offline",
        control_mode="qpos",
        urdf_path=urdf_path,
        ik_end_frame=end_frame,
        ik_eps=ik_eps,
        ik_max_iter=ik_max_iter,
        ik_rotation_weight=ik_rotation_weight,
        max_joint_delta_deg=1e9,
    )
    solver = probe._make_ik_solver(
        reversible=True, regularization_weight=float(regularization_weight)
    )
    return solve_sequence(
        solver,
        target_ee,
        q_current,
        q_hint=q_hint,
        use_hint_warmstart=use_hint_warmstart,
        polish=polish,
        max_step_deg=max_step_deg,
        log=log,
    )
