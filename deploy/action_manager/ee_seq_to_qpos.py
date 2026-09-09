"""
Convert abs-EE action chunks to joint abs via Alicia-D sequence IK, then dispatch.

Inference-compatible: anchors on current robot q, then temporal warmstarts
frame-to-frame (same as offline ``--presolve-ee-to-qpos`` without hint).
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from loguru import logger

from deploy.action_manager.sync_chunk import SyncChunkManager
from deploy.action_manager.async_chunk_ratio import AsyncChunkRatioManager


def _unwrap_joint_path(q_joints: np.ndarray) -> np.ndarray:
    """Make joint angles continuous along time (shortest-path unwrap)."""
    q = np.asarray(q_joints, dtype=np.float64).copy()
    if q.ndim != 2 or q.shape[0] < 2:
        return q
    for i in range(1, q.shape[0]):
        d = (q[i] - q[i - 1] + np.pi) % (2.0 * np.pi) - np.pi
        q[i] = q[i - 1] + d
    return q


def _catmull_tangents(p: np.ndarray) -> np.ndarray:
    """Catmull-Rom tangents on unwrapped waypoints (waypoint-index units)."""
    T = p.shape[0]
    m = np.zeros_like(p)
    if T < 2:
        return m
    if T == 2:
        m[0] = m[1] = p[1] - p[0]
        return m
    m[0] = p[1] - p[0]
    m[-1] = p[-1] - p[-2]
    for i in range(1, T - 1):
        m[i] = 0.5 * (p[i + 1] - p[i - 1])
    return m


def _hermite(p0, p1, m0, m1, t: float):
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def _min_jerk_alpha(t: float) -> float:
    """Smoothstep 0→1 with zero endpoint velocity & acceleration."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * t * (10.0 + t * (-15.0 + t * 6.0))


def _unwrap_lerp_joints(qpos_hd: np.ndarray, factor: int) -> np.ndarray:
    """Dense joint trajectory by shortest-path linear lerp (legacy)."""
    if factor <= 1 or qpos_hd.shape[0] < 2:
        return qpos_hd
    q = np.asarray(qpos_hd, dtype=np.float64)
    rows = []
    for i in range(q.shape[0] - 1):
        a, b = q[i], q[i + 1]
        d = (b[:6] - a[:6] + np.pi) % (2 * np.pi) - np.pi
        for k in range(factor):
            t = k / float(factor)
            qi = a.copy()
            qi[:6] = a[:6] + t * d
            qi[6] = (1.0 - t) * a[6] + t * b[6]
            rows.append(qi)
    rows.append(q[-1].copy())
    return np.asarray(rows, dtype=np.float32)


def _unwrap_cubic_joints(qpos_hd: np.ndarray, factor: int) -> np.ndarray:
    """
    Densify IK waypoints in *joint* space with Catmull-Rom / cubic Hermite.

    Linear lerp has discontinuous velocity at knots (infinite accel spikes).
    Cubic keeps C1 continuity so densified playback has far lower joint accel.
    Gripper uses the same cubic on scalar values (no angle wrap).
    """
    if factor <= 1 or qpos_hd.shape[0] < 2:
        return qpos_hd
    q = np.asarray(qpos_hd, dtype=np.float64)
    pj = _unwrap_joint_path(q[:, :6])
    pg = q[:, 6:7] if q.shape[1] >= 7 else np.zeros((q.shape[0], 1), dtype=np.float64)
    mj = _catmull_tangents(pj)
    mg = _catmull_tangents(pg)
    rows = []
    for i in range(q.shape[0] - 1):
        for k in range(factor):
            t = k / float(factor)
            qi = np.zeros(7, dtype=np.float64)
            qi[:6] = _hermite(pj[i], pj[i + 1], mj[i], mj[i + 1], t)
            qi[6] = float(_hermite(pg[i], pg[i + 1], mg[i], mg[i + 1], t)[0])
            rows.append(qi)
    last = np.zeros(7, dtype=np.float64)
    last[:6] = pj[-1]
    last[6] = float(pg[-1, 0])
    rows.append(last)
    return np.asarray(rows, dtype=np.float32)


def _limit_joint_accel(
    qpos_hd: np.ndarray,
    max_accel_rad_s2: float,
    playback_hz: float,
    n_iters: int = 80,
) -> np.ndarray:
    """
    Soft-limit discrete joint acceleration at the playback rate.

    Gauss–Seidel: for each interior sample, set q[i] so
    ``|q[i+1]-2q[i]+q[i-1]| ≤ max_accel / hz^2`` (endpoints fixed).
    May slightly round sharp corners — that is the intended tradeoff.
    """
    if max_accel_rad_s2 <= 0 or playback_hz <= 0 or qpos_hd.shape[0] < 3:
        return qpos_hd
    q = np.asarray(qpos_hd, dtype=np.float64).copy()
    q[:, :6] = _unwrap_joint_path(q[:, :6])
    max_d2 = float(max_accel_rad_s2) / (float(playback_hz) ** 2)
    if max_d2 <= 0:
        return qpos_hd
    n = q.shape[0]
    for _ in range(max(1, int(n_iters))):
        changed = False
        for i in range(1, n - 1):
            d2 = q[i + 1, :6] - 2.0 * q[i, :6] + q[i - 1, :6]
            over = np.abs(d2) > max_d2 + 1e-12
            if not np.any(over):
                continue
            d2_c = np.clip(d2, -max_d2, max_d2)
            # Exact local projection for this index (neighbors treated as fixed).
            q[i, :6] = np.where(
                over,
                0.5 * (q[i - 1, :6] + q[i + 1, :6] - d2_c),
                q[i, :6],
            )
            changed = True
        if not changed:
            break
    return q.astype(np.float32)


def _densify_joint_traj(
    qpos_hd: np.ndarray,
    factor: int,
    mode: str = "cubic",
    max_accel_deg_s2: float = 0.0,
    playback_hz: Optional[float] = None,
) -> np.ndarray:
    """Joint-space densify (+ optional accel limit). Never interpolates in EE."""
    mode_l = (mode or "cubic").lower()
    if factor > 1:
        if mode_l == "linear":
            out = _unwrap_lerp_joints(qpos_hd, factor)
        else:
            out = _unwrap_cubic_joints(qpos_hd, factor)
    else:
        out = np.asarray(qpos_hd, dtype=np.float32)
    if max_accel_deg_s2 > 0 and playback_hz is not None and playback_hz > 0:
        out = _limit_joint_accel(
            out,
            max_accel_rad_s2=float(np.deg2rad(max_accel_deg_s2)),
            playback_hz=float(playback_hz),
        )
    return out


def _lerp_joint_rows(q0: np.ndarray, q1: np.ndarray, n_steps: int) -> np.ndarray:
    """n_steps min-jerk rows from q0 → q1 (excludes q0, includes q1)."""
    a = np.asarray(q0, dtype=np.float64).reshape(-1).copy()
    b = np.asarray(q1, dtype=np.float64).reshape(-1).copy()
    if a.size < 7:
        a = np.pad(a, (0, 7 - a.size))
    if b.size < 7:
        b = np.pad(b, (0, 7 - b.size))
    n = max(1, int(n_steps))
    d = (b[:6] - a[:6] + np.pi) % (2 * np.pi) - np.pi
    rows = []
    for k in range(1, n + 1):
        t = _min_jerk_alpha(k / float(n))
        qi = a.copy()
        qi[:6] = a[:6] + t * d
        qi[6] = (1.0 - t) * a[6] + t * b[6]
        rows.append(qi)
    return np.asarray(rows, dtype=np.float32)


def _interp_factor_for_hz(chunk_hz: float, playback_hz: float, chunk_T: int = 16) -> int:
    """
    Choose densify factor so T waypoints at ``chunk_hz`` keep the same wall-clock
    duration when played at ``playback_hz``.

    duration ≈ T / chunk_hz
    target_len ≈ duration * playback_hz = T * playback_hz / chunk_hz
    factor ≈ (target_len - 1) / (T - 1)
    """
    if chunk_hz <= 0 or playback_hz <= 0:
        return 1
    T = max(2, int(chunk_T))
    target_len = T * float(playback_hz) / float(chunk_hz)
    factor = int(round((target_len - 1.0) / (T - 1)))
    return max(1, factor)


def _interp_factor_for_min_duration(
    min_duration_s: float, playback_hz: float, chunk_T: int = 16
) -> int:
    """Densify so ((T-1)*f+1)/playback_hz >= min_duration_s."""
    if min_duration_s <= 0 or playback_hz <= 0:
        return 1
    T = max(2, int(chunk_T))
    need = float(min_duration_s) * float(playback_hz)
    return max(1, int(np.ceil((need - 1.0) / (T - 1))))


class EESequenceToQposManager(SyncChunkManager):
    """
    Sync chunk flow: infer EE chunk → sequence IK → dispatch joint abs.
    Inherits SyncChunkManager so playback is never interrupted mid-chunk.

    Safety: after IK, reject/hold if joint jumps or FK error look like thrash
    (common near home / singularity when abs-EE IK flips redundancy branch).
    """

    def __init__(
        self,
        robot_shm_name: str = "alicia_d",
        polish: bool = True,
        ik_eps: float = 0.05,
        ik_max_iter: int = 80,
        ik_rotation_weight: float = 0.1,
        interp_factor: int = 1,
        # Temporal rates: policy waypoints at chunk_hz; control loop at playback_hz
        # (set --pr to the same value). Auto densifies so wall-clock duration of a
        # T-step chunk stays ≈ T/chunk_hz when played at playback_hz.
        chunk_hz: float = 30.0,
        playback_hz: Optional[float] = None,
        # Joint-space densify after IK: cubic (C1) avoids linear corner accel spikes.
        joint_interp: str = "cubic",  # cubic | linear
        # Soft cap on densified joint accel (deg/s^2); 0=off. Uses playback_hz.
        max_joint_accel_deg_s2: float = 900.0,
        # Optional floor on densify (async remote uses this for pipeline headroom).
        min_chunk_duration_s: float = 0.0,
        # EE deadband denoise before IK: suppress sub-threshold jitter only.
        # Steps above deadband pass through unchanged (no precision loss on real motion).
        ee_denoise_enabled: bool = False,
        ee_pos_deadband_m: float = 0.0008,
        ee_rot_deadband_deg: float = 0.25,
        # Post-IK joint-space denoise (preferred vs EE): causal EMA + micro-step hold.
        # alpha=1 / deadband=0 → off. Lower alpha = smoother / more lag.
        joint_ema_alpha: float = 0.4,
        joint_deadband_deg: float = 0.15,
        # Per-frame IK warmstart jump guard (passed to solve_sequence).
        max_ik_step_deg: float = 10.0,
        # Chunk-level safety: hold current q if any limit is exceeded.
        max_q0_delta_deg: float = 20.0,
        max_chunk_dq_deg: float = 15.0,
        max_fk_err: float = 0.10,
        max_ik_fails: int = 6,
        # After this many consecutive hard SAFETY HOLDs, raise IK max_iter.
        # On a hard fail, also retry the same chunk with boosted iters before holding.
        ik_boost_after_holds: int = 2,
        ik_boost_step: int = 40,
        ik_max_iter_cap: int = 400,
        ik_boost_retries: int = 2,
        # If True, clamp consecutive Δq to max_chunk_dq instead of full hold
        # when only mid-chunk steps are large (q0 still ok). Hold still wins
        # when q0_delta / fk_err / fails trip.
        clamp_traj: bool = True,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(debug=debug, **kwargs)
        self.robot_shm_name = str(robot_shm_name)
        self.polish = bool(polish)
        self.ik_eps = float(ik_eps)
        self.ik_max_iter = int(ik_max_iter)
        self.ik_rotation_weight = float(ik_rotation_weight)
        self.ik_boost_after_holds = max(1, int(ik_boost_after_holds))
        self.ik_boost_step = max(0, int(ik_boost_step))
        self.ik_max_iter_cap = max(self.ik_max_iter, int(ik_max_iter_cap))
        self.ik_boost_retries = max(0, int(ik_boost_retries))
        self._active_ik_max_iter = int(self.ik_max_iter)
        self.chunk_hz = float(chunk_hz) if chunk_hz is not None else 30.0
        self.playback_hz = float(playback_hz) if playback_hz not in (None, "", 0, 0.0) else None
        self.min_chunk_duration_s = max(0.0, float(min_chunk_duration_s or 0.0))
        explicit_interp = max(1, int(interp_factor))
        # Floor: explicit --interp>1 and/or min_chunk_duration; never wiped by hz auto.
        self._interp_floor = explicit_interp if explicit_interp > 1 else 1
        if self.min_chunk_duration_s > 0 and self.playback_hz is not None:
            self._interp_floor = max(
                self._interp_floor,
                _interp_factor_for_min_duration(
                    self.min_chunk_duration_s, self.playback_hz, chunk_T=16
                ),
            )
        if self.playback_hz is not None and self.chunk_hz > 0:
            auto_f = _interp_factor_for_hz(self.chunk_hz, self.playback_hz, chunk_T=16)
            self.interp_factor = max(auto_f, self._interp_floor)
        else:
            self.interp_factor = max(explicit_interp, self._interp_floor)
        mode = str(joint_interp or "cubic").lower().strip()
        if mode not in ("cubic", "linear"):
            logger.warning(
                "[EESequenceToQpos] unknown joint_interp={!r}, using cubic", joint_interp
            )
            mode = "cubic"
        self.joint_interp = mode
        self.max_joint_accel_deg_s2 = max(0.0, float(max_joint_accel_deg_s2 or 0.0))
        self.ee_denoise_enabled = bool(ee_denoise_enabled)
        self.ee_pos_deadband_m = max(0.0, float(ee_pos_deadband_m))
        self.ee_rot_deadband_deg = max(0.0, float(ee_rot_deadband_deg))
        self.joint_ema_alpha = float(np.clip(joint_ema_alpha, 0.05, 1.0))
        self.joint_deadband_deg = max(0.0, float(joint_deadband_deg))
        self.max_ik_step_deg = float(max_ik_step_deg)
        self.max_q0_delta_deg = float(max_q0_delta_deg)
        self.max_chunk_dq_deg = float(max_chunk_dq_deg)
        self.max_fk_err = float(max_fk_err)
        self.max_ik_fails = int(max_ik_fails)
        self.clamp_traj = bool(clamp_traj)
        self._robot_shm = None
        self._solver = None
        self._convert_count = 0
        self._last_ee_fp: Optional[bytes] = None
        self._last_ee_row: Optional[np.ndarray] = None
        self._last_joint_chunk = None
        self._last_q_end: Optional[np.ndarray] = None
        self._safety_holds = 0
        if self.ik_boost_step > 0:
            logger.info(
                "[EESequenceToQpos] IK iter boost: after {} holds +{} / retry "
                "(cap={}, retries={})",
                self.ik_boost_after_holds,
                self.ik_boost_step,
                self.ik_max_iter_cap,
                self.ik_boost_retries,
            )
        if self.playback_hz is not None:
            logger.info(
                "[EESequenceToQpos] timing chunk_hz={:.1f} playback_hz={:.1f} → "
                "interp_factor={} joint_interp={} max_accel={:.0f}°/s² "
                "(set --pr to playback_hz)",
                self.chunk_hz,
                self.playback_hz,
                self.interp_factor,
                self.joint_interp,
                self.max_joint_accel_deg_s2,
            )

    def _ensure_robot_shm(self):
        if self._robot_shm is not None:
            return
        from deploy.shm_utils import SharedMemoryChannel

        self._robot_shm = SharedMemoryChannel(name=self.robot_shm_name, is_writer=False)
        logger.info(
            "[EESequenceToQpos] Connected to robot SHM '{}' for q_current",
            self.robot_shm_name,
        )

    def _ensure_solver(self):
        if self._solver is not None:
            self._apply_ik_max_iter(self._active_ik_max_iter)
            return
        from deploy.robot.alicia_d.robot import AliciaD

        probe = AliciaD(
            name="ee_seq_mgr_ik",
            control_mode="qpos",
            ik_eps=self.ik_eps,
            ik_max_iter=self._active_ik_max_iter,
            ik_rotation_weight=self.ik_rotation_weight,
            max_joint_delta_deg=1e9,
        )
        self._solver = probe._make_ik_solver(
            reversible=True, regularization_weight=1.0
        )
        logger.info(
            "[EESequenceToQpos] Sequence IK ready (temporal warmstart only, "
            "polish={}, max_iter={}, interp={})",
            self.polish,
            self._active_ik_max_iter,
            self.interp_factor,
        )

    def _apply_ik_max_iter(self, n: int) -> int:
        """Set solver ``max_iterations``; returns the applied value."""
        n = int(np.clip(int(n), self.ik_max_iter, self.ik_max_iter_cap))
        self._active_ik_max_iter = n
        if self._solver is not None and int(
            getattr(self._solver, "max_iterations", -1)
        ) != n:
            self._solver.max_iterations = n
        return n

    def _boosted_ik_max_iter(self, *, hold_level: int, retry: int = 0) -> int:
        """IK iters after ``hold_level`` consecutive hard fails (+ optional retry)."""
        extra_holds = max(0, int(hold_level) - self.ik_boost_after_holds + 1)
        if int(hold_level) < self.ik_boost_after_holds:
            extra_holds = 0
        n = self.ik_max_iter + (extra_holds + max(0, int(retry))) * self.ik_boost_step
        return min(self.ik_max_iter_cap, max(self.ik_max_iter, n))

    def _reset_ik_iter_boost(self) -> None:
        if self._active_ik_max_iter != self.ik_max_iter:
            logger.info(
                "[EESequenceToQpos] IK recovered — max_iter {} → {}",
                self._active_ik_max_iter,
                self.ik_max_iter,
            )
        self._apply_ik_max_iter(self.ik_max_iter)
        if self.ee_denoise_enabled:
            logger.info(
                "[EESequenceToQpos] EE denoise ON: pos<{:.2f}mm rot<{:.2f}° "
                "(pose only; gripper unchanged)",
                self.ee_pos_deadband_m * 1000.0,
                self.ee_rot_deadband_deg,
            )
        if self.joint_ema_alpha < 1.0 - 1e-9 or self.joint_deadband_deg > 0:
            logger.info(
                "[EESequenceToQpos] joint denoise ON (post-IK): ema_alpha={:.2f} "
                "deadband={:.2f}°",
                self.joint_ema_alpha,
                self.joint_deadband_deg,
            )

    @staticmethod
    def _wrap_rpy_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a - b + np.pi) % (2.0 * np.pi) - np.pi

    def _denoise_ee_chunk(
        self,
        ee: np.ndarray,
        ee_prev: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, int]:
        """Suppress sub-threshold pose jitter; gripper always passes through."""
        if not self.ee_denoise_enabled:
            return ee, 0

        out = np.asarray(ee, dtype=np.float64).copy()
        pos_db = self.ee_pos_deadband_m
        rot_db = np.deg2rad(self.ee_rot_deadband_deg)
        n_suppressed = 0

        prev = (
            np.asarray(ee_prev, dtype=np.float64).reshape(-1).copy()
            if ee_prev is not None
            else out[0].copy()
        )

        def _pose_is_noise(row: np.ndarray) -> bool:
            dpos = float(np.linalg.norm(row[:3] - prev[:3]))
            drpy = float(np.max(np.abs(self._wrap_rpy_delta(row[3:6], prev[3:6]))))
            return dpos < pos_db and drpy < rot_db

        start_t = 0 if ee_prev is not None else 1
        for t in range(start_t, out.shape[0]):
            if _pose_is_noise(out[t]):
                if out.shape[1] >= 7:
                    grip = float(out[t, 6])
                    out[t, :6] = prev[:6]
                    out[t, 6] = grip
                else:
                    out[t, :6] = prev[:6]
                n_suppressed += 1
            prev = out[t].copy()
        return out.astype(np.float32), n_suppressed

    def _smooth_joint_traj(
        self,
        qpos_hd: np.ndarray,
        q_anchor: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Causal joint-space denoise after IK (+ densify).

        For each frame: shortest-path delta from previous smoothed q; hold if
        below ``joint_deadband_deg``; else EMA with ``joint_ema_alpha`` toward
        the raw IK target. Gripper is left unchanged.
        """
        a = self.joint_ema_alpha
        db = np.deg2rad(self.joint_deadband_deg)
        if a >= 1.0 - 1e-9 and db <= 0:
            return qpos_hd

        q = np.asarray(qpos_hd, dtype=np.float64).copy()
        if q.shape[0] == 0:
            return qpos_hd
        if q_anchor is not None:
            prev = np.asarray(q_anchor, dtype=np.float64).reshape(-1)[:6].copy()
        else:
            prev = q[0, :6].copy()

        for i in range(q.shape[0]):
            raw = q[i, :6]
            d = (raw - prev + np.pi) % (2.0 * np.pi) - np.pi
            if db > 0 and float(np.max(np.abs(d))) < db:
                q[i, :6] = prev
            else:
                step = d if a >= 1.0 - 1e-9 else (a * d)
                q[i, :6] = prev + step
            prev = q[i, :6].copy()
        return q.astype(np.float32)

    def _read_q_current(self) -> np.ndarray:
        self._ensure_robot_shm()
        data = self._robot_shm.read(skip_unchanged=False)
        if data is None:
            raise RuntimeError(
                f"[EESequenceToQpos] No data from SHM '{self.robot_shm_name}'"
            )
        qpos = data.get("qpos")
        if qpos is None:
            raise RuntimeError(
                f"[EESequenceToQpos] SHM '{self.robot_shm_name}' missing 'qpos'"
            )
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if q.shape[0] < 6:
            raise RuntimeError(f"[EESequenceToQpos] qpos dim {q.shape[0]} < 6")
        return q[:6].copy()

    @staticmethod
    def _chunk_ctrl(chunk) -> tuple:
        if not chunk:
            return None, None
        first = chunk[0]
        if isinstance(first, np.ndarray) and first.dtype == object and len(first) > 0:
            d0 = first[0]
            if isinstance(d0, dict):
                return d0.get("ctrl_space"), d0.get("ctrl_type")
        return None, None

    @staticmethod
    def _extract_ee_matrix(chunk) -> np.ndarray:
        rows: List[np.ndarray] = []
        for step in chunk:
            if isinstance(step, np.ndarray) and step.dtype == object and len(step) > 0:
                act = step[0].get("action") if isinstance(step[0], dict) else None
            elif isinstance(step, dict):
                act = step.get("action")
            else:
                act = step
            if act is None:
                raise ValueError("EE chunk step missing action")
            rows.append(np.asarray(act, dtype=np.float32).reshape(-1))
        return np.stack(rows, axis=0)

    @staticmethod
    def _build_joint_chunk(qpos_hd: np.ndarray, template_chunk=None) -> list:
        out = []
        for i in range(len(qpos_hd)):
            q = qpos_hd[i]
            out.append(
                np.array(
                    [
                        {
                            "action": np.asarray(q, dtype=np.float32).copy(),
                            "ctrl_space": "joint",
                            "ctrl_type": "abs",
                        }
                    ],
                    dtype=object,
                )
            )
        return out

    def _hold_joint_chunk(self, q_hold: np.ndarray, T: int, gripper: float) -> list:
        q = np.asarray(q_hold, dtype=np.float64).reshape(-1)
        row = np.zeros(7, dtype=np.float32)
        row[:6] = q[:6]
        row[6] = float(gripper)
        qpos_hd = np.repeat(row[None, :], max(1, int(T)), axis=0)
        return self._build_joint_chunk(qpos_hd)

    @staticmethod
    def _joint_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
        da = (np.asarray(a, dtype=np.float64)[:6] - np.asarray(b, dtype=np.float64)[:6]
              + np.pi) % (2 * np.pi) - np.pi
        return float(np.rad2deg(np.max(np.abs(da))))

    def _clamp_joint_traj(self, qpos_hd: np.ndarray, q_anchor: np.ndarray) -> np.ndarray:
        """Limit consecutive shortest-path joint steps to max_chunk_dq_deg."""
        q = np.asarray(qpos_hd, dtype=np.float64).copy()
        max_step = np.deg2rad(self.max_chunk_dq_deg)
        prev = np.asarray(q_anchor, dtype=np.float64).reshape(-1)[:6].copy()
        for t in range(q.shape[0]):
            d = (q[t, :6] - prev + np.pi) % (2 * np.pi) - np.pi
            jump = float(np.max(np.abs(d)))
            if jump > max_step and jump > 1e-9:
                d = d * (max_step / jump)
                q[t, :6] = prev + d
            prev = q[t, :6].copy()
        return q.astype(np.float32)

    def _safety_check(
        self,
        qpos_hd: np.ndarray,
        q_current: np.ndarray,
        stats,
    ) -> Optional[str]:
        """Return reason string if chunk is unsafe; None if ok."""
        q0_deg = self._joint_delta_deg(qpos_hd[0, :6], q_current)
        if q0_deg > self.max_q0_delta_deg:
            return f"q0_delta={q0_deg:.1f}°>{self.max_q0_delta_deg:.1f}°"
        if qpos_hd.shape[0] >= 2:
            dq = np.rad2deg(
                np.abs(
                    (np.diff(qpos_hd[:, :6], axis=0) + np.pi) % (2 * np.pi) - np.pi
                )
            )
            max_dq = float(np.max(dq))
            if max_dq > self.max_chunk_dq_deg:
                return f"chunk_dq={max_dq:.1f}°>{self.max_chunk_dq_deg:.1f}°"
        max_fk = float(getattr(stats, "max_fk_err", 0.0) or 0.0)
        if max_fk > self.max_fk_err:
            return f"fk_err={max_fk:.4f}>{self.max_fk_err:.4f}"
        fails = int(getattr(stats, "fails", 0) or 0)
        if fails > self.max_ik_fails:
            return f"ik_fails={fails}>{self.max_ik_fails}"
        return None

    def _maybe_convert_ee_chunk(self, chunk):
        ctrl_space, ctrl_type = self._chunk_ctrl(chunk)
        cs = str(ctrl_space).lower() if ctrl_space is not None else ""
        ct = str(ctrl_type).lower() if ctrl_type is not None else ""
        if cs == "joint" and ct == "abs":
            return chunk
        if not (cs == "ee" and ct == "abs"):
            logger.warning(
                "[EESequenceToQpos] ctrl_space={!r} ctrl_type={!r} — forcing EE→qpos",
                ctrl_space,
                ctrl_type,
            )

        from deploy.robot.alicia_d.sequence_ik import solve_sequence
        from deploy.robot.alicia_d.robot import (
            _canonicalize_yaw,
            _transform_matrix_from_xyz_rpy,
            _xyz_rpy_from_transform_matrix,
        )

        ee = self._extract_ee_matrix(chunk)
        # Align predicted yaw to current FK branch and unwrap within-chunk
        # (prevents ±π flips from spinning sequence IK).
        # Prefer last commanded joints: Alicia encoder often sticks near home.
        q_for_yaw = (
            self._last_q_end.copy()
            if self._last_q_end is not None
            else self._read_q_current()
        )
        self._ensure_solver()
        yaw_ref = float(
            _xyz_rpy_from_transform_matrix(self._solver.compute_fk(q_for_yaw[:6]))[5]
        )
        ee = ee.copy()
        if self._last_ee_row is not None and ee.shape[1] >= 6:
            prev_y = float(self._last_ee_row[5])
            cur = _canonicalize_yaw(float(ee[0, 5]))
            d = (cur - prev_y + np.pi) % (2 * np.pi) - np.pi
            ee[0, 5] = prev_y + d
        if ee.shape[1] >= 6:
            ee[0, 5] = _canonicalize_yaw(float(ee[0, 5]))
            d0 = (float(ee[0, 5]) - yaw_ref + np.pi) % (2 * np.pi) - np.pi
            ee[0, 5] = yaw_ref + d0
            for t in range(1, ee.shape[0]):
                prev = float(ee[t - 1, 5])
                cur = _canonicalize_yaw(float(ee[t, 5]))
                d = (cur - prev + np.pi) % (2 * np.pi) - np.pi
                ee[t, 5] = prev + d

        ee, n_denoised = self._denoise_ee_chunk(ee, ee_prev=self._last_ee_row)
        if n_denoised > 0:
            self._log_debug(
                "[EESequenceToQpos] EE denoise suppressed {}/{} frames",
                n_denoised,
                ee.shape[0],
            )

        ee_fp = ee.tobytes()
        if ee_fp == self._last_ee_fp and self._last_joint_chunk is not None:
            logger.info(
                "[EESequenceToQpos] identical EE chunk — reuse joints (skip IK)"
            )
            return self._last_joint_chunk

        if ee.shape[0] > 1 and np.allclose(ee, ee[0:1], atol=1e-7):
            q_current = (
                self._last_q_end.copy()
                if self._last_q_end is not None
                else self._read_q_current()
            )
            self._ensure_solver()
            q1, stats = solve_sequence(
                self._solver,
                ee[0:1],
                q_current,
                transform_fn=_transform_matrix_from_xyz_rpy,
                polish=False,
                max_step_deg=self.max_ik_step_deg,
                log=False,
            )
            # Constant EE hold: never jump joints away from current.
            if self._joint_delta_deg(q1[0, :6], q_current) > self.max_q0_delta_deg:
                self._safety_holds += 1
                logger.warning(
                    "[EESequenceToQpos] SAFETY HOLD (constant-EE IK jump {:.1f}°) "
                    "— keep current joints (holds={})",
                    self._joint_delta_deg(q1[0, :6], q_current),
                    self._safety_holds,
                )
                g = float(ee[0, 6]) if ee.shape[1] >= 7 else float(q1[0, 6])
                joint_chunk = self._hold_joint_chunk(q_current, ee.shape[0], g)
                self._last_ee_fp = ee_fp
                self._last_ee_row = ee[-1].copy()
                self._last_joint_chunk = joint_chunk
                self._last_q_end = q_current[:6].copy()
                return joint_chunk
            qpos_hd = np.repeat(q1, ee.shape[0], axis=0)
            qpos_hd = _densify_joint_traj(
                qpos_hd,
                self.interp_factor,
                mode=self.joint_interp,
                max_accel_deg_s2=self.max_joint_accel_deg_s2,
                playback_hz=self.playback_hz,
            )
            joint_chunk = self._build_joint_chunk(qpos_hd)
            self._last_ee_fp = ee_fp
            self._last_ee_row = ee[-1].copy()
            self._last_joint_chunk = joint_chunk
            self._last_q_end = q1[0, :6].copy()
            logger.info("[EESequenceToQpos] hold chunk — single-frame IK + pad")
            return joint_chunk

        q_current = (
            self._last_q_end.copy()
            if self._last_q_end is not None
            else self._read_q_current()
        )

        # Recompute densify from this chunk length; keep floor (explicit/min duration).
        T = int(ee.shape[0])
        auto_f = 1
        if self.playback_hz is not None and self.chunk_hz > 0:
            auto_f = _interp_factor_for_hz(self.chunk_hz, self.playback_hz, chunk_T=T)
        dur_f = 1
        if self.min_chunk_duration_s > 0 and self.playback_hz is not None:
            dur_f = _interp_factor_for_min_duration(
                self.min_chunk_duration_s, self.playback_hz, chunk_T=T
            )
        self.interp_factor = max(auto_f, dur_f, self._interp_floor)

        def _solve_and_post(log_ik: bool):
            self._ensure_solver()
            t0 = time.perf_counter()
            q_raw, st = solve_sequence(
                self._solver,
                ee,
                q_current,
                transform_fn=_transform_matrix_from_xyz_rpy,
                polish=self.polish,
                max_step_deg=self.max_ik_step_deg,
                log=log_ik,
            )
            ms = (time.perf_counter() - t0) * 1000.0
            # Joint-space densify (cubic C1) + optional accel limit — never EE interp.
            q_out = _densify_joint_traj(
                q_raw,
                self.interp_factor,
                mode=self.joint_interp,
                max_accel_deg_s2=self.max_joint_accel_deg_s2,
                playback_hz=self.playback_hz,
            )
            # Joint-space denoise after densify (kills EE/IK micro-jitter at tip).
            q_out = self._smooth_joint_traj(q_out, q_anchor=q_current)
            return q_out, st, ms

        # If previous chunks were holding, start this solve already boosted.
        if self._safety_holds >= self.ik_boost_after_holds and self.ik_boost_step > 0:
            boosted = self._boosted_ik_max_iter(hold_level=self._safety_holds)
            if boosted > self._active_ik_max_iter:
                self._apply_ik_max_iter(boosted)
                logger.warning(
                    "[EESequenceToQpos] consecutive holds={} → IK max_iter={}",
                    self._safety_holds,
                    boosted,
                )

        qpos_hd, stats, ik_ms = _solve_and_post(log_ik=True)
        reason = self._safety_check(qpos_hd, q_current, stats)
        if reason is not None:
            # Soft path: only mid-chunk steps too large → clamp; hard fails → hold.
            hard = (
                reason.startswith("q0_delta")
                or reason.startswith("fk_err")
                or reason.startswith("ik_fails")
            )
            if (not hard) and self.clamp_traj and reason.startswith("chunk_dq"):
                before = self._joint_delta_deg(qpos_hd[-1, :6], q_current)
                qpos_hd = self._clamp_joint_traj(qpos_hd, q_current)
                qpos_hd = self._smooth_joint_traj(qpos_hd, q_anchor=q_current)
                after = self._joint_delta_deg(qpos_hd[-1, :6], q_current)
                logger.warning(
                    "[EESequenceToQpos] SAFETY CLAMP ({}) — limited Δq "
                    "(end_delta {:.1f}°→{:.1f}°)",
                    reason,
                    before,
                    after,
                )
            else:
                # Hard fail: boost iters and retry same EE chunk before HOLD.
                recovered = False
                hold_level = self._safety_holds + 1
                for retry in range(1, self.ik_boost_retries + 1):
                    if self.ik_boost_step <= 0:
                        break
                    new_iter = self._boosted_ik_max_iter(
                        hold_level=hold_level, retry=retry
                    )
                    if new_iter <= self._active_ik_max_iter and retry > 1:
                        break
                    self._apply_ik_max_iter(new_iter)
                    logger.warning(
                        "[EESequenceToQpos] IK retry {}/{} after ({}) with "
                        "max_iter={} (holds→{})",
                        retry,
                        self.ik_boost_retries,
                        reason,
                        new_iter,
                        hold_level,
                    )
                    qpos_hd, stats, ik_ms = _solve_and_post(log_ik=False)
                    reason = self._safety_check(qpos_hd, q_current, stats)
                    if reason is None:
                        recovered = True
                        break
                    hard = (
                        reason.startswith("q0_delta")
                        or reason.startswith("fk_err")
                        or reason.startswith("ik_fails")
                    )
                    if (not hard) and self.clamp_traj and reason.startswith("chunk_dq"):
                        before = self._joint_delta_deg(qpos_hd[-1, :6], q_current)
                        qpos_hd = self._clamp_joint_traj(qpos_hd, q_current)
                        qpos_hd = self._smooth_joint_traj(qpos_hd, q_anchor=q_current)
                        after = self._joint_delta_deg(qpos_hd[-1, :6], q_current)
                        logger.warning(
                            "[EESequenceToQpos] SAFETY CLAMP after boost ({}) — "
                            "limited Δq (end_delta {:.1f}°→{:.1f}°)",
                            reason,
                            before,
                            after,
                        )
                        recovered = True
                        break
                if not recovered:
                    self._safety_holds = hold_level
                    # Keep elevated iters for subsequent chunks while holding.
                    kept = self._boosted_ik_max_iter(hold_level=self._safety_holds)
                    self._apply_ik_max_iter(kept)
                    g = float(ee[0, 6]) if ee.shape[1] >= 7 else float(qpos_hd[0, 6])
                    logger.warning(
                        "[EESequenceToQpos] SAFETY HOLD ({}) — keep current joints "
                        "(holds={}, T={}, ik_max_iter={})",
                        reason,
                        self._safety_holds,
                        ee.shape[0],
                        self._active_ik_max_iter,
                    )
                    joint_chunk = self._hold_joint_chunk(q_current, len(qpos_hd), g)
                    self._last_ee_fp = ee_fp
                    self._last_ee_row = ee[-1].copy()
                    self._last_joint_chunk = joint_chunk
                    self._last_q_end = q_current[:6].copy()
                    self._convert_count += 1
                    return joint_chunk

        # Success path (first try or after boost retry).
        if self._safety_holds > 0 or self._active_ik_max_iter != self.ik_max_iter:
            logger.info(
                "[EESequenceToQpos] IK accepted after holds={} (max_iter={})",
                self._safety_holds,
                self._active_ik_max_iter,
            )
        self._safety_holds = 0
        self._reset_ik_iter_boost()
        self._convert_count += 1
        ee_span = ee.max(axis=0) - ee.min(axis=0)
        logger.info(
            "[EESequenceToQpos] converted chunk #{} → joint/abs "
            "(T={}, ik={:.0f}ms, interp={}, q0_delta_deg={:.2f}, "
            "ee_span_xyz_m={:.4f}/{:.4f}/{:.4f}, ee_span_rpy_deg={:.1f}/{:.1f}/{:.1f}, {})",
            self._convert_count,
            len(qpos_hd),
            ik_ms,
            self.interp_factor,
            self._joint_delta_deg(qpos_hd[0, :6], q_current),
            float(ee_span[0]),
            float(ee_span[1]),
            float(ee_span[2]),
            float(np.rad2deg(ee_span[3])),
            float(np.rad2deg(ee_span[4])),
            float(np.rad2deg(ee_span[5])),
            stats.as_dict(),
        )
        joint_chunk = self._build_joint_chunk(qpos_hd)
        self._last_ee_fp = ee_fp
        self._last_ee_row = ee[-1].copy()
        self._last_joint_chunk = joint_chunk
        self._last_q_end = np.asarray(qpos_hd[-1, :6], dtype=np.float64).copy()
        return joint_chunk

    def put(self, chunk, timestamp: float = None):
        if chunk is not None:
            chunk = self._maybe_convert_ee_chunk(chunk)
        return super().put(chunk, timestamp=timestamp)

    def reset(self):
        self._last_ee_fp = None
        self._last_ee_row = None
        self._last_joint_chunk = None
        self._last_q_end = None
        self._convert_count = 0
        self._safety_holds = 0
        self._reset_ik_iter_boost()
        return super().reset()


class EESequenceToQposAsyncManager(EESequenceToQposManager):
    """
    EE sequence IK + async chunk-ratio prefetch.

    Unlike :class:`EESequenceToQposManager` (strict sync), triggers the next
    inference when ``current_step / chunk_len >= chunk_ratio`` so IK+infer can
    overlap with the tail of the current chunk playback.

    Extra smoothness / gap fixes:
    - ``chunk_ratio``: fraction of the *post-interp* playback buffer
      (``len(_chunk_buffer)`` after IK densify + stitch), e.g. 0.7 of 31 steps
      → trigger at step 22. Early ratios / ``prefetch_on_put`` stale the obs and
      cause retract/oscillation.
    - ``skip_prefix_steps``: drop this many *original* policy frames from each
      new chunk before playback (``-1`` = auto from ``chunk_ratio``, e.g. 0.7→4).
      Avoids retract when late-prefetched plans restart near an older pose.
    - ``stitch_steps``: after skip, min-jerk joint blend from current pose →
      first kept frame (length ``stitch_steps * interp_factor`` densified steps).
    - ``hold_on_underrun``: keep publishing the last joint cmd while waiting
    - ``prefetch_on_put``: optional immediate re-infer on put (off by default)
    - ``min_chunk_duration_s``: joint densify floor (slower / stabler / RTT headroom)
    """

    def __init__(
        self,
        chunk_ratio: float = 0.7,
        skip_prefix_steps: int = 8,
        stitch_steps: int = 4,
        hold_on_underrun: bool = True,
        prefetch_on_put: bool = False,
        # Stretch joint densify (~2s/chunk @ 30Hz → interp≈4). Set 0 for real-time.
        min_chunk_duration_s: float = 2.0,
        max_align_trim_frac: float = 0.25,
        # Re-trigger only if an in-flight infer has been silent this long (remote hang).
        infer_retry_s: float = 5.0,
        **kwargs,
    ):
        # Default playback to dataset rate when caller did not set it.
        kwargs.setdefault("playback_hz", kwargs.get("chunk_hz", 30.0))
        kwargs.setdefault("min_chunk_duration_s", min_chunk_duration_s)
        super().__init__(**kwargs)
        if not (0.0 < chunk_ratio <= 1.0):
            raise ValueError(f"chunk_ratio must be in (0, 1], got {chunk_ratio!r}")
        self.chunk_ratio = float(chunk_ratio)
        self.skip_prefix_steps = int(skip_prefix_steps)
        self.stitch_steps = max(0, int(stitch_steps))
        self.hold_on_underrun = bool(hold_on_underrun)
        self.prefetch_on_put = bool(prefetch_on_put)
        self.max_align_trim_frac = float(np.clip(max_align_trim_frac, 0.0, 0.95))
        self.infer_retry_s = max(0.5, float(infer_retry_s))
        self._ratio_prefetch_sent = False
        self._prefetch_now = False
        self._underrun_trigger_sent = False
        self._infer_inflight = False
        self._infer_inflight_t0 = 0.0
        self._last_emitted_q: Optional[np.ndarray] = None
        self._last_emitted_step = None
        self._underrun_holds = 0
        pr = float(self.playback_hz or 30.0)
        t16 = (16 - 1) * self.interp_factor + 1
        rem_s = (1.0 - self.chunk_ratio) * t16 / pr
        skip_auto = self._auto_skip_prefix_orig(policy_T=16)
        logger.info(
            "[EESequenceToQposAsync] prefetch@{:.0f}% of *post-interp* buffer "
            "(skip_prefix={}{} stitch={} on_put={} hold={} interp={} "
            "T16→{}steps≈{:.2f}s @ {:.0f}Hz, remain≈{:.2f}s, infer_retry={:.1f}s)",
            self.chunk_ratio * 100.0,
            skip_auto if self.skip_prefix_steps < 0 else self.skip_prefix_steps,
            "(auto)" if self.skip_prefix_steps < 0 else "",
            self.stitch_steps,
            self.prefetch_on_put,
            self.hold_on_underrun,
            self.interp_factor,
            t16,
            t16 / pr,
            pr,
            rem_s,
            self.infer_retry_s,
        )

    def _mark_infer_sent(self) -> None:
        self._infer_inflight = True
        self._infer_inflight_t0 = time.perf_counter()

    def _clear_infer_inflight(self) -> None:
        self._infer_inflight = False
        self._infer_inflight_t0 = 0.0

    def _can_send_infer(self) -> bool:
        """Single-flight: at most one remote infer; retry only after infer_retry_s."""
        if not self._infer_inflight:
            return True
        return (time.perf_counter() - self._infer_inflight_t0) >= self.infer_retry_s

    @staticmethod
    def _step_to_q(step) -> Optional[np.ndarray]:
        if isinstance(step, np.ndarray) and step.dtype == object and len(step) > 0:
            d0 = step[0]
            if isinstance(d0, dict) and d0.get("action") is not None:
                return np.asarray(d0["action"], dtype=np.float64).reshape(-1)
        if isinstance(step, dict) and step.get("action") is not None:
            return np.asarray(step["action"], dtype=np.float64).reshape(-1)
        return None

    def _auto_skip_prefix_orig(self, policy_T: int = 16) -> int:
        """Original-frame skip count from chunk_ratio (fallback when skip_prefix_steps<0)."""
        T = max(4, int(policy_T))
        # Prefer ~half of early horizon; cap so enough tail remains.
        n = int(round(self.chunk_ratio * 0.5 * T))
        return int(np.clip(n, 2, max(2, T // 2)))

    def _skip_chunk_prefix(self, chunk: list) -> list:
        """
        Drop the first N *original* policy frames of a densified chunk.

        At chunk_ratio≈0.7 the new plan's early frames often pull back toward an
        older pose; skipping them then blending into the kept head removes retract.
        """
        if not chunk or self._last_emitted_q is None:
            return chunk
        f = max(1, int(self.interp_factor))
        # Densified length L ≈ (T-1)*f+1  →  T ≈ (L-1)/f + 1
        T_est = max(2, int(round((len(chunk) - 1) / float(f) + 1.0)))
        if self.skip_prefix_steps < 0:
            n_orig = self._auto_skip_prefix_orig(policy_T=T_est)
        else:
            n_orig = max(0, int(self.skip_prefix_steps))
        if n_orig <= 0:
            return chunk
        skip_d = n_orig * f
        # Keep at least 2 densified frames for blend + motion.
        skip_d = min(skip_d, max(0, len(chunk) - 2))
        if skip_d <= 0:
            return chunk
        kept = chunk[skip_d:]
        logger.info(
            "[EESequenceToQposAsync] skip_prefix n_orig={}×interp={} → "
            "drop densified[:{}] len {}→{} (ratio={:.2f})",
            n_orig,
            f,
            skip_d,
            len(chunk),
            len(kept),
            self.chunk_ratio,
        )
        return kept

    def _stitch_blend(self, chunk: list) -> list:
        """
        After prefix skip: min-jerk joint blend from current pose → chunk[0],
        then play ``chunk[1:]``. Blend length = ``stitch_steps * interp_factor``.
        """
        n_orig = int(self.stitch_steps)
        if n_orig <= 0 or not chunk or self._last_emitted_q is None:
            return chunk
        f = max(1, int(self.interp_factor))
        n = n_orig * f
        q_cur = np.asarray(self._last_emitted_q, dtype=np.float64).reshape(-1)
        q_tgt = self._step_to_q(chunk[0])
        if q_tgt is None or q_tgt.size < 6:
            return chunk
        gap = self._joint_delta_deg(q_tgt[:6], q_cur[:6])
        if gap < 0.3:
            return chunk
        if gap < 2.0:
            n = min(n, max(1, f))
        morph_q = _lerp_joint_rows(q_cur, q_tgt, n)  # min-jerk, lands on q_tgt
        morph = self._build_joint_chunk(morph_q)
        rest = chunk[1:]
        logger.info(
            "[EESequenceToQposAsync] stitch blend n_orig={}×interp={} → "
            "land on kept[0] gap={:.2f}° len {}→{}",
            n_orig,
            f,
            gap,
            len(chunk),
            len(morph) + len(rest),
        )
        return morph + rest

    def select_action(self):
        # Skip SyncChunkManager (sync-only); use BasicActionManager async loop
        # with underrun hold so --pr stays steady across chunk gaps.
        if self._inference_ctx is None:
            raise RuntimeError("InferenceContext not set. Call set_inference_context() first.")

        mact_list = self._inference_ctx.poll_action_chunks()
        if mact_list:
            self._stats["chunks_received"] += 1
            self.put(mact_list, timestamp=time.perf_counter())

        buffer_empty = self.is_empty()
        need_infer = self.should_infer()
        # Single-flight remote infer. Re-trigger only if silent > infer_retry_s
        # (avoids stacking requests after tunnel drop → late-run death spiral).
        if need_infer and self._can_send_infer():
            if self._infer_inflight:
                logger.warning(
                    "[EESequenceToQposAsync] infer retry after {:.1f}s silence "
                    "(prefetch)",
                    time.perf_counter() - self._infer_inflight_t0,
                )
            with self._lock:
                blen = (
                    len(self._chunk_buffer) if self._chunk_buffer is not None else 0
                )
                step = self.current_step
            logger.info(
                "[EESequenceToQposAsync] prefetch @ step {}/{} "
                "(post-interp, ratio={:.2f})",
                step,
                blen,
                self.chunk_ratio,
            )
            self._inference_ctx.send_trigger(self.t)
            self._stats["triggers_sent"] += 1
            self._mark_infer_sent()
            self._ratio_prefetch_sent = True
            self._prefetch_now = False
        elif buffer_empty and self._can_send_infer():
            if self._infer_inflight:
                logger.warning(
                    "[EESequenceToQposAsync] infer retry after {:.1f}s silence "
                    "(underrun)",
                    time.perf_counter() - self._infer_inflight_t0,
                )
            self._inference_ctx.send_trigger(self.t)
            self._stats["triggers_sent"] += 1
            self._mark_infer_sent()
            self._underrun_trigger_sent = True

        if buffer_empty:
            # Soft underrun: keep last joint cmd (no stall) while next chunk loads.
            if (
                self.hold_on_underrun
                and self._last_emitted_step is not None
            ):
                # Opportunistically drain a just-arrived chunk without blocking.
                mact_list = self._inference_ctx.poll_action_chunks()
                if mact_list:
                    self._stats["chunks_received"] += 1
                    self.put(mact_list, timestamp=time.perf_counter())
                if self.is_empty():
                    self._underrun_holds += 1
                    if self._underrun_holds == 1 or self._underrun_holds % 30 == 0:
                        pr = float(self.playback_hz or 30.0)
                        with self._lock:
                            blen = (
                                len(self._chunk_buffer)
                                if self._chunk_buffer is not None
                                else 0
                            )
                        # remain uses configured ratio × typical densified length
                        rem_s = (1.0 - self.chunk_ratio) * max(
                            blen, (16 - 1) * self.interp_factor + 1
                        ) / pr
                        aged = (
                            time.perf_counter() - self._infer_inflight_t0
                            if self._infer_inflight
                            else -1.0
                        )
                        logger.warning(
                            "[EESequenceToQposAsync] underrun hold x{} — "
                            "waiting for chunk (ratio={:.2f} remain≈{:.2f}s, "
                            "inflight_age={:.1f}s); check SSH tunnel / remote",
                            self._underrun_holds,
                            self.chunk_ratio,
                            rem_s,
                            aged,
                        )
                    self.t += 1
                    return self._last_emitted_step
            self._wait_for_action_chunk(timeout=3600.0)

        action = self.get()
        if action is None:
            raise RuntimeError("ActionManager returned None - no action available")
        q = self._step_to_q(action)
        if q is not None:
            self._last_emitted_q = q.copy()
        self._last_emitted_step = action
        self._underrun_holds = 0
        self._underrun_trigger_sent = False
        self.t += 1
        return action

    def put(self, chunk, timestamp: float = None):
        if chunk is not None:
            chunk = self._maybe_convert_ee_chunk(chunk)
        # IMPORTANT: skip SyncChunkManager.put (it *discards* mid-play arrivals).
        # Async must replace via BasicActionManager.put.
        _store = super(SyncChunkManager, self).put

        if chunk is None or len(chunk) == 0:
            _store(chunk, timestamp=timestamp)
            with self._lock:
                self._ratio_prefetch_sent = False
            return

        was_underrun = self.is_empty() or self._underrun_holds > 0
        if was_underrun:
            logger.info(
                "[EESequenceToQposAsync] post-underrun put (T={})",
                len(chunk),
            )

        # 1) Drop stale open-loop prefix (ratio-scaled) to avoid retract.
        # 2) Min-jerk joint blend from current q → first kept frame.
        # Align-trim is skipped when prefix-skip is enabled (would double-cut).
        use_prefix_skip = self.skip_prefix_steps != 0
        aligned = chunk
        if use_prefix_skip:
            aligned = self._skip_chunk_prefix(aligned)
        elif not was_underrun:
            aligned, start_j, best_d = AsyncChunkRatioManager._align_chunk_to_reference(
                self, chunk
            )
            max_trim = int(len(chunk) * self.max_align_trim_frac)
            if start_j > max_trim:
                aligned, start_j, best_d = chunk, 0, None
            elif best_d is not None and (start_j > 0 or len(aligned) != len(chunk)):
                self._log_debug(
                    "async chunk align: start_j={}/{} best_dist={:.6f} len {} -> {}",
                    start_j,
                    len(chunk),
                    best_d,
                    len(chunk),
                    len(aligned),
                )

        aligned = self._stitch_blend(aligned)
        _store(aligned, timestamp=timestamp)
        with self._lock:
            self._ratio_prefetch_sent = False
            self._underrun_trigger_sent = False
            self._clear_infer_inflight()
            # Fire next infer ASAP so remote RTT overlaps the whole chunk.
            if self.prefetch_on_put and len(aligned) > 0:
                self._prefetch_now = True

    def should_infer(self) -> bool:
        """True when we want a prefetch; select_action enforces single-flight send."""
        with self._lock:
            if self._prefetch_now and not self._ratio_prefetch_sent:
                return True
            if self._chunk_buffer is None:
                return False
            chunk_len = len(self._chunk_buffer)
            if chunk_len == 0:
                return False
            if self._ratio_prefetch_sent:
                return False
            return (self.current_step / chunk_len) >= self.chunk_ratio

    def reset(self):
        with self._lock:
            self._ratio_prefetch_sent = False
            self._prefetch_now = False
            self._underrun_trigger_sent = False
            self._clear_infer_inflight()
        self._last_emitted_q = None
        self._last_emitted_step = None
        self._underrun_holds = 0
        return super().reset()
