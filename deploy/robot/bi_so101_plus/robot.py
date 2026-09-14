"""
Bimanual SO101 Plus — two independent So101Plus arms + Quest3 dual rel_ee.

Action layout (14D, same as Quest3 arm_mode=dual / Bessica):
  [L_xyz(3), L_rpy(3), L_grip, R_xyz(3), R_rpy(3), R_grip]

Each hand uses its own squeeze flag (left_unsqueeze_active / right_unsqueeze_active).
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from deploy.robot.base import BaseRobot
from deploy.robot.so101_plus.robot import So101Plus, N_JOINTS
from deploy.utils import RateLimiter


class BiSo101Plus(BaseRobot):
    """Compose left + right ``So101Plus`` for 14D dual VR teleop."""

    ARM_CLS = So101Plus

    CONTROL_MODES = ("qpos", "delta_ee", "rel_ee")

    def __init__(
        self,
        name: str = "bi_so101_plus",
        max_size_mb: int = 64,
        fps: float = 100.0,
        control_shm_name: Optional[str] = None,
        left_arm_port: str = "/dev/ttyACM0",
        right_arm_port: str = "/dev/ttyACM1",
        left_arm_id: str = "so101_plus_left",
        right_arm_id: str = "so101_plus_right",
        control_mode: str = "rel_ee",
        # Shared So101Plus kwargs (IK / teleop feel) — forwarded to both arms.
        urdf_path: Optional[str] = None,
        ik_end_frame: str = "gripper_body",
        ik_ee_offset_xyz: Optional[List[float]] = None,
        ik_ee_offset_rpy: Optional[List[float]] = None,
        ik_eps: float = 0.01,
        ik_max_iter: int = 60,
        ik_max_solve_time_s: float = 0.05,
        ik_rotation_weight: float = 0.3,
        ik_position_only: bool = False,
        ik_joint_reg_weights: Optional[List[float]] = None,
        ik_regularize_to_warmstart: bool = True,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_scale: float = 1.0,
        gripper_absolute: bool = True,
        max_joint_delta_deg: float = 30.0,
        clip_joint_delta: bool = True,
        joint_smooth_alpha: float = 0.15,
        joint_stop_hold: bool = True,
        joint_still_deg: float = 0.12,
        joint_still_frames: int = 5,
        joint_move_deg: float = 0.40,
        joint_teleop_brake_deg: float = 2.0,
        joint_teleop_brake_speed_deg_s: float = 80.0,
        joint_signs: Optional[List[int]] = None,
        home_duration_s: float = 1.5,
        home_step_sleep: float = 0.03,
        left_home_qpos: Optional[List[float]] = None,
        right_home_qpos: Optional[List[float]] = None,
        debug: bool = False,
        gripper_trace: bool = False,
        diag_log: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)

        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Invalid control_mode '{control_mode}'. Must be one of {self.CONTROL_MODES}")
        self.control_mode = control_mode
        self.gripper_absolute = bool(gripper_absolute)
        self.gripper_scale = float(gripper_scale)
        self.debug = bool(debug)
        self.home_duration_s = max(0.5, float(home_duration_s))
        self.home_step_sleep = max(0.01, float(home_step_sleep))

        shared = dict(
            max_size_mb=max_size_mb,
            fps=fps,
            control_shm_name=None,  # Bi owns SHM; arms never read teleop SHM
            control_mode=control_mode,
            urdf_path=urdf_path,
            ik_end_frame=ik_end_frame,
            ik_ee_offset_xyz=ik_ee_offset_xyz,
            ik_ee_offset_rpy=ik_ee_offset_rpy,
            ik_eps=ik_eps,
            ik_max_iter=ik_max_iter,
            ik_max_solve_time_s=ik_max_solve_time_s,
            ik_rotation_weight=ik_rotation_weight,
            ik_position_only=ik_position_only,
            ik_joint_reg_weights=ik_joint_reg_weights,
            ik_regularize_to_warmstart=ik_regularize_to_warmstart,
            position_scale=position_scale,
            rotation_scale=rotation_scale,
            gripper_scale=gripper_scale,
            gripper_absolute=gripper_absolute,
            max_joint_delta_deg=max_joint_delta_deg,
            clip_joint_delta=clip_joint_delta,
            joint_smooth_alpha=joint_smooth_alpha,
            joint_stop_hold=joint_stop_hold,
            joint_still_deg=joint_still_deg,
            joint_still_frames=joint_still_frames,
            joint_move_deg=joint_move_deg,
            joint_teleop_brake_deg=joint_teleop_brake_deg,
            joint_teleop_brake_speed_deg_s=joint_teleop_brake_speed_deg_s,
            joint_signs=joint_signs,
            home_duration_s=home_duration_s,
            home_step_sleep=home_step_sleep,
            debug=debug,
            gripper_trace=gripper_trace,
            diag_log=diag_log,
        )
        # Drop unused YAML extras (cameras etc.)
        _ = kwargs

        self.left = self.ARM_CLS(
            name=f"{name}_left",
            com=left_arm_port,
            robot_id=left_arm_id,
            home_qpos=left_home_qpos,
            **shared,
        )
        self.right = self.ARM_CLS(
            name=f"{name}_right",
            com=right_arm_port,
            robot_id=right_arm_id,
            home_qpos=right_home_qpos,
            **shared,
        )

        self._prev_l_unsq = False
        self._prev_r_unsq = False
        self._last_left_cmd: Optional[np.ndarray] = None
        self._last_right_cmd: Optional[np.ndarray] = None

        logger.info(
            f"[BiSo101Plus] mode={control_mode} left={left_arm_port}({left_arm_id}) "
            f"right={right_arm_port}({right_arm_id})"
        )

    # ---------------------------------------------------------------- lifecycle
    def connect(self) -> bool:
        ok_l = self.left.connect()
        ok_r = self.right.connect()
        if not (ok_l and ok_r):
            logger.error(f"[BiSo101Plus] connect failed: left={ok_l} right={ok_r}")
            return False
        return True

    def shutdown(self):
        try:
            self.left.shutdown()
        except Exception:
            pass
        try:
            self.right.shutdown()
        except Exception:
            pass

    def close(self):
        super().close()
        self.shutdown()

    def is_running(self) -> bool:
        return bool(self.left.is_running() and self.right.is_running())

    def get_action_dim(self) -> int:
        if self.control_mode == "qpos":
            return 2 * N_JOINTS  # 14
        return 14

    # ------------------------------------------------------------- observation
    def get_observation(self) -> Optional[Dict[str, Any]]:
        try:
            lo = self.left.get_observation()
            ro = self.right.get_observation()
            if lo is None or ro is None:
                return None
            lq = np.asarray(lo["qpos"], dtype=np.float64).reshape(-1)
            rq = np.asarray(ro["qpos"], dtype=np.float64).reshape(-1)
            return {
                "qpos": np.concatenate([lq, rq]),
                "left_qpos": lq,
                "right_qpos": rq,
                "left": lo,
                "right": ro,
            }
        except Exception as e:
            logger.error(f"[BiSo101Plus] get_observation error: {e}")
            traceback.print_exc()
            return None

    def set_home(self):
        """Slow joint interpolate both arms to each arm's captured home_qpos."""
        # Ensure homes exist (captured at connect if not set in YAML).
        if self.left._home_qpos is None:
            self.left.capture_home_from_current()
        if self.right._home_qpos is None:
            self.right.capture_home_from_current()

        try:
            l_obs = self.left.get_observation()
            r_obs = self.right.get_observation()
            l_start = (
                np.asarray(l_obs["qpos"], dtype=np.float64).reshape(-1)[:N_JOINTS].copy()
                if l_obs is not None and l_obs.get("qpos") is not None
                else self.left._home_qpos.copy()
            )
            r_start = (
                np.asarray(r_obs["qpos"], dtype=np.float64).reshape(-1)[:N_JOINTS].copy()
                if r_obs is not None and r_obs.get("qpos") is not None
                else self.right._home_qpos.copy()
            )
        except Exception:
            l_start = self.left._home_qpos.copy()
            r_start = self.right._home_qpos.copy()

        l_home = self.left._home_qpos.copy()
        r_home = self.right._home_qpos.copy()
        n_steps = max(1, int(np.ceil(self.home_duration_s / self.home_step_sleep)))
        logger.info(
            f"[BiSo101Plus] go_home interpolate {n_steps} steps "
            f"(~{self.home_duration_s:.1f}s, both arms → captured home)"
        )
        for alpha in np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)[1:]:
            l_t = l_start + (l_home - l_start) * alpha
            r_t = r_start + (r_home - r_start) * alpha
            self.publish_action(np.concatenate([l_t, r_t]))
            time.sleep(self.home_step_sleep)

        self.publish_action(np.concatenate([l_home, r_home]))
        self.left._sync_state_to_qpos(l_home)
        self.right._sync_state_to_qpos(r_home)

        self._prev_l_unsq = False
        self._prev_r_unsq = False
        self.left._rel_ee_anchor_pose = None
        self.right._rel_ee_anchor_pose = None
        self.left._rel_ee_vr_unsqueeze_active = False
        self.right._rel_ee_vr_unsqueeze_active = False
        time.sleep(min(0.3, self.home_step_sleep * 4))
        logger.info("[BiSo101Plus] go_home finished")

    def publish_action(self, action: np.ndarray) -> None:
        """Publish 14D normalized joint cmd: [left7, right7]."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if len(a) != 2 * N_JOINTS:
            logger.warning(f"[BiSo101Plus] publish expects {2 * N_JOINTS}D, got {len(a)}")
            return
        self.left.publish_action(a[:N_JOINTS])
        self.right.publish_action(a[N_JOINTS:])
        self._last_left_cmd = a[:N_JOINTS].copy()
        self._last_right_cmd = a[N_JOINTS:].copy()

    # ---------------------------------------------------------- process_action
    def _split_arm_dict(
        self,
        action_dict: dict,
        arm_7: np.ndarray,
        unsqueeze: bool,
        anchor_just_set: bool,
    ) -> dict:
        return {
            "action": np.asarray(arm_7, dtype=np.float64).reshape(-1),
            "unsqueeze_active": bool(unsqueeze),
            "anchor_just_set": bool(anchor_just_set),
            "go_home": False,
        }

    def process_action(self, action_dict: dict) -> dict:
        if self.control_mode == "qpos":
            return action_dict

        raw = action_dict.get("action", None)
        if raw is None:
            return action_dict
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        if len(raw) != 14:
            logger.warning(f"[BiSo101Plus] {self.control_mode} expects 14D, got {len(raw)}D")
            return action_dict

        l_unsq = bool(
            action_dict.get("left_unsqueeze_active", action_dict.get("unsqueeze_active", False))
        )
        r_unsq = bool(
            action_dict.get("right_unsqueeze_active", action_dict.get("unsqueeze_active", False))
        )
        l_edge = bool(l_unsq and not self._prev_l_unsq)
        r_edge = bool(r_unsq and not self._prev_r_unsq)
        self._prev_l_unsq = l_unsq
        self._prev_r_unsq = r_unsq

        left_in = self._split_arm_dict(action_dict, raw[:7], l_unsq, l_edge)
        right_in = self._split_arm_dict(action_dict, raw[7:14], r_unsq, r_edge)

        left_out = self.left.process_action(left_in)
        right_out = self.right.process_action(right_in)

        l_cmd = left_out.get("action", None)
        r_cmd = right_out.get("action", None)

        if self.debug:
            now = time.time()
            if not hasattr(self, "_last_diag_log_t") or (now - self._last_diag_log_t) >= 0.5:
                self._last_diag_log_t = now
                lee = float(np.linalg.norm(raw[:6]))
                ree = float(np.linalg.norm(raw[7:13]))
                logger.info(
                    f"[BiDiag] L_unsq={l_unsq} R_unsq={r_unsq} "
                    f"|Lee|={lee:.4f} |Ree|={ree:.4f} "
                    f"L_cmd={'ok' if l_cmd is not None else 'None'} "
                    f"R_cmd={'ok' if r_cmd is not None else 'None'}"
                )

        # Hold last command on a frozen / failed arm so the other can still move.
        if l_cmd is None:
            l_cmd = self._last_left_cmd
        else:
            self._last_left_cmd = np.asarray(l_cmd, dtype=np.float64).copy()
        if r_cmd is None:
            r_cmd = self._last_right_cmd
        else:
            self._last_right_cmd = np.asarray(r_cmd, dtype=np.float64).copy()

        if l_cmd is None and r_cmd is None:
            action_dict["action"] = None
            return action_dict

        # Never synthesize an all-zero joint command here: normalized zero is
        # the middle of the calibrated range and would pull an idle arm away
        # from its startup pose. If no safe hold target exists, skip the frame.
        if l_cmd is None or r_cmd is None:
            action_dict["action"] = None
            return action_dict

        action_dict["action"] = np.concatenate(
            [np.asarray(l_cmd, dtype=np.float64).reshape(-1)[:N_JOINTS],
             np.asarray(r_cmd, dtype=np.float64).reshape(-1)[:N_JOINTS]]
        )
        return action_dict

    def _apply_gripper_absolute(self, arm: So101Plus, grip_cmd: float, unsqueeze: bool) -> None:
        g_cmd = float(grip_cmd) * arm.gripper_scale
        if arm.gripper_absolute:
            if unsqueeze:
                arm._target_gripper = float(np.clip(g_cmd * 100.0, 0.0, 100.0))
        else:
            arm._target_gripper = float(np.clip(arm._target_gripper + g_cmd * 100.0, 0.0, 100.0))

    # -------------------------------------------------------------------- start
    def start(self):
        """Dual EE loop: read 14D Quest action, run per-arm So101Plus IK, publish both."""
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)

        if self.control_shm_name is not None and self.control_shm is None:
            for i in range(10):
                try:
                    self.control_shm = self.connect_to_existing_shm(self.control_shm_name)
                    logger.info(f"[BiSo101Plus] Connected to control SHM: {self.control_shm_name}")
                    break
                except ValueError as e:
                    if i < 9:
                        logger.warning(
                            f"[BiSo101Plus] Waiting for control SHM '{self.control_shm_name}'... ({i + 1}/10)"
                        )
                        time.sleep(0.5)
                    else:
                        logger.error(f"[BiSo101Plus] Failed to connect to control SHM: {e}")

        if self.control_mode == "qpos":
            return super().start()

        # Warm IK from the measured startup pose and seed both hold commands.
        # This makes the first VR frame relative to the actual arm poses.
        startup_obs = self.get_observation()
        if startup_obs is not None:
            self._last_left_cmd = np.asarray(startup_obs["left_qpos"], dtype=np.float64).copy()
            self._last_right_cmd = np.asarray(startup_obs["right_qpos"], dtype=np.float64).copy()
        _ = self.left.ik_solver
        _ = self.right.ik_solver

        self.is_running = True
        rate_limiter = RateLimiter()
        data = self.get_data()
        if data is not None:
            self.write_data(data)

        logger.info(f"[BiSo101Plus] Starting dual EE loop (mode={self.control_mode}, fps={self.fps})")

        while self.is_running:
            action = self.read_action()
            if action is not None:
                if action.get("cmd") == "reset" or action.get("go_home", False):
                    logger.info("[BiSo101Plus] Go-home command received")
                    self.set_home()
                    continue

                raw = action.get("action", None)
                l_unsq = bool(
                    action.get("left_unsqueeze_active", action.get("unsqueeze_active", False))
                )
                r_unsq = bool(
                    action.get("right_unsqueeze_active", action.get("unsqueeze_active", False))
                )

                if raw is not None and len(raw) >= 14:
                    for arm, g_idx, unsq in (
                        (self.left, 6, l_unsq),
                        (self.right, 13, r_unsq),
                    ):
                        tg_before = float(arm._target_gripper)
                        self._apply_gripper_absolute(arm, float(raw[g_idx]), unsq)
                        if abs(arm._target_gripper - tg_before) > 0.5 and arm._target_q_ik is not None:
                            q_hold = (
                                arm._smooth_q_ik
                                if arm._smooth_q_ik is not None
                                else arm._target_q_ik
                            )
                            arm.publish_action(
                                arm._ik6_gripper_to_hw7_norm(q_hold, arm._target_gripper)
                            )

                t0 = time.perf_counter()
                action = self.process_action(action)
                ik_ms = (time.perf_counter() - t0) * 1000.0
                if ik_ms > 20.0:
                    if not hasattr(self, "_last_ik_slow_log") or (time.time() - self._last_ik_slow_log) >= 1.0:
                        logger.warning(f"[BiSo101Plus] Slow dual IK: {ik_ms:.1f}ms")
                        self._last_ik_slow_log = time.time()

                out = action.get("action", None)
                if out is not None:
                    out = np.asarray(out, dtype=np.float64).copy()
                    if len(out) == 2 * N_JOINTS:
                        out[6] = self.left._target_gripper
                        out[13] = self.right._target_gripper
                    self.publish_action(out)

            data = self.get_data()
            if data is not None:
                self.write_data(data)
            rate_limiter.sleep(self.fps)

    def get_data(self):
        obs = self.get_observation()
        if obs is None:
            return None
        return {"qpos": obs["qpos"]}

    def obs2meta(self, device_data):
        qpos = device_data.get("qpos")
        if qpos is None:
            return {}
        return {"state": np.asarray(qpos, dtype=np.float32)}

    def meta2act(self, mact):
        return mact
