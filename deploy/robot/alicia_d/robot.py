"""
Alicia-D Robot for ILStudio

Integrates Alicia-D robot arm with ILStudio framework using:
- Official alicia_d_sdk for hardware communication (via pip install)
- Piper-style global IK (Pinocchio + scipy / optional casadi+ipopt)
- SharedMemory for observation/action communication

Control modes:
1. qpos: Direct joint position control (6D arm joints + 1D gripper)
2. delta_ee: 7D delta end-effector control with IK (dx, dy, dz, drx, dry, drz, dgripper)
3. rel_ee: 7D relative end-effector control with IK (relative to anchor pose)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from deploy.robot.base import BaseRobot
from benchmark.base import MetaObs
from loguru import logger


def _transform_matrix_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform matrix from xyz + rpy."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rot_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    rot_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rot_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    T = np.eye(4)
    T[:3, :3] = rot_z @ rot_y @ rot_x
    T[:3, 3] = xyz
    return T


def _canonicalize_yaw(yaw: float, near_pi_eps: float = 0.05) -> float:
    """Wrap yaw to (-π, π], mapping the +π branch cut to -π.

    Training demos store home yaw near -π; tiny FK noise can flip to +π.
    Same rotation, but zscore(±π) differs by ~2.6σ and breaks EE ACT/IK.
    Only used when converting FK matrices → RPY for state_ee (not teleop matrix IK).
    """
    y = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
    if y > np.pi - near_pi_eps:
        y -= 2.0 * np.pi
    return y


def _xyz_rpy_from_transform_matrix(T: np.ndarray) -> np.ndarray:
    """Inverse of `_transform_matrix_from_xyz_rpy` (extrinsic XYZ, R=Rz@Ry@Rx)."""
    R = np.asarray(T, dtype=np.float64)[:3, :3]
    x, y, z = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
    cy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    pitch = float(np.arctan2(-R[2, 0], cy))
    if cy > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = 0.0
    yaw = _canonicalize_yaw(yaw)
    return np.array([x, y, z, roll, pitch, yaw], dtype=np.float64)


# ============================================================================
# Alicia-D Robot Class
# ============================================================================

class AliciaD(BaseRobot):
    """
    Alicia-D Robot for ILStudio
    
    Uses alicia_d_sdk.hardware for low-level communication.
    
    Supports three control modes:
    1. "qpos" (default): Direct joint position control (7D: 6 arm + 1 gripper)
    2. "delta_ee": 7D delta EE control with global IK
    3. "rel_ee": 7D relative EE control with IK (relative to anchor pose)
    """
    
    # Control modes
    CONTROL_MODES = ("qpos", "delta_ee", "rel_ee")
    
    def __init__(
        self,
        name: str = "alicia_d",
        max_size_mb: int = 64,
        fps: float = 200.0,
        control_shm_name: Optional[str] = None,
        port: str = "",
        urdf_path: Optional[str] = None,
        control_mode: str = "qpos",
        gripper_type: str = "50mm",
        ik_end_frame: str = "link6",
        ik_ee_offset_xyz: Optional[list] = None,
        ik_ee_offset_rpy: Optional[list] = None,
        ik_ee_offset_matrix: Optional[list] = None,
        ik_eps: float = 2e-3,
        ik_max_iter: int = 100,
        # Teleop scipy IK wall-clock budget (seconds). 0/None = disabled. Exceed → hold last q.
        ik_max_solve_time_s: float = 0.0,
        ik_rotation_weight: float = 0.3,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_scale: float = 1.0,
        # If True, teleop action[6] is absolute open amount in [0,1] (1=open).
        # If False, action[6] is a per-frame delta (legacy ramp — feels lagged).
        gripper_absolute: bool = False,
        speed_deg_s: float = 30.0,
        gripper_speed_deg_s: float = 483.4,
        max_joint_delta_deg: float = 10.0,
        # EMA on teleop EE→IK joints: 1.0=off; lower=more inertia (silky but laggy).
        joint_smooth_alpha: float = 1.0,
        # Unused legacy (kept for YAML compat).
        joint_settle_deg: float = 0.25,
        # Inference / direct-qpos: skip tiny rewrites (stop hunting).
        joint_cmd_deadband_deg: float = 0.12,
        joint_creep_deg: float = 1.5,
        joint_creep_speed_deg_s: float = 45.0,
        # Teleop stop-hold: when IK target barely moves for N frames, freeze cmd
        # (kills stop jitter from VR/IK noise + re-commanding at high speed).
        # Does NOT engage while moving, so feel stays silky.
        joint_stop_hold: bool = False,
        joint_still_deg: float = 0.12,
        joint_still_frames: int = 5,
        joint_move_deg: float = 0.40,
        # While slowing (not yet held), soft-brake servo speed for small steps.
        joint_teleop_brake_deg: float = 2.0,
        joint_teleop_brake_speed_deg_s: float = 80.0,
        debug: bool = False,
        gripper_trace: bool = False,
        **kwargs
    ):
        """Initialize Alicia-D Robot. Extra YAML keys (legacy realtime IK) are ignored."""
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        
        self.port = port
        self.debug = debug
        self.gripper_trace = gripper_trace
        self.gripper_type = gripper_type
        
        # Control mode
        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Invalid control_mode '{control_mode}'. Must be one of {self.CONTROL_MODES}")
        self.control_mode = control_mode
        
        # Delta EE scaling
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.gripper_scale = gripper_scale
        self.gripper_absolute = bool(gripper_absolute)
        
        # Speed settings
        self.speed_deg_s = speed_deg_s
        self.gripper_speed_deg_s = gripper_speed_deg_s
        self.joint_creep_speed_deg_s = float(joint_creep_speed_deg_s)
        self.joint_teleop_brake_speed_deg_s = float(joint_teleop_brake_speed_deg_s)
        
        # Safety settings
        self.max_joint_delta_rad = np.deg2rad(max_joint_delta_deg)
        self.joint_smooth_alpha = float(np.clip(joint_smooth_alpha, 0.05, 1.0))
        self.joint_settle_rad = np.deg2rad(float(joint_settle_deg))
        self.joint_cmd_deadband_rad = np.deg2rad(float(joint_cmd_deadband_deg))
        self.joint_creep_rad = np.deg2rad(float(joint_creep_deg))
        self.joint_stop_hold = bool(joint_stop_hold)
        self.joint_still_rad = np.deg2rad(float(joint_still_deg))
        self.joint_still_frames = max(1, int(joint_still_frames))
        self.joint_move_rad = np.deg2rad(float(joint_move_deg))
        self.joint_teleop_brake_rad = np.deg2rad(float(joint_teleop_brake_deg))
        
        # Find URDF path
        if urdf_path is None:
            # Use local URDF file (bundled with this module)
            local_urdf = Path(__file__).parent / "urdf" / f"Alicia_D_v5_6_gripper_{gripper_type}.urdf"
            if local_urdf.exists():
                urdf_path = str(local_urdf)
            else:
                # Fallback: try synriard if available
                try:
                    from synriard import get_model_path
                    urdf_path = str(get_model_path("Alicia_D", version="v5_6", variant=f"gripper_{gripper_type}", model_format="urdf"))
                except ImportError:
                    raise FileNotFoundError(
                        f"URDF file not found at {local_urdf}. "
                        "Please provide urdf_path argument or install synriard package."
                    )
        self.urdf_path = urdf_path
        
        # Robot driver instance (initialized in connect())
        self._driver = None
        
        if ik_ee_offset_matrix is not None:
            ee_offset_matrix = np.array(ik_ee_offset_matrix, dtype=np.float64)
        else:
            ee_offset_xyz = np.array(ik_ee_offset_xyz or [0.0, 0.0, 0.0], dtype=np.float64)
            ee_offset_rpy = np.array(ik_ee_offset_rpy or [0.0, 0.0, 0.0], dtype=np.float64)
            ee_offset_matrix = _transform_matrix_from_xyz_rpy(ee_offset_xyz, ee_offset_rpy)

        self._ik_solver = None  # teleop IK (reg→0)
        # Reversible IK for inference only — NOT created during teleop/collect (avoids 2x casadi).
        self._replay_ik_solver = None
        self._sequence_ik_solver = None  # chunk EE→qpos (strong warmstart reg)
        self._ik_settings = {
            'end_frame': ik_end_frame,
            'ee_offset_matrix': ee_offset_matrix,
            'eps': ik_eps,
            'max_iter': ik_max_iter,
            'rotation_weight': ik_rotation_weight,
            'max_solve_time_s': float(ik_max_solve_time_s) if ik_max_solve_time_s else None,
        }
        
        # State tracking
        self._current_joint_angles: Optional[np.ndarray] = None
        self._current_gripper: Optional[float] = None
        self._target_qpos: Optional[np.ndarray] = None  # last IK / logical joint target
        self._smooth_qpos: Optional[np.ndarray] = None  # last teleop-smoothed / logical cmd
        self._last_sent_qpos: Optional[np.ndarray] = None  # last joints written to driver
        self._last_sent_gripper: Optional[int] = None
        self._target_gripper: float = 500.0
        self._prev_ik_qpos: Optional[np.ndarray] = None
        self._still_count: int = 0
        self._teleop_holding_still: bool = False
        self._lock = threading.RLock()  # Use RLock to allow reentrant acquisition (process_action -> refresh_anchor)
        
        # rel_ee mode: anchor pose tracking
        self._rel_ee_anchor_pose: Optional[np.ndarray] = None
        self._rel_ee_anchor_gripper: float = 500.0
        self._rel_ee_vr_unsqueeze_active: bool = False
        self._rel_ee_anchor_initialized: bool = False
        
        # rel_ee mode: anchor offset for dynamic re-anchoring on IK failure.
        # When IK fails, the VR-side relative pose that caused the failure is absorbed
        # into this offset. Subsequent VR relative poses are corrected by subtracting
        # the offset, so the robot effectively re-anchors to its last reachable pose
        # without requiring the user to release and re-squeeze.
        self._rel_ee_pos_offset: np.ndarray = np.zeros(3)   # position offset (robot frame, scaled)
        self._rel_ee_rot_offset: np.ndarray = np.zeros(3)   # euler offset (robot frame, scaled)
        
        # Connection state
        self._connected = False
        self.is_running = False
        
        logger.info(f"[AliciaD] Initialized with control_mode={control_mode}, fps={fps}")
        if control_mode in ("delta_ee", "rel_ee"):
            logger.info(f"[AliciaD] Delta EE scales: pos={position_scale}, rot={rotation_scale}, gripper={gripper_scale}")
            logger.info(
                f"[AliciaD] IK (piper_global): end_frame={ik_end_frame}, "
                f"eps={ik_eps}, max_iter={ik_max_iter}, rotation_weight={ik_rotation_weight}"
            )
            if self.joint_smooth_alpha < 1.0 - 1e-9:
                logger.info(
                    f"[AliciaD] Joint command EMA enabled: alpha={self.joint_smooth_alpha} "
                    f"(lower=smoother/more inertia; 1.0=off)"
                )
            if self.joint_stop_hold:
                logger.info(
                    f"[AliciaD] Teleop stop-hold ON: still<{np.rad2deg(self.joint_still_rad):.2f}° "
                    f"x{self.joint_still_frames} → hold; resume>{np.rad2deg(self.joint_move_rad):.2f}°; "
                    f"brake<{np.rad2deg(self.joint_teleop_brake_rad):.1f}°@{self.joint_teleop_brake_speed_deg_s:.0f}deg/s"
                )
        logger.info(
            f"[AliciaD] Stop-hunt guards: "
            f"deadband={np.rad2deg(self.joint_cmd_deadband_rad):.2f}° "
            f"creep<{np.rad2deg(self.joint_creep_rad):.1f}°@{self.joint_creep_speed_deg_s:.0f}deg/s"
        )
    
    def _make_ik_solver(self, *, reversible: bool, regularization_weight: Optional[float] = None):
        from deploy.robot.alicia_d.piper_global_ik import PiperStyleGlobalIKSolver

        s = self._ik_settings
        # Sequence IK uses a stronger warmstart pull to stay on one redundancy branch.
        reg_w = 0.01 if regularization_weight is None else float(regularization_weight)
        return PiperStyleGlobalIKSolver(
            urdf_path=self.urdf_path,
            end_frame=s['end_frame'],
            ee_offset_matrix=s['ee_offset_matrix'],
            position_weight=1.0,
            rotation_weight=max(float(s['rotation_weight']), 1e-6),
            total_cost_scale=20.0,
            regularization_weight=reg_w,
            max_iterations=s['max_iter'],
            tol=max(float(s['eps']), 1e-6),
            # Teleop: allow jump-reset. Reversible/inference: never reset warm-start to zero.
            jump_reset_threshold_deg=(
                1e9 if reversible else np.rad2deg(self.max_joint_delta_rad)
            ),
            regularize_to_warmstart=reversible,
            exact_when_converged=reversible,
            # Time budget only for teleop (keeps control loop from hitching).
            max_solve_time_s=(None if reversible else s.get('max_solve_time_s')),
        )

    @property
    def ik_solver(self):
        """Teleop IK (original Piper-style: regularize toward zero)."""
        if self._ik_solver is None:
            s = self._ik_settings
            self._ik_solver = self._make_ik_solver(reversible=False)
            logger.info(
                f"[AliciaD] Teleop IK initialized (piper_global): "
                f"eps={s['eps']}, max_iter={s['max_iter']}, end_frame={s['end_frame']}, "
                f"rotation_weight={s['rotation_weight']}, "
                f"max_solve_time_s={s.get('max_solve_time_s')}"
            )
        return self._ik_solver

    @property
    def replay_ik_solver(self):
        """Reversible IK for inference EE control (lazy; not used in teleop loop)."""
        if self._replay_ik_solver is None:
            s = self._ik_settings
            self._replay_ik_solver = self._make_ik_solver(reversible=True)
            logger.info(
                f"[AliciaD] Replay/inference IK initialized (reversible): "
                f"eps={s['eps']}, max_iter={s['max_iter']}, end_frame={s['end_frame']}"
            )
        return self._replay_ik_solver

    def _compute_fk_T(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        FK for recorded state_ee.

        Uses teleop ik_solver's Pinocchio model (same URDF / ee frame as reversible IK).
        Do NOT touch replay_ik_solver here — that would double-load casadi during collect.
        """
        return self.ik_solver.compute_fk(joint_angles)
    
    def connect(self) -> bool:
        """Connect to robot using low-level hardware SDK."""
        if self._connected:
            return True
        
        try:
            # Import low-level hardware SDK
            import sys
            # Ensure sdk path is in sys.path
            sdk_path = os.path.join(os.path.dirname(__file__))
            if sdk_path not in sys.path:
                sys.path.insert(0, sdk_path)
                
            from alicia_d_sdk.hardware import ServoDriver
            
            # Create driver instance
            logger.info(f"[AliciaD] Connecting via low-level SDK on port '{self.port}'...")
            self._driver = ServoDriver(
                port=self.port,
                debug_mode=self.debug
            )
            
            # Connect
            if not self._driver.connect():
                logger.error("[AliciaD] Failed to connect to robot hardware")
                return False
            self.set_home(speed_deg_s=self.speed_deg_s)
            # Wait for valid state
            logger.info("[AliciaD] Waiting for valid state...")
            if not self._driver.wait_for_valid_state(timeout=2.0):
                 logger.warning("[AliciaD] Initial state read timeout, but continuing...")

            # Get initial state
            # Try to acquire info explicitly to ensure we have data
            self._driver.acquire_info("joint_gripper", wait=True, timeout=1.0)
            state = self._driver.data_parser.get_joint_state()
            
            if state is None or state.angles is None:
                logger.warning("[AliciaD] Initial state read failed")
            else:
                self._current_joint_angles = np.array(state.angles, dtype=np.float64)
                self._current_gripper = state.gripper
                self._target_qpos = self._current_joint_angles.copy()
                self._smooth_qpos = self._current_joint_angles.copy()
                self._last_sent_qpos = self._current_joint_angles.copy()
                if self._current_gripper is not None:
                    self._last_sent_gripper = int(self._current_gripper)
                self._target_gripper = self._current_gripper
            
            # Initialize IK solver with current joint angles
            if self.control_mode in ("delta_ee", "rel_ee") and self._current_joint_angles is not None:
                self.ik_solver.reset(self._current_joint_angles)
                self.ik_solver.set_reference(self._current_joint_angles)
                logger.info(f"[AliciaD] IK solver initialized with joints (deg): {np.rad2deg(self._current_joint_angles)}")
            
            self._connected = True
            logger.info(f"[AliciaD] Connected to robot successfully")
            if self._current_joint_angles is not None:
                logger.info(f"[AliciaD] Current joints (deg): {np.rad2deg(self._current_joint_angles)}")
            
            return True
            
        except Exception as e:
            logger.error(f"[AliciaD] Connection error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_observation(self) -> Optional[Dict[str, Any]]:
        """
        Get complete observation data (ALWAYS fresh, no caching).
        
        For imitation learning data collection, observations must be
        real-time ground truth, not cached values.
        """
        if not self._connected or self._driver is None:
            return None
        
        try:
            # Actively request joint data from hardware (rate-limited to avoid serial flooding)
            # This ensures we get fresh data, not stale cache from connect() time.
            now = time.time()
            if not hasattr(self, '_last_joint_query_time'):
                self._last_joint_query_time = 0.0
            # Query at most ~50Hz (every 20ms) to avoid flooding serial
            if now - self._last_joint_query_time >= 0.02:
                self._driver.acquire_info("joint_gripper", wait=False)
                self._last_joint_query_time = now
            
            # Get fresh state from SDK (non-blocking read from parser, updated by thread)
            # The ServoDriver has a background thread that updates state
            state = self._driver.data_parser.get_joint_state()
            
            if state is None or state.angles is None:
                return None
            
            joint_angles = np.array(state.angles, dtype=np.float64)
            gripper = state.gripper
            
            # Update internal tracking
            self._current_joint_angles = joint_angles
            self._current_gripper = gripper

            # Encoder often sticks near home on this arm — prefer last commanded joints
            # for policy state (matches teleop recording which used commanded/FK-consistent q).
            if self._smooth_qpos is not None and len(self._smooth_qpos) >= 6:
                obs_joints = np.asarray(self._smooth_qpos[:6], dtype=np.float64)
            elif self._target_qpos is not None and len(self._target_qpos) >= 6:
                obs_joints = np.asarray(self._target_qpos[:6], dtype=np.float64)
            else:
                obs_joints = joint_angles
            obs_grip = (
                float(self._target_gripper) / 1000.0
                if self._target_gripper is not None
                else float(gripper) / 1000.0
            )
            
            # Combine arm joints and gripper (normalized to 0-1)
            # Hardware gripper is 0-1000
            # NOTE: state_ee is NOT computed here (avoids collect stutter). It is added at
            # dataset save time via enrich_saved_frame() using the reversible FK pipeline.
            qpos = np.concatenate([obs_joints, [obs_grip]])

            return {
                'qpos': qpos.astype(np.float32),
                'joint_angles': joint_angles.astype(np.float32),
                'gripper': gripper,
                'timestamp': state.timestamp,
            }
        except Exception as e:
            logger.error(f"[AliciaD] get_observation error: {e}")
            return None
    
    def process_action(self, action_dict: dict) -> dict:
        """
        Process action based on control mode.
        
        For qpos mode: pass through (7D: 6 joint angles in rad + gripper 0-1)
        For delta_ee mode: convert 7D delta to joint positions using IK
        For rel_ee mode: convert 7D relative pose (from anchor) to joint positions using IK
        """
        if self.control_mode == "qpos":
            return action_dict
        
        # delta_ee and rel_ee modes
        action = action_dict.get('action', None)
        if action is None:
            return action_dict
        
        action = np.array(action, dtype=np.float64)
        
        if len(action) != 7:
            logger.warning(f"[AliciaD] {self.control_mode} expects 7D action, got {len(action)}D")
            return action_dict
        
        # Parse action: [dx, dy, dz, droll, dpitch, dyaw, dgripper]
        # Gripper delta is applied in start() before this call (delta_ee / rel_ee).
        pos_component = action[:3] * self.position_scale
        euler_component = action[3:6] * self.rotation_scale
        
        # Check if action is essentially zero
        if np.allclose(action, 0, atol=1e-6):
            # In rel_ee mode, we MUST process zero actions if they signal a state change
            # (start or stop) to keep the robot's anchor state synchronized with the teleop.
            if self.control_mode == "rel_ee":
                input_unsqueeze = action_dict.get('unsqueeze_active', True)
                anchor_set = action_dict.get('anchor_just_set', False)
                internal_unsqueeze = self._rel_ee_vr_unsqueeze_active
                
                # Pass through if:
                # 1. Anchor just set (Start of movement)
                # 2. Falling edge: Internal True -> Input False (End of movement)
                if anchor_set or (internal_unsqueeze and not input_unsqueeze):
                    pass # Continue to process_action logic (state change frame)
                else:
                    action_dict['action'] = None
                    return action_dict
            else:
                # For delta_ee/qpos, zero action means nothing to do
                action_dict['action'] = None
                return action_dict
        
        with self._lock:
            if self.control_mode == "delta_ee":
                # delta_ee mode: apply delta to IK solver's INTERNAL state (not actual joints)
                # This allows accumulation of deltas for smooth motion
                # Only sync with actual joints on first action or when explicitly requested
                
                # Initialize IK state if this is the first action
                if not hasattr(self, '_ik_initialized') or not self._ik_initialized:
                    if self._current_joint_angles is not None:
                        self.ik_solver.reset(self._current_joint_angles)
                        self._ik_initialized = True
                        logger.info(f"[AliciaD] delta_ee: IK initialized with actual joints (deg): {np.rad2deg(self._current_joint_angles)}")
                        logger.info(f"[AliciaD] delta_ee: initial EE pos: {self.ik_solver.get_current_pose()[:3, 3]}")
                    else:
                        logger.warning("[AliciaD] delta_ee: No current joint angles for IK init")
                        action_dict['action'] = None
                        return action_dict
                
                # Get current EE from IK solver's INTERNAL state (accumulated)
                current_T = self.ik_solver.get_current_pose()
                
                # Build target pose by applying delta to accumulated position
                target_T = current_T.copy()
                target_T[:3, 3] += pos_component
                
                # Convert euler delta to rotation matrix (base frame delta rotation)
                if np.linalg.norm(euler_component) > 1e-10:
                    from scipy.spatial.transform import Rotation as R
                    R_delta_base = R.from_euler('xyz', euler_component).as_matrix()
                    target_T[:3, :3] = R_delta_base @ current_T[:3, :3]
                
                # Gripper: _target_gripper already includes this frame's teleop delta (see start() loop).
                new_gripper = float(np.clip(self._target_gripper, 0, 1000))
                
            elif self.control_mode == "rel_ee":
                try:
                    # rel_ee mode: apply relative pose to ANCHOR
                    vr_unsqueeze_active = action_dict.get('unsqueeze_active', True)
                    anchor_just_set = action_dict.get('anchor_just_set', False)
                    
                    # Detect VR unsqueeze state transitions
                    vr_was_active = self._rel_ee_vr_unsqueeze_active
                    self._rel_ee_vr_unsqueeze_active = vr_unsqueeze_active
                    
                    if vr_unsqueeze_active and not vr_was_active and not anchor_just_set:
                        logger.warning("[AliciaD] rel_ee: unsqueeze rising edge but anchor_just_set=False (teleop may be missing press frame).")

                    if not vr_unsqueeze_active:
                        # Freeze: snap EMA to last IK target so residual lag does not crawl/wiggle.
                        if self._target_qpos is not None:
                            self._smooth_qpos = np.asarray(
                                self._target_qpos[:6], dtype=np.float64
                            ).copy()
                        self._teleop_holding_still = False
                        self._still_count = 0
                        self._prev_ik_qpos = None
                        action_dict['action'] = None
                        return action_dict

                    # Determine trigger reason
                    trigger_reason = ""
                    if anchor_just_set: 
                        trigger_reason = "anchor_just_set"
                    elif (vr_unsqueeze_active and not vr_was_active): 
                        trigger_reason = "unsqueeze_edge"

                    # Set anchor when triggered (fresh squeeze = full reset)
                    if trigger_reason:
                        success = self.refresh_rel_ee_anchor(use_actual_joints=True)
                        if not success:
                            logger.warning("[AliciaD] rel_ee: Failed to refresh anchor. Aborting action.")
                            action_dict['action'] = None
                            return action_dict
                        # Clear offset on fresh anchor (VR and robot are fully synced)
                        self._rel_ee_pos_offset = np.zeros(3)
                        self._rel_ee_rot_offset = np.zeros(3)
                        # Safety: do not publish motion on the press frame (zero action).
                        if anchor_just_set and np.allclose(pos_component, 0, atol=1e-8) and np.allclose(euler_component, 0, atol=1e-8):
                            action_dict['action'] = None
                            return action_dict
                    
                    # If no anchor yet, try to set it now
                    if self._rel_ee_anchor_pose is None:
                        success = self.refresh_rel_ee_anchor(use_actual_joints=True)
                        if not success:
                            logger.warning("[AliciaD] rel_ee: Failed to set initial anchor")
                            action_dict['action'] = None
                            return action_dict
                        self._rel_ee_pos_offset = np.zeros(3)
                        self._rel_ee_rot_offset = np.zeros(3)
                    
                    # Apply offset correction: subtract the accumulated offset from VR's
                    # relative pose so that the robot's target stays within reachable space.
                    corrected_pos = pos_component - self._rel_ee_pos_offset
                    corrected_euler = euler_component - self._rel_ee_rot_offset
                    
                    # Build target pose by applying corrected relative transform to anchor
                    target_T = self._rel_ee_anchor_pose.copy()
                    target_T[:3, 3] += corrected_pos
                    
                    # Apply relative rotation (in base frame)
                    if np.linalg.norm(corrected_euler) > 1e-10:
                        from scipy.spatial.transform import Rotation as R
                        R_rel_base = R.from_euler('xyz', corrected_euler).as_matrix()
                        target_T[:3, :3] = R_rel_base @ self._rel_ee_anchor_pose[:3, :3]
                    
                    # Save raw (uncorrected) VR relative pose for potential offset absorption
                    self._rel_ee_last_raw_pos = pos_component.copy()
                    self._rel_ee_last_raw_euler = euler_component.copy()
                    
                    # Gripper: _target_gripper already includes this frame's teleop delta (see start() loop).
                    new_gripper = float(np.clip(self._target_gripper, 0, 1000))
                    
                except Exception as e:
                    logger.error(f"[AliciaD] rel_ee process error: {e}")
                    import traceback
                    traceback.print_exc()
                    action_dict['action'] = None
                    return action_dict
            
            else:
                return action_dict
            
            # Solve IK with target pose
            success, q_arm, iters, err = self.ik_solver.solve(target_T)
            timed_out = bool(getattr(self.ik_solver, "_last_solve_timed_out", False))

            # IK Recovery: only if failed AND not time-budget (recovery would double the hitch).
            # Prefer _target_qpos (last commanded) over _current_joint_angles (encoder)
            # because the hardware encoder is unreliable and often reports near-zero.
            if not success and not timed_out:
                recovery_joints = self._target_qpos if self._target_qpos is not None else self._current_joint_angles
                if recovery_joints is not None:
                    self.ik_solver.reset(recovery_joints)
                    success, q_arm, iters, err = self.ik_solver.solve(target_T)
            
            # SAFETY: If IK failed, reject the action and dynamically re-anchor (rel_ee mode)
            if not success:
                if self.control_mode == "rel_ee" and hasattr(self, '_rel_ee_last_raw_pos'):
                    # Dynamic re-anchor: absorb the current VR relative pose into offset
                    # so that subsequent frames start from the last reachable position.
                    self._rel_ee_pos_offset = self._rel_ee_last_raw_pos.copy()
                    self._rel_ee_rot_offset = self._rel_ee_last_raw_euler.copy()
                    # Refresh robot anchor to last successfully commanded pose
                    self._reanchor_to_last_good_pose()
                    self._warning_throttled_ik(
                        f"[AliciaD] IK failed, re-anchored: err={err:.4e}, target_pos={target_T[:3,3]}"
                    )
                else:
                    self._warning_throttled_ik(
                        f"[AliciaD] IK failed (unreachable target): err={err:.4e}, iters={iters}, target_pos={target_T[:3,3]}"
                    )
                action_dict['action'] = None
                return action_dict
            
            # SAFETY: Reject IK solution if it jumps too far from last known position.
            # This catches "flip solutions" where IK converges to a valid but distant config.
            reference_q = self._target_qpos if self._target_qpos is not None else self._current_joint_angles
            if reference_q is not None:
                ik_jump = np.max(np.abs(q_arm - reference_q))
                if ik_jump > self.max_joint_delta_rad:
                    if self.control_mode == "rel_ee" and hasattr(self, '_rel_ee_last_raw_pos'):
                        self._rel_ee_pos_offset = self._rel_ee_last_raw_pos.copy()
                        self._rel_ee_rot_offset = self._rel_ee_last_raw_euler.copy()
                        self._reanchor_to_last_good_pose()
                    else:
                        self.ik_solver.reset(reference_q)
                    self._warning_throttled_ik(
                        f"[AliciaD] IK solution REJECTED (jump): max joint jump {np.rad2deg(ik_jump):.1f}° "
                        f"exceeds limit {np.rad2deg(self.max_joint_delta_rad):.1f}°"
                    )
                    action_dict['action'] = None
                    return action_dict
            
            # IK succeeded: clear any accumulated offset drift (successful convergence
            # means robot and VR are in sync at this corrected pose)
            # NOTE: We do NOT clear offset here. The offset stays until the user
            # re-squeezes (full anchor reset). This ensures continuity.
            
            # Store raw IK target for jump rejection / re-anchor (not EMA'd).
            self._target_qpos = q_arm.copy()
            self._target_gripper = new_gripper

            # Motion-gated stop-hold: if IK barely moves for N frames, freeze joint
            # cmd so VR/IK noise does not re-excite servos at high speed.
            ik_step = float("inf")
            if self._prev_ik_qpos is not None:
                d_ik = (q_arm - self._prev_ik_qpos + np.pi) % (2.0 * np.pi) - np.pi
                ik_step = float(np.max(np.abs(d_ik)))
            self._prev_ik_qpos = q_arm.copy()

            if self.joint_stop_hold:
                if ik_step >= self.joint_move_rad:
                    if self._teleop_holding_still:
                        logger.debug("[AliciaD] stop-hold resume (IK moved)")
                    self._teleop_holding_still = False
                    self._still_count = 0
                elif ik_step < self.joint_still_rad:
                    self._still_count += 1
                    if (
                        not self._teleop_holding_still
                        and self._still_count >= self.joint_still_frames
                    ):
                        self._teleop_holding_still = True
                        # Lock to current smoothed cmd (already near rest).
                        if self._smooth_qpos is not None:
                            self._smooth_qpos = np.asarray(
                                self._smooth_qpos[:6], dtype=np.float64
                            ).copy()

            if self.joint_stop_hold and self._teleop_holding_still:
                # Hold: do not chase noisy IK while hand is still.
                if self._smooth_qpos is not None and len(self._smooth_qpos) >= 6:
                    q_out = np.asarray(self._smooth_qpos[:6], dtype=np.float64)
                else:
                    q_out = q_arm
            else:
                # Joint EMA on teleop EE→IK output (silky while moving).
                a = self.joint_smooth_alpha
                if a < 1.0 - 1e-9 and self._smooth_qpos is not None and len(self._smooth_qpos) >= 6:
                    prev = np.asarray(self._smooth_qpos[:6], dtype=np.float64)
                    err = (q_arm - prev + np.pi) % (2.0 * np.pi) - np.pi
                    q_out = prev + a * err
                else:
                    q_out = q_arm
                self._smooth_qpos = np.asarray(q_out, dtype=np.float64).copy()

            # Build output action: [6 joint angles, gripper_normalized]
            output_action = np.concatenate([q_out, [new_gripper / 1000.0]])
            action_dict['action'] = output_action
        
        return action_dict
    
    def _reanchor_to_last_good_pose(self):
        """
        Re-anchor to the last successfully commanded pose (IK failure recovery).
        
        This refreshes the robot-side anchor to the last known reachable EE pose,
        so that subsequent VR relative poses (after offset correction) will target
        positions near the robot's actual position instead of the unreachable one.
        """
        joints = self._target_qpos
        if joints is not None:
            fk_pose = self.ik_solver.compute_fk(joints)
            self._rel_ee_anchor_pose = fk_pose.copy()
            self.ik_solver.reset(joints)
            self.ik_solver.set_reference(joints, fk_pose)
    
    def _warning_throttled_ik(self, msg: str):
        """Throttled IK warning to avoid spamming at 200Hz."""
        now = time.time()
        if not hasattr(self, '_last_ik_warn_time'):
            self._last_ik_warn_time = 0.0
            self._ik_warn_suppressed = 0
        if now - self._last_ik_warn_time >= 1.0:
            if self._ik_warn_suppressed > 0:
                msg += f" (suppressed {self._ik_warn_suppressed} repeats)"
            logger.warning(msg)
            self._last_ik_warn_time = now
            self._ik_warn_suppressed = 0
        else:
            self._ik_warn_suppressed += 1
    
    def refresh_rel_ee_anchor(self, use_actual_joints: bool = True) -> bool:
        """
        Refresh the rel_ee anchor to current EE pose.
        
        The anchor is computed using forward kinematics from actual joint observations.
        """
        if self.control_mode != "rel_ee":
            return False
        
        with self._lock:
            # FORCE UPDATE: Get fresh observation immediately
            self.get_observation()
            
            # Determine which joint source to use for anchor FK calculation.
            #
            # IMPORTANT: This robot's hardware encoder does NOT reliably track 
            # commanded positions - it often reports near-zero (home) values 
            # even after the robot has moved. Therefore we PREFER _target_qpos 
            # (the last successfully commanded joints) over _current_joint_angles 
            # (hardware encoder readout).
            #
            # Priority:
            # 1. _target_qpos (last commanded) - most reliable
            # 2. _current_joint_angles (hardware encoder) - fallback only
            joints_to_use = None
            
            if self._target_qpos is not None:
                joints_to_use = self._target_qpos
            elif self._current_joint_angles is not None:
                joints_to_use = self._current_joint_angles
            
            if joints_to_use is not None:
                fk_pose = self.ik_solver.compute_fk(joints_to_use)
                self._rel_ee_anchor_pose = fk_pose.copy()
                self.ik_solver.reset(joints_to_use)
                self.ik_solver.set_reference(joints_to_use, fk_pose)
            else:
                logger.warning("[AliciaD] refresh_anchor failed: No joints available")
                return False
            
            self._rel_ee_anchor_gripper = self._current_gripper if self._current_gripper is not None else self._target_gripper
            self._rel_ee_anchor_initialized = True
            
            return True
    
    def publish_action(self, action: np.ndarray, *, from_ik_teleop: bool = False):
        """
        Publish action to robot with safety checks (no joint EMA here).

        Joint EMA runs in process_action after EE IK (teleop only). Inference paths
        (qpos / set_state_ee / set_joints) call this directly and must stay unsmoothed.

        Args:
            action: 7D array [6 joint angles in rad, gripper normalized 0-1]
            from_ik_teleop: If True, keep _target_qpos as raw IK from process_action.
        """
        if not self._connected or self._driver is None:
            logger.error("[AliciaD] Not connected")
            return
        
        try:
            if len(action) < 7:
                logger.error(f"[AliciaD] Invalid action dimension: {len(action)}")
                return
            
            target_joints = np.array(action[:6], dtype=np.float64)
            # Convert gripper from 0-1 to 0-1000
            gripper_value = int(action[6] * 1000)
            gripper_value = np.clip(gripper_value, 0, 1000)

            def _wrap_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                """Shortest signed angle a-b on each joint (rad)."""
                return (a - b + np.pi) % (2.0 * np.pi) - np.pi

            # Safety: accept if near last IK target OR last teleop-smoothed cmd
            # (gripper-only publishes hold smoothed joints, which may lag IK target).
            refs = []
            if self._target_qpos is not None:
                refs.append(self._target_qpos)
            if self._smooth_qpos is not None:
                refs.append(self._smooth_qpos)
            if self._current_joint_angles is not None:
                refs.append(self._current_joint_angles)
            if not refs:
                logger.warning("[AliciaD] No reference joint angles available for safety check, skipping action")
                return

            # Unwrap target toward nearest ref so ±π equivalents are not treated as 180° jumps.
            best_ref = min(
                refs,
                key=lambda r: float(np.max(np.abs(_wrap_delta(target_joints, np.asarray(r, dtype=np.float64)[:6])))),
            )
            best_ref = np.asarray(best_ref, dtype=np.float64)[:6]
            target_joints = best_ref + _wrap_delta(target_joints, best_ref)

            if not any(
                np.max(np.abs(_wrap_delta(target_joints, np.asarray(ref, dtype=np.float64)[:6])))
                <= self.max_joint_delta_rad
                for ref in refs
            ):
                max_delta = max(
                    float(np.max(np.abs(_wrap_delta(target_joints, np.asarray(ref, dtype=np.float64)[:6]))))
                    for ref in refs
                )
                logger.warning(
                    f"[AliciaD] Action REJECTED: max joint delta {np.rad2deg(max_delta):.1f}° "
                    f"exceeds limit {np.rad2deg(self.max_joint_delta_rad):.1f}°"
                )
                return

            if not from_ik_teleop:
                # Direct joint / inference command: own the logical target.
                self._target_qpos = target_joints.copy()
                self._smooth_qpos = target_joints.copy()
            elif self._smooth_qpos is None:
                self._smooth_qpos = target_joints.copy()

            speed = float(self.speed_deg_s)
            sent_ref = (
                self._last_sent_qpos
                if self._last_sent_qpos is not None
                else best_ref
            )
            step = _wrap_delta(target_joints, np.asarray(sent_ref, dtype=np.float64)[:6])
            step_mag = float(np.max(np.abs(step)))
            grip_changed = (
                self._last_sent_gripper is None
                or abs(int(gripper_value) - int(self._last_sent_gripper)) > 0
            )

            if from_ik_teleop:
                # Teleop: while stop-holding, do not re-command identical pose
                # (high-speed re-issue is what causes settle jitter).
                if self.joint_stop_hold and self._teleop_holding_still:
                    if step_mag < max(self.joint_still_rad, 1e-6) and not grip_changed:
                        return
                elif (
                    self.joint_teleop_brake_rad > 0
                    and step_mag < self.joint_teleop_brake_rad
                    and step_mag > 1e-9
                ):
                    # Soft brake while decelerating into a stop (still streaming).
                    speed = min(speed, float(self.joint_teleop_brake_speed_deg_s))
            elif self.joint_cmd_deadband_rad > 0:
                # Inference / direct qpos stop-hunt.
                if step_mag < self.joint_cmd_deadband_rad and not grip_changed:
                    return
                if step_mag < self.joint_creep_rad:
                    speed = min(speed, float(self.joint_creep_speed_deg_s))

            self._driver.set_joint_and_gripper(
                joint_angles=target_joints.tolist(),
                gripper_value=gripper_value,
                speed_deg_s=speed,
                gripper_speed_deg_s=self.gripper_speed_deg_s
            )
            self._last_sent_qpos = target_joints.copy()
            self._last_sent_gripper = int(gripper_value)
            self._target_gripper = gripper_value
            
        except Exception as e:
            logger.error(f"[AliciaD] publish_action error: {e}")
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        return 7  # 6 arm joints + 1 gripper
    
    def is_running(self) -> bool:
        """Check if robot is running."""
        # Check if driver is connected and update thread is running
        if self._driver is None:
            return False
        return self._connected and self._driver.serial_comm.is_connected()
    
    def shutdown(self):
        """Shutdown robot."""
        try:
            if self._driver is not None:
                self._driver.disconnect()
                self._driver = None
            self._connected = False
            logger.info("[AliciaD] Shutdown complete")
        except Exception as e:
            logger.error(f"[AliciaD] Shutdown error: {e}")
    
    def start(self):
        """
        Start the robot in subprocess.
        
        IMPORTANT: Connection must be established in the subprocess, not in the main process.
        This is because serial connections cannot be inherited across process boundaries
        (especially with 'spawn' start method).
        """
        from deploy.utils import RateLimiter
        
        # Create shared memory for observations
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)
        
        # Connect to control shm (teleop) with retry - it may not be ready yet
        if self.control_shm_name is not None and self.control_shm is None:
            max_retries = 10
            for i in range(max_retries):
                try:
                    self.control_shm = self.connect_to_existing_shm(self.control_shm_name)
                    logger.info(f"Connected to control SHM: {self.control_shm_name}")
                    break
                except ValueError as e:
                    if i < max_retries - 1:
                        logger.warning(f"Waiting for control SHM '{self.control_shm_name}'... ({i+1}/{max_retries})")
                        time.sleep(0.5)
                    else:
                        logger.error(f"Failed to connect to control SHM: {e}")
        
        # IMPORTANT: Connect to robot hardware in subprocess
        # Serial connections cannot be inherited from main process
        if not self._connected:
            logger.info("[AliciaD] Connecting to robot in subprocess...")
            if not self.connect():
                logger.error("[AliciaD] Failed to connect to robot in subprocess")
                return
        
        self.is_running = True
        rate_limiter = RateLimiter()
        
        logger.info(
            f"[AliciaD] Starting main loop (mode={self.control_mode}, fps={self.fps})"
        )
        if self.gripper_trace:
            logger.warning(
                "[AliciaD] gripper_trace=ON: logging [GripperTrace:robot] lines; "
                "disable via gripper_trace: false in robot YAML after debugging."
            )
        
        # Idle counter for auto-resetting unsqueeze state
        _action_idle_count = 0
        _trace_prev_unsqueeze: Optional[bool] = None
        
        while self.is_running:
            # Read and process action from teleop (arrives at teleop rate, e.g. 50Hz)
            action = self.read_action()
            if action is not None:
                _action_idle_count = 0

                
                # go_home: VR B button or eval_real pause menu (r+Enter)
                if action.get("cmd") == "reset" or action.get("go_home", False):
                    logger.info("[AliciaD] Go-home command received, moving to home position...")
                    if self.gripper_trace:
                        logger.info(
                            f"[GripperTrace:robot] GO_HOME rx: tgt_pre={self._target_gripper:.1f} "
                            f"cur_hw={self._current_gripper}"
                        )
                    self.set_home(speed_deg_s=self.speed_deg_s)
                    # Reset IK / anchor state so next movement starts fresh
                    self._ik_initialized = False
                    self._rel_ee_anchor_pose = None
                    self._rel_ee_vr_unsqueeze_active = False
                    self._rel_ee_pos_offset = np.zeros(3)
                    self._rel_ee_rot_offset = np.zeros(3)
                    if self.gripper_trace:
                        logger.info(
                            f"[GripperTrace:robot] GO_HOME after set_home: tgt={self._target_gripper:.1f} "
                            f"tq0={self._target_qpos[:3] if self._target_qpos is not None else None}"
                        )
                    continue
                
                unsqz = bool(action.get("unsqueeze_active", False))
                if self.gripper_trace and _trace_prev_unsqueeze is not None and unsqz != _trace_prev_unsqueeze:
                    edge = "UNFREEZE" if unsqz else "FREEZE"
                    ra6 = None
                    raw = action.get("action")
                    if raw is not None and len(raw) >= 7:
                        ra6 = float(raw[6])
                    logger.info(
                        f"[GripperTrace:robot] {edge}: tgt={self._target_gripper:.1f} raw_a6={ra6} "
                        f"go_home={action.get('go_home')} anchor_set={action.get('anchor_just_set')}"
                    )
                _trace_prev_unsqueeze = unsqz

                # Extract gripper BEFORE process_action.
                # Critical: publish gripper immediately — do NOT wait for IK (scipy can take
                # 100–500ms+ per solve and makes trigger release feel laggy).
                raw_action = action.get('action', None)
                tg_before = float(self._target_gripper)
                unsqz_now = bool(action.get("unsqueeze_active", False))
                if (
                    self.control_mode in ("delta_ee", "rel_ee")
                    and raw_action is not None
                    and len(raw_action) >= 7
                ):
                    g_cmd = float(raw_action[6]) * self.gripper_scale
                    if self.gripper_absolute:
                        # Absolute open amount [0,1]; only apply while squeezing so freeze holds pose.
                        if unsqz_now:
                            self._target_gripper = float(np.clip(g_cmd * 1000.0, 0, 1000))
                            if self.gripper_trace and abs(self._target_gripper - tg_before) > 0.5:
                                logger.info(
                                    f"[GripperTrace:robot] absolute: raw6={float(raw_action[6]):.5f} "
                                    f"tgt {tg_before:.1f}->{self._target_gripper:.1f}"
                                )
                    else:
                        self._target_gripper = float(np.clip(
                            self._target_gripper + g_cmd * 1000, 0, 1000
                        ))
                        if self.gripper_trace and (
                            abs(g_cmd) > 1e-9 or abs(self._target_gripper - tg_before) > 0.5
                        ):
                            logger.info(
                                f"[GripperTrace:robot] accum: raw6={float(raw_action[6]):.5f} "
                                f"scale={self.gripper_scale} d_scaled={g_cmd:.5f} "
                                f"tgt {tg_before:.1f}->{self._target_gripper:.1f}"
                            )

                gripper_changed = abs(self._target_gripper - tg_before) > 0.5
                if (
                    gripper_changed
                    and self.control_mode in ("delta_ee", "rel_ee")
                    and self._target_qpos is not None
                ):
                    # Hold last smoothed joints when only gripper changes (avoids joint snap).
                    q_hold = self._smooth_qpos if self._smooth_qpos is not None else self._target_qpos
                    out_g = np.concatenate([
                        np.asarray(q_hold, dtype=np.float64)[:6],
                        [self._target_gripper / 1000.0],
                    ])
                    if self.gripper_trace:
                        logger.info(
                            f"[GripperTrace:robot] publish_gripper_immediate: out6={out_g[6]:.4f} "
                            f"tgt={self._target_gripper:.1f}"
                        )
                    self.publish_action(out_g, from_ik_teleop=True)
                
                t_ik0 = time.perf_counter()
                action = self.process_action(action)
                ik_ms = (time.perf_counter() - t_ik0) * 1000.0
                if ik_ms > 12.0:
                    if not hasattr(self, "_last_ik_slow_log") or (time.time() - self._last_ik_slow_log) >= 1.0:
                        to = bool(getattr(self.ik_solver, "_last_solve_timed_out", False))
                        nit = getattr(self.ik_solver, "_last_solve_iterations", "?")
                        logger.warning(
                            f"[AliciaD] Slow process_action/IK: {ik_ms:.1f}ms "
                            f"(iters={nit}, timed_out={to})"
                        )
                        self._last_ik_slow_log = time.time()
                action_array = action.get('action', None)
                if action_array is not None:
                    out = np.asarray(action_array, dtype=np.float64).copy()
                    if self.control_mode == "qpos":
                        pass  # gripper already absolute in teleop command
                    else:
                        out[6] = self._target_gripper / 1000.0
                    if self.gripper_trace:
                        logger.info(
                            f"[GripperTrace:robot] publish: out6={out[6]:.4f} tgt_raw={self._target_gripper:.1f} "
                            f"unsqz={action.get('unsqueeze_active')}"
                        )
                    # EE teleop: EMA already applied in process_action; qpos teleop: no EMA.
                    self.publish_action(
                        out,
                        from_ik_teleop=(self.control_mode in ("delta_ee", "rel_ee")),
                    )
                elif self.gripper_trace and self.control_mode in ("delta_ee", "rel_ee") and not gripper_changed:
                    ra6 = float(raw_action[6]) if raw_action is not None and len(raw_action) >= 7 else None
                    logger.info(
                        f"[GripperTrace:robot] no_publish (IK/skip): tgt={self._target_gripper:.1f} raw6={ra6}"
                    )
            else:
                _action_idle_count += 1
                # If idle for >0.1s (20 frames at 200Hz), assume teleop stopped sending
                if _action_idle_count > 20 and self._rel_ee_vr_unsqueeze_active:
                    self._rel_ee_vr_unsqueeze_active = False
            
            # Get observation and write to shm
            data = self.get_data()
            if data is not None:
                self.write_data(data)
            
            rate_limiter.sleep(self.fps)
    
    @staticmethod
    def obs2meta(device_data: dict) -> dict:
        """Extract robot state from this device's SHM data.

        Must be static/classmethod: ``create_obs2meta_func`` calls ``cls.obs2meta(data)``.
        Always fills ``state_joint`` (qpos). Computes ``state_ee`` via FK when missing
        so EE policies can use MetaPolicy(ctrl_space=ee).
        """
        if device_data is None:
            return {}
        qpos = np.asarray(
            device_data.get("qpos", np.zeros(7, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if qpos.size < 7:
            qpos = np.pad(qpos, (0, max(0, 7 - qpos.size))).astype(np.float32)
        meta = {
            "state": qpos.copy(),
            "state_joint": qpos.copy(),
        }
        state_ee = device_data.get("state_ee")
        if state_ee is None and qpos.size >= 6:
            try:
                solver = AliciaD._get_save_fk_solver()
                T = solver.compute_fk(qpos[:6].astype(np.float64))
                pose6 = _xyz_rpy_from_transform_matrix(T)
                state_ee = np.concatenate(
                    [pose6, [float(qpos[6]) if qpos.size >= 7 else 0.0]]
                ).astype(np.float32)
            except Exception as e:
                logger.warning(
                    "[AliciaD.obs2meta] FK→state_ee failed (EE policies will error): {}",
                    e,
                )
                state_ee = None
        if state_ee is not None:
            meta["state_ee"] = np.asarray(state_ee, dtype=np.float32)
        return meta
    
    def meta2act(self, mact) -> np.ndarray:
        """Convert MetaAction to robot action."""
        return mact.get('action', np.zeros(7))
    
    # ==================== Convenience Methods ====================
    
    def set_home(self, speed_deg_s: float = 10.0):
        """Move robot to home position (all zeros) with gripper fully open (1000)."""
        if self._driver is not None:
            home_joints = [0.0] * 6
            gripper_open = 1000
            self._driver.set_joint_and_gripper(
                joint_angles=home_joints,
                gripper_value=gripper_open,
                speed_deg_s=speed_deg_s,
            )
            # Keep command state in sync — otherwise the next publish_action uses a stale
            # _target_gripper (e.g. pre-home closed) and immediately re-clamps the gripper
            # even when the teleop sends zero gripper delta.
            self._target_qpos = np.zeros(6, dtype=np.float64)
            self._smooth_qpos = np.zeros(6, dtype=np.float64)
            self._last_sent_qpos = np.zeros(6, dtype=np.float64)
            self._last_sent_gripper = int(gripper_open)
            self._prev_ik_qpos = None
            self._still_count = 0
            self._teleop_holding_still = False
            self._target_gripper = float(gripper_open)
            self._current_gripper = float(gripper_open)
    
    def set_joints(self, joint_angles: np.ndarray, gripper: Optional[float] = None):
        """
        Set joint angles directly.
        
        Args:
            joint_angles: 6D array of joint angles in radians
            gripper: Gripper value (0-1), or None to keep current
        """
        if gripper is None:
            gripper = self._target_gripper / 1000.0 if self._target_gripper else 0.5
        
        action = np.concatenate([joint_angles, [gripper]])
        self.publish_action(action)
    
    def set_pose(self, target_pose: np.ndarray, gripper: Optional[float] = None) -> bool:
        """
        Set end-effector pose using teleop IK (original Piper-style).
        For recorded state_ee / policy inference, use set_state_ee instead.
        """
        success, q_arm, iters, err = self.ik_solver.solve(target_pose)
        
        if gripper is None:
            gripper = self._target_gripper / 1000.0 if self._target_gripper else 0.5
        
        action = np.concatenate([q_arm, [gripper]])
        self.publish_action(action)
        
        return success

    # Cache for save-time FK (main/collect process; not the teleop subprocess).
    _save_fk_solver = None
    _save_fk_key: Optional[tuple] = None

    @classmethod
    def _resolve_urdf_path(cls, device_args: Optional[Dict[str, Any]] = None) -> str:
        args = device_args or {}
        urdf_path = args.get("urdf_path")
        gripper_type = args.get("gripper_type", "50mm")
        if urdf_path:
            return str(urdf_path)
        local_urdf = Path(__file__).parent / "urdf" / f"Alicia_D_v5_6_gripper_{gripper_type}.urdf"
        if local_urdf.exists():
            return str(local_urdf)
        try:
            from synriard import get_model_path
            return str(
                get_model_path(
                    "Alicia_D",
                    version="v5_6",
                    variant=f"gripper_{gripper_type}",
                    model_format="urdf",
                )
            )
        except ImportError as e:
            raise FileNotFoundError(f"URDF not found at {local_urdf}") from e

    @classmethod
    def _get_save_fk_solver(cls, device_args: Optional[Dict[str, Any]] = None):
        """Lazy reversible-geometry FK solver for dataset save (not used in teleop loop)."""
        from deploy.robot.alicia_d.piper_global_ik import PiperStyleGlobalIKSolver

        args = device_args or {}
        urdf_path = cls._resolve_urdf_path(args)
        end_frame = args.get("ik_end_frame", "link6")
        if args.get("ik_ee_offset_matrix") is not None:
            ee_offset = np.asarray(args["ik_ee_offset_matrix"], dtype=np.float64)
        else:
            ee_offset = _transform_matrix_from_xyz_rpy(
                np.asarray(args.get("ik_ee_offset_xyz") or [0.0, 0.0, 0.0], dtype=np.float64),
                np.asarray(args.get("ik_ee_offset_rpy") or [0.0, 0.0, 0.0], dtype=np.float64),
            )
        key = (urdf_path, end_frame, tuple(np.round(ee_offset.ravel(), 9).tolist()))
        if cls._save_fk_solver is None or cls._save_fk_key != key:
            cls._save_fk_solver = PiperStyleGlobalIKSolver(
                urdf_path=urdf_path,
                end_frame=end_frame,
                ee_offset_matrix=ee_offset,
                position_weight=1.0,
                rotation_weight=max(float(args.get("ik_rotation_weight", 0.1)), 1e-6),
                total_cost_scale=20.0,
                regularization_weight=0.01,
                max_iterations=int(args.get("ik_max_iter", 200)),
                tol=max(float(args.get("ik_eps", 0.05)), 1e-6),
                jump_reset_threshold_deg=1e9,
                regularize_to_warmstart=True,
                exact_when_converged=True,
            )
            cls._save_fk_key = key
            logger.info(
                "[AliciaD] Save-time reversible FK ready (end_frame={}, urdf={})",
                end_frame,
                urdf_path,
            )
        return cls._save_fk_solver

    @classmethod
    def enrich_saved_frame(
        cls,
        device_data: dict,
        device_args: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Dataset-save hook (collect process): add state_ee from recorded qpos.

        state_ee = [x, y, z, roll, pitch, yaw, gripper] where gripper is qpos[6]
        (normalized open amount in [0,1]). Uses reversible FK geometry so inference
        can call set_state_ee / replay_ik_solver consistently.
        """
        if not isinstance(device_data, dict):
            return device_data
        qpos = device_data.get("qpos")
        if qpos is None:
            return device_data
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.size < 6:
            return device_data

        try:
            solver = cls._get_save_fk_solver(device_args)
            T = solver.compute_fk(qpos[:6])
            pose6 = _xyz_rpy_from_transform_matrix(T)
            if qpos.size >= 7:
                gripper = float(qpos[6])
            else:
                # Fallback: hardware gripper 0-1000 → normalize
                g = device_data.get("gripper", 0.0)
                gripper = float(g) / 1000.0 if float(g) > 1.0 + 1e-6 else float(g)
            state_ee = np.concatenate([pose6, [gripper]]).astype(np.float32)
        except Exception as e:
            logger.warning("[AliciaD] enrich_saved_frame failed: {}", e)
            return device_data

        out = dict(device_data)
        out["state_ee"] = state_ee
        return out

    def state_ee_from_joints(self, joint_angles: Optional[np.ndarray] = None) -> np.ndarray:
        """FK → state_ee [x,y,z,roll,pitch,yaw] (record frame; same as inference FK)."""
        if joint_angles is None:
            joint_angles = self._current_joint_angles
        if joint_angles is None:
            raise RuntimeError("No joint angles available for state_ee FK")
        T = self._compute_fk_T(joint_angles)
        return _xyz_rpy_from_transform_matrix(T).astype(np.float32)

    def set_state_ee(
        self,
        state_ee: np.ndarray,
        gripper: Optional[float] = None,
        q_init: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Inference / replay: absolute EE [x,y,z,roll,pitch,yaw] → joints via reversible IK.

        Warm-start from q_init (default: current joints) so recorded state_ee maps back
        to the same configuration when reachable.
        """
        state_ee = np.asarray(state_ee, dtype=np.float64).reshape(-1)
        if state_ee.shape[0] < 6:
            raise ValueError(f"state_ee must be at least 6D, got shape {state_ee.shape}")
        target_pose = _transform_matrix_from_xyz_rpy(state_ee[:3], state_ee[3:6])

        if q_init is None:
            q_init = self._current_joint_angles
        if q_init is None:
            q_init = self.replay_ik_solver.get_current_q()

        success, q_arm, iters, err = self.replay_ik_solver.solve(target_pose, q_init=q_init)

        if gripper is None:
            if state_ee.shape[0] >= 7:
                gripper = float(state_ee[6])
            else:
                gripper = self._target_gripper / 1000.0 if self._target_gripper else 0.5

        action = np.concatenate([q_arm, [gripper]])
        self.publish_action(action)
        return success

    def solve_ee_sequence(
        self,
        ee_chunk: np.ndarray,
        q_current: Optional[np.ndarray] = None,
        *,
        q_hint: Optional[np.ndarray] = None,
        use_hint_warmstart: bool = False,
        polish: bool = True,
        log: bool = True,
    ) -> Tuple[np.ndarray, Any]:
        """
        Sequence EE→qpos (chunk-level redundancy resolution).

        Anchors to ``q_current`` (default: measured joints), then temporal IK.
        Does not publish — caller dispatches the returned qpos chunk.
        """
        from deploy.robot.alicia_d.sequence_ik import solve_sequence

        if q_current is None:
            q_current = self._current_joint_angles
        if q_current is None:
            q_current = self.replay_ik_solver.get_current_q()
        # Dedicated sequence solver (stronger warmstart reg) — do not reuse teleop solver.
        if getattr(self, "_sequence_ik_solver", None) is None:
            self._sequence_ik_solver = self._make_ik_solver(
                reversible=True, regularization_weight=1.0
            )
        return solve_sequence(
            self._sequence_ik_solver,
            ee_chunk,
            q_current,
            transform_fn=_transform_matrix_from_xyz_rpy,
            q_hint=q_hint,
            use_hint_warmstart=use_hint_warmstart,
            polish=polish,
            log=log,
        )
    
    def get_pose(self) -> np.ndarray:
        """Current EE pose (4x4) in the recorded frame (teleop FK model; same geometry)."""
        if self._current_joint_angles is not None:
            return self._compute_fk_T(self._current_joint_angles)
        return self.ik_solver.get_current_pose()


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Alicia-D Robot")
    parser.add_argument('--port', type=str, default='', help='Serial port (empty for auto-detect)')
    parser.add_argument('--mode', type=str, default='qpos', choices=['qpos', 'delta_ee', 'rel_ee'], help='Control mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    print(f"Creating Alicia-D robot on port '{args.port}' with mode={args.mode}")
    
    robot = AliciaD(
        port=args.port,
        control_mode=args.mode,
        debug=args.debug,
    )
    
    if not robot.connect():
        print("Failed to connect to robot")
        exit(1)
    
    print("Connected! Reading state...")
    
    try:
        for i in range(10):
            obs = robot.get_observation()
            if obs is not None:
                qpos = obs['qpos']
                print(f"[{i}] joints (deg): {np.rad2deg(qpos[:6])}, gripper: {qpos[6]:.3f}")
                state_ee = obs.get('state_ee')
                if state_ee is not None:
                    print(
                        f"     state_ee xyz=[{state_ee[0]:.4f}, {state_ee[1]:.4f}, {state_ee[2]:.4f}] "
                        f"rpy=[{state_ee[3]:.3f}, {state_ee[4]:.3f}, {state_ee[5]:.3f}]"
                    )
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.shutdown()
        print("Done.")
