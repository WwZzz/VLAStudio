"""
BiSO101 Follower Robot with Camera Integration and Delta EE Control

Supports two control modes:
1. qpos: Direct joint position control (normalized space) - 12D (6 left + 6 right)
2. delta_ee: 14D delta EE control with IK (7D per arm: dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper)
"""

import numpy as np
import traceback
import time
import threading
import multiprocessing as mp
from multiprocessing import Queue
from typing import Optional, List
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
try:
    from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
except ImportError:
    from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .bi_so101_follower import BiSO101FollowerConfig, BiSO101Follower
from vlastudio.deploy.robot.base import BaseRobot
from vlastudio.benchmark.base import MetaObs
from loguru import logger

# Import kinematics from so101_sim (shared implementation)
from vlastudio.deploy.robot.so101.kinematics import create_so101, lerobot_FK, lerobot_IK


# Default control limits in radians (matching so101)
DEFAULT_QLIMIT_MIN = [-2.1, -3.1, -0.0, -1.375, -1.57, -0.15]
DEFAULT_QLIMIT_MAX = [2.1, 0.0, 3.1, 1.475, 3.1, 1.5]

# Default gpos limits (end-effector pose limits)
DEFAULT_GLIMIT_MIN = [0.125, -0.4, 0.046, -3.1, -0.75, -1.5]
DEFAULT_GLIMIT_MAX = [0.340, 0.4, 0.23, 2.0, 1.57, 1.5]

# Default joint signs (1 = normal, -1 = reversed)
DEFAULT_JOINT_SIGNS = [1, 1, 1, 1, 1, 1]

# Home pose (normalized qpos): [left(6), right(6)].
# Measured from the current real robot startup pose.
DEFAULT_HOME_QPOS = [
    -4.094430099594248,
    -97.98319327731093,
    100.0,
    71.20756338633433,
    -53.69841681806385,
    2.9277813923227063,
    -4.309392265193367,
    -96.13282891971417,
    100.0,
    70.41685342895562,
    -55.01171569903671,
    3.3159947984395317,
]


def _camera_frame_to_chw_u8(frame: np.ndarray) -> np.ndarray:
    """
    Normalize a single camera frame to uint8 CHW (3, H, W).

    OpenCV / LeRobot typically yield HWC (H, W, 3). Placeholders may use CHW (3, H, W).
    Stacking HWC then doing transpose(0, 3, 1, 2) breaks if any frame is already CHW
    (e.g. head missing and only np.zeros((3, H, W)) is used), which yields tensors
    like (K, H, C, W) and ACT/ResNet errors (e.g. 640 vs 3 on channel/spatial dims).
    """
    if frame.ndim != 3:
        raise ValueError(f"Camera frame must be rank-3, got shape {getattr(frame, 'shape', None)}")
    frame = np.asarray(frame)
    if frame.shape[-1] == 3:
        return np.ascontiguousarray(frame.transpose(2, 0, 1))
    if frame.shape[0] == 3:
        return np.ascontiguousarray(frame)
    raise ValueError(
        f"Expected HWC (H, W, 3) or CHW (3, H, W) camera frame, got shape {frame.shape}"
    )


class BiSo101Follower(BaseRobot):
    """
    Bimanual SO101 Follower Robot with Camera Integration
    
    Supports two control modes:
    1. "qpos" (default): Direct joint position control in normalized space (12D)
    2. "delta_ee": 14D delta EE control with IK (7D per arm)
    """

    _left_motors = ["left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex", "left_wrist_flex", "left_wrist_roll", "left_gripper"]
    _right_motors = ["right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex", "right_wrist_flex", "right_wrist_roll", "right_gripper"]
    # Valid control modes
    CONTROL_MODES = ("delta_ee", "qpos")
    
    def __init__(self,
                 name: str = "bi_so101_follower",
                 max_size_mb: int = 64,
                 fps: float = 100.0,
                 control_shm_name: Optional[str] = None,
                 left_arm_port: str = "/dev/ttyACM0",
                 right_arm_port: str = "/dev/ttyACM3",
                 robot_id: str = "bi_so101_follower_arm",
                 left_arm_id: Optional[str] = None,
                 right_arm_id: Optional[str] = None,
                 camera_configs: dict = {},
                 calibration_dir: Optional[str] = None,
                 control_mode: str = "qpos",
                 position_scale: float = 1.0,
                 rotation_scale: float = 1.0,
                 gripper_scale: float = 1.0,
                 qlimit_min: Optional[List[float]] = None,
                 qlimit_max: Optional[List[float]] = None,
                 glimit_min: Optional[List[float]] = None,
                 glimit_max: Optional[List[float]] = None,
                 joint_signs: Optional[List[int]] = None,
                 left_delta_signs: Optional[List[int]] = None,
                 right_delta_signs: Optional[List[int]] = None,
                 safety_check: bool = True,
                 max_joint_delta: float = 0.15,
                 home_qpos: Optional[List[float]] = DEFAULT_HOME_QPOS,
                 reset_step_rad: Optional[float] = None,
                 reset_step_sleep: float = 0.03,
                 reset_final_hold_s: float = 0.2,
                 ik_tolerance: float = 0.05,
                 zero_threshold: float = 0.01,
                 debug: bool = False,
                 **kwargs):
        """
        Initialize the Bimanual SO101 Follower robot with camera
        
        Args:
            name: Name for the shared memory segment
            max_size_mb: Maximum size of shared memory in MB
            fps: Control frequency in Hz
            control_shm_name: Name of the control shared memory (for receiving actions)
            left_arm_port: Communication port for the left arm
            right_arm_port: Communication port for the right arm
            robot_id: Identifier for the robot
            left_arm_id: Custom ID for left arm (for sharing calibration)
            right_arm_id: Custom ID for right arm
            camera_configs: Dictionary of camera configurations
            calibration_dir: Optional calibration directory path
            control_mode: Control mode - "qpos" (direct joint 12D) or "delta_ee" (14D delta with IK)
            position_scale: Scale factor for position deltas (delta_ee mode)
            rotation_scale: Scale factor for rotation deltas (delta_ee mode)
            gripper_scale: Scale factor for gripper deltas (delta_ee mode)
            qlimit_min: Minimum joint limits in radians (6D, shared for both arms)
            qlimit_max: Maximum joint limits in radians (6D, shared for both arms)
            glimit_min: Minimum end-effector pose limits (6D)
            glimit_max: Maximum end-effector pose limits (6D)
            joint_signs: Joint direction signs (6D, shared for both arms)
            left_delta_signs: Delta action direction signs for left arm (7D)
            right_delta_signs: Delta action direction signs for right arm (7D)
            safety_check: Enable safety validation (FK-IK consistency, max joint delta)
            max_joint_delta: Maximum allowed joint change per step (radians)
            ik_tolerance: Maximum allowed FK-IK round-trip error (radians)
            zero_threshold: Deadzone threshold - actions smaller than this are treated as zero
            debug: Enable debug output (print qpos, gpos, actions each step)
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        
        self.left_arm_port = left_arm_port
        self.right_arm_port = right_arm_port
        self.robot_id = robot_id
        
        # Validate and set control mode
        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Invalid control_mode '{control_mode}'. Must be one of {self.CONTROL_MODES}")
        self.control_mode = control_mode
        
        # Action scaling (for delta_ee mode)
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.gripper_scale = gripper_scale
        
        # Joint and end-effector limits (in radians) - shared for both arms
        self.qlimit_min = np.array(qlimit_min if qlimit_min is not None else DEFAULT_QLIMIT_MIN)
        self.qlimit_max = np.array(qlimit_max if qlimit_max is not None else DEFAULT_QLIMIT_MAX)
        self.glimit_min = np.array(glimit_min if glimit_min is not None else DEFAULT_GLIMIT_MIN)
        self.glimit_max = np.array(glimit_max if glimit_max is not None else DEFAULT_GLIMIT_MAX)
        self.joint_signs = np.array(joint_signs if joint_signs is not None else DEFAULT_JOINT_SIGNS)
        
        # Delta action signs for each arm (for reversing input directions)
        # Format: [dx, dy, dz, droll, dpitch, dyaw, dgripper]
        self.left_delta_signs = np.array(left_delta_signs if left_delta_signs is not None else [1, 1, 1, 1, 1, 1, 1])
        self.right_delta_signs = np.array(right_delta_signs if right_delta_signs is not None else [1, 1, 1, 1, 1, 1, 1])
        
        # Kinematics (for delta_ee mode) - one for each arm
        self.robot_kin = create_so101()
        
        # Safety parameters
        self.safety_check = safety_check
        self.max_joint_delta = max_joint_delta
        self.reset_step_rad = float(reset_step_rad) if reset_step_rad is not None else min(max_joint_delta, 0.08)
        self.reset_step_sleep = float(reset_step_sleep)
        self.reset_final_hold_s = float(reset_final_hold_s)
        self.ik_tolerance = ik_tolerance
        self._left_safety_verified = False
        self._right_safety_verified = False
        self._blocked_count = 0
        
        # Deadzone threshold for delta_ee mode
        self.zero_threshold = zero_threshold
        
        # Target state in radians (for delta_ee mode)
        # Separate state for left and right arms
        self.left_target_qpos_rad = None
        self.left_target_gpos = None
        self.right_target_qpos_rad = None
        self.right_target_gpos = None
        self._lock = threading.Lock()
        
        # Normalization offset calibration (separate for each arm)
        self._left_norm_offset = np.zeros(6, dtype=np.float64)
        self._right_norm_offset = np.zeros(6, dtype=np.float64)
        self._left_is_norm_calibrated = False
        self._right_is_norm_calibrated = False
        
        # State synchronization parameters
        # Periodically sync internal state with real robot feedback to prevent drift
        self._sync_counter = 0
        self._sync_interval = 50  # Sync every N observations
        self._sync_threshold = 0.1  # Sync if joint error > threshold (radians, ~5.7 degrees)
        self._last_left_real_qpos_rad = None
        self._last_right_real_qpos_rad = None
        self.home_qpos = None if home_qpos is None else np.asarray(home_qpos, dtype=np.float64)
        if self.home_qpos is not None:
            if self.home_qpos.shape != (12,):
                raise ValueError(f"home_qpos must be shape (12,), got {self.home_qpos.shape}")
            self.left_home_qpos = self.home_qpos[:6].copy()
            self.right_home_qpos = self.home_qpos[6:12].copy()
            self.left_home_qpos_rad = self._normalized_to_radians(self.left_home_qpos)
            self.right_home_qpos_rad = self._normalized_to_radians(self.right_home_qpos)
        else:
            self.left_home_qpos = None
            self.right_home_qpos = None
            self.left_home_qpos_rad = None
            self.right_home_qpos_rad = None
        
        # Debug mode
        self.debug = debug
        self._debug_step = 0
        
        logger.info("[BiSo101Follower] Control mode: {}", control_mode)
        if control_mode == "delta_ee":
            logger.info("[BiSo101Follower] Action dimension: 14D (7D left + 7D right)")
        else:
            logger.info("[BiSo101Follower] Action dimension: 12D (6D left + 6D right)")
        if safety_check:
            logger.info(
                "[BiSo101Follower] Safety check ENABLED: max_joint_delta={:.3f} rad, ik_tolerance={:.3f} rad",
                max_joint_delta,
                ik_tolerance,
            )
        
        # Create camera configurations
        self.camera_configs_dict = {}
        self.cameras = {}
        
        # BiSO101 arm part - using lerobot support
        robot_config = BiSO101FollowerConfig(
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            id=robot_id,
            left_arm_id=left_arm_id,
            right_arm_id=right_arm_id,
            cameras=self.camera_configs_dict
        )
        if calibration_dir:
            robot_config.calibration_dir = Path(calibration_dir)
        self._robot = BiSO101Follower(robot_config)
        
        # Connect to robot
        retry_counts = 0
        max_connect_retry = 10
        while not self.connect():
            logger.info("Retrying for {} time...", retry_counts)
            retry_counts += 1
            if retry_counts > max_connect_retry:
                raise RuntimeError("Failed to connect to robot after max retries")
            time.sleep(1)
    
    def connect(self):
        """Connect robot and cameras"""
        try:
            if not self._robot.is_connected:
                self._robot.connect()
        except DeviceAlreadyConnectedError as e:
            logger.info("Robot already connected: {}", e)
            pass
        except Exception as e:
            logger.error("Failed to connect to robot due to {}", e)
            traceback.print_exc()
            return False
        logger.info("Robot connected")
        return True
    
    def get_action_dim(self):
        """Get action dimension based on control mode"""
        if self.control_mode == "delta_ee":
            return 14  # 7D left + 7D right
        else:  # qpos
            return len(self._left_motors) + len(self._right_motors)  # 12D
    
    def _normalized_to_radians(self, normalized: np.ndarray) -> np.ndarray:
        """Convert normalized joint values to radians (for a single arm)"""
        radians = np.zeros(6, dtype=np.float64)
        for i in range(5):  # Joints 0-4: RANGE_M100_100 [-100, 100]
            qmin = self.qlimit_min[i]
            qmax = self.qlimit_max[i]
            norm_signed = normalized[i] * self.joint_signs[i]
            norm_clamped = np.clip(norm_signed, -100.0, 100.0)
            radians[i] = (norm_clamped + 100.0) / 200.0 * (qmax - qmin) + qmin
        
        # Joint 5 (gripper): RANGE_0_100 [0, 100]
        qmin = self.qlimit_min[5]
        qmax = self.qlimit_max[5]
        norm_signed = normalized[5] * self.joint_signs[5]
        norm_clamped = np.clip(norm_signed, 0.0, 100.0)
        radians[5] = norm_clamped / 100.0 * (qmax - qmin) + qmin
        
        return radians
    
    def _radians_to_normalized(self, radians: np.ndarray, norm_offset: np.ndarray, is_calibrated: bool) -> np.ndarray:
        """Convert radians to normalized joint values (for a single arm)"""
        normalized = np.zeros(6, dtype=np.float64)
        for i in range(5):  # Joints 0-4: RANGE_M100_100 [-100, 100]
            qmin = self.qlimit_min[i]
            qmax = self.qlimit_max[i]
            rad_clamped = np.clip(radians[i], qmin, qmax)
            normalized[i] = ((rad_clamped - qmin) / (qmax - qmin)) * 200.0 - 100.0
            normalized[i] *= self.joint_signs[i]
        
        # Joint 5 (gripper): RANGE_0_100 [0, 100]
        qmin = self.qlimit_min[5]
        qmax = self.qlimit_max[5]
        rad_clamped = np.clip(radians[5], qmin, qmax)
        normalized[5] = ((rad_clamped - qmin) / (qmax - qmin)) * 100.0
        normalized[5] *= self.joint_signs[5]
        
        # Apply calibration offset if calibrated
        if is_calibrated:
            normalized = normalized + norm_offset
        
        return normalized
    
    def _verify_fk_ik_consistency(self, qpos_rad: np.ndarray, arm_name: str) -> bool:
        """Safety check: Verify FK-IK round-trip consistency for one arm"""
        try:
            gpos = lerobot_FK(qpos_rad[1:5], robot=self.robot_kin)
            qpos_recovered, ik_success = lerobot_IK(qpos_rad[1:5], gpos, robot=self.robot_kin)
            
            if not ik_success:
                print(f"[SAFETY-{arm_name}] FK-IK verification FAILED: IK did not converge")
                return False
            
            error = np.abs(qpos_rad[1:5] - qpos_recovered[:4])
            max_error = np.max(error)
            
            if max_error > self.ik_tolerance:
                print(f"[SAFETY-{arm_name}] FK-IK verification FAILED: max_error={max_error:.4f} rad > tolerance={self.ik_tolerance:.4f} rad")
                return False
            
            print(f"[SAFETY-{arm_name}] FK-IK verification PASSED: max_error={max_error:.6f} rad")
            return True
            
        except Exception as e:
            print(f"[SAFETY-{arm_name}] FK-IK verification FAILED with exception: {e}")
            return False
    
    def _apply_deadzone(self, action: np.ndarray) -> np.ndarray:
        """Apply deadzone threshold to action"""
        result = action.copy()
        for i in range(len(result)):
            if abs(result[i]) < self.zero_threshold:
                result[i] = 0.0
        return result
    
    def _sync_state_with_real(self, real_qpos_rad: np.ndarray, arm_name: str):
        """
        Synchronize internal target state with real robot feedback for one arm.
        
        This prevents drift between internal state (which accumulates delta commands)
        and real robot state (which may not fully execute each command).
        
        Args:
            real_qpos_rad: Current real robot joint positions in radians (6D)
            arm_name: "LEFT" or "RIGHT"
        """
        with self._lock:
            if arm_name == "LEFT":
                target_qpos_rad = self.left_target_qpos_rad
            else:
                target_qpos_rad = self.right_target_qpos_rad
            
            if target_qpos_rad is None:
                return
            
            # Compute error between internal target and real position
            error = np.abs(target_qpos_rad - real_qpos_rad)
            max_error = np.max(error)
            max_error_joint = np.argmax(error)
            
            # If error exceeds threshold, sync internal state to real
            if max_error > self._sync_threshold:
                joint_names = ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']
                if self.debug:
                    print(f"[SYNC-{arm_name}] State drift detected: max_error={max_error:.4f} rad at {joint_names[max_error_joint]}")
                    print(f"       Internal target: {target_qpos_rad}")
                    print(f"       Real position:   {real_qpos_rad}")
                
                # Sync: update internal state to match real robot
                new_gpos = lerobot_FK(real_qpos_rad[1:5], robot=self.robot_kin)
                
                if arm_name == "LEFT":
                    self.left_target_qpos_rad = real_qpos_rad.copy()
                    self.left_target_gpos = new_gpos
                else:
                    self.right_target_qpos_rad = real_qpos_rad.copy()
                    self.right_target_gpos = new_gpos
                
                if self.debug:
                    print(f"       Synced gpos:     {new_gpos}")
                    print(f"[SYNC-{arm_name}] Internal state synchronized with real robot")
    
    def _calibrate_normalization(self, qpos_normalized: np.ndarray, qpos_rad: np.ndarray, 
                                  arm_name: str, norm_offset: np.ndarray, is_calibrated: bool) -> tuple:
        """Calibrate normalization offset for one arm"""
        qpos_normalized_recovered = self._radians_to_normalized(qpos_rad, np.zeros(6), False)
        offset = qpos_normalized - qpos_normalized_recovered
        
        print(f"\n[CALIBRATION-{arm_name}] Normalization calibration complete!")
        print(f"              Original normalized:  {qpos_normalized}")
        print(f"              Recovered normalized: {qpos_normalized_recovered}")
        print(f"              Offset:               {offset}")
        
        max_offset = np.max(np.abs(offset))
        if max_offset > 1.0:
            print(f"              WARNING: Large offset detected ({max_offset:.2f})!")
        else:
            print(f"              Offset is small (max={max_offset:.4f}), calibration looks good.")
        
        return offset, True

    def _extract_arm_qpos_from_obs(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract left/right 6D normalized joint positions from robot observation."""
        left_qpos = np.array([obs[m + '.pos'] for m in self._left_motors], dtype=np.float64)
        right_qpos = np.array([obs[m + '.pos'] for m in self._right_motors], dtype=np.float64)
        return left_qpos, right_qpos

    def _read_joint_state(self):
        """Read the latest robot joint state in both normalized and radian space."""
        obs = self._robot.get_observation()
        left_qpos, right_qpos = self._extract_arm_qpos_from_obs(obs)
        left_qpos_rad = self._normalized_to_radians(left_qpos)
        right_qpos_rad = self._normalized_to_radians(right_qpos)
        return obs, left_qpos, right_qpos, left_qpos_rad, right_qpos_rad

    def _clear_pending_action(self) -> None:
        """Drop any queued action so reset is not immediately overwritten."""
        pending_lock = getattr(self, "_pending_action_lock", None)
        if pending_lock is not None and hasattr(self, "_pending_action"):
            with pending_lock:
                self._pending_action = None

    def reset(self):
        """Return both arms to the configured home pose with joint interpolation."""
        self._clear_pending_action()
        if self.left_home_qpos is None or self.right_home_qpos is None:
            raise RuntimeError("home_qpos is not configured; cannot reset robot.")

        _, left_qpos, right_qpos, left_qpos_rad, right_qpos_rad = self._read_joint_state()

        home_left_qpos = self.left_home_qpos.copy()
        home_right_qpos = self.right_home_qpos.copy()
        home_left_qpos_rad = self.left_home_qpos_rad.copy()
        home_right_qpos_rad = self.right_home_qpos_rad.copy()

        max_rad_delta = max(
            float(np.max(np.abs(home_left_qpos_rad - left_qpos_rad))),
            float(np.max(np.abs(home_right_qpos_rad - right_qpos_rad))),
        )
        step_rad = max(float(self.reset_step_rad), 1e-4)
        n_steps = max(1, int(np.ceil(max_rad_delta / step_rad)))

        logger.info(
            "[BiSo101Follower] Resetting to home pose with {} interpolation steps "
            "(max_delta={:.4f} rad, step={:.4f} rad)",
            n_steps,
            max_rad_delta,
            step_rad,
        )

        for alpha in np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)[1:]:
            target_left = left_qpos + (home_left_qpos - left_qpos) * alpha
            target_right = right_qpos + (home_right_qpos - right_qpos) * alpha
            self.publish_action(np.concatenate([target_left, target_right]))
            time.sleep(self.reset_step_sleep)

        self.publish_action(np.concatenate([home_left_qpos, home_right_qpos]))
        time.sleep(self.reset_final_hold_s)

        try:
            _, _, _, final_left_qpos_rad, final_right_qpos_rad = self._read_joint_state()
        except Exception:
            final_left_qpos_rad = home_left_qpos_rad
            final_right_qpos_rad = home_right_qpos_rad

        with self._lock:
            self.left_target_qpos_rad = final_left_qpos_rad.copy()
            self.right_target_qpos_rad = final_right_qpos_rad.copy()
            self.left_target_gpos = lerobot_FK(final_left_qpos_rad[1:5], robot=self.robot_kin)
            self.right_target_gpos = lerobot_FK(final_right_qpos_rad[1:5], robot=self.robot_kin)

        self._last_left_real_qpos_rad = final_left_qpos_rad.copy()
        self._last_right_real_qpos_rad = final_right_qpos_rad.copy()
        logger.info("[BiSo101Follower] Reset to home pose finished")
    
    def get_observation(self):
        """Get complete observation data (including camera images)"""
        try:
            obs, left_qpos, right_qpos, left_qpos_rad, right_qpos_rad = self._read_joint_state()
            self._last_left_real_qpos_rad = left_qpos_rad.copy()
            self._last_right_real_qpos_rad = right_qpos_rad.copy()
            
            # Update internal state for delta_ee mode
            if self.control_mode == "delta_ee":
                # Calibration for each arm
                if not self._left_is_norm_calibrated:
                    self._left_norm_offset, self._left_is_norm_calibrated = self._calibrate_normalization(
                        left_qpos, left_qpos_rad, "LEFT", self._left_norm_offset, self._left_is_norm_calibrated)
                
                if not self._right_is_norm_calibrated:
                    self._right_norm_offset, self._right_is_norm_calibrated = self._calibrate_normalization(
                        right_qpos, right_qpos_rad, "RIGHT", self._right_norm_offset, self._right_is_norm_calibrated)
                
                # First-time initialization of target state
                if self.left_target_qpos_rad is None:
                    print("\n" + "="*60)
                    print("[INIT-LEFT] Initializing left arm target state...")
                    print(f"            qpos (normalized): {left_qpos}")
                    print(f"            qpos (radians):    {left_qpos_rad}")
                    
                    with self._lock:
                        self.left_target_qpos_rad = left_qpos_rad.copy()
                        self.left_target_gpos = lerobot_FK(left_qpos_rad[1:5], robot=self.robot_kin)
                    
                    print(f"            Initial gpos:      {self.left_target_gpos}")
                    
                    if self.safety_check:
                        if self._verify_fk_ik_consistency(left_qpos_rad, "LEFT"):
                            self._left_safety_verified = True
                        else:
                            print("[SAFETY-LEFT] WARNING: FK-IK consistency check FAILED!")
                    else:
                        self._left_safety_verified = True
                    print("="*60 + "\n")
                
                if self.right_target_qpos_rad is None:
                    print("\n" + "="*60)
                    print("[INIT-RIGHT] Initializing right arm target state...")
                    print(f"             qpos (normalized): {right_qpos}")
                    print(f"             qpos (radians):    {right_qpos_rad}")
                    
                    with self._lock:
                        self.right_target_qpos_rad = right_qpos_rad.copy()
                        self.right_target_gpos = lerobot_FK(right_qpos_rad[1:5], robot=self.robot_kin)
                    
                    print(f"             Initial gpos:      {self.right_target_gpos}")
                    
                    if self.safety_check:
                        if self._verify_fk_ik_consistency(right_qpos_rad, "RIGHT"):
                            self._right_safety_verified = True
                        else:
                            print("[SAFETY-RIGHT] WARNING: FK-IK consistency check FAILED!")
                    else:
                        self._right_safety_verified = True
                    print("="*60 + "\n")
                
                # Periodic state synchronization for both arms (moved outside of else!)
                if self.left_target_qpos_rad is not None and self.right_target_qpos_rad is not None:
                    self._sync_counter += 1
                    if self._sync_counter >= self._sync_interval:
                        self._sync_counter = 0
                        self._sync_state_with_real(left_qpos_rad, "LEFT")
                        self._sync_state_with_real(right_qpos_rad, "RIGHT")
            
            return {'qpos': np.concatenate([left_qpos, right_qpos]), **obs}
        except Exception as e:
            print(f"Error getting observation: {e}")
            return None
    
    def _process_arm_delta_action(self, action_7d: np.ndarray, delta_signs: np.ndarray,
                                   target_qpos_rad: np.ndarray, target_gpos: np.ndarray,
                                   norm_offset: np.ndarray, is_calibrated: bool,
                                   arm_name: str) -> tuple:
        """
        Process 7D delta action for one arm and return target normalized qpos
        
        Returns: (target_qpos_normalized, new_target_qpos_rad, new_target_gpos, success)
        """
        # Apply delta signs and scale
        action_signed = action_7d * delta_signs
        delta_pos = action_signed[0:3] * self.position_scale
        delta_rot = action_signed[3:6] * self.rotation_scale
        delta_gripper = action_signed[6] * self.gripper_scale
        
        current_gpos = target_gpos.copy()
        current_qpos_rad = target_qpos_rad.copy()
        
        # Check if all deltas are zero
        total_delta = np.sum(np.abs(delta_pos)) + np.sum(np.abs(delta_rot)) + np.abs(delta_gripper)
        if total_delta < 1e-10:
            return None, target_qpos_rad, target_gpos, True  # No change needed
        
        # Compute new target gpos (world-frame XY motion)
        angle_curr = current_qpos_rad[0]  # Base rotation
        forward_curr = current_gpos[0]    # Forward distance
        
        x_curr = forward_curr * np.cos(angle_curr)
        y_curr = forward_curr * np.sin(angle_curr)
        
        x_new = x_curr + delta_pos[0]
        y_new = y_curr + delta_pos[1]
        
        theta_update = np.arctan2(y_new, x_new) - np.arctan2(y_curr, x_curr)
        forward_update = np.sqrt(x_new**2 + y_new**2) - np.sqrt(x_curr**2 + y_curr**2)
        
        new_target_gpos = current_gpos.copy()
        new_target_gpos[0] += forward_update  # Forward
        new_target_gpos[2] += delta_pos[2]    # Height (Z)
        new_target_gpos[3] += delta_rot[0]    # Roll
        new_target_gpos[4] += delta_rot[1]    # Pitch
        
        # Compute IK for joints 1-4 (fast_mode=True - MUST be ultra-fast due to GIL)
        fd_qpos = current_qpos_rad[1:5]
        qpos_inv, ik_success = lerobot_IK(fd_qpos, new_target_gpos, robot=self.robot_kin, fast_mode=True)
        
        if not ik_success:
            # IK failed - try ONE smaller movement (0.3x scale)
            # Keep fallback minimal to avoid blocking
            scaled_gpos = current_gpos.copy()
            scaled_gpos[0] += forward_update * 0.3
            scaled_gpos[2] += delta_pos[2] * 0.3
            scaled_gpos[3] += delta_rot[0] * 0.3
            scaled_gpos[4] += delta_rot[1] * 0.3
            
            qpos_inv, ik_success = lerobot_IK(fd_qpos, scaled_gpos, robot=self.robot_kin, fast_mode=True)
            if ik_success:
                new_target_gpos = scaled_gpos
                theta_update *= 0.3
            else:
                # Still failed - skip this frame
                return None, target_qpos_rad, target_gpos, False
        
        # Build target joint positions
        new_target_qpos_rad = current_qpos_rad.copy()
        new_target_qpos_rad[0] += theta_update
        new_target_qpos_rad[1:5] = qpos_inv[:4]
        new_target_qpos_rad[5] += delta_gripper
        
        # Convert to normalized space
        target_qpos_normalized = self._radians_to_normalized(new_target_qpos_rad, norm_offset, is_calibrated)
        
        # Clip in normalized space
        target_qpos_normalized[:5] = np.clip(target_qpos_normalized[:5], -100.0, 100.0)
        target_qpos_normalized[5] = np.clip(target_qpos_normalized[5], 0.0, 100.0)
        
        return target_qpos_normalized, new_target_qpos_rad, new_target_gpos, True
    
    def process_action(self, action_dict: dict) -> dict:
        """
        Process action based on control mode
        
        For qpos mode: pass through (12D normalized values: 6 left + 6 right)
        For delta_ee mode: convert 14D delta to 12D normalized joint positions
        """
        if self.control_mode == "qpos":
            return action_dict
        
        # delta_ee mode
        action = action_dict.get('action', None)
        if action is None:
            return action_dict
        
        action = np.array(action, dtype=np.float64)
        
        # Apply deadzone threshold
        if self.zero_threshold > 0:
            action = self._apply_deadzone(action)
        
        # If action is all zeros after deadzone, skip processing entirely
        if np.allclose(action, 0):
            action_dict['action'] = None
            return action_dict
        
        # Safety check: Block if FK-IK not verified for either arm
        if self.safety_check and (not self._left_safety_verified or not self._right_safety_verified):
            self._blocked_count += 1
            if self._blocked_count % 100 == 1:
                print(f"[SAFETY] Action BLOCKED (#{self._blocked_count}): FK-IK consistency not verified. "
                      f"Left verified: {self._left_safety_verified}, Right verified: {self._right_safety_verified}")
            action_dict['action'] = None
            return action_dict
        
        if len(action) != 14:
            print(f"[BiSo101Follower] Warning: delta_ee mode expects 14D action, got {len(action)}D")
            return action_dict
        
        # Split action for left and right arms
        left_action = action[:7]
        right_action = action[7:]
        
        with self._lock:
            if self.left_target_qpos_rad is None or self.right_target_qpos_rad is None:
                return action_dict
            
            left_target_qpos_rad = self.left_target_qpos_rad.copy()
            left_target_gpos = self.left_target_gpos.copy()
            right_target_qpos_rad = self.right_target_qpos_rad.copy()
            right_target_gpos = self.right_target_gpos.copy()
        
        # Process left arm
        left_qpos_normalized, new_left_qpos_rad, new_left_gpos, left_success = self._process_arm_delta_action(
            left_action, self.left_delta_signs, left_target_qpos_rad, left_target_gpos,
            self._left_norm_offset, self._left_is_norm_calibrated, "LEFT"
        )
        
        # Process right arm
        right_qpos_normalized, new_right_qpos_rad, new_right_gpos, right_success = self._process_arm_delta_action(
            right_action, self.right_delta_signs, right_target_qpos_rad, right_target_gpos,
            self._right_norm_offset, self._right_is_norm_calibrated, "RIGHT"
        )
        
        # Update internal target state
        with self._lock:
            if left_success and left_qpos_normalized is not None:
                self.left_target_qpos_rad = new_left_qpos_rad.copy()
                self.left_target_gpos = new_left_gpos.copy()
            elif not left_success and np.any(left_action != 0):
                # IK failed on non-zero action - reset to real robot position
                if self._last_left_real_qpos_rad is not None:
                    self.left_target_qpos_rad = self._last_left_real_qpos_rad.copy()
                    self.left_target_gpos = lerobot_FK(self._last_left_real_qpos_rad[1:5], robot=self.robot_kin)
            
            if right_success and right_qpos_normalized is not None:
                self.right_target_qpos_rad = new_right_qpos_rad.copy()
                self.right_target_gpos = new_right_gpos.copy()
            elif not right_success and np.any(right_action != 0):
                # IK failed on non-zero action - reset to real robot position
                if self._last_right_real_qpos_rad is not None:
                    self.right_target_qpos_rad = self._last_right_real_qpos_rad.copy()
                    self.right_target_gpos = lerobot_FK(self._last_right_real_qpos_rad[1:5], robot=self.robot_kin)
        
        # Combine actions for both arms
        if left_qpos_normalized is None and right_qpos_normalized is None:
            # Both arms have no change
            action_dict['action'] = None
        else:
            # Use current normalized position for arms with no change
            if left_qpos_normalized is None:
                left_qpos_normalized = self._radians_to_normalized(
                    left_target_qpos_rad, self._left_norm_offset, self._left_is_norm_calibrated)
            if right_qpos_normalized is None:
                right_qpos_normalized = self._radians_to_normalized(
                    right_target_qpos_rad, self._right_norm_offset, self._right_is_norm_calibrated)
            
            action_dict['action'] = np.concatenate([left_qpos_normalized, right_qpos_normalized])
        
        # Debug output
        if self.debug and action_dict.get('action') is not None:
            self._debug_step += 1
            if self._debug_step % 10 == 1:
                print(f"\n[DEBUG step {self._debug_step}] ================")
                print(f"  Input action (14D): {action}")
                print(f"  Left arm action:    {left_action}")
                print(f"  Right arm action:   {right_action}")
                print(f"  ---")
                print(f"  Left target gpos:   {new_left_gpos}")
                print(f"  Right target gpos:  {new_right_gpos}")
                if action_dict.get('action') is not None:
                    output = action_dict['action']
                    print(f"  Output (12D norm):  L={output[:6]}, R={output[6:]}")
        
        return action_dict
    
    @classmethod
    def obs2meta(cls, device_data):
        """Extract robot state from this device's SHM data.

        Args:
            device_data: dict from SharedMemoryChannel (motor positions, etc.)

        Returns:
            dict with 'state' key containing qpos array, or empty dict.
        """
        if device_data is None:
            return {}
        left_qpos = np.array([device_data[m + '.pos'] for m in cls._left_motors], dtype=np.float32)
        right_qpos = np.array([device_data[m + '.pos'] for m in cls._right_motors], dtype=np.float32)
        return {'state': np.concatenate([left_qpos, right_qpos])}
    
    def start(self):
        """
        Start the robot with async IK processing.
        
        Uses multiprocessing to run IK computation in a separate process,
        preventing it from blocking observation publishing due to GIL.
        
        Main loop (this process): read observation -> write to SHM -> read processed action -> execute
        IK process: read raw action -> compute IK -> write processed action
        """
        from vlastudio.deploy.utils import RateLimiter
        from loguru import logger
        
        # Create shared memory for robot observations
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)
        
        # Connect to control shm (teleop) with retry
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
        
        # Use sync mode (async IK has too much multiprocessing overhead)
        # IK with fast_mode=True is fast enough for real-time control
        self._start_sync()
    
    def _start_sync(self):
        """
        Start with observation publishing decoupled from IK computation.
        
        Uses a background thread for IK to prevent blocking observation publishing.
        This is different from full async (multiprocessing) - we use threading here
        which has lower overhead but is still affected by GIL. However, since we
        release GIL during sleep and I/O, observations can still be published
        while IK is computing.
        """
        from vlastudio.deploy.utils import RateLimiter
        from loguru import logger
        import queue
        
        self.is_running = True
        rate_limiter = RateLimiter()
        
        # Queue for passing actions to IK thread
        action_queue = queue.Queue(maxsize=1)
        # Store latest processed action
        self._pending_action = None
        self._pending_action_lock = threading.Lock()
        
        def ik_worker():
            """Background thread for IK computation"""
            while self.is_running:
                try:
                    # Get action with timeout to allow checking is_running
                    try:
                        raw_action = action_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    
                    # Process action (IK computation)
                    processed = self.process_action(raw_action)
                    action_array = processed.get('action', None) if processed else None
                    
                    # Store result
                    with self._pending_action_lock:
                        self._pending_action = action_array
                        
                except Exception as e:
                    print(f"[IK Thread] Error: {e}")
        
        # Start IK worker thread
        ik_thread = threading.Thread(target=ik_worker, daemon=True)
        ik_thread.start()
        logger.info(f"[{self.name}] Started IK worker thread")
        
        try:
            while self.is_running:
                # 1. Get and publish observation (NEVER blocked by IK)
                data = self.get_data()
                if data is not None:
                    self.write_data(data)

                # 2. Read control message / raw action and handle reset commands first
                raw_action = self.read_action()
                if self.handle_control_message(raw_action):
                    rate_limiter.sleep(self.fps)
                    continue
                
                # 3. Execute any pending processed action
                with self._pending_action_lock:
                    if self._pending_action is not None:
                        self.publish_action(self._pending_action)
                        self._pending_action = None
                
                # 4. Read new raw action and send to IK thread (non-blocking)
                if raw_action is not None:
                    try:
                        # Clear old action, keep only latest
                        while not action_queue.empty():
                            try:
                                action_queue.get_nowait()
                            except queue.Empty:
                                break
                        action_queue.put_nowait(raw_action)
                    except queue.Full:
                        pass  # Skip if queue full
                
                rate_limiter.sleep(self.fps)
        finally:
            self.is_running = False
            ik_thread.join(timeout=1.0)
    
    def _start_async_ik(self):
        """
        Async start with IK in separate process.
        
        Observation publishing is never blocked by IK computation.
        """
        from vlastudio.deploy.utils import RateLimiter
        from loguru import logger
        
        # Create queues for IK process communication
        action_in_queue = mp.Queue(maxsize=2)
        action_out_queue = mp.Queue(maxsize=2)
        
        # Start IK worker process
        ik_process = mp.Process(
            target=self._ik_worker,
            args=(action_in_queue, action_out_queue),
            daemon=True
        )
        ik_process.start()
        logger.info(f"[{self.name}] Started async IK process")
        
        self.is_running = True
        rate_limiter = RateLimiter()
        pending_result = None  # Latest result from IK worker
        
        try:
            while self.is_running:
                # 1. Get and publish observation (NEVER blocked by IK)
                data = self.get_data()
                if data is not None:
                    self.write_data(data)

                # 2. Handle control commands before executing / queuing actions
                raw_action = self.read_action()
                if self.handle_control_message(raw_action):
                    pending_result = None
                    rate_limiter.sleep(self.fps)
                    continue
                
                # 3. Check for processed result from IK process (non-blocking)
                try:
                    while not action_out_queue.empty():
                        pending_result = action_out_queue.get_nowait()
                except:
                    pass
                
                # 4. Execute pending action and update state
                if pending_result is not None:
                    processed_action = pending_result.get('action', {})
                    action_array = processed_action.get('action', None) if processed_action else None
                    
                    if action_array is not None:
                        self.publish_action(action_array)
                    
                    # Update internal state from IK worker result
                    with self._lock:
                        if pending_result.get('left_target_qpos_rad') is not None:
                            self.left_target_qpos_rad = np.array(pending_result['left_target_qpos_rad'])
                            self.left_target_gpos = np.array(pending_result['left_target_gpos'])
                        if pending_result.get('right_target_qpos_rad') is not None:
                            self.right_target_qpos_rad = np.array(pending_result['right_target_qpos_rad'])
                            self.right_target_gpos = np.array(pending_result['right_target_gpos'])
                    
                    pending_result = None
                
                # 5. Read new raw action and send to IK process with state (non-blocking)
                if raw_action is not None:
                    # Package action with current state for IK worker
                    with self._lock:
                        if self.left_target_qpos_rad is not None and self.right_target_qpos_rad is not None:
                            msg = {
                                'action': raw_action,
                                'left_target_qpos_rad': self.left_target_qpos_rad.copy(),
                                'left_target_gpos': self.left_target_gpos.copy(),
                                'right_target_qpos_rad': self.right_target_qpos_rad.copy(),
                                'right_target_gpos': self.right_target_gpos.copy(),
                                'left_norm_offset': self._left_norm_offset.copy(),
                                'right_norm_offset': self._right_norm_offset.copy(),
                                'left_is_norm_calibrated': self._left_is_norm_calibrated,
                                'right_is_norm_calibrated': self._right_is_norm_calibrated,
                                # Also send real robot state for fallback reset
                                'left_real_qpos_rad': self._last_left_real_qpos_rad.copy() if self._last_left_real_qpos_rad is not None else None,
                                'right_real_qpos_rad': self._last_right_real_qpos_rad.copy() if self._last_right_real_qpos_rad is not None else None,
                            }
                            try:
                                # Clear old messages, keep only latest
                                while not action_in_queue.empty():
                                    try:
                                        action_in_queue.get_nowait()
                                    except:
                                        break
                                action_in_queue.put_nowait(msg)
                            except:
                                pass  # Queue full, skip
                
                rate_limiter.sleep(self.fps)
        finally:
            # Cleanup
            action_in_queue.put(None)  # Signal worker to stop
            ik_process.join(timeout=1.0)
            if ik_process.is_alive():
                ik_process.terminate()
            logger.info(f"[{self.name}] Stopped async IK process")
    
    def _ik_worker(self, action_in_queue: Queue, action_out_queue: Queue):
        """
        IK worker process - runs in separate process to avoid GIL.
        
        Receives messages containing raw action + state, computes IK, outputs processed actions.
        """
        # Re-initialize kinematics in this process
        from vlastudio.deploy.robot.so101.kinematics import create_so101, lerobot_FK, lerobot_IK
        self.robot_kin = create_so101()
        
        # Skip safety check in worker (verified in main process)
        self._left_safety_verified = True
        self._right_safety_verified = True
        
        while True:
            try:
                # Get message (blocking with timeout)
                try:
                    msg = action_in_queue.get(timeout=0.1)
                except:
                    continue
                
                if msg is None:  # Shutdown signal
                    break
                
                # Unpack state from message
                raw_action = msg['action']
                self.left_target_qpos_rad = msg['left_target_qpos_rad']
                self.left_target_gpos = msg['left_target_gpos']
                self.right_target_qpos_rad = msg['right_target_qpos_rad']
                self.right_target_gpos = msg['right_target_gpos']
                self._left_norm_offset = msg['left_norm_offset']
                self._right_norm_offset = msg['right_norm_offset']
                self._left_is_norm_calibrated = msg['left_is_norm_calibrated']
                self._right_is_norm_calibrated = msg['right_is_norm_calibrated']
                left_real_qpos_rad = msg.get('left_real_qpos_rad', None)
                right_real_qpos_rad = msg.get('right_real_qpos_rad', None)
                
                # Check if input action is non-zero (before processing)
                input_action = raw_action.get('action', None)
                has_nonzero_input = input_action is not None and not np.allclose(input_action, 0)
                
                # Process action (IK computation)
                processed_action = self.process_action(raw_action)
                
                # If IK failed AND there was a non-zero input, reset internal state
                # Don't reset for zero actions (that's normal - no movement requested)
                action_array = processed_action.get('action', None) if processed_action else None
                ik_failed = action_array is None and has_nonzero_input
                if ik_failed:
                    # IK failed on real movement attempt - reset state to real robot position
                    reset_done = False
                    if left_real_qpos_rad is not None:
                        self.left_target_qpos_rad = left_real_qpos_rad.copy()
                        self.left_target_gpos = lerobot_FK(left_real_qpos_rad[1:5], robot=self.robot_kin)
                        reset_done = True
                    if right_real_qpos_rad is not None:
                        self.right_target_qpos_rad = right_real_qpos_rad.copy()
                        self.right_target_gpos = lerobot_FK(right_real_qpos_rad[1:5], robot=self.robot_kin)
                        reset_done = True
                    if reset_done:
                        print(f"[IK Worker] IK failed on movement -> RESET state to real robot position", flush=True)
                
                # Package result with updated state
                result = {
                    'action': processed_action,
                    'left_target_qpos_rad': self.left_target_qpos_rad,
                    'left_target_gpos': self.left_target_gpos,
                    'right_target_qpos_rad': self.right_target_qpos_rad,
                    'right_target_gpos': self.right_target_gpos,
                }
                
                # Output result (non-blocking, keep latest)
                try:
                    while not action_out_queue.empty():
                        try:
                            action_out_queue.get_nowait()
                        except:
                            break
                    action_out_queue.put_nowait(result)
                except:
                    pass
                    
            except Exception as e:
                import traceback
                print(f"[IK Worker] Error: {e}", flush=True)
                traceback.print_exc()
                continue
    
    def shutdown(self):
        """Shutdown robot and cameras"""
        if self._robot.is_connected:
            self._robot.disconnect()
    
    def close(self):
        """Close robot"""
        super().close()
        if self._robot.is_connected:
            self._robot.disconnect()
    
    def publish_action(self, action: np.ndarray):
        """Publish action to robot (12D: 6 left + 6 right)"""
        try:
            left_action = {mname+'.pos': action[i] for i, mname in enumerate(self._left_motors)}
            right_action = {mname+'.pos': action[i+len(self._left_motors)] for i, mname in enumerate(self._right_motors)}
            action_dict = {**left_action, **right_action}
            self._robot.send_action(action_dict)
        except Exception as e:
            pass
    
    def is_running(self):
        """Check if robot is running"""
        return self._robot.is_connected
    
    def save_episode(self, file_path: str, observations: list, actions: list):
        """Save episode data to HDF5 file"""
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
                        group.create_dataset('data', data=np.stack(data_list))
                    else:
                        group.create_dataset(key_prefix, data=np.stack(data_list))
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not stack data for key '{key_prefix}'. Skipping. Error: {e}")
        
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('actions', data=np.array(actions, dtype=np.float32))
            obs_group = f.create_group('observations')
            if observations:
                write_group(obs_group, observations)


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    import importlib
    import threading
    import yaml
    from pathlib import Path

    from vlastudio.deploy.shm_utils import SharedMemoryChannel

    # Load config
    cfg_path = Path(__file__).resolve().parents[3] / "configs" / "robot" / "biso101.yaml"
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
    device_config = raw[0] if isinstance(raw, list) else raw

    # Create device in main process
    device_type = device_config["type"]
    module_name, class_name = device_type.rsplit(".", 1)
    module = importlib.import_module(module_name)
    device_class = getattr(module, class_name)
    device = device_class(**device_config["args"])

    shm_name = device_config["args"].get("name", device_config["args"]["robot_id"])
    start_thread = threading.Thread(target=device.start, daemon=True)
    start_thread.start()

    time.sleep(0.5)
    print("Reading from BiSo101 Follower SHM (Ctrl+C to stop)...")

    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=15.0)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None:
                arr = data.get("qpos")
                if arr is not None:
                    n = len(arr)
                    if n >= 12:
                        left, right = arr[:6], arr[6:12]
                        print(
                            f"  left: [{left[0]:.3f}, {left[1]:.3f}, {left[2]:.3f}, "
                            f"{left[3]:.3f}, {left[4]:.3f}, {left[5]:.3f}]  "
                            f"right: [{right[0]:.3f}, {right[1]:.3f}, {right[2]:.3f}, "
                            f"{right[3]:.3f}, {right[4]:.3f}, {right[5]:.3f}]",
                            end="\r",
                            flush=True,
                        )
                    else:
                        print(f"  qpos: {arr}", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        device.close()
        start_thread.join(timeout=2.0)
        print("Done.")
