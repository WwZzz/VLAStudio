#!/usr/bin/env python3
"""
Quest3 VR Teleoperator - VR teleoperation controller for ILStudio
Uses Quest3 VR headset for robot teleoperation via XLeVR
"""

import os
import sys
import asyncio
import threading
import time
import numpy as np
from typing import Optional, Literal, Dict, Any, Union, Sequence
from scipy.spatial.transform import Rotation as R

from loguru import logger

from deploy.teleoperator.base import BaseTeleopDevice
from deploy.teleoperator.quest_keep_awake import QuestKeepAwake
from deploy.utils import RateLimiter

# Set the absolute path to the XLeVR folder
XLEVR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils', 'XLeVR'))


def setup_xlevr_environment():
    """Setup XLeVR environment"""
    if XLEVR_PATH not in sys.path:
        sys.path.insert(0, XLEVR_PATH)
    os.chdir(XLEVR_PATH)
    os.environ['PYTHONPATH'] = f"{XLEVR_PATH}:{os.environ.get('PYTHONPATH', '')}"


class Quest3Teleop(BaseTeleopDevice):
    """
    Quest3 VR Teleoperator for ILStudio.
    
    Captures VR controller poses and converts them to end-effector pose deltas.
    
    VR Controller Button Mapping:
    =============================
    - unsqueeze (side grip): Enables pose delta transmission.
      Hold to move the robot arm. When released, pose deltas are zeroed.
      In code: is_unsqueeze_pressed, _left_unsqueeze_active, _right_unsqueeze_active
      
    - GRIPPER (YAML: gripper_use_trigger, gripper_trigger_mode, gripper_a_maps_to_open, gripper_b_maps_to_close):
      * Quest 扳机: WebXR gamepad.buttons[0].value 为 0~1 模拟量（见 XLeVR vr_app.js）；无则回退二值。
      * gripper_trigger_mode (仅 gripper_delta_mode=True): delta_binary | analog_travel | a_open_trigger_close
        - analog_travel: 与 delta_binary 在松/满端点一致，中间行程连续。
        - a_open_trigger_close: A/X 张、扳机（>0.5）合；无可靠模拟量时用此，并设 gripper_a_maps_to_open: true。
      * Alicia 典型: analog_travel + A 不参与夹爪；B = 仅 go_home。
      * Freeze: 不读扳机夹爪；B/Y 不作夹爪闭合。
    
    Action format:
        Single arm: [dx, dy, dz, droll, dpitch, dyaw, gripper] (7D)
        Dual arm: [left_dx, left_dy, left_dz, left_droll, left_dpitch, left_dyaw, left_gripper,
                   right_dx, right_dy, right_dz, right_droll, right_dpitch, right_dyaw, right_gripper] (14D)
    """
    
    # Supported teleop modes
    TELEOP_MODES = ("delta_ee", "rel_ee")
    GRIPPER_TRIGGER_MODES = frozenset({"delta_binary", "analog_travel", "a_open_trigger_close"})
    
    def __init__(
        self,
        name: str = "quest3_teleop",
        max_size_mb: int = 1,
        fps: float = 50.0,
        mode: str = 'delta_ee',
        arm_mode: Literal["left", "right", "dual"] = "dual",
        position_scale: Union[float, Sequence[float]] = 1.0,
        rotation_scale: float = 1.0,
        connect_timeout: float = 60.0,
        calibration_transform: Optional[list] = None,
        gripper_default_closed: bool = True,
        gripper_delta_mode: bool = False,
        gripper_delta_value: float = 1.0,
        gripper_button_control: bool = True,  # Ignored; kept for YAML compatibility
        gripper_use_trigger: bool = False,
        gripper_a_maps_to_open: bool = True,
        gripper_b_maps_to_close: bool = True,
        gripper_trigger_mode: Literal["delta_binary", "analog_travel", "a_open_trigger_close"] = "delta_binary",
        gripper_trigger_analog_scale: float = 1.0,
        gripper_trigger_analog_deadband: float = 0.06,
        # If True: swap open↔close on the gripper channel (absolute: g→1-g; delta: g→-g).
        # Use with absolute mode so trigger released = closed when hardware expects it.
        gripper_invert: bool = False,
        debug: bool = False,
        gripper_trace: bool = False,
        keep_screen_awake: bool = True,
        keep_screen_awake_interval: float = 5.0,
        publish_thumbsticks: bool = False,
        adb_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Quest3 VR teleoperator.
        
        Args:
            name: Name for shared memory
            max_size_mb: Max shared memory size in MB
            fps: Control frequency in Hz
            mode: Teleop mode:
                  - "delta_ee": Frame-to-frame delta mode. Each action is the pose change 
                                between consecutive frames.
                  - "rel_ee": Relative to anchor mode. When unsqueeze is activated, the VR 
                              pose at that moment becomes the anchor. All subsequent actions 
                              are relative poses from the anchor until re-freeze.
            arm_mode: "left", "right", or "dual" for which arm(s) to control
            position_scale: Position scale after calibration_transform (robot XYZ).
                            float → all axes; [sx, sy, sz] → per-axis (Z usually up/down).
            rotation_scale: Scale factor for rotation delta
            connect_timeout: Timeout in seconds for VR connection
            calibration_transform: 3x3 matrix to transform VR coordinates to robot coordinates
                                   Applied to both position (as linear transform) and rotation
                                   (as similarity transform: R_robot = T @ R_vr @ T^T)
                                   This matrix handles both axis remapping AND sign corrections.
                                   If None, no transformation is applied (identity matrix)
            gripper_default_closed: With trigger+delta: trigger pressed vs released open/close convention
            gripper_delta_mode: If True, per-frame deltas; else absolute 0/1 hints
            gripper_delta_value: Step size for delta mode
            gripper_button_control: Ignored (kept for YAML compatibility)
            gripper_use_trigger: If True, trigger drives gripper while unsqueeze (ignored while frozen)
            gripper_a_maps_to_open: If False, A/X do not add open delta (trigger-only gripper)
            gripper_b_maps_to_close: If False, B/Y never emit close delta (use B for go_home only on Alicia)
            gripper_trigger_mode: delta_binary | analog_travel | a_open_trigger_close (delta mode only)
            gripper_trigger_analog_scale: Scales analog_travel contribution
            gripper_trigger_analog_deadband: |tr-0.5| below this → no trigger delta (analog_travel)
            gripper_invert: Invert open/close on output (absolute: 1-g; delta: -g)
            debug: Enable debug output
            gripper_trace: Log [GripperTrace:teleop] on each SHM write (pair with robot gripper_trace)
            keep_screen_awake: Try to keep Quest display awake via adb prox_close while teleoping
            keep_screen_awake_interval: Seconds between keep-awake pulses
            adb_path: Optional explicit path to adb binary
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps)
        
        # Validate and store teleop mode
        if mode not in self.TELEOP_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {self.TELEOP_MODES}")
        self.mode = mode
        
        self.arm_mode = arm_mode
        self.position_scale = self._normalize_position_scale(position_scale)
        self.rotation_scale = rotation_scale
        self.connect_timeout = connect_timeout
        self.gripper_default_closed = gripper_default_closed
        self.gripper_delta_mode = gripper_delta_mode
        self.gripper_delta_value = gripper_delta_value
        self.gripper_button_control = gripper_button_control
        self.gripper_use_trigger = gripper_use_trigger
        self.gripper_a_maps_to_open = gripper_a_maps_to_open
        self.gripper_b_maps_to_close = gripper_b_maps_to_close
        if gripper_trigger_mode not in self.GRIPPER_TRIGGER_MODES:
            raise ValueError(
                f"Invalid gripper_trigger_mode '{gripper_trigger_mode}'. "
                f"Must be one of {sorted(self.GRIPPER_TRIGGER_MODES)}"
            )
        self.gripper_trigger_mode = gripper_trigger_mode
        self.gripper_trigger_analog_scale = float(gripper_trigger_analog_scale)
        self.gripper_trigger_analog_deadband = float(gripper_trigger_analog_deadband)
        self.gripper_invert = bool(gripper_invert)
        self.debug = debug
        self.gripper_trace = gripper_trace
        self.keep_screen_awake = keep_screen_awake
        self.publish_thumbsticks = bool(publish_thumbsticks)
        self._keep_awake = QuestKeepAwake(
            enabled=keep_screen_awake,
            interval_s=keep_screen_awake_interval,
            adb_path=adb_path,
        )
        self._closed = False
        
        # Calibration transform matrix (3x3)
        # Used for both position and rotation coordinate transformation
        # Applied AFTER left-handed to right-handed conversion (if enabled)
        # This matrix should include both axis remapping AND any sign corrections
        if calibration_transform is not None:
            self.calibration_transform = np.array(calibration_transform, dtype=np.float64)
            if self.calibration_transform.shape != (3, 3):
                raise ValueError(f"calibration_transform must be 3x3 matrix, got {self.calibration_transform.shape}")
        else:
            self.calibration_transform = np.eye(3)  # Identity matrix (no transform)
        
        # VR Monitor instance
        self.vr_monitor = None
        self._vr_thread = None
        self._async_loop = None
        
        # State tracking for delta calculation (delta_ee mode: prev frame, rel_ee mode: anchor)
        self._left_prev_position = None
        self._left_prev_quaternion = None
        self._left_unsqueeze_active = False  # VR side grip state - enables pose transmission
        self._left_last_goal_id = None  # Track if goal changed
        
        self._right_prev_position = None
        self._right_prev_quaternion = None
        self._right_unsqueeze_active = False  # VR side grip state - enables pose transmission
        self._right_last_goal_id = None  # Track if goal changed
        
        # B button go-home: edge detection (only trigger once per press)
        self._right_b_was_pressed = False
        self._left_b_was_pressed = False
        
        # Anchor positions for rel_ee mode (set when unsqueeze is first activated)
        self._left_anchor_position = None
        self._left_anchor_quaternion = None
        self._right_anchor_position = None
        self._right_anchor_quaternion = None
        
        # Legacy gripper tracking (unused for output; gripper deltas come from buttons each frame)
        self._left_gripper = 0.0
        self._right_gripper = 0.0
        
        # Action dimension
        if arm_mode == "dual":
            self.action_dim = 14  # 7 per arm
        else:
            self.action_dim = 7
    
    def _run_vr_monitor_async(self):
        """Run VR monitor in a separate thread with its own event loop."""
        self._async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._async_loop)
        
        try:
            self._async_loop.run_until_complete(self.vr_monitor.start_monitoring())
        except Exception as e:
            if self.debug:
                print(f"[Quest3Teleop] VR monitor stopped: {e}")
        finally:
            self._async_loop.close()
    
    def connect(self) -> bool:
        """
        Connect to VR headset and wait for pose data.
        
        Returns:
            True if connected successfully, False otherwise
        """
        print("[Quest3Teleop] Initializing VR connection...")

        # Best-effort: keep Quest awake while removed (adb). Warns at most once on failure.
        self._keep_awake.start()
        
        # Setup XLeVR environment
        setup_xlevr_environment()
        
        # Import VRMonitor from the local module
        from vr_monitor import VRMonitor
        
        # Create VR monitor
        self.vr_monitor = VRMonitor()
        
        # Start VR monitor in background thread
        self._vr_thread = threading.Thread(target=self._run_vr_monitor_async, daemon=True)
        self._vr_thread.start()
        
        # Wait for VR connection (check for headset data)
        print("[Quest3Teleop] Waiting for VR headset connection...")
        start_time = time.time()
        
        while time.time() - start_time < self.connect_timeout:
            # Check if headset data is available
            goals = self.vr_monitor.get_latest_goal_nowait()
            if goals and goals.get("has_headset", False):
                print("[Quest3Teleop] VR headset connected successfully!")
                return True
            
            time.sleep(0.1)
        
        print(f"[Quest3Teleop] VR connection timeout after {self.connect_timeout}s")
        self._keep_awake.stop()
        return False
    
    def get_data(self) -> Optional[dict]:
        """
        Get VR controller data.
        
        Returns:
            Dictionary containing VR pose data, or None if not available
        """
        if self.vr_monitor is None:
            return None
        
        goals = self.vr_monitor.get_latest_goal_nowait()
        if goals is None:
            return None
        
        return {
            "left": goals.get("left"),
            "right": goals.get("right"),
            "headset": goals.get("headset"),
            "has_left": goals.get("has_left", False),
            "has_right": goals.get("has_right", False),
            "has_headset": goals.get("has_headset", False),
        }

    @staticmethod
    def _normalize_position_scale(scale: Union[float, Sequence[float]]) -> np.ndarray:
        """Return (3,) robot-frame XYZ scales. Scalar broadcasts to all axes."""
        if isinstance(scale, (list, tuple, np.ndarray)):
            arr = np.asarray(scale, dtype=np.float64).reshape(-1)
            if arr.size != 3:
                raise ValueError(
                    f"position_scale list must have 3 elements [sx, sy, sz], got {arr.size}"
                )
            return arr
        return np.full(3, float(scale), dtype=np.float64)

    def _apply_gripper_invert(self, gripper_value: float) -> float:
        """Optionally swap open↔close on the gripper channel."""
        g = float(gripper_value)
        if not self.gripper_invert:
            return g
        if self.gripper_delta_mode:
            return -g
        return float(np.clip(1.0 - g, 0.0, 1.0))
    
    def _compute_arm_delta(
        self,
        goal,
        prev_position: Optional[np.ndarray],
        prev_quaternion: Optional[np.ndarray],
        unsqueeze_was_active: bool,
        is_right_hand: bool = True,
        anchor_position: Optional[np.ndarray] = None,
        anchor_quaternion: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Compute pose delta for one arm.
        
        Args:
            prev_position: Previous frame position (for delta_ee mode reference tracking)
            prev_quaternion: Previous frame quaternion (for delta_ee mode reference tracking)
            unsqueeze_was_active: Whether VR side grip was active last frame
            is_right_hand: True for right hand (A button), False for left hand (X button)
            anchor_position: Anchor position for rel_ee mode (set when unsqueeze first activated)
            anchor_quaternion: Anchor quaternion for rel_ee mode
        
        Returns:
            (delta_pose, new_position, new_quaternion, new_unsqueeze_active, gripper_value, new_anchor_pos, new_anchor_quat)
            
            - delta_pose/rel_pose: [dx, dy, dz, droll, dpitch, dyaw] (6D)
              - delta_ee mode: Frame-to-frame change, only non-zero when unsqueeze active
              - rel_ee mode: Relative to anchor pose, only non-zero when unsqueeze active
            - new_unsqueeze_active: VR side grip state (for pose transmission)
            - gripper_value: Robot GRIPPER control value
            - new_anchor_pos/quat: Updated anchor for rel_ee mode (only set on unsqueeze activation)
        """
        delta_pose = np.zeros(6)
        # Default gripper value: no change in button control mode
        gripper_value = 0.0
        new_position = prev_position
        new_quaternion = prev_quaternion
        new_unsqueeze_active = unsqueeze_was_active
        new_anchor_position = anchor_position
        new_anchor_quaternion = anchor_quaternion
        
        if goal is None:
            return delta_pose, new_position, new_quaternion, new_unsqueeze_active, gripper_value, new_anchor_position, new_anchor_quaternion
        
        # Get current position (use vr_position for raw VR coordinates)
        current_position = None
        if goal.metadata and "vr_position" in goal.metadata:
            # Use raw VR position for delta calculation
            current_position = np.array(goal.metadata["vr_position"])
        elif goal.target_position is not None:
            current_position = np.array(goal.target_position)
        
        # Get current quaternion from metadata
        current_quaternion = None
        if goal.metadata and "quaternion" in goal.metadata:
            quat = goal.metadata["quaternion"]
            if quat and isinstance(quat, dict) and all(k in quat for k in ['x', 'y', 'z', 'w']):
                # Quaternion format: [x, y, z, w]
                current_quaternion = np.array([quat['x'], quat['y'], quat['z'], quat['w']])
        
        # ============================================================
        # Squeeze (side grip) - unfreezes pose delta transmission
        # When pressed, position/rotation deltas are computed and sent
        # This is NOT the robot gripper! This is the VR controller's side button.
        # ============================================================
        is_unsqueeze_pressed = False
        if goal.metadata:
            # Check grip_active field (VR side grip state from XLeVR)
            is_unsqueeze_pressed = goal.metadata.get("grip_active", False)
            
            # Also check buttons dict for unsqueeze
            buttons = goal.metadata.get("buttons", {})
            
            # XLeVR sends "squeeze"; some paths use "unsqueeze"
            if buttons and (
                buttons.get("unsqueeze", False) or buttons.get("squeeze", False)
            ):
                is_unsqueeze_pressed = True
            
            # ============================================================
            # GRIPPER: trigger (while unsqueeze) + optional A/X open + optional B/Y close.
            # B never affects gripper when gripper_b_maps_to_close is False (go_home only).
            # Frozen: no trigger gripper; no B/Y close.
            # ============================================================
            if self.gripper_delta_mode:
                raw_tr = goal.metadata.get("trigger_value", goal.metadata.get("trigger", 0.0))
                try:
                    trv = float(raw_tr if raw_tr is not None else 0.0)
                except (TypeError, ValueError):
                    trv = 0.0
                trv = max(0.0, min(1.0, trv))
                trigger_active = bool(goal.metadata.get("trigger_active", False))
                if not trigger_active and buttons:
                    tb = buttons.get("trigger", False)
                    if isinstance(tb, (int, float)):
                        trigger_active = float(tb) > 0.5
                    else:
                        trigger_active = bool(tb)
                if not trigger_active and trv > 0.5:
                    trigger_active = True

                if is_right_hand:
                    open_button = (
                        buttons.get("a", False) or buttons.get("button_a", False) or buttons.get("A", False)
                    )
                    close_button = (
                        (buttons.get("b", False) or buttons.get("button_b", False) or buttons.get("B", False))
                        if self.gripper_b_maps_to_close
                        else False
                    )
                else:
                    open_button = (
                        buttons.get("x", False) or buttons.get("button_x", False) or buttons.get("X", False)
                    )
                    close_button = (
                        (buttons.get("y", False) or buttons.get("button_y", False) or buttons.get("Y", False))
                        if self.gripper_b_maps_to_close
                        else False
                    )
                if not is_unsqueeze_pressed:
                    close_button = False

                g = 0.0
                if self.gripper_use_trigger and is_unsqueeze_pressed:
                    mode = self.gripper_trigger_mode
                    if mode == "analog_travel":
                        if abs(trv - 0.5) < self.gripper_trigger_analog_deadband:
                            trig_cmd = 0.0
                        else:
                            # (0.5 - tr): 拉满 tr→1 与松开 tr→0 相对 delta_binary 取反后的开/合方向
                            trig_cmd = (0.5 - trv) * 2.0
                        if not self.gripper_default_closed:
                            trig_cmd = -trig_cmd
                        g += trig_cmd * self.gripper_delta_value * self.gripper_trigger_analog_scale
                    elif mode == "a_open_trigger_close":
                        if trigger_active:
                            g += self.gripper_delta_value
                    else:
                        if self.gripper_default_closed:
                            g += -self.gripper_delta_value if trigger_active else self.gripper_delta_value
                        else:
                            g += self.gripper_delta_value if trigger_active else -self.gripper_delta_value
                if open_button and self.gripper_a_maps_to_open:
                    g += self.gripper_delta_value
                if close_button and is_unsqueeze_pressed:
                    g -= self.gripper_delta_value
                if open_button and close_button and self.gripper_b_maps_to_close:
                    g = 0.0
                gripper_value = float(g)
            else:
                raw_tr = goal.metadata.get("trigger_value", goal.metadata.get("trigger", 0.0))
                try:
                    trv = float(raw_tr if raw_tr is not None else 0.0)
                except (TypeError, ValueError):
                    trv = 0.0
                trv = max(0.0, min(1.0, trv))
                trigger_active = bool(goal.metadata.get("trigger_active", False))
                if not trigger_active and buttons:
                    tb = buttons.get("trigger", False)
                    if isinstance(tb, (int, float)):
                        trigger_active = float(tb) > 0.5
                    else:
                        trigger_active = bool(tb)
                if not trigger_active and trv > 0.5:
                    trigger_active = True
                if is_right_hand:
                    open_button = (
                        buttons.get("a", False) or buttons.get("button_a", False) or buttons.get("A", False)
                    )
                    close_button = (
                        (buttons.get("b", False) or buttons.get("button_b", False) or buttons.get("B", False))
                        if self.gripper_b_maps_to_close
                        else False
                    )
                else:
                    open_button = (
                        buttons.get("x", False) or buttons.get("button_x", False) or buttons.get("X", False)
                    )
                    close_button = (
                        (buttons.get("y", False) or buttons.get("button_y", False) or buttons.get("Y", False))
                        if self.gripper_b_maps_to_close
                        else False
                    )
                if not is_unsqueeze_pressed:
                    close_button = False

                if self.gripper_use_trigger and is_unsqueeze_pressed:
                    # Absolute open amount [0,1] (1=fully open). Tracks trigger continuously.
                    if self.gripper_default_closed:
                        gripper_value = 1.0 - trv
                    else:
                        gripper_value = trv
                else:
                    # Frozen: placeholder (robot ignores when gripper_absolute and not unsqueeze)
                    gripper_value = 1.0 if not self.gripper_default_closed else 0.0
                if open_button and self.gripper_a_maps_to_open and not close_button:
                    gripper_value = 1.0
                elif close_button and not open_button:
                    gripper_value = 0.0
        
        # Handle unsqueeze (side grip) state transitions - enables pose transmission
        if current_position is not None:
            if is_unsqueeze_pressed:
                if not unsqueeze_was_active:
                    # unsqueeze just activated - set current position as reference/anchor
                    # Don't send delta on first frame
                    new_position = current_position.copy()
                    new_quaternion = current_quaternion.copy() if current_quaternion is not None else None
                    new_unsqueeze_active = True
                    
                    # For rel_ee mode: set anchor position
                    if self.mode == "rel_ee":
                        new_anchor_position = current_position.copy()
                        new_anchor_quaternion = current_quaternion.copy() if current_quaternion is not None else None
                    return delta_pose, new_position, new_quaternion, new_unsqueeze_active, gripper_value, new_anchor_position, new_anchor_quaternion
                else:
                    # unsqueeze held - compute delta/relative pose based on mode
                    if self.mode == "delta_ee" and prev_position is not None:
                        # delta_ee mode: compute delta from previous frame in PREVIOUS FRAME'S LOCAL FRAME
                        # This ensures consistent movement direction relative to hand orientation
                        
                        # Get previous frame rotation matrix (VR world -> prev local)
                        if prev_quaternion is not None:
                            R_prev_world = R.from_quat(prev_quaternion).as_matrix()
                        else:
                            R_prev_world = np.eye(3)
                        
                        # Position delta in VR world frame
                        pos_diff_world = current_position - prev_position
                        
                        # Transform to previous frame's LOCAL frame
                        pos_delta_local = R_prev_world.T @ pos_diff_world
                        
                        # Apply calibration transform, then per-axis scale in robot XYZ
                        pos_delta_robot = self.calibration_transform @ pos_delta_local
                        delta_pose[:3] = pos_delta_robot * self.position_scale
                        
                        # Rotation delta in local frame
                        if current_quaternion is not None and prev_quaternion is not None:
                            try:
                                R_current_world = R.from_quat(current_quaternion).as_matrix()
                                
                                # Relative rotation in prev frame's local:
                                # R_rel_local = R_prev^T @ R_current
                                R_rel_local = R_prev_world.T @ R_current_world
                                
                                # Apply calibration transform
                                T = self.calibration_transform
                                R_rel_robot = T @ R_rel_local @ T.T
                                
                                euler_delta_robot = R.from_matrix(R_rel_robot).as_euler('xyz', degrees=False)
                                delta_pose[3:6] = euler_delta_robot * self.rotation_scale
                            except Exception as e:
                                if self.debug:
                                    print(f"[Quest3Teleop] Rotation delta error: {e}")
                        
                        # Update reference for next frame
                        new_position = current_position.copy()
                        new_quaternion = current_quaternion.copy() if current_quaternion is not None else None
                    
                    elif self.mode == "rel_ee" and anchor_position is not None:
                        # rel_ee mode: compute relative pose from anchor in ANCHOR'S LOCAL FRAME
                        # This ensures that hand movement "forward" always maps to robot "forward"
                        # regardless of how the user rotates during freeze
                        
                        # Get anchor rotation matrix (VR world -> anchor local)
                        if anchor_quaternion is not None:
                            R_anchor_world = R.from_quat(anchor_quaternion).as_matrix()
                        else:
                            R_anchor_world = np.eye(3)
                        
                        # Position difference in VR world frame
                        pos_diff_world = current_position - anchor_position
                        
                        # Transform position difference to anchor's LOCAL frame
                        # R_anchor_world.T rotates from world to anchor-local
                        pos_rel_local = R_anchor_world.T @ pos_diff_world
                        
                        # Apply calibration transform, then per-axis scale in robot XYZ
                        # (Z is typically up/down after calibration_transform)
                        pos_rel_robot = self.calibration_transform @ pos_rel_local
                        delta_pose[:3] = pos_rel_robot * self.position_scale
                        
                        # Rotation relative to anchor in anchor's local frame
                        if current_quaternion is not None and anchor_quaternion is not None:
                            try:
                                R_current_world = R.from_quat(current_quaternion).as_matrix()
                                
                                # Relative rotation in anchor's local frame:
                                # R_rel_local = R_anchor^T @ R_current
                                # This represents how the hand has rotated relative to its initial orientation
                                R_rel_local = R_anchor_world.T @ R_current_world
                                
                                # Apply calibration transform to convert to robot frame
                                T = self.calibration_transform
                                R_rel_robot = T @ R_rel_local @ T.T
                                
                                euler_rel_robot = R.from_matrix(R_rel_robot).as_euler('xyz', degrees=False)
                                delta_pose[3:6] = euler_rel_robot * self.rotation_scale
                            except Exception as e:
                                if self.debug:
                                    print(f"[Quest3Teleop] Rotation rel error: {e}")
                        
                        # Update prev position for tracking (but anchor stays fixed)
                        new_position = current_position.copy()
                        new_quaternion = current_quaternion.copy() if current_quaternion is not None else None
                        # Anchor remains unchanged until next unsqueeze activation
                    
                    new_unsqueeze_active = True
            else:
                # unsqueeze not pressed - pose frozen, update reference for next activation
                new_position = current_position.copy()
                new_quaternion = current_quaternion.copy() if current_quaternion is not None else None
                new_unsqueeze_active = False
                # In rel_ee mode, clear anchor when unsqueeze is released (will be reset on next activation)
                if self.mode == "rel_ee":
                    new_anchor_position = None
                    new_anchor_quaternion = None
        
        return delta_pose, new_position, new_quaternion, new_unsqueeze_active, gripper_value, new_anchor_position, new_anchor_quaternion

    def _get_unsqueeze_pressed(self, goal) -> bool:
        """Extract current unsqueeze (side grip) state from goal metadata."""
        if goal is None or not goal.metadata:
            return False
        if goal.metadata.get("grip_active", False):
            return True
        buttons = goal.metadata.get("buttons", {})
        return bool(
            buttons
            and (buttons.get("unsqueeze", False) or buttons.get("squeeze", False))
        )

    @staticmethod
    def _get_thumbstick(goal) -> list[float]:
        if goal is None or not goal.metadata:
            return [0.0, 0.0]
        stick = goal.metadata.get("thumbstick", {}) or {}
        try:
            x = float(stick.get("x", 0.0))
            y = float(stick.get("y", 0.0))
        except (TypeError, ValueError, AttributeError):
            return [0.0, 0.0]
        return [float(np.clip(x, -1.0, 1.0)), float(np.clip(y, -1.0, 1.0))]
    
    def convert_data_to_action(self, data: dict) -> tuple:
        """
        Convert VR data to robot action.
        
        Args:
            data: Dictionary containing VR pose data
            
        Returns:
            (action_dict, should_write): Action dict and whether to write to shm
            action_dict contains:
                - action: The action array
                - unsqueeze_active: Whether VR is currently transmitting (side grip pressed)
                - anchor_just_set: Whether anchor was just set this frame (first frame of unsqueeze)
            should_write is True whenever there's new VR data
            
        Note:
            - unsqueeze (side grip): Enables pose delta transmission
            - Gripper: A/X open, B/Y close only; trigger ignored.
            - B (right) / Y (left) rising edge = go_home (no side-grip required)
        """
        should_write = False
        has_new_data = False
        anchor_just_set = False  # True on first frame of unsqueeze activation
        
        # Track previous unsqueeze states to detect transitions
        left_was_active = self._left_unsqueeze_active
        right_was_active = self._right_unsqueeze_active
        
        if self.arm_mode == "dual":
            # No trigger: default gripper channel is no delta (delta mode) or neutral absolute
            if self.gripper_delta_mode:
                default_gripper = 0.0
            else:
                # Output space: 0=closed / 1=open (do not run through gripper_invert).
                default_gripper = 0.0 if self.gripper_default_closed else 1.0
            action = np.zeros(14)
            action[6] = default_gripper   # Left gripper default
            action[13] = default_gripper  # Right gripper default
            
            # Left arm — always recompute from latest goal (do not gate on object id;
            # stale id + zeros in action[0:6] left that arm frozen in dual mode).
            left_goal = data.get("left")
            if left_goal is not None:
                (left_delta, self._left_prev_position, self._left_prev_quaternion,
                 self._left_unsqueeze_active, left_gripper,
                 self._left_anchor_position, self._left_anchor_quaternion) = self._compute_arm_delta(
                    left_goal,
                    self._left_prev_position,
                    self._left_prev_quaternion,
                    self._left_unsqueeze_active,
                    is_right_hand=False,
                    anchor_position=self._left_anchor_position,
                    anchor_quaternion=self._left_anchor_quaternion,
                )
                action[0:6] = left_delta
                action[6] = self._apply_gripper_invert(left_gripper)
                has_new_data = True
                if self._left_unsqueeze_active and not left_was_active:
                    anchor_just_set = True

            # Right arm — same always-update path
            right_goal = data.get("right")
            if right_goal is not None:
                (right_delta, self._right_prev_position, self._right_prev_quaternion,
                 self._right_unsqueeze_active, right_gripper,
                 self._right_anchor_position, self._right_anchor_quaternion) = self._compute_arm_delta(
                    right_goal,
                    self._right_prev_position,
                    self._right_prev_quaternion,
                    self._right_unsqueeze_active,
                    is_right_hand=True,
                    anchor_position=self._right_anchor_position,
                    anchor_quaternion=self._right_anchor_quaternion,
                )
                action[7:13] = right_delta
                action[13] = self._apply_gripper_invert(right_gripper)
                has_new_data = True
                if self._right_unsqueeze_active and not right_was_active:
                    anchor_just_set = True

            # Only write when VR unsqueeze is active AND there's new data
            # OR if unsqueeze just turned off (send one last frame to signal robot)
            unsqueeze_active = self._left_unsqueeze_active or self._right_unsqueeze_active
            just_released = (left_was_active and not self._left_unsqueeze_active) or \
                          (right_was_active and not self._right_unsqueeze_active)

            should_write = (has_new_data and unsqueeze_active) or just_released

        elif self.arm_mode == "left":
            if self.gripper_delta_mode:
                default_gripper = 0.0
            else:
                default_gripper = 0.0 if self.gripper_default_closed else 1.0
            action = np.zeros(7)
            action[6] = default_gripper
            
            left_goal = data.get("left")
            left_goal_id = id(left_goal) if left_goal else None
            left_is_new = (left_goal_id != self._left_last_goal_id) and (left_goal is not None)
            self._left_last_goal_id = left_goal_id
            
            left_unsqueeze_now = self._get_unsqueeze_pressed(left_goal)
            left_should_update = left_is_new or (left_unsqueeze_now != left_was_active)
            if left_should_update:
                (left_delta, self._left_prev_position, self._left_prev_quaternion,
                 self._left_unsqueeze_active, left_gripper,
                 self._left_anchor_position, self._left_anchor_quaternion) = self._compute_arm_delta(
                    left_goal,
                    self._left_prev_position,
                    self._left_prev_quaternion,
                    self._left_unsqueeze_active,
                    is_right_hand=False,  # Left hand: X button
                    anchor_position=self._left_anchor_position,
                    anchor_quaternion=self._left_anchor_quaternion,
                )
                action[0:6] = left_delta
                action[6] = self._apply_gripper_invert(left_gripper)
                has_new_data = True
                
                # Detect anchor just set (unsqueeze just activated)
                if self._left_unsqueeze_active and not left_was_active:
                    anchor_just_set = True
            
            unsqueeze_active = self._left_unsqueeze_active
            just_released = left_was_active and not self._left_unsqueeze_active
            should_write = (has_new_data and unsqueeze_active) or just_released
            
        else:  # right
            if self.gripper_delta_mode:
                default_gripper = 0.0
            else:
                default_gripper = 0.0 if self.gripper_default_closed else 1.0
            action = np.zeros(7)
            action[6] = default_gripper
            right_goal = data.get("right")
            right_goal_id = id(right_goal) if right_goal else None
            right_is_new = (right_goal_id != self._right_last_goal_id) and (right_goal is not None)
            self._right_last_goal_id = right_goal_id
            
            right_unsqueeze_now = self._get_unsqueeze_pressed(right_goal)
            right_should_update = right_is_new or (right_unsqueeze_now != right_was_active)
            if right_should_update:
                (right_delta, self._right_prev_position, self._right_prev_quaternion,
                 self._right_unsqueeze_active, right_gripper,
                 self._right_anchor_position, self._right_anchor_quaternion) = self._compute_arm_delta(
                    right_goal,
                    self._right_prev_position,
                    self._right_prev_quaternion,
                    self._right_unsqueeze_active,
                    is_right_hand=True,  # Right hand: A button
                    anchor_position=self._right_anchor_position,
                    anchor_quaternion=self._right_anchor_quaternion,
                )
                action[0:6] = right_delta
                action[6] = self._apply_gripper_invert(right_gripper)
                has_new_data = True
                
                # Detect anchor just set (unsqueeze just activated)
                if self._right_unsqueeze_active and not right_was_active:
                    anchor_just_set = True
            
            unsqueeze_active = self._right_unsqueeze_active
            just_released = right_was_active and not self._right_unsqueeze_active
            should_write = (has_new_data and unsqueeze_active) or just_released
        
        # ============================================================
        # While frozen: still write if action carries non-zero gripper delta (e.g. trigger path
        # is off, but A/B gripper modes may still set a6).
        # ============================================================
        final_unsqueeze_for_gripper = unsqueeze_active if 'unsqueeze_active' in dir() else (self._left_unsqueeze_active or self._right_unsqueeze_active)
        if not final_unsqueeze_for_gripper:
            # Non-zero gripper channel while frozen (e.g. legacy button gripper)
            gripper_value_in_action = action[6] if self.arm_mode != "dual" else (action[6] or action[13])
            if abs(gripper_value_in_action) > 1e-6:
                should_write = True
        
        # ============================================================
        # B / Y go-home: always allowed (no side-grip / unsqueeze required).
        # Rising edge only. gripper_b_maps_to_close=False → B/Y never close gripper.
        # ============================================================
        go_home = False
        final_unsqueeze = unsqueeze_active if 'unsqueeze_active' in dir() else (self._left_unsqueeze_active or self._right_unsqueeze_active)

        for hand_key in (
            ["right"] if self.arm_mode == "right" else
            ["left"] if self.arm_mode == "left" else
            ["right", "left"]
        ):
            goal = data.get(hand_key)
            if goal is not None and goal.metadata:
                buttons = goal.metadata.get("buttons", {}) or {}
                if hand_key == "right":
                    b_pressed = bool(
                        buttons.get("b", False)
                        or buttons.get("button_b", False)
                        or buttons.get("B", False)
                    )
                    if b_pressed and not self._right_b_was_pressed:
                        go_home = True
                    self._right_b_was_pressed = b_pressed
                else:
                    b_pressed = bool(
                        buttons.get("y", False)
                        or buttons.get("button_y", False)
                        or buttons.get("Y", False)
                    )
                    if b_pressed and not self._left_b_was_pressed:
                        go_home = True
                    self._left_b_was_pressed = b_pressed

        if go_home:
            should_write = True
            # Drop teleop so robot can finish go-home without fighting VR deltas.
            self._left_unsqueeze_active = False
            self._right_unsqueeze_active = False
            final_unsqueeze = False
            self._left_anchor_position = None
            self._left_anchor_quaternion = None
            self._right_anchor_position = None
            self._right_anchor_quaternion = None

        # Return action dict with state information for robot side anchor management
        action_dict = {
            "action": action,
            "unsqueeze_active": final_unsqueeze,
            "anchor_just_set": False if go_home else anchor_just_set,
            "go_home": go_home,
        }
        # Per-hand squeeze (dual rel_ee / bimanual robots need independent anchor refresh)
        if self.arm_mode == "dual":
            action_dict["left_unsqueeze_active"] = self._left_unsqueeze_active
            action_dict["right_unsqueeze_active"] = self._right_unsqueeze_active
        elif self.arm_mode == "left":
            action_dict["left_unsqueeze_active"] = self._left_unsqueeze_active
            action_dict["right_unsqueeze_active"] = False
        else:
            action_dict["left_unsqueeze_active"] = False
            action_dict["right_unsqueeze_active"] = self._right_unsqueeze_active

        # XLeRobot uses both sticks even while arm side-grips are released.
        # Keep this opt-in so existing arm-only Quest configurations retain
        # their original shared-memory write behavior.
        if self.publish_thumbsticks:
            action_dict["left_thumbstick"] = self._get_thumbstick(data.get("left"))
            action_dict["right_thumbstick"] = self._get_thumbstick(data.get("right"))
            if data.get("left") is not None or data.get("right") is not None:
                should_write = True

        if self.debug:
            now = time.time()
            if not hasattr(self, "_last_diag_log_t") or (now - self._last_diag_log_t) >= 0.5:
                self._last_diag_log_t = now
                left_goal = data.get("left") if isinstance(data, dict) else None
                right_goal = data.get("right") if isinstance(data, dict) else None
                l_grip_raw = False
                r_grip_raw = False
                if left_goal is not None and left_goal.metadata:
                    l_grip_raw = bool(
                        left_goal.metadata.get("grip_active", False)
                        or (left_goal.metadata.get("buttons") or {}).get("squeeze", False)
                        or (left_goal.metadata.get("buttons") or {}).get("unsqueeze", False)
                    )
                if right_goal is not None and right_goal.metadata:
                    r_grip_raw = bool(
                        right_goal.metadata.get("grip_active", False)
                        or (right_goal.metadata.get("buttons") or {}).get("squeeze", False)
                        or (right_goal.metadata.get("buttons") or {}).get("unsqueeze", False)
                    )
                if self.arm_mode == "dual":
                    lee = float(np.linalg.norm(action[:6]))
                    ree = float(np.linalg.norm(action[7:13]))
                    logger.info(
                        f"[Quest3Diag] hasL={left_goal is not None} hasR={right_goal is not None} "
                        f"L_grip_raw={l_grip_raw} R_grip_raw={r_grip_raw} "
                        f"L_unsq={self._left_unsqueeze_active} R_unsq={self._right_unsqueeze_active} "
                        f"|Lee|={lee:.4f} |Ree|={ree:.4f} go_home={go_home} write={should_write}"
                    )
                else:
                    logger.info(
                        f"[Quest3Diag] unsqz={final_unsqueeze} |ee|={float(np.linalg.norm(action[:6])):.4f} "
                        f"go_home={go_home} write={should_write}"
                    )

        if self.gripper_trace and should_write:
            if self.arm_mode == "dual":
                logger.info(
                    f"[GripperTrace:teleop] shm_write L6={float(action[6]):.5f} R13={float(action[13]):.5f} "
                    f"unsqz={final_unsqueeze} go_home={go_home} anchor_set={anchor_just_set}"
                )
            else:
                logger.info(
                    f"[GripperTrace:teleop] shm_write a6={float(action[6]):.5f} "
                    f"unsqz={final_unsqueeze} go_home={go_home} anchor_set={anchor_just_set}"
                )

        return action_dict, should_write
    
    def start(self):
        """Start the Quest3 teleoperator."""
        import signal
        
        print(f"[Quest3Teleop] Starting, name={self.name}, mode={self.arm_mode}")
        if self.gripper_trace:
            logger.warning(
                "[Quest3Teleop] gripper_trace=ON: logging [GripperTrace:teleop] on each SHM write"
            )
        
        # Connect to VR first
        if not self.connect():
            raise RuntimeError("[Quest3Teleop] Failed to connect to VR headset")
        
        # Create shared memory for output
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)
        
        # Signal: only request loop exit. Do NOT close() here — adb restore must run
        # in finally without being interrupted by raise KeyboardInterrupt / double-close.
        def cleanup_handler(signum, frame):
            self.is_running = False

        signal.signal(signal.SIGTERM, cleanup_handler)
        signal.signal(signal.SIGINT, cleanup_handler)
        
        # Main loop
        self.is_running = True
        rate_limiter = RateLimiter()
        self._write_count = 0
        
        try:
            while self.is_running:
                data = self.get_data()
                
                if data is not None:
                    action_dict, should_write = self.convert_data_to_action(data)
                    # Only write to shm when unsqueeze button is pressed AND there's new VR data
                    if should_write:
                        # Write action dict with state info (action, unsqueeze_active, anchor_just_set)
                        self.write_data_to_shm(action_dict)
                        self._write_count += 1
                        
                rate_limiter.sleep(self.fps)
        finally:
            self.close()
    
    def close(self):
        """Close the teleoperator (idempotent)."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.is_running = False

        # Stop keep-awake / restore proximity first (must finish before parent kill).
        if getattr(self, "_keep_awake", None) is not None:
            try:
                self._keep_awake.stop()
            except BaseException as e:
                logger.warning(f"[Quest3Teleop] keep_awake.stop failed: {e!r}")
        
        # Stop VR monitor
        if self.vr_monitor is not None:
            self.vr_monitor.is_running = False
        
        # Wait for VR thread to finish
        if self._vr_thread is not None and self._vr_thread.is_alive():
            self._vr_thread.join(timeout=2.0)
        
        try:
            super().close()
        except Exception as e:
            logger.warning(f"[Quest3Teleop] super().close failed: {e!r}")
        print("[Quest3Teleop] Closed")


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import multiprocessing as mp
    from deploy.base import start_device
    from deploy.shm_utils import SharedMemoryChannel
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Quest3 VR Teleoperator")
    parser.add_argument("--mode", "-m", type=str, default="right", 
                        choices=["left", "right", "dual"],
                        help="Arm mode: left, right, or dual (default: right)")
    parser.add_argument("--timeout", "-t", type=float, default=60.0,
                        help="VR connection timeout in seconds (default: 60)")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--test-buttons", action="store_true",
                        help="Test button data capture (direct VR access)")
    args = parser.parse_args()
    
    # Button test mode: directly access VR data without starting device process
    if args.test_buttons:
        print("=== Button Test Mode ===")
        print("Press buttons on VR controller to see their states.")
        print("Available buttons: trigger, a, b, x, y, unsqueeze (side grip)\n")
        
        # Create a test instance to access VR data directly
        test_teleop = Quest3Teleop(
            name="quest3_test",
            arm_mode=args.mode,
            debug=True,
            gripper_button_control=True,
            gripper_delta_mode=True,
        )
        
        if not test_teleop.connect():
            print("Failed to connect to VR. Exiting.")
            sys.exit(1)
        
        try:
            last_print_time = 0
            print_interval = 0.1  # Print every 100ms
            
            while True:
                vr_data = test_teleop.get_data()
                current_time = time.time()
                
                # Collect data from both hands
                right_data = None
                left_data = None
                
                if vr_data is not None:
                    # Check right controller buttons
                    if vr_data.get("has_right") and vr_data.get("right"):
                        right_goal = vr_data["right"]
                        if right_goal and right_goal.metadata:
                            buttons = right_goal.metadata.get("buttons", {})
                            right_data = {
                                "buttons": buttons,
                                "trigger_active": right_goal.metadata.get("trigger_active", False),
                                "trigger_value": right_goal.metadata.get("trigger_value", 0.0),
                                "grip_active": right_goal.metadata.get("grip_active", False),
                            }
                    
                    # Check left controller buttons
                    if vr_data.get("has_left") and vr_data.get("left"):
                        left_goal = vr_data["left"]
                        if left_goal and left_goal.metadata:
                            buttons = left_goal.metadata.get("buttons", {})
                            left_data = {
                                "buttons": buttons,
                                "trigger_active": left_goal.metadata.get("trigger_active", False),
                                "trigger_value": left_goal.metadata.get("trigger_value", 0.0),
                                "grip_active": left_goal.metadata.get("grip_active", False),
                            }
                
                # Print both hands together every 100ms
                if current_time - last_print_time >= print_interval:
                    print("\n" + "="*70)
                    print("DUAL HAND BUTTON STATES")
                    print("="*70)
                    
                    # Right hand
                    if right_data:
                        print("\nRIGHT CONTROLLER:")
                        print(f"  trigger_active: {right_data['trigger_active']}")
                        print(f"  trigger_value: {right_data['trigger_value']:.3f}")
                        print(f"  grip_active (unsqueeze): {right_data['grip_active']}")
                        print(f"  buttons dict: {right_data['buttons']}")
                        if right_data['buttons']:
                            print("  Individual buttons:")
                            for btn_name, btn_state in right_data['buttons'].items():
                                status = "✓" if btn_state else "✗"
                                print(f"    {status} {btn_name}: {btn_state}")
                        else:
                            print("  WARNING: buttons dict is empty!")
                    else:
                        print("\nRIGHT CONTROLLER: Not available")
                    
                    # Left hand
                    if left_data:
                        print("\nLEFT CONTROLLER:")
                        print(f"  trigger_active: {left_data['trigger_active']}")
                        print(f"  trigger_value: {left_data['trigger_value']:.3f}")
                        print(f"  grip_active (unsqueeze): {left_data['grip_active']}")
                        print(f"  buttons dict: {left_data['buttons']}")
                        if left_data['buttons']:
                            print("  Individual buttons:")
                            for btn_name, btn_state in left_data['buttons'].items():
                                status = "✓" if btn_state else "✗"
                                print(f"    {status} {btn_name}: {btn_state}")
                        else:
                            print("  WARNING: buttons dict is empty!")
                    else:
                        print("\nLEFT CONTROLLER: Not available")
                    
                    print("="*70)
                    last_print_time = current_time
                
                time.sleep(0.01)  # Small sleep to avoid CPU spinning
        except KeyboardInterrupt:
            print("\nStopping button test...")
        finally:
            test_teleop.close()
        sys.exit(0)
    
    print(f"Starting Quest3 Teleop in {args.mode.upper()} mode...")
    
    # Device configuration
    device_config = {
        "type": "deploy.teleoperator.quest3.Quest3Teleop",
        "name": "quest3_teleop",
        "args": {
            "name": "quest3_teleop",
            "fps": 50.0,
            "arm_mode": args.mode,
            "position_scale": 1.0,
            "rotation_scale": 1.0,
            "connect_timeout": args.timeout,
            "debug": args.debug,
        }
    }
    
    shm_name = device_config["args"]["name"]
    proc = mp.Process(target=start_device, args=(device_config,))
    proc.start()
    
    time.sleep(2.0)
    print("Reading from SHM (Ctrl+C to stop)...")
    print("Press and hold unsqueeze button on VR controller to send pose deltas.\n")
    
    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=args.timeout + 10)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None and "action" in data:
                arr = data["action"]
                if len(arr) >= 14:
                    # Dual arm mode
                    print(
                        f"L: [{arr[0]:+.3f},{arr[1]:+.3f},{arr[2]:+.3f}] rot:[{arr[3]:+.2f},{arr[4]:+.2f},{arr[5]:+.2f}] g:{arr[6]:.3f} | "
                        f"R: [{arr[7]:+.3f},{arr[8]:+.3f},{arr[9]:+.3f}] rot:[{arr[10]:+.2f},{arr[11]:+.2f},{arr[12]:+.2f}] g:{arr[13]:.3f}",
                        end="\r",
                        flush=True,
                    )
                elif len(arr) >= 7:
                    # Single arm mode
                    print(
                        f"pos: [{arr[0]:+.3f},{arr[1]:+.3f},{arr[2]:+.3f}] "
                        f"rot: [{arr[3]:+.2f},{arr[4]:+.2f},{arr[5]:+.2f}] "
                        f"grip: {arr[6]:.3f}",
                        end="\r",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
        print("Done.")
