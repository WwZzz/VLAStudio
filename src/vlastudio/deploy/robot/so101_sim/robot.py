"""
SO101 Simulation Robot with Multiple Control Modes.
"""

import time
import threading
from pathlib import Path
from typing import Dict, Optional

import mujoco
import numpy as np

from vlastudio.deploy.simulation.mujoco.base import MujocoDeviceBase
from .kinematics import create_so101, lerobot_FK, lerobot_IK


# Joint names in MuJoCo model
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Default control limits (can be overridden via config)
DEFAULT_QLIMIT_MIN = [-2.1, -3.1, -0.0, -1.375, -1.57, -0.15]
DEFAULT_QLIMIT_MAX = [2.1, 0.0, 3.1, 1.475, 3.1, 1.5]

# Default joint signs (1 = normal, -1 = reversed)
DEFAULT_JOINT_SIGNS = [1, 1, 1, 1, 1, 1]

# Default gpos limits (end-effector pose limits)
DEFAULT_GLIMIT_MIN = [0.125, -0.4, 0.046, -3.1, -0.75, -1.5]
DEFAULT_GLIMIT_MAX = [0.340, 0.4, 0.23, 2.0, 1.57, 1.5]

# Default initial positions
DEFAULT_INIT_QPOS = [0.0, -3.14, 3.14, 0.0, -1.57, -0.157]

# `table_block.xml`: table_scene at (0.48,0,0.575), top half-thickness 0.025 → surface z=0.60; top half-size x=0.34.
_TABLE_BLOCK_TOP_Z = 0.575 + 0.025
_TABLE_BLOCK_BASE_Z_CLEARANCE = 0.004
_TABLE_BLOCK_CENTER_X = 0.48
_TABLE_BLOCK_HALF_X = 0.34
# Bessica root sits near world origin; place SO101 on the tabletop edge closest to that side (-world X from table center).
_TABLE_BLOCK_SO101_EDGE_INSET = 0.06
# Match `bessica_d.xml` bessica_root quat (wxyz): body +X → world +Y, same table-facing convention.
_TABLE_BLOCK_SO101_BASE_QUAT_WXYZ = np.array([0.707107, 0.0, 0.0, 0.707107], dtype=np.float64)


def _draw_viewer_coordinate_frame(viewer, pos, axis_length=0.05, axis_radius=0.002):
    axes_and_colors = [
        (np.array([0, 1, 0]), [1, 0, 0, 1]),
        (np.array([-1, 0, 0]), [0, 1, 0, 1]),
        (np.array([0, 0, 1]), [0, 0, 1, 1]),
    ]
    for axis, color in axes_and_colors:
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).flatten(),
            np.array(color, dtype=np.float32),
        )
        mujoco.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            axis_radius,
            pos,
            pos + axis * axis_length,
        )
        viewer.user_scn.ngeom += 1


class So101SimRobot(MujocoDeviceBase):
    """
    SO101 Simulation Robot with Multiple Control Modes
    
    Supports two control modes:
    
    1. "delta_ee" (default): 7D delta actions
       - action[0:3]: Delta position (dx, dy, dz) in world frame
       - action[3:6]: Delta orientation (d_roll, d_pitch, d_yaw)
       - action[6]: Delta gripper position
       - Uses Inverse Kinematics to compute joint positions
    
    2. "qpos": 6D absolute joint positions
       - action[0:6]: Joint positions [q1, q2, ..., q5]
       - Direct joint control without IK
    """
    
    # Valid control modes
    CONTROL_MODES = ("delta_ee", "qpos")
    
    # Valid qpos input units
    QPOS_INPUT_UNITS = ("normalized", "radians")
    
    def __init__(self,
                 name: str = "so101_sim",
                 max_size_mb: int = 64,
                 fps: float = 50.0,
                 control_shm_name: Optional[str] = None,
                 xml_path: Optional[str] = None,
                 scene_name: Optional[str] = None,
                 scene_xml_path: Optional[str] = None,
                 camera_names: Optional[list] = None,
                 camera_width: int = 640,
                 camera_height: int = 480,
                 control_mode: str = "delta_ee",
                 qpos_input_unit: str = "normalized",
                 position_scale: float = 0.001,
                 rotation_scale: float = 0.01,
                 gripper_scale: float = 0.01,
                 qlimit_min: Optional[list] = None,
                 qlimit_max: Optional[list] = None,
                 joint_signs: Optional[list] = None,
                 init_qpos: Optional[list] = None,
                 **kwargs):
        """
        Initialize SO101 Simulation Robot
        
        Args:
            name: Name for the shared memory segment
            max_size_mb: Maximum size of shared memory in MB
            fps: Control frequency in Hz
            control_shm_name: Name of the control shared memory
            xml_path: Path to MuJoCo XML model file
            control_mode: Control mode, one of:
                - "delta_ee": 7D delta EE control (dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper)
                - "qpos": 6D absolute joint position control
            qpos_input_unit: Input unit for qpos mode, one of:
                - "normalized": [-100, 100] range (lerobot format, default)
                - "radians": Direct radians input
            position_scale: Scale factor for position deltas (delta_ee mode only)
            rotation_scale: Scale factor for rotation deltas (delta_ee mode only)
            gripper_scale: Scale factor for gripper deltas (delta_ee mode only)
            qlimit_min: Minimum joint limits in radians (6D list)
            qlimit_max: Maximum joint limits in radians (6D list)
            joint_signs: Joint direction signs (6D list, 1=normal, -1=reversed)
            init_qpos: Initial joint positions in radians (6D list)
        """
        # Joint limits and signs (configurable)
        self.qlimit_min = np.array(qlimit_min if qlimit_min is not None else DEFAULT_QLIMIT_MIN)
        self.qlimit_max = np.array(qlimit_max if qlimit_max is not None else DEFAULT_QLIMIT_MAX)
        self.joint_signs = np.array(joint_signs if joint_signs is not None else DEFAULT_JOINT_SIGNS)
        self.init_qpos = np.array(init_qpos if init_qpos is not None else DEFAULT_INIT_QPOS)
        
        # Validate and set control mode
        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Invalid control_mode '{control_mode}'. Must be one of {self.CONTROL_MODES}")
        self.control_mode = control_mode
        
        # Validate and set qpos input unit
        if qpos_input_unit not in self.QPOS_INPUT_UNITS:
            raise ValueError(f"Invalid qpos_input_unit '{qpos_input_unit}'. Must be one of {self.QPOS_INPUT_UNITS}")
        self.qpos_input_unit = qpos_input_unit
        
        print(f"[So101SimRobot] Control mode: {control_mode}" + 
              (f", qpos_input_unit: {qpos_input_unit}" if control_mode == "qpos" else ""))

        # Action scaling
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.gripper_scale = gripper_scale

        # Kinematics
        self.robot_kin = create_so101()

        # Current state
        self.current_qpos = self.init_qpos.copy()

        self.current_gpos = lerobot_FK(self.current_qpos[1:5], robot=self.robot_kin)
        self.target_gpos = self.current_gpos.copy()

        super().__init__(
            name=name,
            max_size_mb=max_size_mb,
            fps=fps,
            control_shm_name=control_shm_name,
            xml_path=xml_path,
            scene_name=scene_name,
            scene_xml_path=scene_xml_path,
            camera_names=camera_names,
            camera_width=camera_width,
            camera_height=camera_height,
            **kwargs,
        )
        print("[So101SimRobot] Initialized successfully")

    def _default_robot_xml_path(self) -> str:
        module_dir = Path(__file__).parent
        return str(module_dir / "mujoco_model" / "so_101.xml")

    def _configure_loaded_model(self) -> None:
        if self.scene_name != "table_block":
            return
        try:
            base_body_id = self.mjmodel.body("Base").id
        except KeyError:
            return
        # On tabletop near Bessica (origin): left edge + inset, same base yaw as bessica_root.
        bx = _TABLE_BLOCK_CENTER_X - _TABLE_BLOCK_HALF_X + _TABLE_BLOCK_SO101_EDGE_INSET
        bz = _TABLE_BLOCK_TOP_Z + _TABLE_BLOCK_BASE_Z_CLEARANCE
        self.mjmodel.body_pos[base_body_id] = np.array([bx, 0.0, bz], dtype=np.float64)
        self.mjmodel.body_quat[base_body_id] = _TABLE_BLOCK_SO101_BASE_QUAT_WXYZ.copy()

    def _build_joint_indices(self) -> None:
        self.qpos_indices = np.array([
            self.mjmodel.jnt_qposadr[self.mjmodel.joint(name).id]
            for name in JOINT_NAMES
        ])
        # Position actuators: set ctrl to targets and mj_step so PD generates torque at contacts.
        # Never set qpos == ctrl immediately before mj_step — zero error => zero torque => deep penetration.
        self.actuator_indices = np.array(
            [self.mjmodel.actuator(name).id for name in JOINT_NAMES],
            dtype=np.int32,
        )
        # Match ~50Hz control when model timestep is 0.001 (see so_101.xml <option>).
        self._physics_substeps_per_command = 20
        self.qvel_indices = np.array(
            [self.mjmodel.jnt_dofadr[self.mjmodel.joint(name).id] for name in JOINT_NAMES],
            dtype=np.int32,
        )

    def _apply_qpos_preview_for_static_guard_locked(self, q6: np.ndarray) -> None:
        self.mjdata.qpos[self.qpos_indices] = q6
        self.mjdata.ctrl[self.actuator_indices] = q6
        self.mjdata.qvel[:] = 0.0
        mujoco.mj_forward(self.mjmodel, self.mjdata)

    def _apply_initial_state(self) -> None:
        self.mjdata.qpos[self.qpos_indices] = self.current_qpos
        self.mjdata.ctrl[self.actuator_indices] = self.current_qpos
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        self.current_gpos = lerobot_FK(self.current_qpos[1:5], robot=self.robot_kin)
        self.target_gpos = self.current_gpos.copy()
        self._safe_commanded_qpos = self.current_qpos.copy()

    def _setup_viewer_overlay_state(self) -> None:
        try:
            self._viewer_tcp_site_id = self.mjmodel.site("tcp").id
        except KeyError:
            self._viewer_tcp_site_id = None

    def _draw_viewer_overlay_locked(self, viewer) -> None:
        if getattr(self, "_viewer_tcp_site_id", None) is None:
            return
        _draw_viewer_coordinate_frame(viewer, self.mjdata.site_xpos[self._viewer_tcp_site_id].copy(), 0.07, 0.003)
    
    def get_action_dim(self) -> int:
        """Return action dimension based on control mode"""
        if self.control_mode == "delta_ee":
            return 7  # 6-DOF pose delta + gripper
        else:  # qpos
            return 6  # 6 joint positions
    
    def _get_robot_observation_core(self) -> Dict[str, np.ndarray]:
        qpos = self.current_qpos.copy()
        gpos = self.current_gpos.copy()
        return {
            'qpos': qpos,
            'gpos': gpos,  # End-effector pose [x, y, z, roll, pitch, yaw]
        }

    def _process_robot_action(self, action_dict: dict) -> dict:
        """
        Process action based on control mode
        
        Args:
            action_dict: Dictionary containing 'action' key
            
        Returns:
            Modified action_dict with joint position targets
        """
        action = action_dict.get('action', None)
        if action is None:
            return action_dict
        
        action = np.array(action, dtype=np.float64)
        
        if self.control_mode == "qpos":
            return self._process_qpos_action(action_dict, action)
        
        return self._process_delta_ee_action(action_dict, action)
    
    def _process_qpos_action(self, action_dict: dict, action: np.ndarray) -> dict:
        """
        Process 6D absolute joint position action
        
        Args:
            action_dict: Original action dictionary
            action: 6D joint positions [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]
                    Can be in normalized [-100, 100] range or radians, depending on input_unit
            
        Returns:
            Modified action_dict with clipped joint positions in radians
        """
        if len(action) != 6:
            print(f"[So101SimRobot] Warning: qpos mode expects 6D action, got {len(action)}D")
            return action_dict
        
        action = np.array(action, dtype=np.float64)
        
        # Convert from normalized [-100, 100] to radians if needed
        if self.qpos_input_unit == "normalized":
            target_qpos = self._normalized_to_radians(action)
        else:  # radians
            target_qpos = action.copy()
        
        # Clip to joint limits
        for i in range(6):
            target_qpos[i] = np.clip(target_qpos[i], self.qlimit_min[i], self.qlimit_max[i])
        
        action_dict['action'] = target_qpos
        return action_dict
    
    def _normalized_to_radians(self, normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized joint values to radians
        
        IMPORTANT: SO101 Leader uses different normalization for different joints:
        - Joints 0-4 (body): MotorNormMode.RANGE_M100_100, range [-100, 100]
        - Joint 5 (gripper): MotorNormMode.RANGE_0_100, range [0, 100]
        
        Also applies joint_signs to handle reversed joints.
        
        Conversion formulas:
        - RANGE_M100_100: radians = (normalized + 100) / 200 * (max - min) + min
        - RANGE_0_100: radians = normalized / 100 * (max - min) + min
        
        Args:
            normalized: 6D array with normalized values
            
        Returns:
            6D array with values in radians
        """
        radians = np.zeros(6, dtype=np.float64)
        for i in range(5):  # Joints 0-4: RANGE_M100_100 [-100, 100]
            qmin = self.qlimit_min[i]
            qmax = self.qlimit_max[i]
            # Apply joint sign before conversion (reversed joints have inverted normalized values)
            norm_signed = normalized[i] * self.joint_signs[i]
            norm_clamped = np.clip(norm_signed, -100.0, 100.0)
            radians[i] = (norm_clamped + 100.0) / 200.0 * (qmax - qmin) + qmin
        
        # Joint 5 (gripper/Jaw): RANGE_0_100 [0, 100]
        qmin = self.qlimit_min[5]
        qmax = self.qlimit_max[5]
        norm_signed = normalized[5] * self.joint_signs[5]
        norm_clamped = np.clip(norm_signed, 0.0, 100.0)
        radians[5] = norm_clamped / 100.0 * (qmax - qmin) + qmin
        
        return radians
    
    def _radians_to_normalized(self, radians: np.ndarray) -> np.ndarray:
        """
        Convert radians to normalized joint values
        
        IMPORTANT: SO101 Leader uses different normalization for different joints:
        - Joints 0-4 (body): MotorNormMode.RANGE_M100_100, range [-100, 100]
        - Joint 5 (gripper): MotorNormMode.RANGE_0_100, range [0, 100]
        
        Also applies joint_signs to handle reversed joints.
        
        Args:
            radians: 6D array with values in radians
            
        Returns:
            6D array with normalized values
        """
        normalized = np.zeros(6, dtype=np.float64)
        for i in range(5):  # Joints 0-4: RANGE_M100_100 [-100, 100]
            qmin = self.qlimit_min[i]
            qmax = self.qlimit_max[i]
            rad_clamped = np.clip(radians[i], qmin, qmax)
            # Apply joint sign after conversion
            normalized[i] = ((rad_clamped - qmin) / (qmax - qmin)) * 200.0 - 100.0
            normalized[i] *= self.joint_signs[i]
        
        # Joint 5 (gripper/Jaw): RANGE_0_100 [0, 100]
        qmin = self.qlimit_min[5]
        qmax = self.qlimit_max[5]
        rad_clamped = np.clip(radians[5], qmin, qmax)
        normalized[5] = ((rad_clamped - qmin) / (qmax - qmin)) * 100.0
        normalized[5] *= self.joint_signs[5]
        
        return normalized
    
    def _process_delta_ee_action(self, action_dict: dict, action: np.ndarray) -> dict:
        """
        Process 7D delta EE action and convert to joint positions using IK
        
        Args:
            action_dict: Original action dictionary
            action: 7D delta action [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]
            
        Returns:
            Modified action_dict with joint position targets
        """
        if len(action) != 7:
            print(f"[So101SimRobot] Warning: delta_ee mode expects 7D action, got {len(action)}D")
            return action_dict
        
        # Extract and scale deltas
        delta_pos = action[0:3] * self.position_scale
        delta_rot = action[3:6] * self.rotation_scale
        delta_gripper = action[6] * self.gripper_scale
        
        # CRITICAL: If all deltas are zero, skip IK computation entirely
        # This avoids IK numerical drift - IK can return slightly different solutions
        # even for the same target pose, causing the robot to drift when idle
        total_delta = np.sum(np.abs(delta_pos)) + np.sum(np.abs(delta_rot)) + np.abs(delta_gripper)
        if total_delta < 1e-10:
            # No movement requested, set action to None to keep current position
            action_dict['action'] = None
            return action_dict
        
        with self._lock:
            # Current end-effector pose
            current_gpos = self.current_gpos.copy()
            current_qpos = self.current_qpos.copy()
            
            # Compute new target gpos
            # Handle XY motion in world frame (like keyboard_control.py)
            angle_curr = current_qpos[0]  # Base rotation
            forward_curr = current_gpos[0]  # Forward distance
            
            # Current XY in world frame
            x_curr = forward_curr * np.cos(angle_curr)
            y_curr = forward_curr * np.sin(angle_curr)
            
            # Apply delta in world frame
            x_new = x_curr + delta_pos[0]
            y_new = y_curr + delta_pos[1]
            
            # Convert back to polar coordinates
            theta_update = np.arctan2(y_new, x_new) - np.arctan2(y_curr, x_curr)
            forward_update = np.sqrt(x_new**2 + y_new**2) - np.sqrt(x_curr**2 + y_curr**2)
            
            # Update target gpos
            target_gpos = current_gpos.copy()
            target_gpos[0] += forward_update  # Forward
            target_gpos[2] += delta_pos[2]  # Height (Z)
            target_gpos[3] += delta_rot[0]  # Roll
            target_gpos[4] += delta_rot[1]  # Pitch
            # target_gpos[5] unused here, handled by wrist roll joint
            
            # Update base rotation
            target_qpos = current_qpos.copy()
            target_qpos[0] += theta_update
            
            # Clip to limits
            target_qpos[0] = np.clip(target_qpos[0], self.qlimit_min[0], self.qlimit_max[0])
            for i in range(6):
                target_gpos[i] = np.clip(target_gpos[i], DEFAULT_GLIMIT_MIN[i], DEFAULT_GLIMIT_MAX[i])
            
            # Compute IK for joints 1-4 (Pitch, Elbow, Wrist_Pitch, Wrist_Roll)
            fd_qpos = current_qpos[1:5]
            qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=self.robot_kin)
            
            if ik_success:
                # Update target joint positions
                target_qpos[1:5] = qpos_inv[:4]
                
                # Update gripper
                target_qpos[5] = current_qpos[5] + delta_gripper
                target_qpos[5] = np.clip(target_qpos[5], self.qlimit_min[5], self.qlimit_max[5])
                
                # Store new state
                self.target_gpos = target_gpos.copy()
                
                # Return joint position action
                action_dict['action'] = target_qpos
            else:
                # IK failed, keep current position
                action_dict['action'] = current_qpos
        
        return action_dict
    
    def _publish_robot_action(self, action: np.ndarray):
        """Execute joint position action in simulation"""
        with self._lock:
            action = np.asarray(action, dtype=np.float64).reshape(6)
            action = self._interpolate_pose_avoiding_static_contact_locked(
                self._safe_commanded_qpos,
                action,
                self._apply_qpos_preview_for_static_guard_locked,
            )
            self.mjdata.ctrl[self.actuator_indices] = action
            for _ in range(self._physics_substeps_per_command):
                mujoco.mj_step(self.mjmodel, self.mjdata)
            q = self.mjdata.qpos[self.qpos_indices].copy()
            self.current_qpos = q
            self._safe_commanded_qpos = q.copy()
            self.current_gpos = lerobot_FK(q[1:5], robot=self.robot_kin)
    
    def meta2act(self, mact):
        """Convert MetaAction to robot action"""
        default_dim = 7 if self.control_mode == "delta_ee" else 6
        return mact.get('action', np.zeros(default_dim))
    
    def obs2meta(self, device_data):
        """Extract robot state from this device's SHM data."""
        if device_data is None:
            return {}
        qpos = device_data.get('qpos', np.zeros(6, dtype=np.float32))
        return {'state': np.asarray(qpos, dtype=np.float32)}


# ==============================================================================
# Test
# ==============================================================================

def _test_action_writer(control_shm_name: str, mode: str, stop_event):
    """
    Write sine wave control actions to control_shm.
    Runs in the main process while robot.start() runs in a subprocess.
    """
    from vlastudio.deploy.shm_utils import SharedMemoryChannel
    
    # Create control SHM as writer
    control_shm = SharedMemoryChannel(name=control_shm_name, max_size_mb=1, is_writer=True)
    print(f"[ActionWriter] Control SHM '{control_shm_name}' created, writing {mode} actions...")
    
    t = 0
    try:
        while not stop_event.is_set():
            if mode == "delta_ee":
                # Generate test action (7D delta)
                action = np.array([
                    0.1 * np.sin(t),  # dx
                    0.0,  # dy
                    0.05 * np.sin(t * 0.5),  # dz
                    0.0,  # d_roll
                    0.0,  # d_pitch
                    0.0,  # d_yaw
                    0.0,  # d_gripper
                ])
            else:  # qpos mode
                # Generate test action (6D absolute joint positions in radians)
                action = np.array([
                    0.5 * np.sin(t * 0.5),  # Rotation
                    -3.14 + 0.3 * np.sin(t * 0.3),  # Pitch
                    3.14 - 0.3 * np.sin(t * 0.3),  # Elbow
                    0.2 * np.sin(t * 0.4),  # Wrist_Pitch
                    -1.57 + 0.5 * np.sin(t * 0.2),  # Wrist_Roll
                    0.5 * (1 + np.sin(t)),  # Jaw (gripper)
                ])
            
            control_shm.write({"action": action})
            t += 0.02
            time.sleep(0.02)
    finally:
        control_shm.destroy()
        print("[ActionWriter] Stopped")


def _run_robot_process(robot_config: dict):
    """
    Run robot.start() in a subprocess.
    """
    from vlastudio.deploy.base import start_device
    start_device(robot_config)


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp
    import threading
    
    # CRITICAL: Import shm_utils FIRST to patch resource_tracker
    import vlastudio.deploy.shm_utils; from vlastudio import deploy  # noqa: F401
    
    parser = argparse.ArgumentParser(description="Test SO101 Simulation Robot")
    parser.add_argument("--mode", "-m", type=str, default="delta_ee", 
                        choices=["delta_ee", "qpos"], help="Control mode")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Enable visualization (starts Visualizer in separate process)")
    args = parser.parse_args()
    
    # Names for shared memory segments
    robot_shm_name = "so101_sim_test"
    control_shm_name = "so101_sim_test_ctrl"
    
    # Clean up any orphaned SHM
    from vlastudio.deploy.shm_utils import cleanup_all_shm
    cleanup_all_shm([robot_shm_name, control_shm_name])
    
    # Robot config
    robot_config = {
        "type": "deploy.robot.so101_sim.So101SimRobot",
        "args": {
            "name": robot_shm_name,
            "max_size_mb": 64,
            "fps": 50.0,
            "control_shm_name": control_shm_name,
            "control_mode": args.mode,
            "qpos_input_unit": "radians",  # Use radians for test
            "position_scale": 0.005,
            "rotation_scale": 0.02,
        }
    }
    
    print(f"Testing SO101 Simulation Robot (mode: {args.mode})")
    print("=" * 50)
    
    # Create stop event for action writer thread
    stop_event = threading.Event()
    
    # Start action writer in a thread (writes to control_shm)
    action_thread = threading.Thread(
        target=_test_action_writer,
        args=(control_shm_name, args.mode, stop_event),
        daemon=True
    )
    action_thread.start()
    
    # Give control_shm time to be created
    time.sleep(0.5)
    
    # Start robot in subprocess (reads from control_shm, writes to robot_shm)
    robot_proc = mp.Process(target=_run_robot_process, args=(robot_config,))
    robot_proc.start()
    print(f"[Main] Robot subprocess started (PID: {robot_proc.pid})")
    
    # Wait for robot SHM to be ready
    time.sleep(1.0)
    
    # Start visualizer in subprocess if requested (reads from robot_shm)
    viz_proc = None
    if args.visualize:
        from vlastudio.deploy.visualizer.base import start_visualizer
        viz_proc = mp.Process(
            target=start_visualizer,
            args=("deploy.robot.so101_sim.Visualizer", robot_shm_name),
            daemon=True
        )
        viz_proc.start()
        print(f"[Main] Visualizer subprocess started (PID: {viz_proc.pid})")
    
    print("\nPress Ctrl+C to stop...\n")
    
    try:
        # Monitor robot SHM and print state
        from vlastudio.deploy.shm_utils import SharedMemoryChannel
        time.sleep(0.5)
        robot_shm = SharedMemoryChannel(robot_shm_name, is_writer=False, timeout=10.0)
        
        while robot_proc.is_alive():
            data = robot_shm.read(blocking=False, skip_unchanged=True)
            if data is not None:
                qpos = data.get('qpos', np.zeros(6))
                gpos = data.get('gpos', np.zeros(6))
                print(f"  qpos: [{qpos[0]:+.2f}, {qpos[1]:+.2f}, {qpos[2]:+.2f}, ...] "
                      f"gpos: [{gpos[0]:+.3f}, {gpos[1]:+.3f}, {gpos[2]:+.3f}, ...]", end="\r")
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\n[Main] Stopping...")
    finally:
        # Stop action writer
        stop_event.set()
        action_thread.join(timeout=1.0)
        
        # Stop visualizer
        if viz_proc is not None:
            viz_proc.terminate()
            viz_proc.join(timeout=2.0)
            if viz_proc.is_alive():
                viz_proc.kill()
        
        # Stop robot
        robot_proc.terminate()
        robot_proc.join(timeout=2.0)
        if robot_proc.is_alive():
            robot_proc.kill()
        
        # Cleanup SHM
        cleanup_all_shm([robot_shm_name, control_shm_name])
        print("[Main] Done.")
