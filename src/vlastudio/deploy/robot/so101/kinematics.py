"""
SO101 Kinematics Module for Real Robot
Forward Kinematics (FK) and Inverse Kinematics (IK) for SO101 robot arm
Based on lerobot-kinematics implementation

This version includes fast_mode for real-time control on physical robots.
"""

import numpy as np
import math
from math import sqrt
from scipy.spatial.transform import Rotation as R

# Precision rounding functions
def atan2(first, second):
    return round(math.atan2(first, second), 6)

def sin(radians_angle):
    return round(math.sin(radians_angle), 6)

def cos(radians_angle):
    return round(math.cos(radians_angle), 6)

def acos(value):
    return round(math.acos(np.clip(value, -1.0, 1.0)), 6)


class ET:
    """Elementary Transform class for building robot kinematics chain"""
    
    def __init__(self, transform_type, value=None, is_joint=False):
        self.transform_type = transform_type  # 'tx', 'ty', 'tz', 'Rx', 'Ry', 'Rz'
        self.value = value  # Fixed value for static transforms
        self.is_joint = is_joint  # True if this is a joint (variable)
        self.next = None
        
    @staticmethod
    def tx(value):
        return ET('tx', value, is_joint=False)
    
    @staticmethod
    def ty(value):
        return ET('ty', value, is_joint=False)
    
    @staticmethod
    def tz(value):
        return ET('tz', value, is_joint=False)
    
    @staticmethod
    def Rx():
        return ET('Rx', None, is_joint=True)
    
    @staticmethod
    def Ry():
        return ET('Ry', None, is_joint=True)
    
    @staticmethod
    def Rz():
        return ET('Rz', None, is_joint=True)
    
    def __mul__(self, other):
        """Chain two ETs together"""
        chain = ETChain()
        chain.add(self)
        if isinstance(other, ETChain):
            for et in other.elements:
                chain.add(et)
        else:
            chain.add(other)
        return chain
    
    def get_transform(self, q=None):
        """Get 4x4 transformation matrix"""
        T = np.eye(4)
        if self.transform_type == 'tx':
            T[0, 3] = self.value
        elif self.transform_type == 'ty':
            T[1, 3] = self.value
        elif self.transform_type == 'tz':
            T[2, 3] = self.value
        elif self.transform_type == 'Rx':
            angle = q if q is not None else 0
            c, s = np.cos(angle), np.sin(angle)
            T[1, 1] = c
            T[1, 2] = -s
            T[2, 1] = s
            T[2, 2] = c
        elif self.transform_type == 'Ry':
            angle = q if q is not None else 0
            c, s = np.cos(angle), np.sin(angle)
            T[0, 0] = c
            T[0, 2] = s
            T[2, 0] = -s
            T[2, 2] = c
        elif self.transform_type == 'Rz':
            angle = q if q is not None else 0
            c, s = np.cos(angle), np.sin(angle)
            T[0, 0] = c
            T[0, 1] = -s
            T[1, 0] = s
            T[1, 1] = c
        return T


class ETChain:
    """Chain of Elementary Transforms representing a robot"""
    
    def __init__(self):
        self.elements = []
        self.qlim = None
        self.q = None  # Current joint positions
        
    def add(self, et):
        self.elements.append(et)
        
    def __mul__(self, other):
        chain = ETChain()
        for et in self.elements:
            chain.add(et)
        if isinstance(other, ETChain):
            for et in other.elements:
                chain.add(et)
        else:
            chain.add(other)
        return chain
    
    def fkine(self, q):
        """Forward kinematics - compute end-effector pose from joint angles"""
        T = np.eye(4)
        joint_idx = 0
        for et in self.elements:
            if et.is_joint:
                T = T @ et.get_transform(q[joint_idx])
                joint_idx += 1
            else:
                T = T @ et.get_transform()
        return SE3(T)
    
    def jacob0(self, q):
        """Compute geometric Jacobian in base frame"""
        n = len(q)
        J = np.zeros((6, n))
        
        # Compute transforms to each joint
        T = np.eye(4)
        transforms = [T.copy()]
        joint_idx = 0
        
        for et in self.elements:
            if et.is_joint:
                T = T @ et.get_transform(q[joint_idx])
                joint_idx += 1
            else:
                T = T @ et.get_transform()
            if et.is_joint:
                transforms.append(T.copy())
        
        # End-effector position
        T_ee = self.fkine(q).A
        p_ee = T_ee[:3, 3]
        
        # Compute Jacobian columns
        joint_idx = 0
        T = np.eye(4)
        for i, et in enumerate(self.elements):
            if et.is_joint:
                # Get axis of rotation
                if et.transform_type == 'Rx':
                    axis = T[:3, 0]
                elif et.transform_type == 'Ry':
                    axis = T[:3, 1]
                elif et.transform_type == 'Rz':
                    axis = T[:3, 2]
                else:
                    axis = np.zeros(3)
                
                # Position of this joint
                p_joint = T[:3, 3]
                
                # Linear velocity contribution
                J[:3, joint_idx] = np.cross(axis, p_ee - p_joint)
                # Angular velocity contribution
                J[3:, joint_idx] = axis
                
                T = T @ et.get_transform(q[joint_idx])
                joint_idx += 1
            else:
                T = T @ et.get_transform()
        
        return J
    
    def ikine_LM(self, Tep, q0, ilimit=50, slimit=10, tol=1e-6):
        """Levenberg-Marquardt IK solver"""
        q = np.array(q0, dtype=float)
        
        for _ in range(slimit):
            for _ in range(ilimit):
                T_current = self.fkine(q)
                
                # Compute pose error
                e = self._pose_error(Tep, T_current.A)
                
                if np.linalg.norm(e) < tol:
                    return IKSolution(q, True)
                
                # Compute Jacobian
                J = self.jacob0(q)
                
                # Levenberg-Marquardt step
                lambda_val = 0.01
                Jt = J.T
                H = Jt @ J + lambda_val * np.eye(len(q))
                dq = np.linalg.solve(H, Jt @ e)
                
                q = q + dq
                
                # Apply joint limits if defined
                if self.qlim is not None:
                    q = np.clip(q, self.qlim[0], self.qlim[1])
        
        return IKSolution(q, False)
    
    def _pose_error(self, Tep, T_current):
        """Compute 6D pose error (position + orientation)"""
        # Position error
        e_pos = Tep[:3, 3] - T_current[:3, 3]
        
        # Orientation error using angle-axis
        R_err = Tep[:3, :3] @ T_current[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        
        if angle < 1e-6:
            e_rot = np.zeros(3)
        else:
            axis = np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ]) / (2 * np.sin(angle))
            e_rot = angle * axis
        
        return np.concatenate([e_pos, e_rot])


class SE3:
    """SE3 pose representation"""
    
    def __init__(self, T):
        if isinstance(T, np.ndarray):
            self._T = T
        else:
            self._T = np.eye(4)
    
    @property
    def A(self):
        return self._T
    
    @property
    def t(self):
        return self._T[:3, 3]
    
    @property
    def R(self):
        return self._T[:3, :3]


class IKSolution:
    """IK solution container"""
    
    def __init__(self, q, success):
        self.q = q
        self.success = success


def create_so101():
    """Create SO101 robot kinematic model (4 DOF for FK/IK, excluding base rotation and gripper)"""
    # Joint 2: Pitch
    E4 = ET.tx(0.02943)
    E5 = ET.tz(0.05504)
    E6 = ET.Ry()
    
    # Joint 3: Elbow
    E7 = ET.tx(0.1127)
    E8 = ET.tz(-0.02798)
    E9 = ET.Ry()

    # Joint 4: Wrist Pitch
    E10 = ET.tx(0.13504)
    E11 = ET.tz(0.00519)
    E12 = ET.Ry()
    
    # Joint 5: Wrist Roll
    E13 = ET.tx(0.0593)
    E14 = ET.tz(0.00996)
    E15 = ET.Rx()
    
    so101 = E4 * E5 * E6 * E7 * E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15
    
    # Set joint limits
    so101.qlim = [[-3.14158, -0.2, -1.5, -3.14158], 
                  [0.2, 3.14158, 1.5, 3.14158]]
    
    return so101


def lerobot_FK(qpos_data, robot):
    """
    Forward Kinematics: compute end-effector pose from joint angles
    
    Args:
        qpos_data: Joint positions [Pitch, Elbow, Wrist_Pitch, Wrist_Roll] (4 DOF)
        robot: Robot kinematic model
        
    Returns:
        np.array: [X, Y, Z, roll, pitch, yaw] end-effector pose
    """
    if len(qpos_data) != len(robot.qlim[0]):
        raise Exception("Joint dimensions mismatch")
    
    T = robot.fkine(qpos_data)
    X, Y, Z = T.t
    R_mat = T.R
    
    # Extract Euler angles (ZYX convention)
    beta = atan2(-R_mat[2, 0], sqrt(R_mat[0, 0]**2 + R_mat[1, 0]**2))
    
    if cos(beta) != 0:
        alpha = atan2(R_mat[1, 0] / cos(beta), R_mat[0, 0] / cos(beta))
        gamma = atan2(R_mat[2, 1] / cos(beta), R_mat[2, 2] / cos(beta))
    else:
        alpha = 0
        gamma = atan2(R_mat[0, 1], R_mat[1, 1])
    
    return np.array([X, Y, Z, gamma, beta, alpha])


def lerobot_IK(q_now, target_pose, robot, fast_mode=False):
    """
    Inverse Kinematics: compute joint angles from end-effector pose
    
    Args:
        q_now: Current joint positions (4 DOF)
        target_pose: Target [X, Y, Z, roll, pitch, yaw]
        robot: Robot kinematic model
        fast_mode: If True, use minimal iterations for real-time control (< 1ms)
                   If False, use multiple attempts for higher success rate
        
    Returns:
        tuple: (joint_positions, success)
    """
    if len(q_now) != len(robot.qlim[0]):
        raise Exception("Joint dimensions mismatch")
    
    x, y, z, roll, pitch, yaw = target_pose
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    R_mat = r.as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = [x, y, z]
    
    q_now = np.array(q_now, dtype=np.float64)
    
    if fast_mode:
        # Ultra-fast mode: Single attempt with minimal iterations
        # For real-time control, we expect small deltas from current position
        # Target: < 1ms execution time
        sol = robot.ikine_LM(
            Tep=T,
            q0=q_now,
            ilimit=10,   # Minimal iterations - small deltas converge quickly
            slimit=1,    # Single search attempt
            tol=5e-3     # Looser tolerance for speed
        )
        return sol.q, sol.success
    
    else:
        # Standard mode: Multiple attempts with different parameters
        # Try 1: Standard LM with current position as seed
        sol = robot.ikine_LM(
            Tep=T,
            q0=q_now,
            ilimit=30,
            slimit=10,
            tol=1e-3
        )
        if sol.success:
            return sol.q, True
        
        # Try 2: Looser tolerance
        sol = robot.ikine_LM(
            Tep=T,
            q0=q_now,
            ilimit=50,
            slimit=20,
            tol=5e-3
        )
        if sol.success:
            return sol.q, True
        
        # Try 3: Random perturbation of seed
        for _ in range(3):
            q_perturbed = q_now + np.random.uniform(-0.1, 0.1, len(q_now))
            q_perturbed = np.clip(q_perturbed, robot.qlim[0], robot.qlim[1])
            sol = robot.ikine_LM(
                Tep=T,
                q0=q_perturbed,
                ilimit=50,
                slimit=10,
                tol=5e-3
            )
            if sol.success:
                return sol.q, True
        
        # All attempts failed
        return -1 * np.ones(len(q_now)), False


def compute_jacobian(qpos_data, robot):
    """Compute Jacobian matrix"""
    if len(qpos_data) != len(robot.qlim[0]):
        raise Exception("Joint dimensions mismatch")
    return robot.jacob0(qpos_data)


def manipulability(J):
    """Compute manipulability measure"""
    s = np.linalg.svd(J, compute_uv=False)
    m = sqrt(np.prod(s))
    condition = s[0] / s[-1] if s[-1] > 1e-10 else float('inf')
    return m, condition
