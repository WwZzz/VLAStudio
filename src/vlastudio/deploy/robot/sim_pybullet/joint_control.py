"""
PyBullet Absolute Joint Control Robot
A PyBullet-based robot with absolute joint position control
"""

import pybullet as p
import pybullet_data
import numpy as np
from typing import Dict, Any, List, Optional

from vlastudio.deploy.robot.base import BaseRobot
from vlastudio.deploy.utils import RateLimiter


class AbsJointRobot(BaseRobot):
    """
    PyBullet-based absolute joint control robot implementing BaseRobot API
    """

    def __init__(self,
                 name: str = "pybullet_joint",
                 max_size_mb: int = 64,
                 fps: float = 240.0,
                 control_shm_name: Optional[str] = None,
                 robot_urdf_path: str = "franka_panda/panda.urdf",
                 initial_joint_positions: List[float] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
                 gripper_joint_indices: List[int] = [9, 10],
                 gripper_width: float = 0.04,
                 use_gui: bool = True,
                 **kwargs):
        """
        Initialize PyBullet joint control robot
        
        Args:
            name: Name for the shared memory segment
            max_size_mb: Maximum size of shared memory in MB
            fps: Control frequency in Hz
            control_shm_name: Name of the control shared memory
            robot_urdf_path: Path to the robot URDF file
            initial_joint_positions: Initial joint positions
            gripper_joint_indices: Gripper joint indices
            gripper_width: Maximum gripper width
            use_gui: Whether to use PyBullet GUI
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        
        self.use_gui = use_gui
        self.physics_client = None
        self.robot_id = None

        self._robot_urdf_path = robot_urdf_path
        self._initial_joint_positions = initial_joint_positions
        self.gripper_joint_indices = gripper_joint_indices
        self.gripper_width = gripper_width

        # Get controllable joint indices
        _temp_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        _temp_robot_id = p.loadURDF(self._robot_urdf_path, useFixedBase=True, physicsClientId=_temp_client)
        self._num_joints = p.getNumJoints(_temp_robot_id, physicsClientId=_temp_client)
        self._controllable_joints = [i for i in range(self._num_joints) if p.getJointInfo(
            _temp_robot_id, i, physicsClientId=_temp_client)[2] != p.JOINT_FIXED]
        p.disconnect(physicsClientId=_temp_client)

        self._rate_limiter = RateLimiter()

    def connect(self):
        """连接到PyBullet物理服务器并加载机器人。"""
        if self.physics_client is not None:
            print("Already connected to PyBullet.")
            return True
        try:
            print("Connecting to PyBullet...")
            client_mode = p.GUI if self.use_gui else p.DIRECT
            self.physics_client = p.connect(client_mode)

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)

            # 加载地面和机器人
            p.loadURDF("plane.urdf")
            self.robot_id = p.loadURDF(self._robot_urdf_path, [0, 0, 0], useFixedBase=True)

            # 重置机器人到初始姿态
            for i, pos in zip(self._controllable_joints, self._initial_joint_positions):
                p.resetJointState(self.robot_id, i, pos)

            print(f"Robot '{self._robot_urdf_path}' loaded with ID: {self.robot_id}")
            return True
        except Exception as e:
            print(f"Failed to connect to PyBullet: {e}")
            return False

    def get_action_dim(self):
        """返回动作空间的维度（关节数+1个夹爪）"""
        return len(self._controllable_joints) + 1

    def get_observation(self) -> Dict[str, Any]:
        """
        获取观测，包括关节位置和夹爪宽度。
        """
        if not self.is_running():
            raise ConnectionError("PyBullet is not running. Call connect() first.")

        joint_states = p.getJointStates(self.robot_id, self._controllable_joints)
        qpos = np.array([state[0] for state in joint_states])

        # 夹爪状态
        gripper_states = p.getJointStates(self.robot_id, self.gripper_joint_indices)
        gripper_pos = np.mean([state[0] for state in gripper_states])

        obs = {
            'qpos': qpos,
            'gripper': gripper_pos
        }
        return obs

    def publish_action(self, action: np.ndarray):
        """
        绝对关节控制：action[:n]为关节角度，action[-1]为夹爪开合（0~1）。
        """
        # print("publish_action", action)
        if not self.is_running():
            raise ConnectionError("PyBullet is not running.")

        joint_targets = action[:len(self._controllable_joints)]
        gripper_target = float(action[-1]) * self.gripper_width

        # print(f"Setting joint targets: {joint_targets}")
        # print(f"Gripper target: {gripper_target}")

        MAX_FORCE = 100
        # print("joint len",len(self._controllable_joints))
        for i, joint_id in enumerate(self._controllable_joints):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_targets[i],
                force=MAX_FORCE,
            )
        for joint_id in self.gripper_joint_indices:
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=gripper_target,
                force=20
            )
        p.stepSimulation()

    def is_running(self) -> bool:
        """检查PyBullet仿真是否仍在运行。"""
        return self.physics_client is not None and p.isConnected(self.physics_client)

    def shutdown(self):
        """Disconnect from PyBullet server"""
        if self.is_running():
            print("Disconnecting from PyBullet...")
            p.disconnect(self.physics_client)
            self.physics_client = None

    def close(self):
        """Close the robot"""
        super().close()
        self.shutdown()
