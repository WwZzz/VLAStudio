"""
PyBullet Delta End-Effector Control Robot
A PyBullet-based robot implementation controlled by end-effector deltas
"""

import pybullet as p
import pybullet_data
import numpy as np
from typing import Dict, Any, List, Optional

from vlastudio.deploy.robot.base import BaseRobot
from vlastudio.deploy.utils import RateLimiter


class DeltaEERobot(BaseRobot):
    """
    A PyBullet-based robot implementation controlled by end-effector deltas.
    This class encapsulates PyBullet interaction logic for connecting to simulation,
    getting observations, publishing actions, and shutdown.
    """

    def __init__(self,
                 name: str = "pybullet_delta_ee",
                 max_size_mb: int = 64,
                 fps: float = 240.0,
                 control_shm_name: Optional[str] = None,
                 robot_urdf_path: str = "franka_panda/panda.urdf",
                 ee_link_index: int = 8,
                 initial_joint_positions: List[float] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
                 gripper_joint_indices: List[int] = [9, 10],
                 gripper_width: float = 0.04,
                 use_gui: bool = True,
                 **kwargs):
        """
        Initialize PyBullet robot environment
        
        Args:
            name: Name for the shared memory segment
            max_size_mb: Maximum size of shared memory in MB
            fps: Control frequency in Hz
            control_shm_name: Name of the control shared memory
            robot_urdf_path: Path to the robot URDF file
            ee_link_index: End-effector link index
            initial_joint_positions: Initial joint positions
            gripper_joint_indices: Gripper joint indices
            gripper_width: Maximum gripper width
            use_gui: Whether to use PyBullet GUI
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        
        self.use_gui = use_gui
        self.physics_client = None
        self.robot_id = None

        # Robot parameters
        self._robot_urdf_path = robot_urdf_path
        self._ee_link_index = ee_link_index
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

        # Visual parameters
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.5, 0, 0.5],
            distance=1.5,
            yaw=90,
            pitch=-20,
            roll=0,
            upAxisIndex=2
        )
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0
        )

        # Rate limiter
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
        return 8 # 7 joints + 1 gripper
    
    def get_observation(self) -> Dict[str, Any]:
        """
        从PyBullet获取多模态观测数据。

        Returns:
            Dict[str, Any]: 包含关节位置和图像的观测字典。
        """
        if not self.is_running():
            raise ConnectionError("PyBullet is not running. Call connect() first.")

        # 1. 获取关节位置 (qpos)
        joint_states = p.getJointStates(self.robot_id, self._controllable_joints)
        qpos = np.array([state[0] for state in joint_states])

        # 2. 获取图像
        width, height, rgb_img, _, _ = p.getCameraImage(
            width=224,
            height=224,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        # PyBullet返回RGBA，我们需要转换为RGB并重塑
        rgb_array = np.array(rgb_img, dtype=np.uint8)
        rgb_array = rgb_array.reshape((height, width, 4))
        rgb_array = rgb_array[:, :, :3]  # 丢弃 alpha 通道

        # 组装观测字典，以匹配基类中 obs2meta 的期望格式
        obs = {
            'qpos': qpos,
            'image': {
                'front_camera': rgb_array
            }
        }
        return obs

    def publish_action(self, action: np.ndarray):
        """
        发布一个新的动作来控制机器人。

        Args:
            action (np.ndarray): 一个3D或6D的向量。
                                 - action[:3]: 期望的EE位置变化 (dx, dy, dz)。
                                 - action[3:6]: (可选) 期望的EE姿态变化 (d_roll, d_pitch, d_yaw)。
        """
        if not self.is_running():
            raise ConnectionError("PyBullet is not running.")

        # --- 位置控制 (与之前相同) ---
        delta_pos = action[:3]
        ee_state = p.getLinkState(self.robot_id, self._ee_link_index, computeForwardKinematics=True)
        current_pos = np.array(ee_state[4])
        current_orn_quat = np.array(ee_state[5])  # 当前姿态是一个四元数 [x, y, z, w]
        target_pos = current_pos + delta_pos
        
        # print(f"Delta position: {delta_pos}")
        # print(f"Current position: {current_pos}")
        # print(f"Target position: {target_pos}")

        # --- 姿态控制 (新增部分) ---
        # 默认目标姿态为当前姿态
        target_orn_quat = current_orn_quat

        # 如果action包含姿态信息
        if len(action) >= 6:
            delta_orn_euler = action[3:6]

            # 1. 将欧拉角增量转换为四元数增量
            delta_orn_quat = p.getQuaternionFromEuler(delta_orn_euler)

            # 2. 将当前姿态四元数与增量四元数相乘，得到目标姿态
            # p.multiplyTransforms 返回组合后的 (位置, 姿态)
            # 我们只需要姿态部分，即索引为1的结果
            _, target_orn_quat = p.multiplyTransforms(
                positionA=[0, 0, 0],  # 位置不重要
                orientationA=current_orn_quat,  # 基础姿态
                positionB=[0, 0, 0],  # 位置不重要
                orientationB=delta_orn_quat  # 要应用的旋转
            )
            target_gripper_pos = self.gripper_width * action[-1]
        else:
            # If no gripper action provided, keep current gripper position
            gripper_states = p.getJointStates(self.robot_id, self.gripper_joint_indices)
            target_gripper_pos = np.mean([state[0] for state in gripper_states])

        # print(f"Gripper target: {target_gripper_pos}")

        # --- 逆运动学 (IK) ---
        # 使用新的目标位置和目标姿态
        target_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self._ee_link_index,
            targetPosition=target_pos,
            targetOrientation=target_orn_quat,  # <--- 使用计算出的目标姿态
            # 添加一些求解器参数可以提高稳定性和成功率
            solver=0,
            maxNumIterations=100,
            residualThreshold=.01
        )
        
        # print(f"Target joint positions: {target_joint_positions[:len(self._controllable_joints)]}")

        # --- 电机控制 (与之前相同) ---
        MAX_FORCE = 100
        for i, joint_id in enumerate(self._controllable_joints):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_joint_positions[i],
                force=MAX_FORCE,
            )
        # 为每个抓夹关节设置目标位置
        for joint_id in self.gripper_joint_indices:
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_gripper_pos,
                force=20  # 抓夹的力量可以小一些
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