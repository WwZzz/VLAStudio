"""
Agilex Aloha Robot Interface
ROS-based robot interface for Agilex Aloha dual-arm robot
"""

import numpy as np
import rospy
import traceback
import time
import os
import h5py
from types import SimpleNamespace
from typing import Dict, Optional, Any

from vlastudio.deploy.robot.base import BaseRobot
from .rosoperator import RosOperator, RosTeleOperator


class AgilexAloha(BaseRobot):
    """
    Agilex Aloha dual-arm robot interface using ROS
    """
    def __init__(self,
                 name: str = "agilex_aloha",
                 max_size_mb: int = 64,
                 fps: float = 40.0,
                 control_shm_name: Optional[str] = None,
                 init_pos: dict = {},
                 ros_operator: dict = {},
                 limit_pos: dict = {},
                 **kwargs):
        """
        Initialize the Agilex Aloha robot
        
        Args:
            name: Name for the shared memory segment
            max_size_mb: Maximum size of shared memory in MB
            fps: Control frequency in Hz
            control_shm_name: Name of the control shared memory
            init_pos: Initial positions for left and right arms
            ros_operator: ROS operator configuration
            limit_pos: Position limits
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        self.ros_operator = RosOperator(SimpleNamespace(**ros_operator))
        self.init_pos = init_pos
        self.limit_pos = limit_pos

    def reset(self):
        try:
            left = self.init_pos.get('left', None)
            right = self.init_pos.get('right', None)
            if left is not None and right is not None:
                self.ros_operator.puppet_arm_publish_continuous(left, right)
            return True
        except Exception as e:
            print(f"Failed to reset robot to initial pos: \n Left: {left} \n Right: {right}")
            traceback.print_exc()
            return False

    def connect(self):
        res = self.reset()
        time.sleep(3)
        return res
    
    def get_observation(self) -> Dict[str, Any]:
        """
        """
        result = self.ros_operator.get_frame()
        if not result: return False
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result

        obs = {}
        # organize image
        obs['image'] = dict(primary=img_front, wrist_left=img_left, wrist_right=img_right)
        # organize depth
        if self.ros_operator.args.use_depth_image:
            obs['depth'] = dict(primary=img_front_depth, wrist_left=img_left_depth, wrist_right=img_right_depth)
        obs['qpos'] = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0).astype(np.float32)
        obs['qvel'] = np.concatenate((np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0).astype(np.float32)
        obs['effort'] = np.concatenate((np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0).astype(np.float32)
        if self.ros_operator.args.use_robot_base:
            obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]
        return obs

    def get_action_dim(self):
        return 14
    
    def publish_action(self, action: np.ndarray):
        """
        Simulates publishing an action command to the robot and validates the action format.
        """
        left_action, right_action = action[:7], action[7:14]
        self.ros_operator.puppet_arm_publish(left_action, right_action)
        if self.ros_operator.args.use_robot_base:
            vel_action = action[14:16]
            self.ros_operator.robot_base_publish(vel_action)

    def is_running(self) -> bool:
        """
        Checks if the robot system is still running.
        A threading event is used to simulate system shutdown.
        """
        return not rospy.is_shutdown()

    def shutdown(self):
        """Shutdown the robot"""
        rospy.signal_shutdown("Shutdown robot node")

    def close(self):
        """Close the robot"""
        super().close()
        rospy.signal_shutdown("Shutdown robot node")


class AgilexAlohaTele(BaseRobot):
    """
    Agilex Aloha dual-arm robot with teleoperation support
    """
    def __init__(self,
                 name: str = "agilex_aloha_tele",
                 max_size_mb: int = 64,
                 fps: float = 40.0,
                 control_shm_name: Optional[str] = None,
                 init_pos: dict = {},
                 ros_operator: dict = {},
                 limit_pos: dict = {},
                 use_master_action: bool = False,
                 use_ee_space: bool = False,
                 **kwargs):
        """
        Initialize the Agilex Aloha robot with teleoperation
        
        Args:
            name: Name for the shared memory segment
            max_size_mb: Maximum size of shared memory in MB
            fps: Control frequency in Hz
            control_shm_name: Name of the control shared memory
            init_pos: Initial positions for left and right arms
            ros_operator: ROS operator configuration
            limit_pos: Position limits
            use_master_action: Whether to use master arm action
            use_ee_space: Whether to use end-effector space
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        self.ros_operator = RosTeleOperator(SimpleNamespace(**ros_operator))
        self.init_pos = init_pos
        self.limit_pos = limit_pos
        self.use_master_action = use_master_action
        self.use_ee_space = use_ee_space

    def connect(self):
        return True
    
    def get_observation(self) -> Dict[str, Any]:
        """
        """
        result = self.ros_operator.get_frame()
        if not result: return False
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right,master_arm_left, master_arm_right, robot_base, puppet_eef_left, puppet_eef_right,) = result

        obs = {}
        # organize image
        obs['image'] = dict(primary=img_front, wrist_left=img_left, wrist_right=img_right)
        # organize depth
        if self.ros_operator.args.use_depth_image:
            obs['depth'] = dict(primary=img_front_depth, wrist_left=img_left_depth, wrist_right=img_right_depth)
        obs['qpos'] = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0).astype(np.float32)
        obs['qvel'] = np.concatenate((np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0).astype(np.float32)
        
        obs['effort'] = np.concatenate([
            np.array([puppet_eef_left.pose.position.x, puppet_eef_left.pose.position.y, puppet_eef_left.pose.position.z, ]), 
            np.array([puppet_eef_left.pose.orientation.x, puppet_eef_left.pose.orientation.y, puppet_eef_left.pose.orientation.z,puppet_eef_left.pose.orientation.w,]), 
            np.array([puppet_arm_left.position[-1]]), # left gripper
            np.array([puppet_eef_right.pose.position.x, puppet_eef_right.pose.position.y, puppet_eef_right.pose.position.z]), 
            np.array([puppet_eef_right.pose.orientation.x,puppet_eef_right.pose.orientation.y,puppet_eef_right.pose.orientation.z,puppet_eef_right.pose.orientation.w,] ),
            np.array([puppet_arm_right.position[-1]]), # right gripper
            ], axis=0)
        
        if self.ros_operator.args.use_robot_base:
            obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]

        obs['qpos_master'] = np.concatenate((
            np.array(master_arm_left.position), 
            np.array(master_arm_right.position)), 
            axis=0).astype(np.float32)
        return obs

    def get_action_dim(self):
        return 14
    
    def publish_action(self, action: np.ndarray):
        """
        Simulates publishing an action command to the robot and validates the action format.
        """
        left_action, right_action = action[:7], action[7:14]
        self.ros_operator.puppet_arm_publish(left_action, right_action)
        if self.ros_operator.args.use_robot_base:
            vel_action = action[14:16]
            self.ros_operator.robot_base_publish(vel_action)

    def is_running(self) -> bool:
        """
        Checks if the robot system is still running.
        A threading event is used to simulate system shutdown.
        """
        return not rospy.is_shutdown()

    def shutdown(self):
        """Shutdown the robot"""
        rospy.signal_shutdown("Shutdown robot node")

    def close(self):
        """Close the robot"""
        super().close()
        rospy.signal_shutdown("Shutdown robot node")

    def save_episode(self, file_path, observations, actions=None):
        """Save observations into episode"""
        save_dir = os.path.dirname(file_path)
        os.makedirs(save_dir, exist_ok=True)
        # generate actions
        if self.use_ee_space:
            assert not self.use_master_action, "Master arms's ee_pose cannot be recorded"
            action_key = 'effort'
        else:
            if self.use_master_action:
                action_key = 'qpos_master'
            else:
                action_key = 'qpos'
        with h5py.File(file_path, 'w') as f:
            actions = np.stack([obs[action_key] for obs in observations]).astype(np.float32)
            f.create_dataset('actions', data=actions)

            obs_group = f.create_group('observations')
            if observations:
                for key in observations[0].keys():
                    if key=='image': continue
                    data_list = [obs[key] for obs in observations]
                    try:
                        obs_group.create_dataset(key, data=np.stack(data_list))
                    except (TypeError, ValueError) as e:
                        print(f"Warning: Could not stack data for key '{key}'. Skipping. Error: {e}")
                images = observations[0]['image']
                img_group = obs_group.create_group('image')
                for img_key in images.keys():
                    data_list = [obs['image'][img_key] for obs in observations]
                    try:
                        img_group.create_dataset(img_key, data=np.stack(data_list).astype(np.uint8))
                    except Exception as e:
                        print(f"Failed to save images {img_key} due to error: {e}")




