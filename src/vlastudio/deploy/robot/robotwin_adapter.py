"""Control the gripper TCP while preserving RoboTwin's native EE action format."""
import numpy as np
from scipy.spatial.transform import Rotation

from .benchmark_adapters import CartesianAdapter


class RoboTwinAdapter(CartesianAdapter):
    def reset(self, env, observation):
        super().reset(env, observation)
        self.control_frequency = float(getattr(env.config, 'control_freq', 20))
        if self.arms != 2 or self.rotation != 'quat_wxyz' or self.native_type != 'abs':
            raise ValueError('RoboTwin EE control requires two absolute xyz + WXYZ + gripper targets')
        native = self._pose_matrices(self.target.reshape(2, 8)[:, :7])
        tcp = self._pose_matrices(env.get_tcp_poses())
        # Calibrate the rigid tool transform from the loaded embodiment, not a
        # hard-coded distance. Reset rebinds it when the robot is recreated.
        self.native_to_tcp = np.linalg.inv(native) @ tcp
        self.tcp_to_native = np.linalg.inv(self.native_to_tcp)
        self.tcp_controls = CartesianAdapter(self.config)
        self.tcp_controls.target = self._change_control_point(self.target, self.native_to_tcp)

    @staticmethod
    def _pose_matrices(poses):
        poses = np.asarray(poses, dtype=float).reshape(2, 7)
        if not np.all(np.isfinite(poses)):
            raise ValueError('Non-finite RoboTwin tool pose')
        matrices = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
        matrices[:, :3, :3] = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
        matrices[:, :3, 3] = poses[:, :3]
        return matrices

    def _change_control_point(self, actions, transform):
        result = np.asarray(actions, dtype=float).reshape(2, 8).copy()
        poses = self._pose_matrices(result[:, :7]) @ transform
        result[:, :3] = poses[:, :3, 3]
        result[:, 3:7] = Rotation.from_matrix(poses[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        return result.reshape(-1)

    def teleop_action(self, message, observation):
        # Keep the requested TCP position during rotation and hold. Reanchoring
        # every frame would integrate the native servo's tracking error as motion.
        tcp_action = self.tcp_controls.teleop_action(message, observation)['action']
        self.target = self._change_control_point(tcp_action, self.tcp_to_native)
        return {'action': self.target.copy()}
