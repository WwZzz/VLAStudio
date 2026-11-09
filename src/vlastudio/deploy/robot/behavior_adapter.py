"""Map keyboard EE increments and mobile-base velocity to native OG controller slots."""
import numpy as np
from scipy.spatial.transform import Rotation

from vlastudio.benchmark.base import MetaAction


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


class BehaviorAdapter:
    arms, width, space, native_type = 2, 7, 'ee', 'delta'

    def __init__(self, config):
        self.config = dict(config)
        self.target = None

    def reset(self, env, observation):
        self.env, self.robot = env, env.robot
        self.control_frequency = env.control_frequency
        self.action_dim = self.robot.action_dim
        self.indices = {name: array(idx) for name, idx in self.robot.controller_action_idx.items()}
        self.target = env.no_op_action()
        self.opening = np.array([(self.target[self.indices['gripper_' + arm]][0] + 1) / 2
                                 for arm in self.robot.arm_names])
        if len(self.indices['base']) != 3 or len(self.robot.arm_names) != self.arms:
            raise ValueError('BEHAVIOR teleoperation requires a holonomic base and two arms')
        if any(len(self.indices['arm_' + arm]) != 6 for arm in self.robot.arm_names):
            raise ValueError('BEHAVIOR teleoperation requires six-dimensional IK controllers')

    def idle_action(self):
        result = self.target.copy()
        result[self.indices['base']] = 0
        for arm in self.robot.arm_names:
            result[self.indices['arm_' + arm]] = 0
        return result

    def teleop_action(self, message, observation):
        if message.get('ctrl_space', 'ee') != 'ee' or message.get('ctrl_type', 'delta') != 'delta':
            raise ValueError('BEHAVIOR teleoperation expects ee/delta increments')
        delta = np.asarray(message['action'], dtype=float).reshape(self.arms, 7)
        base = np.asarray(message.get('base_velocity', [0, 0, 0]), dtype=float).reshape(3)
        if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(base)):
            raise ValueError('Non-finite teleoperation command')
        if message.get('base_frame', 'robot') != 'robot':
            raise ValueError('Base velocity must use the robot frame')
        result = self.idle_action()
        limit = np.asarray(self.config.get('base_velocity_limit', [.25, .25, .5]))
        result[self.indices['base']] = np.clip(base, -limit, limit)
        world_to_base = Rotation.from_quat(array(self.robot.get_position_orientation()[1])).inv()
        for i, arm in enumerate(self.robot.arm_names):
            # The UI axes are world-aligned; OG IK commands are relative to the robot base.
            result[self.indices['arm_' + arm]] = np.r_[world_to_base.apply(delta[i, :3]),
                                                        world_to_base.apply(delta[i, 3:6])]
            self.opening[i] = np.clip(self.opening[i] + delta[i, 6], 0, 1)
            result[self.indices['gripper_' + arm]] = 2 * self.opening[i] - 1
        self.target = result.copy()
        return dict(action=result)

    def environment_action(self, action):
        return MetaAction(action=action.copy(), ctrl_space=self.space, ctrl_type=self.native_type)
