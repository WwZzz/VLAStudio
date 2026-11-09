"""RoboTwin absolute xyz + WXYZ targets, anchored to measured poses."""
import numpy as np

from .benchmark_adapters import CartesianAdapter


class RoboTwinAdapter(CartesianAdapter):
    def reset(self, env, observation):
        super().reset(env, observation)
        self.control_frequency = float(getattr(env.config, 'control_freq', 20))
        if self.arms != 2 or self.width != 8 or self.native_type != 'abs':
            raise ValueError('RoboTwin EE control requires two absolute xyz + WXYZ + gripper targets')

    def teleop_action(self, message, observation):
        self.target.reshape(2, 8)[:, :7] = np.asarray(observation['state_ee']).reshape(2, 8)[:, :7]
        return super().teleop_action(message, observation)
