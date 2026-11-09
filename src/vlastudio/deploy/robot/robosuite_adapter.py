"""robosuite 1.5 BASIC / world OSC_POSE keyboard control."""
import numpy as np

from .benchmark_adapters import CartesianAdapter


class RobosuiteAdapter(CartesianAdapter):
    def reset(self, env, observation):
        super().reset(env, observation)
        base = env.env.base_env
        if self.arms != len(base.robots) or self.width != 7 or self.native_type != 'delta':
            raise ValueError('robosuite profile requires one 7D OSC action per robot')
        scales = []
        for robot in base.robots:
            controllers = getattr(robot, 'part_controllers', {})
            controller = controllers.get('right')
            if (controller is None or controller.name != 'OSC_POSE'
                    or controller.input_type != 'delta' or controller.input_ref_frame != 'world'
                    or list(robot._action_split_indexes.items()) != [
                        ('right', (0, 6)), ('right_gripper', (6, 7))]):
                raise ValueError('Use robosuite 1.5 BASIC with world-frame delta OSC_POSE and a 1D gripper')
            low, high = np.asarray(controller.output_min), np.asarray(controller.output_max)
            in_low, in_high = np.asarray(controller.input_min), np.asarray(controller.input_max)
            if (low.size != 6 or np.any(high <= 0) or not np.allclose(high, -low)
                    or not np.allclose(in_high, -in_low)):
                raise ValueError('Keyboard control requires symmetric 6D OSC input/output bounds')
            scales.append((in_high - in_low) / (high - low))
        self.scales = np.stack(scales)
        self.low, self.high = base.action_spec
        self.control_frequency = float(base.control_freq)
        if getattr(env.config, 'input_mode', 'ee_delta') == 'native':
            # Native idle must hold the open gripper in native units too.
            self.target = self.environment_action(self.target).action

    def teleop_action(self, message, observation):
        result = super().teleop_action(message, observation)
        self.target.reshape(self.arms, 7)[:, :6] *= self.scales
        result['action'] = self.target.copy()
        return result

    def environment_action(self, action):
        result = super().environment_action(action)
        result.action = np.clip(result.action, self.low, self.high)
        return result
