"""RLBench interactive control in one owning process, with explicit cameras."""
import numpy as np
import time
from loguru import logger
from vlastudio.benchmark.rlbench import RLBenchEnv


class RLBenchInteractiveEnv(RLBenchEnv):
    def _create_rlbench_env(self):
        import os
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        if self.ctrl_space != 'ee' or self.ctrl_type != 'abs':
            raise ValueError('Interactive RLBench requires absolute EE control')
        observation = ObservationConfig()
        observation.set_all(False)
        observation.set_all_low_dim(True)
        for name in self.camera_names:
            camera = getattr(observation, name + '_camera')
            camera.rgb = True
            camera.image_size = list(self.image_size)
        mode = getattr(self.config, 'arm_mode', 'ik')
        modes = {'ik': EndEffectorPoseViaIK, 'planning': EndEffectorPoseViaPlanning}
        if mode not in modes:
            raise ValueError('RLBench arm_mode must be ik or planning')
        env = Environment(MoveArmThenGripper(modes[mode](), Discrete()),
                          obs_config=observation, headless=self.headless)
        if 'COPPELIASIM_ROOT' in os.environ:
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.environ['COPPELIASIM_ROOT']
        env.launch()
        self.min_action = self.max_action = None
        return env

    def obs2meta(self, obs):
        result = super().obs2meta(obs)
        result.state_joint = np.r_[obs.joint_positions, obs.gripper_open].astype(np.float32)
        result.state_ee = np.r_[obs.gripper_pose, obs.gripper_open].astype(np.float32)
        return result

    def reset(self):
        np.random.seed(int(getattr(self.config, 'seed', 7)))
        return super().reset()

    def step(self, action):
        from rlbench.backend.exceptions import InvalidActionError
        try:
            obs, reward, terminate = self.task.step(self.meta2act(action))
        except InvalidActionError as error:
            # An unreachable keyboard increment must not lock out reverse motion.
            native = getattr(self.config, 'input_mode', 'ee_delta') == 'native'
            now = time.monotonic()
            if now - getattr(self, '_last_rejection_log', 0.) > 2:
                logger.warning('RLBench rejected an unreachable target ({}): {}',
                               'policy paused; Reset to continue' if native else 'try another direction', error)
                self._last_rejection_log = now
            self.prev_obs = self.obs2meta(self.task.get_observation())
            return self.prev_obs, 0., native, False, {'success': False, 'action_rejected': True}
        success, _ = self.task._task.success()
        self.prev_obs = self.obs2meta(obs)
        return self.prev_obs, reward, bool(terminate and not success), False, {'success': bool(success)}

    def close(self):
        self.force_close()
