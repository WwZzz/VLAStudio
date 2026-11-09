"""Integration regression: interactive control must survive the native horizon."""
import copy
from pathlib import Path
import numpy as np
import pytest
import yaml

pytest.importorskip('metaworld')

from vlastudio.deploy.robot.benchmark_robot import BenchmarkRobot


@pytest.mark.parametrize('task', ['reach-v3', 'push-v3'])
def test_motion_after_native_horizon_and_reset(task):
    from metaworld.sawyer_xyz_env import SawyerXYZEnv
    from vlastudio.webui.profiles import config
    args = copy.deepcopy(yaml.safe_load(config('robot', 'metaworld').read_text())[0]['args'])
    args['task'] = task
    # Keep real physics and controller semantics, without expensive test rendering.
    args['environment']['args']['use_camera'] = False
    robot = BenchmarkRobot(**args)
    try:
        robot.connect()
        for _ in range(600):
            robot._step()
            assert not robot.blocked, robot.cached
        before = robot.cached['state_ee'][:3].copy()
        for _ in range(15):
            delta = np.array([.002, 0, 0, 0, 0, 0, 0])
            robot.publish_action(robot.process_action({'action': delta})['action'])
            robot._step()
        assert np.linalg.norm(robot.cached['state_ee'][:3] - before) > .001
        assert robot.steps == 615
        assert not robot.cached['truncated']
        robot.reset()
        assert robot.steps == 0 and not robot.blocked
        robot._step()
        assert robot.steps == 1
        # A deliberately configured device horizon must still work.
        robot.max_steps = 3
        robot._step()
        robot._step()
        assert robot.blocked and robot.cached['truncated']
        assert SawyerXYZEnv.max_path_length == 500  # Original benchmark unchanged.
    finally:
        robot.close()
