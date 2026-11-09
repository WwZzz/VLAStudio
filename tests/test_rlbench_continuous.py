"""Exercise real RLBench IK after the former 500-step deployment horizon."""
import copy
import numpy as np
import pytest
import yaml

pytest.importorskip('rlbench')

from vlastudio.deploy.robot.benchmark_robot import BenchmarkRobot
from vlastudio.webui.profiles import config


@pytest.mark.parametrize('task', ['CloseDrawer', 'ReachTarget'])
def test_keyboard_target_moves_after_600_steps(task):
    args = copy.deepcopy(yaml.safe_load(config('robot', 'rlbench').read_text())[0]['args'])
    args['task'] = task
    args['environment']['args']['camera_names'] = []
    args['observation']['cameras'] = []
    robot = BenchmarkRobot(**args)
    try:
        robot.connect()
        assert robot.max_steps is None
        for _ in range(600):
            robot._step()
            assert not robot.blocked
        # An invalid target is rejected, without permanently disabling teleop.
        invalid = robot.adapter.idle_action()
        invalid[:3] = [10., 10., 10.]
        robot.publish_action(invalid)
        robot._step()
        assert robot.cached['action_rejected'] and not robot.blocked
        before = robot.cached['state_ee'][:3].copy()
        for _ in range(15):
            robot.publish_action(robot.process_action({'action': [-.002, 0, 0, 0, 0, 0, 0]})['action'])
            robot._step()
            assert not robot.blocked
        assert np.linalg.norm(robot.cached['state_ee'][:3] - before) > .01
        assert robot.steps == 616
        assert not robot.cached['truncated']
        robot.reset()
        assert robot.steps == 0 and not robot.blocked
        robot._step()
        assert not robot.blocked
        robot.input_mode = robot.env.config.input_mode = 'native'
        robot.publish_action(invalid)
        robot._step()
        assert robot.blocked and robot.cached['terminated']
    finally:
        robot.close()
