"""Regressions for native policy units and unsupported teleoperation axes."""
from types import SimpleNamespace
import sys
import numpy as np
import pytest

from vlastudio.deploy.robot.benchmark_adapters import MetaWorldAdapter
from vlastudio.deploy.robot.benchmark_adapters import RLBenchAdapter
from vlastudio.deploy.robot.benchmark_robot import BenchmarkRobot


def test_metaworld_physical_delta_and_gripper():
    adapter = MetaWorldAdapter(dict(arms=1, type='delta', space='ee'))
    env = SimpleNamespace(env=SimpleNamespace(unwrapped=SimpleNamespace(action_scale=.01)))
    adapter.reset(env, {})
    action = adapter.teleop_action({'action': [.002, -.003, 0, 0, 0, 0, -.1]}, {})['action']
    np.testing.assert_allclose(action, [.2, -.3, 0, -.8])
    np.testing.assert_allclose(adapter.idle_action(), [0, 0, 0, -.8])
    with pytest.raises(ValueError, match='XYZ'):
        adapter.teleop_action({'action': [0, 0, 0, .01, 0, 0, 0]}, {})


def test_rlbench_deltas_do_not_accumulate_against_unreached_targets():
    adapter = RLBenchAdapter(dict(arms=1, type='abs', space='ee', rotation='quat_xyzw', initial_target='state_ee'))
    measured = np.array([.3, 0, 1.4, 0, 0, 0, 1, 1.])
    obs = {'state_ee': measured}
    adapter.reset(None, obs)
    for _ in range(20):
        action = adapter.teleop_action({'action': [.002, 0, 0, 0, 0, 0, 0]}, obs)['action']
        np.testing.assert_allclose(action[:3], measured[:3] + [.002, 0, 0])


def test_native_policy_does_not_repeat_gripper_conversion():
    robot = BenchmarkRobot('native-units-test', environment={},
                           action=dict(arms=1, space='ee', type='delta', gripper_scale=-2,
                                       gripper_offset=1,
                                       adapter='vlastudio.deploy.robot.benchmark_adapters.LiberoAdapter'),
                           observation={'fields': {'state': 'state'}}, input_mode='native')
    observed = []
    robot.env = SimpleNamespace(step=lambda action: (
        observed.append(action.action.copy()) or {'state': np.zeros(7)},
        0., False, False, {'success': False}))
    robot.adapter.target = np.zeros(7)
    robot.adapter.preview_data = None
    action = np.array([0., 0., 0., 0., 0., 0., -1.])
    robot.publish_action(action)
    robot._step()
    np.testing.assert_array_equal(observed[0], action)
    np.testing.assert_array_equal(robot.cached['executed_action'], action)


def test_recording_uses_executed_native_gripper_units():
    robot = BenchmarkRobot('recording-units-test', environment={},
                           action=dict(arms=1, space='ee', type='delta', gripper_scale=-2,
                                       gripper_offset=1),
                           observation={'fields': {'state': 'state'}})
    robot.env = SimpleNamespace(step=lambda action: ({'state': np.zeros(7)}, 0., False, {}))
    robot.adapter.target = np.zeros(7)
    robot.publish_action(np.array([0., 0., 0., 0., 0., 0., 1.]))
    robot._step()
    assert robot.cached['executed_action'][-1] == -1.


@pytest.mark.skipif(sys.platform == 'win32', reason='WebUI sessions use Linux file descriptors')
def test_joint_profile_is_rejected_for_collection():
    from vlastudio.webui.session import command_line
    with pytest.raises(ValueError, match='mode'):
        command_line('collect', {'robot': 'aloha_sim_joint', 'output': 'invalid-test'})


@pytest.mark.skipif(sys.platform == 'win32', reason='WebUI sessions use Linux file descriptors')
def test_webui_accepts_explicit_cpu_inference():
    from vlastudio.webui.session import command_line
    cmd = command_line('infer', dict(robot='aloha_sim_joint', policy='127.0.0.1:9000', device='cpu'))
    assert cmd[cmd.index('--device') + 1] == 'cpu'


@pytest.mark.skipif(sys.platform == 'win32', reason='WebUI sessions use Linux file descriptors')
def test_simulator_pause_is_not_reported_as_enabled_teleoperation():
    from vlastudio.webui.session import Session
    from vlastudio.webui.bridge import RUN, PHASES
    session = Session()
    session.proc = SimpleNamespace()
    session.box = bytearray([RUN, RUN, PHASES.index('ready'), 0, RUN])
    session.mode, session.robot = 'collect', 'rlbench'
    session.options, session.reset_after_pause = {}, False
    session.gated, session.reset_ok, session.teleop = True, True, 'keyboard_delta_1arm'
    session.keyboard_arms, session.keyboard_bindings, session.keyboard_axes = 1, [], []
    session.preview.devices = {'rlbench': {'paused': True, 'truncated': True}}
    state = session.state()
    assert state['tele'] == 'blocked'
    assert state['simulation'] == {'paused': True, 'reason': 'limit'}
    assert 'record' not in state['actions']
    assert 'home' in state['actions']
    session.preview.devices['rlbench']['paused'] = False
    assert session.state()['tele'] == 'running'
