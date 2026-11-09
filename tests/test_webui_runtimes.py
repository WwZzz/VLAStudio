"""Runtime selection must preserve control FDs but isolate simulator libraries."""
import json
import os
import sys
import pytest

from vlastudio.webui.runtimes import RuntimeLaunch, resolve_runtime


def test_configured_runtime_clears_foreign_graphics_and_python(tmp_path):
    path = tmp_path / 'runtimes.json'
    path.write_text(json.dumps({'runtimes': {'rlbench': {
        'command': [sys.executable], 'extra_paths': ['/selected/packages'],
        'env': {'COPPELIASIM_ROOT': '/selected/CoppeliaSim', 'DISPLAY': ':88'},
    }}}))
    runtime = resolve_runtime('rlbench', registry_path=path, environ={
        'PYTHONPATH': '/wrong/metaworld', 'PYOPENGL_PLATFORM': 'egl',
        'LD_LIBRARY_PATH': '/wrong/libraries', 'VLASTUDIO_WEB_FD': '17',
    })
    assert runtime.prefix == [sys.executable]
    assert runtime.env['COPPELIASIM_ROOT'] == '/selected/CoppeliaSim'
    assert runtime.env['DISPLAY'] == ':88'
    assert runtime.env['PYTHONPATH'].split(os.pathsep)[0] == '/selected/packages'
    assert '/wrong/metaworld' not in runtime.env['PYTHONPATH']
    assert 'LD_LIBRARY_PATH' not in runtime.env
    assert 'PYOPENGL_PLATFORM' not in runtime.env
    assert runtime.env['VLASTUDIO_WEB_FD'] == '17'


def test_missing_runtime_dependencies_fail_before_worker_start():
    runtime = RuntimeLaunch('missing-simulator', [sys.executable], dict(os.environ),
                            imports=['vlastudio_nonexistent_simulator_for_test'])
    with pytest.raises(ValueError, match='vlastudio_nonexistent_simulator_for_test'):
        runtime.probe()


def test_required_environment_is_checked_in_target_runtime():
    env = dict(os.environ)
    env.pop('VLASTUDIO_TEST_REQUIRED_PATH', None)
    runtime = RuntimeLaunch('required-env', [sys.executable], env,
                            required_env=['VLASTUDIO_TEST_REQUIRED_PATH'])
    with pytest.raises(ValueError, match='VLASTUDIO_TEST_REQUIRED_PATH'):
        runtime.probe()


def test_probe_runs_selected_python():
    runtime = RuntimeLaunch('test', [sys.executable], dict(os.environ), imports=['json'])
    assert runtime.probe()['python'] == sys.executable


def test_runtime_command_must_be_an_argument_list(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps({'runtimes': {'rlbench': {'command': 'python -m something'}}}))
    with pytest.raises(ValueError, match='argument list'):
        resolve_runtime('rlbench', registry_path=path)


@pytest.mark.skipif(sys.platform == 'win32', reason='WebUI sessions use Linux file descriptors')
def test_worker_command_uses_selected_launcher():
    from vlastudio.webui.session import command_line
    runtime = RuntimeLaunch('rlbench', ['bash', '/site/run.sh', 'rlbench'], {})
    cmd = command_line('collect', dict(robot='rlbench', output='runtime-test'), runtime)
    assert cmd[:6] == ['bash', '/site/run.sh', 'rlbench', '-u', '-m', 'vlastudio.entrypoints.collect_data']


@pytest.mark.skipif(sys.platform == 'win32', reason='WebUI sessions use Linux file descriptors')
def test_failed_preflight_does_not_start_a_keyboard_or_acquire_devices(monkeypatch):
    import vlastudio.webui.session as sessions
    runtime = RuntimeLaunch('missing-simulator', [sys.executable], dict(os.environ),
                            imports=['vlastudio_nonexistent_simulator_for_test'])
    monkeypatch.setattr(sessions, 'resolve_runtime', lambda profile: runtime)
    session = sessions.Session()
    with pytest.raises(ValueError, match='vlastudio_nonexistent_simulator_for_test'):
        session.start('collect', dict(robot='rlbench', output='preflight-test'))
    assert session.proc is None
    assert session.preview.proc is None
    assert session.locks == []
