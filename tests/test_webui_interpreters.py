import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import threading
import venv

import pytest

pytest.importorskip('fcntl')
from vlastudio.webui import interpreters, settings, runtimes, training
from test_webui_training import storage


@pytest.fixture
def environments(storage):
    entry = dict(id='custom', name='Custom Python', python=sys.executable, env={'EXPERIMENT_VALUE': 'first'})
    data, _ = settings.stored()
    data.update(interpreters=[entry], interpreter_bindings={'training': 'custom'}, interpreter_overrides=[])
    settings.SETTINGS_FILE.write_text(json.dumps(data))
    return entry


def test_roles_and_specific_override_take_precedence(environments):
    data, _ = settings.stored()
    other = dict(environments, id='other', name='Other')
    data['interpreters'].append(other)
    data['interpreter_bindings'].update(default='custom', inference='other')
    data['interpreter_overrides'] = [{'role': 'inference', 'config': '/model/act', 'interpreter': 'custom'}]
    assert interpreters.selection('inference', '/model/act', data)['id'] == 'custom'
    assert interpreters.selection('inference', '/model/pi', data)['id'] == 'other'
    assert interpreters.selection('teleop', '', data)['id'] == 'custom'
    assert interpreters.selection('training', '', data)['id'] == 'custom'


def test_explicit_policy_runtime_does_not_import_simulator(environments, monkeypatch):
    data, _ = settings.stored();data['interpreter_bindings']['inference']='custom'
    settings.SETTINGS_FILE.write_text(json.dumps(data))
    monkeypatch.setattr(runtimes, 'catalog', lambda: {'robot_options': {'sim': {'runtime': 'sim', 'runtime_imports': ['missing_sim_package']}}})
    selected = runtimes.resolve_runtime('sim', role='inference')
    assert selected.prefix == [sys.executable] and not selected.imports
    # A simulation still verifies its own simulator dependencies.
    data['interpreter_bindings']['simulation']='custom';settings.SETTINGS_FILE.write_text(json.dumps(data))
    assert runtimes.resolve_runtime('sim').imports == ['missing_sim_package']


def test_save_round_trip_does_not_disconnect_running_devices(environments):
    studio=SimpleNamespace(lock=threading.RLock(),host=object(),runners={})
    public=settings.public();public['interpreter_bindings']['inference']='custom'
    saved=settings.save(studio, public)
    assert saved['interpreter_bindings']['inference']=='custom'
    assert studio.host is not None and saved['webui_python']==sys.executable
    public['interpreter_bindings']['robot']='custom'
    with pytest.raises(ValueError,match='Settings changed'):
        settings.save(studio,public)


def test_validation_rejects_missing_executable_and_removed_references(environments):
    with pytest.raises(ValueError,match='does not exist'):
        interpreters.validate({'interpreters':[dict(environments,python='/nonexistent/python')]})
    with pytest.raises(ValueError,match='existing interpreter'):
        interpreters.validate({'interpreter_bindings':{'training':'deleted'}})
    with pytest.raises(ValueError,match='absolute Python'):
        interpreters.validate({'interpreters':[dict(environments,python='python -m something')]})


def test_probe_preserves_virtualenv_path_and_cleans_parent_environment(storage, monkeypatch, tmp_path):
    path=tmp_path/'environment'
    venv.EnvBuilder(with_pip=False,system_site_packages=True,symlinks=True).create(path)
    executable=str(path/'bin/python')
    entry=dict(id='venv',name='Isolated',python=executable,env={'EXPERIMENT_VALUE':'chosen'})
    monkeypatch.setenv('VIRTUAL_ENV','/wrong/environment')
    monkeypatch.setenv('CONDA_PREFIX','/wrong/conda')
    launch=interpreters.launch(entry)
    assert 'CONDA_PREFIX' not in launch.env and 'VIRTUAL_ENV' not in launch.env
    assert launch.env['PATH'].split(os.pathsep)[0]==str(path/'bin')
    result=interpreters.probe(entry)
    assert result['python']==executable and result['version'] and result['vlastudio']


def test_training_and_resume_follow_settings_and_record_actual_interpreter(environments, monkeypatch):
    calls=[]
    monkeypatch.setattr(training.subprocess,'Popen',lambda cmd,**kw:(calls.append((cmd,kw)) or SimpleNamespace(wait=lambda:0)))
    run=training.start(dict(name='Environment',policy='sample',task='sample',training='sample'))
    assert run['python']==environments['python'] and run['interpreter']['name']=='Custom Python'
    assert calls[-1][1]['env']['EXPERIMENT_VALUE']=='first'
    folder=training.directory(run['id']);run['status']='failed';training.write(folder/'run.json',run)
    data,_=settings.stored();data['interpreters'][0]['env']['EXPERIMENT_VALUE']='changed';settings.SETTINGS_FILE.write_text(json.dumps(data))
    training.resume(run['id'])
    assert calls[-1][1]['env']['EXPERIMENT_VALUE']=='changed'
    archived=next((folder/'attempts').iterdir())
    assert json.loads((archived/'run.json').read_text())['interpreter']['env']['EXPERIMENT_VALUE']=='first'
    # An explicit Resume selection may switch environments after a dependency failure.
    item=training.read(folder);item['status']='failed';training.write(folder/'run.json',item)
    training.resume(run['id'],interpreter_id='custom')
    assert calls[-1][1]['env']['EXPERIMENT_VALUE']=='changed'


def test_cache_alias_uses_shared_cache_and_preserves_external_environment(tmp_path, monkeypatch):
    from vlastudio import paths
    cache=tmp_path/'shared-cache'
    python=tmp_path/'external-env'/'bin'/'python'
    python.parent.mkdir(parents=True);python.write_text('original executable')
    monkeypatch.setattr(paths,'cache_root',lambda:cache)
    monkeypatch.setattr(sys,'executable',str(python))
    settings.register_interpreter_alias()
    assert settings.interpreter_display_path()=='$VLASTUDIO/envs/external-env/bin/python'
    assert (cache/'envs/external-env/bin/python').samefile(python)
    settings.register_interpreter_alias()
    assert python.read_text()=='original executable'


def test_cache_alias_does_not_overwrite_an_existing_environment(tmp_path, monkeypatch):
    from vlastudio import paths
    cache=tmp_path/'cache';existing=cache/'envs/external-env';existing.mkdir(parents=True)
    (existing/'keep').write_text('existing')
    python=tmp_path/'external-env/bin/python';python.parent.mkdir(parents=True);python.touch()
    monkeypatch.setattr(paths,'cache_root',lambda:cache);monkeypatch.setattr(sys,'executable',str(python))
    settings.register_interpreter_alias()
    assert (existing/'keep').read_text()=='existing'
    assert settings.interpreter_display_path().startswith('$VLASTUDIO/envs/external-env-')
