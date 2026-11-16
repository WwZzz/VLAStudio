import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

pytest.importorskip('fcntl')
from vlastudio.webui import environment_installation as install, managed_environments as managed, interpreters
from vlastudio.deploy.interpreter.manager import Manager, atomic_json


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv('VLASTUDIO_CACHE', str(tmp_path/'cache'))
    return tmp_path/'cache'


@pytest.fixture
def component():
    return dict(id='policy:tiny', kind='policy', label='Tiny', environment='webui-tiny',
                profile={'python':f'{sys.version_info.major}.{sys.version_info.minor}', 'requires_python':'>=3.10', 'requirements':[], 'checks':[]},
                imports=['json'], bindings={'training':['tiny'], 'inference':['tiny','example.policy']}, issue='', licenses={})


def test_valid_imports_skip_installer_and_register_automatic_binding(component, monkeypatch):
    manager=Manager()
    def forbidden(*a,**kw):raise AssertionError('Installer must not run')
    monkeypatch.setattr(manager,'install',forbidden)
    component['profile']['requirements']=['not-actually-needed-for-import==99']
    row={'component':component}
    install.run_component(row,manager,[],'install',lambda:None)
    assert row['status']=='reused'
    selected=interpreters.selection('training','tiny',data={})
    assert selected['python']==sys.executable
    assert interpreters.selection('training','tiny',data={'interpreters':[dict(id='custom',python='/chosen/python')], 'interpreter_bindings':{'training':'custom'}})['python']=='/chosen/python'
    model=manager.cache/'model';model.mkdir();(model/'policy_metadata.json').write_text(json.dumps({'policy_module':'example.policy'}))
    assert managed.choose('inference',str(model))['python']==sys.executable


def test_missing_environment_delegates_to_existing_manager(component, monkeypatch):
    manager=Manager();calls=[];outcomes=iter([(False,'missing'),(True,'ok')])
    monkeypatch.setattr(install,'probe',lambda *a:next(outcomes))
    monkeypatch.setattr(manager,'install',lambda *a,**kw:(calls.append((a,kw)) or 0))
    monkeypatch.setattr(manager,'get',lambda name:dict(python=sys.executable,variables={'SIM_ROOT':'/native'}))
    row={'component':component};install.run_component(row,manager,[],'install',lambda:None)
    assert calls[0][1]['create'] is True and calls[0][0][0]=='webui-tiny'
    assert row['status']=='ready'
    assert managed.choose('training','tiny')['env']=={'SIM_ROOT':'/native'}


def test_license_is_required_only_when_installation_is_needed(component, monkeypatch):
    component['licenses']={'coppeliasim-edu':'https://www.coppeliarobotics.com/licensing'}
    monkeypatch.setattr(install,'probe',lambda *a:(False,'missing'))
    row={'component':component};install.run_component(row,Manager(),[],'install',lambda:None)
    assert row['status']=='needs_license'
    row={'component':component};install.run_component(row,Manager(),[],'check',lambda:None)
    assert row['status']=='missing'


def test_install_does_not_modify_environment_in_use(component, monkeypatch, cache):
    python=cache/'envs/webui-tiny/bin/python';python.parent.mkdir(parents=True);python.touch()
    monkeypatch.setattr(install,'probe',lambda *a:(False,'missing'))
    monkeypatch.setattr(install,'in_use',lambda path:True)
    with pytest.raises(ValueError,match='in use'):
        install.run_component({'component':component},Manager(),[],'install',lambda:None)
    assert python.exists()


def test_probe_failure_after_install_never_registers_ready(component, monkeypatch):
    manager=Manager();monkeypatch.setattr(install,'probe',lambda *a:(False,'cannot import component'))
    monkeypatch.setattr(manager,'install',lambda *a,**kw:0)
    monkeypatch.setattr(manager,'get',lambda *a:dict(python=sys.executable))
    with pytest.raises(ValueError,match='cannot import'):
        install.run_component({'component':component},manager,[],'install',lambda:None)
    assert managed.entries()==[]


def test_queue_recovers_after_one_component_fails(component, monkeypatch):
    root=install.directory();root.mkdir(parents=True)
    job=dict(id='test-job',status='queued',pid=os.getpid(),process_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],action='install',accepted_licenses=[],rows=[{'component':component},{'component':dict(component,id='policy:other')}])
    atomic_json(root/'job.json',job)
    def run(row,*args):
        if row['component']['id']=='policy:tiny':raise RuntimeError('first fails')
        row['status']='ready'
    monkeypatch.setattr(install,'run_component',run);monkeypatch.setattr(install.signal,'signal',lambda *a:None)
    install.work(root/'test-job.json')
    result=install.current();assert result['status']=='completed'
    assert [row['status'] for row in result['rows']]==['failed','ready']


def test_submit_rejects_untrusted_components_and_duplicate_workers(component, monkeypatch):
    monkeypatch.setattr(install,'components',lambda:[component])
    with pytest.raises(ValueError,match='existing components'):install.submit({'components':['../../bad']})
    root=install.directory();root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'job.json',dict(id='busy',status='running',pid=os.getpid(),process_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    with pytest.raises(ValueError,match='already running'):install.submit({'components':[component['id']]})


def test_real_uv_environment_and_package_import(component, cache, tmp_path):
    """Real offline uv creation/install; never downloads model or simulator packages."""
    manager=Manager();uv=manager.uv();destination=cache/'envs/uv-smoke'
    subprocess.run([*uv,'venv','--python',sys.executable,'--system-site-packages',str(destination)],check=True,capture_output=True,timeout=30)
    python=str(destination/'bin/python')
    wheel=tmp_path/'vla_install_smoke-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel,'w') as archive:
        archive.writestr('vla_install_smoke.py','VALUE=42\n')
        archive.writestr('vla_install_smoke-1.0.dist-info/METADATA','Metadata-Version: 2.1\nName: vla-install-smoke\nVersion: 1.0\n')
        archive.writestr('vla_install_smoke-1.0.dist-info/WHEEL','Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        archive.writestr('vla_install_smoke-1.0.dist-info/RECORD','')
    subprocess.run([*uv,'pip','install','--python',python,'--no-deps','--offline',str(wheel)],check=True,capture_output=True,timeout=30)
    # uv may unwrap a parent virtualenv to its base Python. Supply the host's
    # real shared runtime dependencies explicitly for this offline tiny fixture.
    import sysconfig
    component['imports']=['vla_install_smoke'];ok,reason=install.probe(component,dict(python=python,variables={'PYTHONPATH':sysconfig.get_paths()['purelib']}),manager)
    assert ok,reason
    assert Path(python).is_relative_to(cache/'envs')


def test_cancel_stops_import_child_and_keeps_other_processes(component, tmp_path):
    import time
    root=install.directory();root.mkdir(parents=True)
    marker=tmp_path/'probe-pid';plugin=tmp_path/'slow.py'
    plugin.write_text(f'import os,time\nfrom pathlib import Path\nPath({str(marker)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\nclass Thing: pass\n')
    component['imports']=['resolve:'+str(plugin)+':Thing']
    path=root/'cancel-test.json';path.write_text('{}')
    process=subprocess.Popen([sys.executable,'-m','vlastudio.webui.environment_installation',str(path)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    try:
        job=dict(id=path.stem,status='queued',action='check',accepted_licenses=[],rows=[{'component':component,'status':'queued'}],pid=process.pid,process_start=Path(f'/proc/{process.pid}/stat').read_text().split()[21])
        atomic_json(root/'job.json',job)
        deadline=time.monotonic()+15
        while not marker.exists() and time.monotonic()<deadline:time.sleep(.1)
        assert marker.exists(),'Import probe did not start'
        install.cancel({'id':path.stem});assert process.wait(timeout=10)==0
        assert install.current()['status']=='cancelled'
        assert not Path('/proc',marker.read_text()).exists()
        assert Path(f'/proc/{os.getpid()}').exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid,15);process.wait(timeout=10)


def test_environment_catalog_does_not_shadow_robot_catalog(monkeypatch):
    from vlastudio.webui import __main__ as webui
    from test_webui_training import log_server
    import urllib.request
    monkeypatch.setattr(webui,'catalog',lambda **kw:{'robots':['selected-robot']})
    monkeypatch.setattr(install,'catalog',lambda:{'components':[]})
    with log_server() as url:
        base=url.split('/api/')[0]
        with urllib.request.urlopen(base+'/api/settings/environments') as response:
            assert json.load(response)=={'components':[]}
        with urllib.request.urlopen(base+'/api/catalog') as response:
            assert json.load(response)=={'robots':['selected-robot']}
