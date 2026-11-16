"""Server-local interpreter choices; launch arguments never pass through a shell."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROLES = ('default', 'training', 'inference', 'simulation', 'robot', 'teleop')


def validate(data):
    profiles = data.get('interpreters', [])
    bindings = data.get('interpreter_bindings', {})
    overrides = data.get('interpreter_overrides', [])
    if not isinstance(profiles, list) or len(profiles) > 50:
        raise ValueError('Invalid interpreter list')
    from .managed_environments import entries
    ids = set()
    for entry in profiles:
        if not isinstance(entry, dict) or not isinstance(entry.get('id'), str) or not re.fullmatch(r'[\w-]{1,80}', entry['id']) or entry['id'] in ids:
            raise ValueError('Interpreter IDs must be unique')
        ids.add(entry['id'])
        if not isinstance(entry.get('name'), str) or not 1 <= len(entry['name'].strip()) <= 100:
            raise ValueError('Enter an interpreter name')
        executable = entry.get('python')
        if not isinstance(executable, str) or not Path(executable).is_absolute() or any(c in executable for c in '\r\n\0'):
            raise ValueError('Use an absolute Python executable path on the WebUI machine')
        # Keep the venv executable path: resolving symlinks changes Python's prefix.
        if not Path(executable).is_file() or not os.access(executable, os.X_OK):
            raise ValueError('Python executable does not exist or is not executable: ' + executable)
        env = entry.get('env', {})
        if not isinstance(env, dict) or len(env) > 100 or any(not isinstance(k, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', k) or not isinstance(v, str) or '\0' in v for k, v in env.items()):
            raise ValueError('Environment variables must be KEY=value pairs')
    ids.update(entry['id'] for entry in entries())
    if not isinstance(bindings, dict) or any(role not in ROLES or not isinstance(value, str) or value not in ids | {''} for role, value in bindings.items()):
        raise ValueError('Select an existing interpreter for each purpose')
    if not isinstance(overrides, list) or len(overrides) > 200:
        raise ValueError('Invalid interpreter overrides')
    seen = set()
    for row in overrides:
        if not isinstance(row, dict) or row.get('role') not in ROLES[1:] or not isinstance(row.get('config'), str) or not row['config'].strip() or len(row['config']) > 1024 or not isinstance(row.get('interpreter'), str) or row['interpreter'] not in ids:
            raise ValueError('Each configuration override needs a purpose, configuration and interpreter')
        key = (row['role'], row['config'])
        if key in seen:
            raise ValueError('Duplicate interpreter configuration override')
        seen.add(key)
    return dict(interpreters=profiles, interpreter_bindings=bindings, interpreter_overrides=overrides)


def selection(role, key='', data=None):
    if data is None:
        from .settings import stored
        data, _ = stored()
    bindings = data.get('interpreter_bindings', {})
    identifier = next((r['interpreter'] for r in data.get('interpreter_overrides', [])
                       if r['role'] == role and r['config'] == key), None)
    identifier = identifier or bindings.get(role) or bindings.get('default')
    from .managed_environments import entries, choose
    selected = next((p for p in [*data.get('interpreters', []), *entries()] if p['id'] == identifier), None)
    return selected if identifier else choose(role, key)


def launch(entry, environ=None):
    from .runtimes import RuntimeLaunch, _CLEAN_KEYS
    from .profiles import PACKAGE
    from .settings import runtime_environment
    parent = runtime_environment(os.environ if environ is None else environ)
    for key in (*_CLEAN_KEYS, 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'VIRTUAL_ENV'):
        parent.pop(key, None)
    parent.update(entry.get('env', {}))
    parent['PYTHONPATH'] = os.pathsep.join([str(PACKAGE.parent), *([parent['PYTHONPATH']] if parent.get('PYTHONPATH') else [])])
    parent['PATH'] = str(Path(entry['python']).parent) + os.pathsep + parent.get('PATH', '')
    parent['PYTHONNOUSERSITE'] = '1'
    return RuntimeLaunch(entry['name'], [entry['python']], parent)


def probe(entry):
    validate({'interpreters': [dict(entry, id='probe')]})
    runtime = launch(entry)
    script = '''import sys,json,importlib,importlib.util
result={'python':sys.executable,'version':sys.version.split()[0]}
result['packages']={name:importlib.util.find_spec(name) is not None for name in ('openpi','transformers','dm_control','mujoco','pybullet','robosuite','rlbench','omnigibson','h5py','cv2','serial')}
for name in ('vlastudio','torch'):
 try:
  module=importlib.import_module(name);result[name]=getattr(module,'__version__','available')
  if name=='torch':result.update(cuda=module.cuda.is_available(),cuda_version=module.version.cuda)
 except Exception as e:result[name]=type(e).__name__+': '+str(e)
print('INTERPRETER_PROBE='+json.dumps(result))
'''
    try:
        result = subprocess.run([*runtime.prefix, '-c', script], env=runtime.env, capture_output=True, text=True, timeout=45)
        line = next(line for line in reversed(result.stdout.splitlines()) if line.startswith('INTERPRETER_PROBE='))
        if result.returncode:
            raise ValueError('Interpreter check failed')
        return json.loads(line.split('=', 1)[1])
    except (OSError, subprocess.TimeoutExpired, StopIteration, json.JSONDecodeError):
        raise ValueError('Cannot run this interpreter or the check timed out') from None
