"""Durable install queue adapting VLAStudio's existing uv environment manager."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import uuid

from filelock import FileLock
import yaml

from vlastudio.deploy.interpreter.manager import Manager, atomic_json
from vlastudio.core.paths import cache_root, package_source_root
from vlastudio.deploy.interpreter.recipes import environment_for, recipe, combine_profiles, platform_issues
from . import managed_environments
from .profiles import config_roots, ROOT

ACTIVE = ('queued', 'running')


def directory():
    return cache_root() / 'webui' / 'environment-installations'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:10]


def components():
    """Discover actual configurations and merge aliases of one component."""
    from .training import paths
    from vlastudio.deploy.interpreter.installers import describe
    found = {}

    def add(kind, name, profile, modules, bindings, error='', recipe_issue=''):
        identity = kind + ':' + name
        if identity in found and found[identity]['profile'] != profile:
            identity += ':' + fingerprint(profile)
        if identity not in found:
            found[identity] = dict(id=identity, kind=kind, label=name, profile=profile,
                                   imports=sorted(set(modules)), bindings={}, issue=error, recipe_issue=recipe_issue)
        row = found[identity]
        row['imports'] = sorted(set(row['imports'] + modules))
        for role, names in bindings.items():
            row['bindings'].setdefault(role, [])
            row['bindings'][role] = sorted(set(row['bindings'][role] + names))

    for name, path in paths('policy').items():
        try:
            doc = yaml.safe_load(path.read_text(encoding='utf-8'))
            module = doc.get('module_path') or doc.get('type', '')
            family_module = '.'.join(module.split('.')[:3]) if module.startswith('vlastudio.policy.') else module
            label = family_module.removeprefix('vlastudio.policy.') or name
            try:
                profile = environment_for(dict(doc, type=family_module, module_path=family_module), path, check_platform=False)
                issue = ''
                recipe_issue = ''
            except (OSError, ValueError) as exc:
                # A component may already import correctly even without a recipe.
                profile = environment_for({}, path, str(recipe('base')), check_platform=False)
                issue, recipe_issue = '', str(exc)
            add('policy', label, profile, [module if module == family_module else 'resolve:' + module],
                {'training': [name, module, family_module], 'inference': [name, module, family_module]}, issue, recipe_issue)
        except (OSError, ValueError, AttributeError, yaml.YAMLError):
            continue
    seen = set()
    for kind in ('robot', 'teleop'):
        for root in config_roots():
            for path in sorted((root / kind).rglob('*.yaml')):
                name = path.relative_to(root / kind).with_suffix('').as_posix()
                if (kind, name) in seen:
                    continue
                seen.add((kind, name))
                try:
                    doc = yaml.safe_load(path.read_text(encoding='utf-8'))
                    rows = doc if isinstance(doc, list) else [doc]
                    modules = [row['type'].rsplit(':', 1)[0] for row in rows if isinstance(row, dict) and row.get('type')]
                    if not modules:
                        continue
                    options = next((row.get('webui', {}) for row in rows if isinstance(row, dict) and row.get('webui')), {})
                    family = options.get('runtime')
                    simulation = kind == 'robot' and (family or any('sim' in m or 'pybullet' in m for m in modules))
                    category = 'simulation' if simulation else kind
                    label = family or '+'.join(sorted(set(m.split('.')[3] if m.startswith('vlastudio.deploy.') and len(m.split('.')) > 3 else m for m in modules)))
                    try:
                        explicit = options.get('runtime_manifest')
                        explicit = explicit or next((row.get('runtime') for row in rows if isinstance(row, dict) and row.get('runtime')), None)
                        if explicit:
                            profile = environment_for({}, path, explicit, check_platform=False)
                        elif family:
                            family = {'behavior': 'behavior1k'}.get(family, family)
                            profile = environment_for({}, path, str(recipe(family)), check_platform=False)
                        elif any('.sim_pybullet.' in module for module in modules):
                            profile = environment_for({}, path, str(recipe('pybullet')), check_platform=False)
                        elif category != 'simulation' and any(any(key in module for key in ('.so101', '.bi_so101', '.koch.')) for module in modules):
                            profile = environment_for({}, path, str(recipe('device_lerobot')), check_platform=False)
                        else:
                            # Unannotated drivers get the shared runtime; import validation
                            # will surface any undeclared vendor SDK instead of claiming success.
                            profile = environment_for({}, path, str(recipe('base')), check_platform=False)
                            for module in modules:
                                package = package_source_root().joinpath(*module.removeprefix('vlastudio.').split('.')[:-1])
                                metadata = package / 'pyproject.toml'
                                if not metadata.is_file():
                                    metadata = package.parent / 'pyproject.toml'
                                if metadata.is_file():
                                    try:
                                        import tomllib
                                    except ImportError:
                                        import tomli as tomllib
                                    project = tomllib.loads(metadata.read_text(encoding='utf-8'))['project']
                                    profile = combine_profiles(profile, {'python': profile['python'], 'requirements': project.get('dependencies', [])})
                        issue = ''
                    except (OSError, ValueError, KeyError) as exc:
                        profile, issue = None, str(exc)
                    add(category, label, profile, ['resolve:' + module for module in modules],
                        {category: [name]}, issue)
                except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
                    continue
    for row in found.values():
        if row['profile']:
            row['issue'] = '; '.join(platform_issues(row['profile']))
            row['licenses'] = {key: url for step in describe(row['profile']) for key, url in step['licenses'].items()}
        else:
            row['licenses'] = {}
        row['environment'] = ('webui-' + re.sub(r'[^a-z0-9-]', '-', row['id'].lower())[:35] + '-' + fingerprint([row['profile'], row['imports']]))
    return list(found.values())


def alive(job):
    try:
        stat = Path(f"/proc/{job['pid']}/stat").read_text().split()
        return stat[2] != 'Z' and stat[21] == job['process_start']
    except (OSError, KeyError, IndexError):
        return False


def current():
    path = directory() / 'job.json'
    if not path.exists():
        return None
    job = json.loads(path.read_text(encoding='utf-8'))
    if job['status'] in ACTIVE and not alive(job):
        job['status'] = 'interrupted'
    return job


def redact(value):
    value = re.sub(r'(https?://)[^/\s@]+@', r'\1[redacted]@', value)
    value = re.sub(r'(?i)(token|password|authorization|secret)([=: ]+)[^\s&]+', r'\1\2[redacted]', value)
    return re.sub(r'\bhf_[A-Za-z0-9]+\b', '[redacted]', value)


def catalog():
    rows = components()
    job = current()
    registered = managed_environments.read()
    for row in rows:
        previous = next((entry for entry in registered['interpreters'].values() if entry.get('component') == row['id'] and Path(entry['python']).is_file()), None)
        if previous:
            row['saved_result'] = dict(status='registered', python=previous['python'],
                                       message='Previously verified · ' + datetime.fromtimestamp(previous['verified_at'], timezone.utc).isoformat())
    if job:
        job = dict(job, rows=[dict(row, component={k: v for k, v in row['component'].items() if k not in ('profile', 'imports')}) for row in job['rows']])
    return {'components': [{k: v for k, v in row.items() if k not in ('profile', 'imports')} for row in rows],
            'job': job, 'interpreters': managed_environments.entries(), 'cache_root': str(cache_root())}


def submit(payload):
    root = directory();root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / 'queue.lock'), timeout=2):
        previous = current()
        if previous and previous['status'] in ACTIVE:
            raise ValueError('An environment operation is already running')
        rows = {row['id']: row for row in components()}
        identifiers = payload.get('components')
        if not isinstance(identifiers, list) or not identifiers or len(identifiers) > 500 or any(not isinstance(i, str) or i not in rows for i in identifiers):
            raise ValueError('Select existing components')
        action = payload.get('action', 'install')
        if action not in ('check', 'install'):
            raise ValueError('Invalid environment action')
        from vlastudio.deploy.interpreter.installers import LICENSES
        accepted = payload.get('accepted_licenses', [])
        if not isinstance(accepted, list) or any(not isinstance(key, str) or key not in LICENSES for key in accepted):
            raise ValueError('Invalid license selection')
        identifier = uuid.uuid4().hex
        job = dict(id=identifier, status='queued', action=action, created_at=stamp(),
                   rows=[dict(component=rows[i], status='queued') for i in dict.fromkeys(identifiers)],
                   accepted_licenses=accepted)
        # A supervisor is detached from both the page and the WebUI lifetime.
        path = root / (identifier + '.json');atomic_json(path, job)
        from .settings import runtime_environment
        env = runtime_environment(os.environ);env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
        with (root / (identifier + '.log')).open('ab') as log:
            proc = subprocess.Popen([sys.executable, '-u', '-m', 'vlastudio.webui.environment_installation', str(path)],
                                    cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
        job.update(pid=proc.pid, process_start=Path(f'/proc/{proc.pid}/stat').read_text().split()[21])
        # Worker waits for this atomic receipt before touching the queue.
        atomic_json(root / 'job.json', job)
        return {'id': job['id'], 'status': job['status']}


def cancel(payload):
    job = current()
    if not job or job['id'] != payload.get('id') or job['status'] not in ACTIVE or not alive(job):
        raise ValueError('This environment operation is no longer running')
    os.killpg(job['pid'], signal.SIGTERM)
    return {'ok': True}


def probe(component, entry, manager):
    """Actual imports, not strict metadata pins or just find_spec(). No robot instances."""
    from vlastudio.deploy.interpreter.probe import IMPORTS
    imports = ['vlastudio', 'loguru', 'yaml'] + component['imports']
    for check in component['profile'].get('checks', []):
        imports += [check.split(':', 1)[1]] if check.startswith('import:') else IMPORTS.get(check, [])
    profile = component['profile']
    data = dict(imports=list(dict.fromkeys(imports)), python=profile.get('requires_python', '==' + profile['python'] + '.*'))
    code = """import sys,json,importlib
from packaging.specifiers import SpecifierSet
data=json.loads(sys.argv[1])
assert '.'.join(map(str,sys.version_info[:3])) in SpecifierSet(data['python']), 'Python version does not match this component'
for name in data['imports']:
 if name.startswith('resolve:'):
  from vlastudio.utils.extensions import resolve
  resolve(name.split(':',1)[1])
 else:importlib.import_module(name)
print('VLASTUDIO_COMPONENT_IMPORTS_OK')
"""
    env = manager.environment(entry)
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    env.setdefault('MUJOCO_GL', 'egl' if sys.platform == 'linux' else 'glfw')
    process = subprocess.Popen([entry['python'], '-c', code, json.dumps(data)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        output, _ = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill();output, _ = process.communicate()
        return False, 'Import check timed out'
    text = output.decode(errors='replace')
    return process.returncode == 0 and 'VLASTUDIO_COMPONENT_IMPORTS_OK' in text, redact(text[-1500:])


def candidates(component, manager):
    from .interpreters import selection
    result = []
    for role, names in component['bindings'].items():
        for name in names:
            selected = selection(role, name)
            if selected:
                result.append(dict(python=selected['python'], variables=selected.get('env', {})))
        if role in ('simulation', 'robot'):
            from .runtimes import resolve_runtime
            for name in names:
                try:
                    runtime = resolve_runtime(name)
                    if len(runtime.prefix) == 1 and Path(runtime.prefix[0]).is_file():
                        keys = ('MUJOCO_GL', 'PYOPENGL_PLATFORM', 'DISPLAY', 'LD_LIBRARY_PATH',
                                'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_PLUGIN_PATH', 'COPPELIASIM_ROOT')
                        variables = {key: value for key, value in runtime.env.items() if key in keys or key.startswith(('VLASTUDIO_RUNTIME_', 'OMNIGIBSON_', 'LIBERO_'))}
                        if role == 'simulation':
                            variables.setdefault('MUJOCO_GL', 'egl' if sys.platform == 'linux' else 'glfw')
                        result.append(dict(python=runtime.prefix[0], variables=variables))
                except (OSError, ValueError):
                    continue
    entries = manager.entries()
    if component['environment'] in entries:
        result.append(entries[component['environment']])
    result += [entry for entry in entries.values() if entry.get('profile') == component['profile']]
    result.append(dict(python=sys.executable, variables={}))
    seen = set()
    for entry in result:
        key = (entry['python'], json.dumps(entry.get('variables', {}), sort_keys=True))
        if key not in seen and Path(entry['python']).is_file():
            seen.add(key);yield entry


def in_use(python):
    target = str(Path(python).absolute().parent.parent)
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            first = path.read_bytes().split(bytes([0]))[0].decode()
            if first and str(Path(first).absolute().parent.parent) == target:
                return True
        except (OSError, UnicodeError):
            continue
    return False


def run_component(row, manager, accepted, action, update):
    component = row['component']
    if component['issue']:
        row.update(status='unavailable', message=component['issue']);return
    row['status'] = 'checking';update()
    reason = 'No usable interpreter found'
    for entry in candidates(component, manager):
        ok, reason = probe(component, entry, manager)
        if ok:
            ready = managed_environments.register(component, component['environment'], entry['python'], entry.get('variables', {}))
            row.update(status='reused', python=ready['python'], message='Imports passed; installation skipped');return
    if action == 'check':
        row.update(status='missing', message=reason);return
    if component.get('recipe_issue'):
        row.update(status='unavailable', message=component['recipe_issue']);return
    from vlastudio.deploy.interpreter.installers import preflight
    missing = set(component['licenses']) - set(accepted)
    if missing:
        row.update(status='needs_license', message='Review and accept the required licenses before installing');return
    preflight(component['profile'], accepted=accepted)
    name = component['environment']
    destination = manager.cache / 'envs' / name
    python = destination / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if destination.is_symlink():
        raise ValueError('Refusing to install into an external environment alias')
    if python.exists() and in_use(str(python)):
        raise ValueError('This environment is in use. Stop its session before repairing it.')
    if destination.exists() and not python.exists():
        # Preserve an incomplete directory for diagnosis; create a fresh destination.
        name = name[:51] + '-' + uuid.uuid4().hex[:8]
    row.update(status='installing', environment=name);update()
    code = manager.install(name, component['profile'], create=True, package=str(ROOT), accepted=accepted)
    if code:
        raise ValueError('Environment installation or validation failed; open the installation log')
    entry = manager.get(name)
    if component['kind'] == 'simulation':
        entry.setdefault('variables', {}).setdefault('MUJOCO_GL', 'egl' if sys.platform == 'linux' else 'glfw')
    ok, reason = probe(component, entry, manager)
    if not ok:
        raise ValueError(reason)
    ready = managed_environments.register(component, name, entry['python'], entry.get('variables', {}))
    row.update(status='ready', python=ready['python'], message='Installed with uv; imports passed')


def logs(identifier, component_id):
    job = current()
    if not job or job['id'] != identifier:
        raise ValueError('Unknown environment operation')
    row = next((r for r in job['rows'] if r['component']['id'] == component_id), None)
    if row is None:
        raise ValueError('Unknown component')
    name = row.get('environment', row['component']['environment'])
    from vlastudio.deploy.interpreter.manager import valid_name
    valid_name(name)
    path = cache_root() / 'installations' / name / 'install.log'
    if not path.is_file():
        return {'log': row.get('message', '')}
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 65536));value = stream.read().decode(errors='replace')
    return {'log': redact(value)}


def work(path):
    import time
    root = directory()
    for _ in range(100):
        job = current()
        if job and job['id'] == path.stem and alive(job):
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Queue receipt was not committed')
    with FileLock(str(root / 'worker.lock'), timeout=1):
        job.update(status='running', started_at=stamp())
        def update():
            atomic_json(root / 'job.json', job)
        def stop(*_):
            # Only this install worker's descendants, including native check groups.
            import psutil
            children = psutil.Process().children(recursive=True)
            for child in reversed(children):
                try:child.terminate()
                except psutil.NoSuchProcess:pass
            _, remaining = psutil.wait_procs(children, timeout=3)
            for child in remaining:
                try:child.kill()
                except psutil.NoSuchProcess:pass
            job.update(status='cancelled', finished_at=stamp())
            for row in job['rows']:
                if row['status'] in ('queued', 'checking', 'installing'):
                    row['status'] = 'cancelled'
            update()
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
        update();manager = Manager()
        for row in job['rows']:
            try:
                run_component(row, manager, job['accepted_licenses'], job['action'], update)
            except Exception as exc:
                row.update(status='failed', message=redact(str(exc)))
            row['finished_at'] = stamp();update()
        job.update(status='completed', finished_at=stamp());update()


if __name__ == '__main__':
    work(Path(sys.argv[1]))
