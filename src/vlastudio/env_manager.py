"""Explicit environment management; never used as an import-time installer."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from filelock import FileLock

from .paths import cache_root, runtime_env


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def valid_name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name):
        raise ValueError('Environment names must contain 1–64 letters, digits, dots, underscores or hyphens.')
    if name.upper().split('.')[0] in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))}:
        raise ValueError('Reserved environment name')
    return name


class Manager:
    def __init__(self, cache=None):
        self.cache = cache_root(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.registry = self.cache / 'registry.json'
        self.lock = FileLock(str(self.registry) + '.lock', timeout=30)

    def entries(self):
        with self.lock:
            value = json.loads(self.registry.read_text(encoding='utf-8')) if self.registry.exists() else {'schema': 1, 'environments': {}}
            if value.get('schema') != 1:
                raise ValueError('Unsupported environment registry schema')
            entries = value['environments']
            if 'base' not in entries:
                # sys.executable is authoritative; VIRTUAL_ENV may refer to a moved checkout.
                entries['base'] = {'python': sys.executable, 'managed': False, 'status': 'registered', 'checks': {}}
                atomic_json(self.registry, value)
            return entries

    def save(self, name, entry):
        with self.lock:
            value = json.loads(self.registry.read_text(encoding='utf-8')) if self.registry.exists() else {'schema': 1, 'environments': {}}
            value['environments'][name] = entry
            atomic_json(self.registry, value)

    def get(self, name='base'):
        valid_name(name)
        entry = self.entries().get(name)
        if entry is None:
            raise ValueError(f'Unknown environment: {name}. Run vlastudio env list.')
        if not Path(entry['python']).is_file():
            raise ValueError(f"Interpreter no longer exists: {entry['python']}. Reinstall/register this environment.")
        return entry

    def environment(self, entry):
        env = runtime_env(self.cache)
        if os.environ.get('PIP_INDEX_URL'):
            env.setdefault('UV_INDEX_URL', os.environ['PIP_INDEX_URL'])
        if os.environ.get('PIP_EXTRA_INDEX_URL'):
            env.setdefault('UV_EXTRA_INDEX_URL', os.environ['PIP_EXTRA_INDEX_URL'])
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)
        env['PYTHONNOUSERSITE'] = '1'
        executable = Path(entry['python'])
        env['PATH'] = str(executable.parent) + os.pathsep + env.get('PATH', '')
        env.pop('VIRTUAL_ENV', None)
        if (executable.parent.parent / 'pyvenv.cfg').is_file():
            env['VIRTUAL_ENV'] = str(executable.parent.parent)
        for key, value in entry.get('variables', {}).items():
            if key == 'LD_LIBRARY_PATH' and env.get(key):
                # Keep native runtime paths and the current host's graphics paths.
                paths = value.split(os.pathsep) + env[key].split(os.pathsep)
                value = os.pathsep.join(dict.fromkeys(path for path in paths if path))
            env[key] = value
        return env

    def run(self, name, command):
        entry = self.get(name)
        if not command:
            raise ValueError('Supply a command after --, e.g. env run base -- python train.py')
        command = list(command)
        if command[0] in ('python', 'python3'):
            command[0] = entry['python']
        return subprocess.call(command, env=self.environment(entry))

    def uv(self):
        try:
            import uv
            return [uv.find_uv_bin()]
        except ImportError:
            binary = shutil.which('uv')
            if not binary:
                raise ValueError('Install the environment manager dependency: python -m pip install uv')
            return [binary]

    def install(self, name, profile, *, create=False, index=None, package=None,
                accepted=(), assets=False, offline=False, system_packages=False):
        from .env_installers import prepare, finish, preflight, install_system_packages
        from .profiles import platform_issues, combine_profiles
        valid_name(name)
        self.entries()
        # Different invocations may target one interpreter under different names.
        existing = self.entries().get(name)
        destination = self.cache / 'envs' / name
        python = str(destination / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')) if create else sys.executable
        if existing and existing['python'] != python and name != 'base':
            raise ValueError(f'{name} already refers to another interpreter. Use env run {name} -- vlastudio env install ... or a different name.')
        if existing and existing['python'] != python:
            existing = None
        with FileLock(str(self.cache / ('install-' + hashlib.sha256(python.encode()).hexdigest()[:20] + '.lock')), timeout=1):
            # Preserve constraints and native setup when adding a component later.
            if existing and existing.get('profile'):
                profile = combine_profiles(existing['profile'], profile)
            accepted = sorted(set(accepted) | set((existing or {}).get('receipts', {}).get('accepted_licenses', [])))
            issues = platform_issues(profile)
            if issues:
                raise ValueError('; '.join(issues))
            preflight(profile, accepted=accepted, system_packages=system_packages)
            entry = dict(existing or {})
            entry.update(python=python, managed=create or entry.get('managed', False), profile=profile,
                         status='installing', checks={}, updated_at=time.time())
            entry.setdefault('variables', {})
            entry.setdefault('receipts', {})
            entry['receipts']['accepted_licenses'] = accepted
            self.save(name, entry)
            env = self.environment(entry)
            if offline:
                env['UV_OFFLINE'] = '1'
                env['HF_HUB_OFFLINE'] = '1'
            if create:
                # Managed Python includes headers needed by source-built dependencies.
                env.setdefault('UV_PYTHON_PREFERENCE', 'only-managed')
            record = self.cache / 'installations' / name
            record.mkdir(parents=True, exist_ok=True)
            def execute(command, **kwargs):
                with (record / 'install.log').open('a', encoding='utf-8') as log:
                    # Index URLs can include credentials: do not write commands to receipts.
                    result = subprocess.run(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT, **kwargs)
                if result.returncode:
                    raise RuntimeError(f'Installer exited with code {result.returncode}; inspect {record / "install.log"}')
            def pip(requirements, *flags):
                command = [*self.uv(), 'pip', 'install', '--python', python, *flags, *map(str, requirements)]
                if index:
                    command += ['--index-url', index]
                execute(command)
            def stage(value):
                entry['stage'] = value
                self.save(name, entry)
                print(f'{name}: {value}', flush=True)
            try:
                if system_packages:
                    stage('system-packages')
                    if offline:
                        raise ValueError('--system-packages cannot be combined with --offline')
                    install_system_packages(profile, execute)
                if create and not Path(python).exists():
                    stage('python')
                    if destination.exists():
                        raise ValueError(f'Refusing to overwrite incomplete directory {destination}; inspect it or use another name.')
                    execute([*self.uv(), 'venv', '--python', profile['python'], destination])
                from packaging.specifiers import SpecifierSet
                version = subprocess.check_output([python, '-c', 'import platform;print(platform.python_version())'], env=env, text=True).strip()
                if version not in SpecifierSet(profile.get('requires_python', '==' + profile['python'] + '.*')):
                    raise ValueError(f'Python {version} does not match the recipe. Use env create for Python {profile["python"]}.')
                requirements = list(profile.get('requirements', []))
                stage('native-runtimes')
                requirements += prepare(profile, self.cache, env, pip, execute, accepted=accepted)
                requirement_file = record / 'requirements.txt'
                requirement_file.write_text(profile.get('lock') or '\n'.join(requirements) + '\n', encoding='utf-8')
                flags = ['--reinstall-package', 'vlastudio']
                if 'behavior1k' in profile.get('installers', []):
                    flags += ['--extra-index-url', 'https://pypi.nvidia.com']
                if profile.get('overrides'):
                    override = record / 'overrides.txt'
                    override.write_text('\n'.join(profile['overrides']) + '\n', encoding='utf-8')
                    flags += ['--override', str(override)]
                source = package or installed_source()
                print(f'Installing {name} with Python {version}; log: {record / "install.log"}', flush=True)
                stage('packages')
                pip(['-r', str(requirement_file), source], *flags)
                stage('patches-and-assets')
                finish(profile, self.cache, python, env, execute, pip=pip, accepted=accepted, assets=assets)
                execute([*self.uv(), 'pip', 'check', '--python', python])
                entry['receipts']['accepted_licenses'] = sorted(set(accepted))
                entry['variables'] = {k: v for k, v in env.items() if k.startswith(('COPPELIA', 'QT_', 'OMNIGIBSON_', 'OMNI_KIT_', 'VLASTUDIO_RUNTIME_', 'LIBERO_')) or k in ('LD_LIBRARY_PATH', 'CUDA_HOME', 'DISPLAY', 'MUJOCO_GL', 'PYOPENGL_PLATFORM')}
                entry['status'] = 'installed'
                entry['receipts']['profile_sha256'] = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
                entry['receipts']['python_version'] = version
                frozen = subprocess.check_output([*self.uv(), 'pip', 'freeze', '--python', python], env=env, text=True)
                (record / 'installed.txt').write_text(frozen, encoding='utf-8')
                self.save(name, entry)
                stage('validation')
                return self.check(name)
            except BaseException:
                entry['status'] = 'failed'
                self.save(name, entry)
                raise

    def check(self, name='base', smoke=False):
        entry = self.get(name)
        profile = entry.get('profile')
        if not profile:
            raise ValueError(f'{name} is registered but has no installation recipe. Run env install first.')
        record = self.cache / 'installations' / name
        record.mkdir(parents=True, exist_ok=True)
        profile_file = record / 'profile.json'
        atomic_json(profile_file, profile)
        command = [entry['python'], '-m', 'vlastudio.env_probe', str(profile_file)]
        if smoke:
            command.append('--smoke')
        log_path = record / ('smoke.log' if smoke else 'check.log')
        try:
            with log_path.open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, env=self.environment(entry), stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=os.name != 'nt')
                timeout = 1200 if 'behavior1k' in profile.get('checks', []) else 600
                code = process.wait(timeout=timeout if smoke else 180)
        except subprocess.TimeoutExpired:
            if os.name != 'nt':
                os.killpg(process.pid, signal.SIGKILL)
            else:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
            process.wait()
            code = 124
        entry['checks']['smoke' if smoke else 'imports'] = {'code': code, 'time': time.time(), 'log': str(log_path)}
        entry['status'] = ('ready' if smoke else 'imports-passed') if code == 0 else 'check-failed'
        entry['stage'] = 'complete' if code == 0 else 'validation'
        self.save(name, entry)
        print(f'{name}: {entry["status"]}; {log_path}')
        return code


def installed_source():
    dist = metadata.distribution('vlastudio')
    direct = json.loads(dist.read_text('direct_url.json') or '{}')
    if direct.get('url', '').startswith('file:'):
        return direct['url']
    return 'vlastudio==' + dist.version
