"""Site-configured simulator launchers, independent of the dashboard Python."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys

from .profiles import PACKAGE, ROOT, catalog

_CLEAN_KEYS = (
    'PYTHONHOME', 'PYTHONPATH', 'MUJOCO_GL', 'PYOPENGL_PLATFORM',
    'COPPELIASIM_ROOT', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_PLUGIN_PATH',
    'DISPLAY', 'WAYLAND_DISPLAY', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
    'LIBGL_ALWAYS_SOFTWARE', 'LIBGL_DRIVERS_PATH', '__GLX_VENDOR_LIBRARY_NAME',
    '__EGL_VENDOR_LIBRARY_FILENAMES', 'VK_ICD_FILENAMES',
)
_PROBE = """
import importlib, json, os, sys
try:
    for key in json.loads(sys.argv[2]):
        if not os.environ.get(key):
            raise RuntimeError('Missing environment setting: ' + key)
    root = os.environ.get('COPPELIASIM_ROOT')
    if 'COPPELIASIM_ROOT' in json.loads(sys.argv[2]) and not os.path.isdir(root):
        raise RuntimeError('CoppeliaSim directory does not exist: ' + root)
    for name in json.loads(sys.argv[1]):
        importlib.import_module(name)
    result = dict(ok=True, python=sys.executable)
    if sys.argv[3] == 'cuda':
        import torch
        result['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception as error:
    result = dict(ok=False, error=type(error).__name__ + ': ' + str(error))
print('VLASTUDIO_RUNTIME_PROBE=' + json.dumps(result), flush=True)
"""


@dataclass
class RuntimeLaunch:
    name: str
    prefix: list
    env: dict
    imports: list = field(default_factory=list)
    required_env: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def module(self, name):
        return [*self.prefix, '-u', '-m', name]

    def probe(self, check_cuda=False):
        try:
            result = subprocess.run(
                [*self.prefix, '-c', _PROBE, json.dumps(self.imports),
                 json.dumps(self.required_env), 'cuda' if check_cuda else ''],
                cwd=ROOT, env=self.env, capture_output=True, text=True, timeout=45)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ValueError(f'无法启动 {self.name} 运行环境：{error}') from error
        lines = [line for line in result.stdout.splitlines()
                 if line.startswith('VLASTUDIO_RUNTIME_PROBE=')]
        try:
            info = json.loads(lines[-1].split('=', 1)[1]) if lines else {}
        except (ValueError, IndexError):
            info = {}
        if result.returncode or not info.get('ok'):
            reason = info.get('error') or (result.stderr or result.stdout)[-1000:]
            raise ValueError(f'{self.name} 运行环境不可用：{reason}。请检查 WebUI 运行环境配置。')
        self.info = info
        return info


def resolve_runtime(profile, *, registry_path=None, environ=None):
    """Resolve only trusted site configuration; request data cannot supply commands."""
    parent = dict(os.environ if environ is None else environ)
    settings = catalog()['robot_options'].get(profile, {})
    name = settings.get('runtime', 'current')
    selected = registry_path or parent.get('VLASTUDIO_WEBUI_RUNTIMES')
    if selected is None:
        candidate = Path(parent.get('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'vlastudio/webui-runtimes.json'
        selected = candidate if candidate.is_file() else None
    entry = None
    if selected is not None:
        try:
            registry = json.loads(Path(selected).expanduser().read_text(encoding='utf-8'))
            entry = registry.get('runtimes', {}).get(name)
        except (OSError, ValueError, AttributeError) as error:
            raise ValueError(f'无法读取 WebUI 运行环境配置：{error}') from error
    if entry is None:
        # Existing installations continue to use their current interpreter.
        return RuntimeLaunch(name, [sys.executable], parent,
                             list(settings.get('runtime_imports', [])),
                             list(settings.get('runtime_required_env', [])))
    if not isinstance(entry, dict):
        raise ValueError(f'Invalid runtime configuration: {name}')
    prefix = entry.get('command')
    if not isinstance(prefix, list) or not prefix or any(not isinstance(x, str) or not x for x in prefix):
        raise ValueError(f'Runtime {name} needs a nonempty command argument list')
    for key in (*_CLEAN_KEYS, *entry.get('unset', [])):
        parent.pop(key, None)
    env = entry.get('env', {})
    if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError(f'Runtime {name} env must contain string values')
    parent.update(env)
    paths = entry.get('extra_paths', [])
    if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
        raise ValueError(f'Runtime {name} extra_paths must be a list of paths')
    parent['PYTHONPATH'] = os.pathsep.join([*paths, str(PACKAGE.parent), *([parent['PYTHONPATH']] if parent.get('PYTHONPATH') else [])])
    parent['PYTHONNOUSERSITE'] = '1'
    return RuntimeLaunch(name, prefix, parent,
                         list(dict.fromkeys([*settings.get('runtime_imports', []), *entry.get('imports', [])])),
                         list(dict.fromkeys([*settings.get('runtime_required_env', []), *entry.get('required_env', [])])))
