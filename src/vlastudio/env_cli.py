"""Select reusable environments by component configuration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .configuration import read_config
from .paths import cache_root, runtime_env
from .profiles import environment_for
from .runtime import prepare, environment_identity, snapshot

BASE_POLICIES = {'policy.act', 'policy.mlp', 'policy.diffusion_policy'}
BASE_ENVS = {'aloha', 'metaworld', 'libero'}


def select_profile(policy=None, environment=None, manifest=None, remote=False):
    base_path = Path(__file__).parent / 'environments' / 'base.yaml'
    if manifest:
        return 'custom', environment_for({}, base_path, manifest)
    selected = []
    if policy and not remote:
        policy = {'openpi': 'pi0'}.get(policy, policy)
        config, path = read_config('diffusion_policy' if policy == 'dp' else policy, 'policy')
        module = (config.get('type') or '').removeprefix('vlastudio.')
        if module == 'policy.smolvla' and not config.get('runtime'):
            config = dict(config, runtime=str(base_path.with_name('smolvla.yaml')))
        selected.append(('base' if module in BASE_POLICIES and not config.get('runtime') else 'policy', config, path))
    if environment:
        family = {'robotwin': 'robotwin', 'behavior': 'behavior1k', 'behavior1k': 'behavior1k'}.get(environment)
        if family:
            if selected:
                raise ValueError('Use separate policy and simulation environments for this simulator')
            manifest = base_path.with_name(family + '.yaml')
            return family, environment_for({}, manifest, str(manifest))
        config, path = read_config('aloha_transfer' if environment == 'aloha_sim' else environment, 'env')
        configs = config if isinstance(config, list) else [config]
        simple = all((c.get('type', '').removeprefix('vlastudio.benchmark.').split('.')[0] in BASE_ENVS) for c in configs)
        selected.append(('base' if simple else 'env', config, path))
    if all(kind == 'base' for kind, _, _ in selected):
        return 'base', environment_for({}, base_path, str(base_path))
    if len(selected) > 1:
        raise ValueError('These components require separate environments. Run serve with --policy and evaluation with --remote --env, or supply --runtime-manifest.')
    kind, config, path = selected[0]
    if isinstance(config, list):
        raise ValueError('Use --runtime-manifest for a multi-environment configuration')
    profile = environment_for(config, path)
    return kind, profile


def main(argv):
    from . import env_registry
    parser = argparse.ArgumentParser(prog='vlastudio env')
    parser.add_argument('action', choices=['create', 'install', 'activate', 'init', 'prepare', 'run', 'path', 'list'])
    parser.add_argument('name', nargs='?')
    parser.add_argument('-n', '--name', dest='environment_name')
    parser.add_argument('--shell', choices=['bash', 'zsh', 'powershell'])
    parser.add_argument('--policy')
    parser.add_argument('--env')
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--runtime-manifest')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    split = argv.index('--') if '--' in argv else len(argv)
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    cache = cache_root(args.cache_dir)
    if args.action == 'init':
        print(env_registry.hook(args.shell or 'bash'))
        return 0
    if args.action == 'activate':
        try:
            if not args.shell:
                raise ValueError('Initialize the shell first: eval "$(vlastudio env init --shell bash)"; PowerShell: vlastudio env init --shell powershell | Out-String | Invoke-Expression')
            print(env_registry.activation(cache, args.name, args.shell))
            return 0
        except (ValueError, OSError) as error:
            print(str(error), file=sys.stderr)
            return 2
    if args.action == 'list':
        entries = env_registry.read(cache)
        for name, entry in sorted(entries.items()):
            state = 'ready' if Path(entry['python']).is_file() else 'missing'
            print(f"{name}\t{entry['kind']}\t{state}\t{entry['python']}")
        for ready in sorted((cache / 'envs').glob('*/.ready')):
            python = ready.parent / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            if not any(e['python'] == str(python) for e in entries.values()):
                print(f'{ready.parent.name}\tmanaged\t{python}')
        if os.environ.get('VLASTUDIO_BASE_PYTHON'):
            print('base: ' + os.environ['VLASTUDIO_BASE_PYTHON'])
        return 0
    try:
        name, profile = select_profile(args.policy, args.env, args.runtime_manifest, args.remote)
        label = args.environment_name or args.name or (name if name == 'base' else args.policy or args.env or 'custom')
        if args.action == 'install':
            active = os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX')
            current = Path(active) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python') if active else Path(sys.executable)
            python = Path(os.environ.get('VLASTUDIO_ACTIVE_PYTHON') or current)
            if args.dry_run:
                print(json.dumps({'python': str(python), 'profile': profile}, indent=2))
                return 0
            env = runtime_env(cache)
            if args.offline:
                env['UV_OFFLINE'] = '1'
            env_registry.install(profile, python, cache, env)
            label = env_registry.register(cache, label, python, 'installed')
            print(f'{label}: {python}')
            return 0
        key, _ = environment_identity(profile)
        python = cache / 'envs' / key / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        external = os.environ.get('VLASTUDIO_BASE_PYTHON') if name == 'base' and args.action != 'create' else None
        if external:
            python = Path(external).expanduser().resolve()
            if not python.is_file():
                raise ValueError(f'Base interpreter does not exist: {python}')
        if args.dry_run:
            print(json.dumps({'environment': name, 'python': str(python), 'profile': profile}, indent=2))
            return 0
        if args.action == 'path':
            print(python)
            return 0
        env = runtime_env(cache)
        if not external:
            python = prepare(profile, cache, env, args.offline)
        label = env_registry.register(cache, label, python, 'external' if external else 'managed')
        if args.action in ('prepare', 'create'):
            print(f'{label}: {python}')
            return 0
        if not command:
            raise ValueError('Provide a command after --, for example -- python experiment.py')
        app = snapshot(cache)
        env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
        env['VIRTUAL_ENV'] = str(python.parent.parent)
        env['PYTHONPATH'] = os.pathsep.join([str(app), os.getcwd(), env.get('PYTHONPATH', '')])
        if command[0] in ('python', 'python3'):
            command[0] = str(python)
        return subprocess.call(command, env=env)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'vlastudio env: {error}', file=sys.stderr)
        return 2
