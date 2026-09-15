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
    parser = argparse.ArgumentParser(prog='vlastudio env')
    parser.add_argument('action', choices=['prepare', 'run', 'path', 'list'])
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
    if args.action == 'list':
        for ready in sorted((cache / 'envs').glob('*/.ready')):
            print(ready.parent)
        if os.environ.get('VLASTUDIO_BASE_PYTHON'):
            print('base: ' + os.environ['VLASTUDIO_BASE_PYTHON'])
        return 0
    try:
        name, profile = select_profile(args.policy, args.env, args.runtime_manifest, args.remote)
        key, _ = environment_identity(profile)
        python = cache / 'envs' / key / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        external = os.environ.get('VLASTUDIO_BASE_PYTHON') if name == 'base' else None
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
        if args.action == 'prepare':
            print(python)
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
