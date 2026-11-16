"""Explicit environment installation and inspection commands."""
import argparse
import json
from pathlib import Path
import sys

from vlastudio.core.configuration import read_config
from vlastudio.deploy.interpreter.recipes import environment_for


def select_profile(policy=None, environment=None, manifest=None, remote=False,
                   dataset=None, *, check_platform=True):
    from vlastudio.deploy.interpreter.recipes import combine_profiles, dataset_environment, recipe
    if manifest:
        if policy or environment or dataset:
            raise ValueError('Use either a complete --runtime-manifest or component selectors')
        return 'custom', environment_for({}, Path.cwd() / 'config.yaml',
                                         str(Path(manifest).expanduser().resolve()),
                                         check_platform=check_platform)
    selected = []
    if policy and not remote:
        policy = {'openpi': 'pi0', 'dp': 'diffusion_policy'}.get(policy, policy)
        config, path = read_config(policy, 'policy')
        selected.append(('policy', environment_for(config, path, check_platform=check_platform)))
    elif remote:
        selected.append(('remote', environment_for({}, recipe('remote'), str(recipe('remote')),
                                                   check_platform=check_platform)))
    if environment:
        from vlastudio.deploy.interpreter.recipes import ENV_PROFILES
        family = ENV_PROFILES.get(environment) or {'behavior': 'behavior1k', 'aloha_sim': 'aloha'}.get(environment)
        if family:
            profile = environment_for({}, recipe(family), str(recipe(family)), check_platform=check_platform)
        else:
            config, path = read_config('aloha_transfer' if environment == 'aloha_sim' else environment, 'env')
            configs = config if isinstance(config, list) else [config]
            profiles = [environment_for(c, path, check_platform=check_platform) for c in configs]
            profile = profiles[0]
            for extra in profiles[1:]:
                profile = combine_profiles(profile, extra)
        selected.append(('env', profile))
    if dataset:
        config, path = read_config(dataset, 'task')
        # Shared data requirements work on the selected model's Python version.
        profile = dataset_environment(config, path, check_platform=check_platform,
                                      python=selected[0][1]['python'] if selected else None)
        selected.append(('data', profile))
    if not selected:
        profile = environment_for({}, recipe('base'), str(recipe('base')), check_platform=check_platform)
        for family in ('diffusion_policy', 'aloha'):
            profile = combine_profiles(profile, environment_for({}, recipe(family), str(recipe(family)), check_platform=check_platform))
        return 'base', profile
    kind, profile = selected[0]
    for _, extra in selected[1:]:
        profile = combine_profiles(profile, extra)
    return ('combined' if len(selected) > 1 else kind), profile


def main(argv):
    import os
    import subprocess
    from vlastudio.deploy.interpreter.manager import Manager
    # Everything after -- belongs verbatim to the child, including its options.
    command = []
    if '--' in argv:
        split = argv.index('--')
        command, argv = argv[split + 1:], argv[:split]
    parser = argparse.ArgumentParser(prog="vlastudio env")
    parser.add_argument("action", choices=["plan", "check", "catalog", "create", "install", "list", "run", "activate", "deactivate", "path"])
    parser.add_argument('name', nargs='?')
    parser.add_argument('-n', '--name', dest='environment_name')
    parser.add_argument('-i', '--index-url')
    parser.add_argument('--cache-dir')
    parser.add_argument('--package', help='VLAStudio wheel or checkout to install (default: this installation)')
    parser.add_argument('--accept-license', action='append', default=[], help='Explicit license ID from env plan; repeatable')
    parser.add_argument('--assets', action='store_true', help='Download optional simulator datasets after license acceptance')
    parser.add_argument('--system-packages', action='store_true', help='Also install native prerequisites using apt (requires root)')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Also execute GPU/simulator smoke checks')
    parser.add_argument("--policy")
    parser.add_argument("--env")
    parser.add_argument("--dataset", nargs="?", const="sim_transfer_cube_scripted")
    parser.add_argument("--runtime-manifest")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.action not in ('create', 'install', 'plan'):
        parser.error('--dry-run is supported only for create, install and plan')
    if args.smoke and args.action not in ('create', 'install', 'check'):
        parser.error('--smoke requires create, install or check')
    from vlastudio.deploy.interpreter.dependencies import environment_plan, format_plan
    try:
        from vlastudio.deploy.interpreter.recipes import POLICY_PROFILES, ENV_PROFILES
        if args.action == "catalog":
            print(json.dumps({'policies': POLICY_PROFILES, 'environments': ENV_PROFILES}, indent=2))
            return 0
        if args.name and args.environment_name and args.name != args.environment_name:
            raise ValueError('Positional name and --name disagree')
        name = args.environment_name or args.name
        if name is None:
            name = ((args.policy or args.env or 'base') if args.action == 'create' else 'base')
            if '/' in name or '\\' in name:
                name = Path(name).stem
            name = {'act': 'base', 'mlp': 'base', 'dp': 'base', 'diffusion_policy': 'base'}.get(name, name)
        if args.action in ('create', 'install'):
            _, profile = select_profile(args.policy, args.env, args.runtime_manifest, args.remote, args.dataset, check_platform=not args.dry_run)
            if args.dry_run:
                from vlastudio.deploy.interpreter.installers import describe
                print(json.dumps({'name': name, 'profile': profile, 'native': describe(profile)}, indent=2))
                return 0
            manager = Manager(args.cache_dir)
            code = manager.install(name, profile, create=args.action == 'create', index=args.index_url,
                                   package=args.package, accepted=args.accept_license, assets=args.assets, offline=args.offline,
                                   system_packages=args.system_packages)
            return code or (manager.check(name, smoke=True) if args.smoke else 0)
        if args.action in ('list', 'run', 'activate', 'deactivate', 'path') or (args.action == 'check' and (args.name or args.environment_name or not any((args.policy, args.env, args.dataset, args.runtime_manifest, args.remote)))):
            manager = Manager(args.cache_dir)
            if args.action == 'list':
                entries = manager.entries()
                if args.json:
                    print(json.dumps(entries, indent=2))
                else:
                    for key, value in entries.items():
                        state = value['status'] if Path(value['python']).is_file() else 'missing'
                        print(f"{key}\t{state}\t{value['python']}")
                return 0
            if args.action == 'path':
                print(manager.get(name)['python'])
                return 0
            if args.action == 'check':
                return manager.check(name, args.smoke)
            if args.action in ('activate', 'deactivate'):
                name = 'base' if args.action == 'deactivate' else name
                command = [os.environ.get('COMSPEC', 'cmd.exe')] if os.name == 'nt' else [os.environ.get('SHELL', '/bin/bash'), '--norc']
                print(f'Opening {name} in a child shell. Use exit to return to the previous shell.', flush=True)
            return manager.run(name, command)
        plan = environment_plan(policy=args.policy, environment=args.env,
                                dataset=args.dataset, manifest=args.runtime_manifest, remote=args.remote)
        if args.smoke:
            raise ValueError('Use env check NAME --smoke to validate a registered environment.')
        from vlastudio.deploy.interpreter.installers import describe
        plan['native'] = describe(plan['profile'])
        print(json.dumps(plan, indent=2) if args.json else format_plan(plan))
        return 2 if args.action == "check" and plan["issues"] else 0
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 2
