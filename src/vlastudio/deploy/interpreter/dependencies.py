"""Dependency guidance using installed metadata, without importing ML packages."""
import importlib.metadata as metadata
import os
import shlex
import sys


class DependencyError(ImportError):
    """A component's declared runtime is missing or incompatible."""
    def __init__(self, plan):
        self.plan = plan
        super().__init__(format_plan(plan))


def inspect_environment(profile):
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
    from vlastudio.deploy.interpreter.recipes import _requirements, platform_issues
    issues = platform_issues(profile)
    python = profile['python']
    spec = profile.get('requires_python') or (python if python.startswith(('>', '<', '=', '!', '~')) else '==' + python + '.*')
    if Version('.'.join(map(str, sys.version_info[:3]))) not in SpecifierSet(spec):
        issues.append(f"Requires Python {spec}; current Python is {sys.version.split()[0]}")
    requirements = profile.get('requirements', [])
    if profile.get('lock'):
        requirements = [line.strip() for line in profile['lock'].splitlines()
                        if line.strip() and not line.lstrip().startswith('#')]
    requirements = _requirements(requirements, profile.get('overrides', []))
    packages = []
    for value in requirements:
        req = Requirement(value)
        if req.marker and not req.marker.evaluate():
            continue
        try:
            version = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            version = None
        status = 'missing' if version is None else ('version mismatch' if req.specifier and version not in req.specifier else 'installed')
        packages.append({'requirement': value, 'installed': version, 'status': status})
        if status != 'installed':
            issues.append(f'{value}: {status}' + (f' (installed {version})' if version else ''))
    return issues, packages


def _command(tokens):
    if os.name != 'nt':
        return shlex.join(map(str, tokens))
    # PowerShell literal strings, including paths containing spaces or apostrophes.
    def quote(value):
        value = str(value)
        return value if all(c.isalnum() or c in '/\\._:-=' for c in value) else "'" + value.replace("'", "''") + "'"
    return ' '.join(quote(token) for token in tokens)


def environment_plan(*, policy=None, environment=None, dataset=None, manifest=None, remote=False):
    """Read-only installation plan for the calling interpreter."""
    from vlastudio.deploy.interpreter.cli import select_profile
    kind, profile = select_profile(policy, environment, manifest, remote, dataset, check_platform=False)
    issues, packages = inspect_environment(profile)
    selectors = []
    for key, value in (('policy', policy), ('environment', environment), ('dataset', dataset), ('manifest', manifest)):
        if value:
            selectors.append(f'{key}={str(value)!r}')
    if remote:
        selectors.append('remote=True')
    arguments = ', '.join(selectors)
    requirements = [p["requirement"] for p in packages]
    install = _command(["python", "-m", "pip", "install", *requirements])
    env_selectors = []
    for flag, value in (('--policy', policy), ('--env', environment), ('--dataset', dataset), ('--runtime-manifest', manifest)):
        if value:
            env_selectors += [flag, str(value)]
    if remote:
        env_selectors.append('--remote')
    return {"environment": kind, "selection": arguments, "python": sys.executable, "required_python": profile["python"],
            "profile": profile, "issues": issues, "packages": packages,
            "commands": {
                "create": _command(["vlastudio", "env", "create", *env_selectors]),
                "install_current": _command(["vlastudio", "env", "install", *env_selectors]),
                "run": "python run.py",
                "check": _command(["python", "-c", f"import vlastudio as vla; print(vla.environment_plan({arguments}))"]),
            }}


def format_plan(plan):
    lines = [f"VLAStudio runtime: {plan['environment']} (Python {plan['required_python']})",
             f"Current interpreter: {plan['python']}"]
    if plan.get('selection'):
        lines.append('Selected components: ' + plan['selection'])
    if plan['issues']:
        lines.append('Missing or incompatible dependencies:')
        lines.extend('  - ' + issue for issue in plan['issues'])
    else:
        lines.append('Declared package versions are installed (metadata check only).')
    lines.extend(['Create an isolated environment:',
                  '  ' + plan['commands']['create'],
                  'Or install into the calling interpreter:',
                  '  ' + plan['commands']['install_current'],
                  'Then run your own workflow:', '  ' + plan['commands']['run'],
                  'Installation runs only through explicit env commands; Python imports never install packages.'])
    lines.extend('Note: ' + note for note in plan['profile'].get('notes', []))
    for step in plan.get('native', []):
        for key, url in step['licenses'].items():
            lines.append(f'Native license: {url}; explicitly accept with --accept-license {key}')
    lines.append('Checks do not verify CUDA, native libraries, assets, import health, extras or VCS revisions.')
    return '\n'.join(lines)


def require_dependencies(**selectors):
    plan = environment_plan(**selectors)
    if plan['issues']:
        raise DependencyError(plan)
    return plan
