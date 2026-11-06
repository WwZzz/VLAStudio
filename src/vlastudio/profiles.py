"""Composable runtime recipes. Reading them never imports a component."""
import copy
from pathlib import Path
import platform
import sys

import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .paths import package_source_root


POLICY_PROFILES = {
    'act': 'base', 'mlp': 'base', 'diffusion_policy': 'diffusion_policy',
    'remote': 'remote', 'smolvla': 'smolvla', 'openpi': 'openpi',
    'openvla': 'openvla', 'fastwam': 'fastwam',
    'vlnce_cma': 'vlnce_cma', 'recurrent_vlnbert': 'base', 'tcp_b2d': 'base',
}
ENV_PROFILES = {'aloha': 'aloha', 'libero': 'libero', 'metaworld': 'metaworld',
                'tabletop_sim': 'tabletop_sim', 'rlbench': 'rlbench',
                'robotwin': 'robotwin', 'behavior1k': 'behavior1k',
                'habitat': 'habitat', 'matterport3d': 'matterport3d', 'carla': 'carla',
                'robocasa': 'robocasa'}
FIELDS = {'extends', 'python', 'requires_python', 'requirements', 'lockfile', 'lock', 'entrypoint',
          'overrides', 'overlays', 'platforms', 'min_glibc', 'description', 'notes',
          'installers', 'checks'}


def recipe(name):
    return package_source_root() / 'environments' / (name + '.yaml')


def _requirements(parent, child):
    """A child explicitly replaces a parent's package constraints, including pins."""
    def key(value):
        req = Requirement(value)
        return canonicalize_name(req.name), str(req.marker or '')
    names = {key(r) for r in child}
    return [r for r in parent if key(r) not in names] + child


def _inherit(parent, child):
    result = {**parent, **child}
    if 'python' in child and 'requires_python' not in child:
        result.pop('requires_python', None)
    result['requirements'] = _requirements(parent.get('requirements', []), child.get('requirements', []))
    for field in ('overrides', 'overlays', 'notes', 'installers', 'checks'):
        if field in parent or field in child:
            if not isinstance(child.get(field, []), list):
                raise ValueError(f'runtime.{field} must be a list')
            result[field] = list(parent.get(field, []))
            result[field].extend(v for v in child.get(field, []) if v not in result[field])
    return result


def _declaration(value, path, seen):
    if isinstance(value, str):
        target = Path(value).expanduser()
        if not target.is_absolute():
            target = path.parent / target
        path = target.resolve()
        if path in seen:
            raise ValueError(f'Runtime inheritance cycle: {path}')
        seen = (*seen, path)
        value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('runtime must be a mapping or a YAML manifest path')
    unknown = set(value) - FIELDS
    if unknown:
        raise ValueError(f'Unknown runtime fields: {sorted(unknown)}')
    profile = copy.deepcopy(value)
    parents = profile.pop('extends', [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
        raise ValueError('runtime.extends must be a manifest path or list of paths')
    requirements = profile.setdefault('requirements', [])
    if not isinstance(requirements, list) or not all(isinstance(r, str) for r in requirements):
        raise ValueError('runtime.requirements must be a list of requirement strings')
    _requirements([], requirements)
    if profile.get('lockfile'):
        if requirements or profile.get('lock'):
            raise ValueError('Use either a complete runtime.lockfile or runtime.requirements/lock')
        lock = Path(profile.pop('lockfile')).expanduser()
        if not lock.is_absolute():
            lock = path.parent / lock
        profile['lock'] = lock.read_text(encoding='utf-8')
    if '.py:' in profile.get('entrypoint', ''):
        filename, attribute = profile['entrypoint'].rsplit(':', 1)
        entry = Path(filename).expanduser()
        if not entry.is_absolute():
            entry = path.parent / entry
        profile['entrypoint'] = str(entry.resolve()) + ':' + attribute
    result = {}
    for parent in parents:
        inherited = _declaration(parent, path, seen)
        if inherited.get('lock') or profile.get('lock'):
            raise ValueError('Complete lockfiles cannot be extended; supply one final lockfile')
        result = _inherit(result, inherited)
    return _inherit(result, profile)


def platform_issues(profile):
    issues = []
    if profile.get('platforms') and sys.platform not in profile['platforms']:
        issues.append(f"Requires platform {profile['platforms']}; current platform is {sys.platform}")
    if profile.get('min_glibc') and sys.platform.startswith('linux'):
        libc, version = platform.libc_ver()
        if libc != 'glibc' or tuple(map(int, version.split('.'))) < tuple(map(int, profile['min_glibc'].split('.'))):
            issues.append(f"Requires glibc >= {profile['min_glibc']}; found {libc} {version}")
    return issues


def environment_for(config, path, manifest=None, *, check_platform=True):
    declaration = manifest or config.get('runtime')
    if not declaration:
        module = (config.get('type') or config.get('module_path') or '').removeprefix('vlastudio.')
        parts = module.split('.')
        family = POLICY_PROFILES.get(parts[1]) if len(parts) == 2 and parts[0] == 'policy' else None
        if len(parts) >= 2 and parts[0] == 'benchmark':
            family = ENV_PROFILES.get(parts[1])
        if family:
            declaration = str(recipe(family))
        else:
            # Historical models retain their own requirements rather than a
            # guessed compatible base. Surface this limitation in the plan.
            metadata = package_source_root().joinpath(*parts[:2], 'pyproject.toml')
            if len(parts) >= 2 and parts[0] == 'policy' and metadata.is_file():
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib
                project = tomllib.loads(metadata.read_text(encoding='utf-8'))['project']
                from packaging.specifiers import SpecifierSet
                supported = project.get('requires-python', '>=3.10')
                python = next((v for v in ('3.10', '3.11', '3.12') if v in SpecifierSet(supported)), None)
                if python is None:
                    raise ValueError(f'No supported default Python in {supported}; supply --runtime-manifest with an explicit Python version')
                declaration = {'python': python, 'requires_python': supported,
                               'requirements': project.get('dependencies', []),
                               'notes': ['Legacy component requirements; not a validated layered recipe. See ' + str(metadata)]}
            else:
                raise ValueError(
                    f'No runtime recipe for {module!r}. Add runtime: /path/runtime.yaml to the '
                    'component config, or supply --runtime-manifest /path/runtime.yaml. '
                    'Use check_dependencies=False only to create a configuration handle.'
                )
    profile = _declaration(declaration, Path(path), ())
    profile['python'] = str(profile.get('python', '3.10'))
    if check_platform and platform_issues(profile):
        raise ValueError('; '.join(platform_issues(profile)))
    return profile


def combine_profiles(profile, extra):
    """Peer components must coexist; keep constraints for the resolver to check."""
    if profile['python'] != extra['python']:
        raise ValueError('Components require different Python versions; use separate environments/services or one compatible --runtime-manifest')
    if extra.get('entrypoint'):
        raise ValueError('Only the top-level runtime may select the task entrypoint')
    if extra.get('lock'):
        raise ValueError('Use one complete lockfile instead of multiple component lockfiles')
    result = copy.deepcopy(profile)
    if result.get('lock') and extra['requirements']:
        result['requirements'] = [line for line in result.pop('lock').splitlines() if line.strip() and not line.lstrip().startswith('#')]
    for field in ('requirements', 'overrides', 'overlays', 'notes', 'installers', 'checks'):
        if field in result or field in extra:
            result.setdefault(field, [])
            result[field].extend(v for v in extra.get(field, []) if v not in result[field])
    if extra.get('platforms'):
        allowed = set(result.get('platforms', extra['platforms'])) & set(extra['platforms'])
        if not allowed:
            raise ValueError('Components require incompatible platforms; use separate services')
        result['platforms'] = sorted(allowed)
    if extra.get('min_glibc'):
        result['min_glibc'] = max(result.get('min_glibc', '0'), extra['min_glibc'], key=lambda v: tuple(map(int, v.split('.'))))
    if extra.get('requires_python'):
        current = result.get('requires_python', '==' + result['python'] + '.*')
        result['requires_python'] = current + ',' + extra['requires_python']
    return result


def merge_component_runtimes(profile, config, path):
    """Combine explicitly declared nested component runtimes for a worker."""
    result = copy.deepcopy(profile)
    def visit(value):
        nonlocal result
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            if value.get('runtime'):
                result = combine_profiles(result, environment_for(value, path))
            for key, child in value.items():
                if key != 'runtime':
                    visit(child)
    visit(config)
    return result


def dataset_environment(config, path, *, check_platform=True, python=None):
    """Resolve dataset dependencies; navigation replay also needs its native backend."""
    profile = environment_for({}, path, str(recipe('data')), check_platform=check_platform)
    fixed_python = python is not None
    if python:
        profile['python'] = python
    def visit(value):
        nonlocal profile, fixed_python
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            module = str(value.get('type', '')).removeprefix('vlastudio.')
            if value.get('runtime'):
                extra = environment_for(value, path, check_platform=check_platform)
                if not fixed_python:
                    profile['python'] = extra['python']
                    fixed_python = True
                profile = combine_profiles(profile, extra)
            elif module.startswith('data_utils.datasets') and any(s in module.lower() for s in ('droid', 'rlds', 'oxe')):
                profile = combine_profiles(profile, environment_for({}, path, str(recipe('rlds')), check_platform=check_platform))
            elif module.endswith('.WrappedLerobotDataset') or module.startswith('data_utils.datasets.libero_lerobot'):
                profile = combine_profiles(profile, {'python': profile['python'], 'requirements': ['lerobot==0.3.3']})
            elif module.endswith('.R2RDataset'):
                profile = combine_profiles(profile, environment_for({}, path, str(recipe('matterport3d')), check_platform=check_platform))
            elif module.endswith('.VLNCEDataset'):
                for family in ['habitat', 'vlnce_cma']:
                    profile = combine_profiles(profile, environment_for({}, path, str(recipe(family)), check_platform=check_platform))
            for key, child in value.items():
                if key != 'runtime':
                    visit(child)
    visit(config)
    return profile
