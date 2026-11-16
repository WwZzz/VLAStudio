"""Native runtime steps executed only by explicit env create/install commands."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

from filelock import FileLock

LICENSES = {
    'coppeliasim-edu': 'https://www.coppeliarobotics.com/licensing',
    'nvidia-isaac-sim': 'https://docs.omniverse.nvidia.com/isaacsim/latest/common/licenses.html',
    'behavior-dataset': 'https://behavior.stanford.edu/behavior-1k-dataset-license.html',
}
REQUIRED = {'rlbench': ['coppeliasim-edu'], 'behavior1k': ['nvidia-isaac-sim']}
SYSTEM_PACKAGES = {
    'carla': ['libvulkan1', 'libegl1', 'libgl1', 'libglib2.0-0', 'libx11-6',
              'libxcb1', 'libxcb-randr0', 'libxcb-xfixes0', 'libxcb-shape0', 'libdrm2'],
    'robotwin': ['build-essential', 'libvulkan1', 'libgl1', 'libegl1', 'libglib2.0-0'],
    'rlbench': ['build-essential', 'xvfb', 'xauth', 'libgl1', 'libglu1-mesa', 'libxkbcommon-x11-0', 'libxcb-xinerama0'],
    'behavior1k': ['libgl1', 'libegl1', 'libglu1-mesa', 'libxrender1', 'libsm6', 'libice6', 'libxt6', 'libvulkan1'],
}
COPPELIA = 'https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz'


def describe(profile):
    return [{'installer': name, 'licenses': {key: LICENSES[key] for key in REQUIRED.get(name, [])}, 'system_packages': SYSTEM_PACKAGES.get(name, [])}
            for name in profile.get('installers', [])]


def preflight(profile, accepted=(), system_packages=False):
    for name in profile.get('installers', []):
        if name not in ('rlbench', 'behavior1k', 'libero', 'robotwin', 'carla'):
            raise ValueError(f'Unknown native installer {name!r}')
        for license_id in REQUIRED.get(name, []):
            if license_id not in accepted:
                raise ValueError(f'Review {LICENSES[license_id]}, then pass --accept-license {license_id} if eligible and you agree.')
    if 'rlbench' in profile.get('installers', []) and not shutil.which('gcc') and not system_packages:
        raise ValueError('RLBench builds PyRep and requires a C compiler (e.g. apt install build-essential).')
    if 'robotwin' in profile.get('installers', []):
        cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
        if not shutil.which('nvcc') and not (cuda / 'bin/nvcc').is_file():
            raise ValueError('RoboTwin requires the CUDA toolkit (nvcc) to build cuRobo. Set CUDA_HOME to a toolkit compatible with PyTorch; env create does not install GPU drivers.')


def install_system_packages(profile, execute):
    packages = sorted({p for name in profile.get('installers', []) for p in SYSTEM_PACKAGES.get(name, [])})
    if packages:
        if not shutil.which('apt-get') or not hasattr(os, 'geteuid') or os.geteuid() != 0:
            raise ValueError('--system-packages requires root on an apt-based host. Ask your administrator to install: ' + ' '.join(packages))
        execute(['apt-get', 'update'])
        execute(['apt-get', 'install', '-y', '--no-install-recommends', *packages])


def archive(url, destination, cache, *, offline=False):
    """Extract a versioned archive atomically, rejecting paths outside its root."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + '.lock', timeout=600):
        receipt = destination / '.vlastudio-source.json'
        if receipt.exists():
            if json.loads(receipt.read_text())['url'] != url:
                raise ValueError(f'Runtime source changed: {destination}')
            return destination
        if destination.exists():
            raise ValueError(f'Incomplete or unmanaged runtime: {destination}')
        downloads = cache / 'downloads'
        downloads.mkdir(parents=True, exist_ok=True)
        download = downloads / hashlib.sha256(url.encode()).hexdigest()
        if not download.exists():
            if offline:
                raise ValueError(f'Native runtime archive is not cached: {destination.name}')
            temporary = download.with_suffix('.partial')
            request = urllib.request.Request(url, headers={'User-Agent': 'VLAStudio/0.2 (+https://github.com/WwZzz/VLAStudio)'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            temporary.replace(download)
        digest = hashlib.sha256()
        with download.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
            root = Path(temp).resolve()
            with tarfile.open(download) as source:
                for member in source.getmembers():
                    target = (root / member.name).resolve()
                    if not target.is_relative_to(root) or member.isdev():
                        raise ValueError('Unsafe archive member')
                    if member.issym() or member.islnk():
                        link = ((target.parent if member.issym() else root) / member.linkname).resolve()
                        if not link.is_relative_to(root):
                            raise ValueError('Unsafe archive link')
                source.extractall(root, **({'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}))
            children = list(root.iterdir())
            if len(children) != 1 or not children[0].is_dir():
                raise ValueError('Expected a single archive root')
            children[0].rename(destination)
        receipt.write_text(json.dumps({'url': url, 'sha256': digest.hexdigest()}, indent=2), encoding='utf-8')
        return destination


def prepare(profile, cache, env, pip, execute, *, accepted=()):
    requirements = []
    for name in profile.get('installers', []):
        if name == 'robotwin':
            from vlastudio.deploy.interpreter.robotwin import REVISION
            root = archive(f'https://github.com/WwZzz/RoboTwin/archive/{REVISION}.tar.gz', cache / 'runtimes' / ('robotwin-' + REVISION[:8]), cache, offline=env.get('UV_OFFLINE') == '1')
            env['VLASTUDIO_RUNTIME_ROBOTWIN_ROOT'] = str(root)
            env.setdefault('CUDA_HOME', str(Path(shutil.which('nvcc')).parent.parent) if shutil.which('nvcc') else '/usr/local/cuda')
            env.setdefault('MAX_JOBS', '2')
        elif name == 'rlbench':
            if env.get('COPPELIASIM_ROOT'):
                root = Path(env['COPPELIASIM_ROOT']).expanduser().resolve()
                if not (root / 'libcoppeliaSim.so').is_file():
                    raise ValueError(f'COPPELIASIM_ROOT does not contain libcoppeliaSim.so: {root}')
            else:
                root = archive(COPPELIA, cache / 'runtimes' / 'coppeliasim-4.1.0', cache, offline=env.get('UV_OFFLINE') == '1')
            env['COPPELIASIM_ROOT'] = str(root)
            env['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(root)
            env['LD_LIBRARY_PATH'] = str(root) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
            # PyRep's build requires CoppeliaSim before dependency resolution starts.
            pip(['setuptools', 'wheel', 'cffi'])
        elif name == 'behavior1k':
            source = archive('https://github.com/StanfordVL/BEHAVIOR-1K/archive/88454bd04f75dc57c00ab1f1a00bcde1ff505950.tar.gz',
                             cache / 'runtimes' / 'behavior-88454bd0', cache, offline=env.get('UV_OFFLINE') == '1')
            env['VLASTUDIO_RUNTIME_BEHAVIOR_ROOT'] = str(source)
            env['OMNI_KIT_ACCEPT_EULA'] = 'YES'
            env['OMNIGIBSON_HEADLESS'] = 'True'
            env['OMNIGIBSON_DATA_PATH'] = str(cache / 'assets' / 'behavior1k')
            env['OMNIGIBSON_APPDATA_PATH'] = str(cache / 'runtimes' / 'behavior1k-appdata')
            Path(env['OMNIGIBSON_DATA_PATH']).mkdir(parents=True, exist_ok=True)
            Path(env['OMNIGIBSON_APPDATA_PATH']).mkdir(parents=True, exist_ok=True)
            requirements.append('isaacsim[all,extscache]==4.5.0')
        elif name == 'libero':
            root = archive('https://github.com/Lifelong-Robot-Learning/LIBERO/archive/8f1084e3132a39270c3a13ebe37270a43ece2a01.tar.gz', cache / 'runtimes' / 'libero-8f1084e', cache, offline=env.get('UV_OFFLINE') == '1')
            import yaml
            package = root / 'libero' / 'libero'
            config = cache / 'runtimes' / 'libero-config'
            config.mkdir(parents=True, exist_ok=True)
            (config / 'config.yaml').write_text(yaml.safe_dump({
                'benchmark_root': str(package), 'bddl_files': str(package / 'bddl_files'),
                'init_states': str(package / 'init_files'), 'assets': str(package / 'assets'),
                'datasets': str(cache / 'data' / 'libero')}), encoding='utf-8')
            env['LIBERO_ROOT'] = str(root)
            env['LIBERO_CONFIG_PATH'] = str(config)
    return requirements


def finish(profile, cache, python, env, execute, *, pip, accepted=(), assets=False):
    if 'carla' in profile.get('installers', []):
        root = Path(env.get('VLASTUDIO_RUNTIME_CARLA_ROOT', cache/'driving/runtimes/carla-0.9.15'))
        if not (env.get('VLASTUDIO_RUNTIME_CARLA_ROOT') and (root/'CarlaUE4.sh').is_file()):
            command = [python, '-m', 'vlastudio.benchmark.carla.install', '--prefix', str(root)]
            if env.get('UV_OFFLINE') == '1':
                command.append('--offline')
            execute(command)
        env['VLASTUDIO_RUNTIME_CARLA_ROOT'] = str(root)
    if 'behavior1k' in profile.get('installers', []):
        # Upstream resolves app icons and companion files relative to its repository.
        pip(['-e', str(Path(env['VLASTUDIO_RUNTIME_BEHAVIOR_ROOT']) / 'OmniGibson')])
    if 'robotwin' in profile.get('installers', []):
        from vlastudio.deploy.interpreter.robotwin import CUROBO, PYTORCH3D
        # The pinned cuRobo tag uses setuptools-scm; archives have no .git metadata.
        env['SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO'] = '0.7.7'
        pip([*profile['requirements'],
             f'pytorch3d @ https://github.com/facebookresearch/pytorch3d/archive/{PYTORCH3D}.tar.gz',
             f'nvidia-curobo @ https://github.com/NVlabs/curobo/archive/{CUROBO}.tar.gz'], '--no-build-isolation')
        env.pop('SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO', None)
        execute([python, '-m', 'vlastudio.deploy.interpreter.robotwin', 'configure'])
        if assets:
            execute([python, '-m', 'vlastudio.deploy.interpreter.robotwin', 'assets'])
    if 'libero' in profile.get('installers', []):
        # Upstream builds an empty wheel; register the pinned source with this interpreter.
        execute([python, '-c', 'import pathlib,sysconfig,sys; pathlib.Path(sysconfig.get_path("purelib"), "vlastudio_libero.pth").write_text(sys.argv[1]+"\\n", encoding="utf-8")', env['LIBERO_ROOT']])
    # Run overlays under the target interpreter, without importing the ML module.
    for overlay in profile.get('overlays', []):
        execute([python, '-m', 'vlastudio.deploy.interpreter.installers', 'overlay', json.dumps(overlay)])
    if assets and 'behavior1k' in profile.get('installers', []):
        if 'behavior-dataset' not in accepted:
            raise ValueError(f'Review {LICENSES["behavior-dataset"]}; use --assets --accept-license behavior-dataset to download.')
        execute([python, '-m', 'vlastudio.deploy.interpreter.behavior', '--accept-license'])


def apply_overlay(value):
    def locate(name):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError(f'Cannot locate package {name}')
        return Path(next(iter(spec.submodule_search_locations)))
    source_root = locate(value['source_module'])
    source = (source_root / value['source']).resolve()
    if not source.is_relative_to(source_root.resolve()) or not source.is_dir():
        raise ValueError('Invalid overlay source')
    destination = locate(value['target_module'])
    for path in source.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'overlay':
        apply_overlay(json.loads(sys.argv[2]))
