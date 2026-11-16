"""Exercise relocated recipe loading and subprocess imports from outside a checkout."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from vlastudio.core.paths import package_root, package_source_root
from vlastudio.deploy.interpreter.recipes import environment_for, recipe


def test_relocated_recipes_resolve_inheritance_and_package_paths():
    assert package_root() == ROOT / 'src/vlastudio'
    assert package_source_root() == package_root()
    directory = package_root() / 'configs/interpreter'
    manifests = sorted(directory.glob('*.yaml'))
    assert len(manifests) == 23
    for manifest in manifests:
        assert recipe(manifest.stem) == manifest
        profile = environment_for({}, manifest, str(manifest), check_platform=False)
        assert profile['python']
        assert profile['requirements']
    # Explicit relative references in built-in simulation configs also moved.
    from vlastudio.core.configuration import read_config
    for name in ('aloha_transfer', 'aloha_sim.transfer_cube'):
        config, path = read_config(name, 'env')
        profile = environment_for(config, path, check_platform=False)
        assert 'aloha' in profile['checks']


def test_relocated_worker_runs_custom_entrypoint_from_another_directory(tmp_path):
    (tmp_path / 'handler.py').write_text(
        'from pathlib import Path\n'
        'def main(command, args):\n'
        '    Path("result.txt").write_text(command + ":" + ",".join(args))\n'
        '    return 0\n', encoding='utf-8')
    manifest = tmp_path / 'runtime.yaml'
    manifest.write_text('requirements: []\nentrypoint: handler.py:main\n', encoding='utf-8')
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    result = subprocess.run(
        [sys.executable, '-m', 'vlastudio', 'train', '--runtime-manifest', str(manifest),
         '--cache-dir', str(tmp_path / 'cache'), '--example', 'value'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'result.txt').read_text() == 'train:--example,value'


def test_relocated_probe_module_runs_in_registered_interpreter(tmp_path):
    from vlastudio.deploy.interpreter.manager import Manager
    manager = Manager(tmp_path / 'cache')
    profile = {'python': f'{sys.version_info.major}.{sys.version_info.minor}',
               'requirements': [], 'checks': []}
    manager.save('probe', {'python': sys.executable, 'profile': profile, 'checks': {},
                          'variables': {'PYTHONPATH': str(ROOT / 'src')}})
    assert manager.check('probe') == 0
    assert manager.get('probe')['status'] == 'imports-passed'
