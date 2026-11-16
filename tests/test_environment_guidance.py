"""User-facing installation contracts, tested without ML packages or downloads."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import vlastudio as vla
from vlastudio.deploy.interpreter import dependencies
from vlastudio.core.configuration import read_config
from vlastudio.deploy.interpreter.cli import main, select_profile
from vlastudio.deploy.interpreter.recipes import environment_for, recipe, combine_profiles, dataset_environment


def missing(name):
    raise importlib.metadata.PackageNotFoundError(name)


def test_import_uses_only_standard_library():
    script = f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import vlastudio; assert 'yaml' not in sys.modules; assert 'torch' not in sys.modules; assert 'packaging' not in sys.modules"
    subprocess.run([sys.executable, '-S', '-c', script], check=True)


@pytest.mark.parametrize('category, selector, flag', [
    ('policy', 'act', '--policy'), ('dataset', 'sim_transfer_cube_scripted', '--dataset'),
    ('env', 'aloha_transfer', '--env')])
def test_loaders_explain_missing_packages(monkeypatch, category, selector, flag):
    monkeypatch.setattr(dependencies.metadata, 'version', missing)
    with pytest.raises(vla.DependencyError) as caught:
        getattr(vla, 'load_' + category)(selector)
    text = str(caught.value)
    assert 'torch' in text and 'missing' in text
    argument = {'--policy': 'policy', '--dataset': 'dataset', '--env': 'environment'}[flag]
    assert f'{argument}=' in text and selector in text
    assert f'vlastudio env create {flag}' in text
    assert f'vlastudio env install {flag}' in text
    assert 'install_dependencies' not in text
    assert 'python run.py' in text
    assert '-m vlastudio' not in text


def test_handles_can_be_created_for_orchestration(monkeypatch):
    monkeypatch.setattr(dependencies, 'environment_plan', lambda **kw: pytest.fail('Unexpected dependency check'))
    with pytest.raises(ValueError, match='Install dependencies externally'):
        vla.load_policy('act', runtime='managed')
    assert vla.load_policy('act', check_dependencies=False).config_path.is_file()
    assert vla.load_dataset(check_dependencies=False).config_path.is_file()
    assert vla.load_env('aloha_transfer', check_dependencies=False).config_path.is_file()


def test_version_markers_overrides_and_cuda_local_version(monkeypatch):
    versions = {'torch': '2.4.0+cu121', 'numpy': '2.0.0', 'custom': '2.0'}
    monkeypatch.setattr(dependencies.metadata, 'version', lambda n: versions[n])
    issues, packages = dependencies.inspect_environment({
        'python': f'{sys.version_info.major}.{sys.version_info.minor}',
        'requirements': ['torch==2.4.0', 'numpy<2', 'absent; python_version < "3.0"', 'custom==1.0'],
        'overrides': ['custom==2.0']})
    assert len(issues) == 1 and 'numpy<2' in issues[0]
    assert len(packages) == 3
    assert packages[0]['status'] == 'installed'


def test_inheritance_resolves_relative_paths_and_replaces_pins(tmp_path):
    (tmp_path / 'parent.yaml').write_text('python: "3.10"\nrequirements: [torch==2.4.0, numpy<2]\n')
    (tmp_path / 'child.yaml').write_text('extends: parent.yaml\npython: "3.11"\nrequirements: [torch==2.7.1]\nentrypoint: custom.py:main\n')
    profile = environment_for({}, tmp_path / 'config.yaml', str(tmp_path / 'child.yaml'))
    assert profile['requirements'] == ['numpy<2', 'torch==2.7.1']
    assert profile['python'] == '3.11'
    assert profile['entrypoint'] == str(tmp_path / 'custom.py') + ':main'


def test_inheritance_cycle_and_lock_extension_are_rejected(tmp_path):
    path = tmp_path / 'a.yaml'
    path.write_text('extends: a.yaml\n')
    with pytest.raises(ValueError, match='cycle'):
        environment_for({}, path, str(path))
    (tmp_path / 'parent.yaml').write_text('lock: "torch==2.4.0"\n')
    path.write_text('extends: parent.yaml\n')
    with pytest.raises(ValueError, match='lockfiles cannot be extended'):
        environment_for({}, path, str(path))


def test_peer_profiles_do_not_silently_override_conflicts():
    result = combine_profiles({'python': '3.10', 'requirements': ['torch==2.4.0']},
                              {'python': '3.10', 'requirements': ['torch==2.7.1']})
    assert result['requirements'] == ['torch==2.4.0', 'torch==2.7.1']
    with pytest.raises(ValueError, match='different Python'):
        combine_profiles(result, {'python': '3.11', 'requirements': []})


def test_base_and_shared_data_have_no_simulators():
    for name in ('base', 'data', 'smolvla', 'fastwam'):
        profile = environment_for({}, recipe(name), str(recipe(name)), check_platform=False)
        assert not any(r.startswith(('libero ', 'robosuite', 'dm-control', 'mujoco', 'metaworld')) for r in profile['requirements'])


def test_policy_and_simulator_can_be_composed():
    _, profile = select_profile(policy='smolvla', environment='libero.object', dataset='libero.libero_object')
    assert 'torch==2.7.1' in profile['requirements']
    assert not any(r.startswith('torch==2.4') for r in profile['requirements'])
    assert any('LIBERO.git' in r for r in profile['requirements'])


def test_plan_and_check_do_not_create_cache(tmp_path, monkeypatch, capsys):
    cache = tmp_path / 'cache'
    monkeypatch.setenv('VLASTUDIO_CACHE', str(cache))
    monkeypatch.setattr(dependencies.metadata, 'version', missing)
    assert main(['plan', '--policy', 'act', '--json']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['commands']['create'] == 'vlastudio env create --policy act'
    assert plan['profile']['python'] == '3.10'
    assert plan['commands']['install_current'] == 'vlastudio env install --policy act'
    assert main(['check', '--policy', 'act']) == 2
    assert not cache.exists()


def test_platform_requirement_does_not_hide_installation_plan(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    plan = vla.environment_plan(policy='pi0')
    assert any('platform' in issue for issue in plan['issues'])
    assert plan['commands']['install_current'] == 'vlastudio env install --policy pi0'
    assert 'transformers==4.53.2' in plan['profile']['requirements']


def test_legacy_model_metadata_is_discoverable():
    _, profile = select_profile(policy='qwen25dp', check_platform=False)
    assert 'transformers==4.52.4' in profile['requirements']
    assert 'Legacy' in profile['notes'][0]


def test_custom_policy_runtime_has_actionable_guidance(tmp_path, monkeypatch):
    config = tmp_path / 'custom policy.yaml'
    config.write_text('type: mine.policy\nruntime:\n  python: "3.10"\n  requirements: [my-package>=1]\n')
    monkeypatch.setattr(dependencies.metadata, 'version', missing)
    with pytest.raises(vla.DependencyError) as caught:
        vla.load_policy(config)
    assert 'my-package>=1' in str(caught.value)
    assert 'custom policy.yaml' in str(caught.value)


@pytest.mark.parametrize('example, selector', [
    ('07_custom_sim_env_libero_plus', 'env_object'), ('08_fastwam_libero', 'libero_spatial')])
def test_existing_example_runtimes_resolve(example, selector):
    manifest = ROOT / 'examples' / example / 'runtime.yaml'
    profile = environment_for({}, manifest, str(manifest), check_platform=False)
    assert 'torch==2.7.1' in profile['requirements']
    assert 'mujoco==3.3.2' in profile['requirements']
    assert any(r.startswith('pyarrow') for r in profile['requirements'])
    _, combined = select_profile(environment=str(manifest.parent / 'config' / (selector + '.yaml')), check_platform=False)
    assert combined['requirements'] == profile['requirements']


def test_dispatch_uses_current_python_without_installing(monkeypatch):
    from vlastudio.entrypoints import cli
    calls = []
    monkeypatch.setattr(cli, 'execute', lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    assert cli.main(['train', '-p', 'act']) == 0
    args, kwargs = calls[0]
    assert args[2] == Path(sys.executable)
    assert kwargs['isolated'] is False


def test_legacy_dispatch_is_current_by_default(capsys):
    from vlastudio.entrypoints.cli import main as cli_main
    assert cli_main(['train', '-p', 'act', '--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['runtime'] == 'current'
    assert plan['profile'] is None


def test_shared_data_python_follows_model():
    config, path = read_config('sim_transfer_cube_scripted', 'task')
    assert dataset_environment(config, path, python='3.11')['python'] == '3.11'


def test_explicit_current_runtime_manifest_is_used(tmp_path, monkeypatch):
    manifest = tmp_path / 'runtime.yaml'
    manifest.write_text(f'python: "{sys.version_info.major}.{sys.version_info.minor}"\nrequirements: [custom-dep==1.0]\n')
    monkeypatch.setattr(dependencies.metadata, 'version', lambda n: '1.0' if n == 'custom-dep' else missing(n))
    policy = vla.load_policy('act', runtime='current', runtime_manifest=manifest)
    assert policy.runtime_options['runtime_manifest'] == str(manifest)


def test_data_runtime_can_select_its_own_python():
    profile = dataset_environment({'runtime': {'python': '3.11', 'requirements': ['custom-dep==1.0']}}, Path('task.yaml'))
    assert profile['python'] == '3.11'
    assert 'custom-dep==1.0' in profile['requirements']
    with pytest.raises(ValueError, match='different Python'):
        dataset_environment({'runtime': {'python': '3.11'}}, Path('task.yaml'), python='3.10')


def test_child_override_preserves_other_environment_markers():
    from vlastudio.deploy.interpreter.recipes import _requirements
    result = _requirements(['numpy==1.0; sys_platform == "linux"', 'numpy==1.1; sys_platform == "win32"'],
                           ['numpy==2.0; sys_platform == "win32"'])
    assert result == ['numpy==1.0; sys_platform == "linux"', 'numpy==2.0; sys_platform == "win32"']


def test_rlds_requires_tensorflow_but_hdf5_does_not():
    _, rlds = select_profile(dataset='rlds.droid_mini')
    _, hdf5 = select_profile(dataset='sim_transfer_cube_scripted')
    assert any(r.startswith('tensorflow==') for r in rlds['requirements'])
    assert not any(r.startswith(('tensorflow', 'dm-control', 'mujoco')) for r in hdf5['requirements'])


def test_dispatch_manifest_is_metadata_only(capsys):
    from vlastudio.entrypoints.cli import main as cli_main
    assert cli_main(['evaluate', '-m', 'localhost:8765', '-e', 'aloha_transfer',
                     '--runtime-manifest', str(recipe('aloha')), '--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['runtime'] == 'current'
    assert any(r.startswith('mujoco') for r in plan['profile']['requirements'])


def test_linux_glibc_requirement(monkeypatch):
    import platform
    from vlastudio.deploy.interpreter.recipes import platform_issues
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(platform, 'libc_ver', lambda: ('glibc', '2.28'))
    assert 'glibc >= 2.31' in platform_issues({'min_glibc': '2.31'})[0]


def test_library_has_no_package_installation_entrypoint():
    assert not hasattr(vla, 'install_dependencies')
    from vlastudio.deploy.interpreter import runtime
    assert not hasattr(runtime, 'prepare')
    assert not hasattr(runtime, 'install_application')
    assert not (ROOT / 'src/vlastudio/env_registry.py').exists()


def test_default_install_declares_act_dp_and_aloha_requirements():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    from packaging.requirements import Requirement
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    installed = {Requirement(r).name.lower(): Requirement(r) for r in project['dependencies']}
    for name in ('base', 'data', 'diffusion_policy', 'aloha'):
        profile = environment_for({}, recipe(name), str(recipe(name)), check_platform=False)
        for requirement in profile['requirements']:
            req = Requirement(requirement)
            assert req.name.lower() in installed, f'{name}: {requirement} absent from default dependencies'
            assert req.specifier == installed[req.name.lower()].specifier


@pytest.mark.parametrize('name', ['data', 'base', 'smolvla', 'diffusion_policy', 'aloha', 'remote',
                                  'libero', 'metaworld', 'openpi', 'openvla', 'fastwam', 'tabletop_sim', 'rlds'])
def test_external_requirements_match_dependency_metadata(name):
    profile = environment_for({}, recipe(name), str(recipe(name)), check_platform=False)
    lines = (ROOT / 'requirements' / (name + '.txt')).read_text().splitlines()
    assert [line for line in lines if line and not line.startswith('#')] == profile['requirements']


def test_robot_library_calls_forward_configuration_and_policy(tmp_path, monkeypatch):
    from vlastudio.core import api
    calls = []
    monkeypatch.setattr(api, '_run', lambda *args: calls.append(args))
    vla.collect('so101', teleop='so101_leader', output_dir=tmp_path / 'data',
                task='pick up the cube', visualize=True)
    command, args, options = calls.pop()
    assert command == 'collect' and options['runtime'] == 'current'
    assert args[args.index('--dataset-format') + 1] == 'hdf5'
    assert args[args.index('--task') + 1] == 'pick up the cube'
    assert args[args.index('-r') + 1].endswith('so101.yaml')
    assert args[args.index('-t') + 1].endswith('so101_leader.yaml')
    assert '--visualize' in args
    vla.infer('so101', vla.connect_policy('localhost:5000'), output_dir=tmp_path / 'eval')
    command, args, options = calls.pop()
    assert command == 'infer' and options['runtime'] == 'current'
    assert args[args.index('-m') + 1] == 'localhost:5000'
    assert args[args.index('-am') + 1].endswith('basic.yaml')
    policy = vla.load_policy('act', checkpoint=tmp_path / 'checkpoint', check_dependencies=False)
    vla.infer('so101', policy)
    assert calls[-1][1][calls[-1][1].index('-m') + 1] == str(tmp_path / 'checkpoint')
    with pytest.raises(ValueError, match='frequency'):
        vla.collect('so101', output_dir=tmp_path, frequency=0)
    with pytest.raises(ValueError, match='saved Policy'):
        vla.infer('so101', object())


def test_example05_selects_separate_model_and_simulator_recipes():
    _, model = select_profile(policy='pi05', dataset='tabletop_sim.dish_drainer', check_platform=False)
    _, client = select_profile(environment='tabletop_sim.dish_drainer', remote=True, check_platform=False)
    assert model['python'] == '3.11'
    assert client['python'] == '3.10'
    assert 'mujoco==3.2.0' in client['requirements']
    assert 'dm-control==1.0.21' in client['requirements']
    assert not any(req.startswith('mujoco') for req in model['requirements'])
