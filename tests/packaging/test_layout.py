import json
from pathlib import Path
import subprocess
import sys

from vlastudio.paths import package_root
from vlastudio.profiles import environment_for


def test_real_package_layout():
    root = package_root()
    for name in ('policy', 'benchmark', 'data_utils', 'deploy', 'configs', 'utils'):
        assert (root / name).is_dir()
    assert not (root / '_legacy').exists()


def test_legacy_alias_preserves_module_identity():
    code = '''
import importlib, sys
import vlastudio
assert 'policy' not in sys.modules
from vlastudio.compat import enable_legacy_imports
enable_legacy_imports()
old = importlib.import_module('policy')
new = importlib.import_module('vlastudio.policy')
assert old is new
assert new.__spec__.name == 'vlastudio.policy'
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_namespaced_policy_uses_same_runtime():
    path = package_root() / 'configs/policy/act.yaml'
    assert environment_for({'type': 'vlastudio.policy.act'}, path) == environment_for({'type': 'policy.act'}, path)
