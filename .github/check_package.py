"""Check the installed distribution without a source checkout on sys.path."""
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory() as workdir:
    subprocess.run([sys.executable, '-I', '-c', '''
import importlib.metadata
import sys
import vlastudio as vla
from vlastudio.core.paths import package_root

assert vla.__version__ == importlib.metadata.version('vlastudio')
assert 'torch' not in sys.modules
root = package_root()
assert 'site-packages' in root.parts, root
for handle in (
    vla.load_policy('act'),
    vla.load_dataset('sim_transfer_cube_scripted'),
    vla.load_env('aloha_transfer'),
):
    assert handle.config_path.is_file(), handle
    assert root in handle.config_path.parents, handle
assert (root / 'configs' / 'interpreter' / 'base.yaml').is_file()
assert list((root / 'benchmark' / 'aloha' / 'assets').rglob('*.xml'))
assert list((root / 'benchmark' / 'aloha' / 'assets').rglob('*.stl'))
assert vla.connect_policy('127.0.0.1:8000').address == '127.0.0.1:8000'
# A plain wheel install must support the base workflow without env install.
import torch
import vlastudio.policy.act
import vlastudio.policy.diffusion_policy
import vlastudio.benchmark.aloha
import pyarrow.parquet
import av
layer = torch.nn.Linear(3, 2)
layer(torch.ones(1, 3)).sum().backward()
assert layer.weight.grad is not None
print('Default base dependencies, policy imports, CPU backward, aliases and assets: PASS')
'''], cwd=workdir, check=True)
    subprocess.run([sys.executable, '-I', '-m', 'vlastudio', '--help'],
                   cwd=workdir, check=True)
    subprocess.run([sys.executable, '-I', '-m', 'vlastudio', 'env', 'create',
                    '--policy', 'openpi', '-n', 'pi05', '--dry-run'], cwd=workdir, check=True)
    subprocess.run([sys.executable, '-I', '-m', 'vlastudio', 'env', 'run', 'base',
                    '--cache-dir', workdir, '--', 'python', '-c',
                    'import vlastudio,sys; assert "torch" not in sys.modules'], cwd=workdir, check=True)
