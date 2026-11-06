"""Optional integration tests requiring RoboCasa assets and image HDF5 files."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

DATA = os.environ.get('VLASTUDIO_ROBOCASA_SMOKE_DATA')
pytestmark = pytest.mark.skipif(not DATA, reason='Set VLASTUDIO_ROBOCASA_SMOKE_DATA for native kitchen smoke tests')


@pytest.mark.parametrize('task', ['RestockPantry', 'MicrowaveThawing'])
def test_real_kitchen_reset_render_and_native_step(task):
    from vlastudio.benchmark.base import MetaAction
    from vlastudio.benchmark.robocasa import RoboCasaEnv
    from vlastudio.data_utils.datasets import RoboCasaDataset
    path = Path(DATA) / (task + '.hdf5')
    dataset = RoboCasaDataset([str(path)], chunk_size=50)
    sample = dataset[0]
    assert sample['action'].shape == (50, 12)
    env = RoboCasaEnv(SimpleNamespace(task=task, dataset_path=str(path), max_timesteps=2, seed=10000))
    try:
        obs = env.reset()
        assert obs.state.shape == (8,)
        assert obs.image.shape == (3, 3, 128, 128)
        assert obs.image.dtype.name == 'uint8' and obs.raw_lang
        _, _, done, info = env.step(MetaAction(action=sample['action'][0].numpy()))
        assert isinstance(done, bool) and isinstance(info['success'], bool)
        if not done:
            _, _, done, info = env.step(MetaAction(action=sample['action'][0].numpy()))
            assert done and (info['success'] or info['timeout'])
    finally:
        env.close()
