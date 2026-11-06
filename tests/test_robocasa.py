"""RoboCasa integration contracts, without kitchen assets or native simulator."""
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlastudio.benchmark.base import MetaAction, MetaObs
from vlastudio.benchmark.robocasa import RoboCasaEnv, evaluate
from vlastudio.benchmark.robocasa.observations import CAMERAS, state_vector
from vlastudio.data_utils.datasets import RoboCasaDataset


def hdf5(path):
    with h5py.File(path, 'w') as f:
        data = f.create_group('data')
        data.attrs['env_args'] = json.dumps({'env_name': 'RestockPantry', 'env_kwargs': {'control_freq': 20}})
        for i, length in enumerate([3, 2]):
            d = data.create_group(f'demo_{i}')
            d.attrs['ep_meta'] = json.dumps({'lang': f'Pick item {i}'})
            d['actions'] = np.full((length, 12), i + 1, np.float32)
            o = d.create_group('obs')
            o['robot0_eef_pos'] = np.full((length, 3), i, np.float32)
            o['robot0_eef_quat'] = np.tile([0., 0., 0., 1.], (length, 1))
            o['robot0_gripper_qpos'] = np.zeros((length, 2), np.float32)
            for j, camera in enumerate(CAMERAS):
                images = np.full((length, 4, 5, 3), j, np.uint8)
                images[:, 0] = 99
                o[camera + '_image'] = images
        masks = f.create_group('mask')
        masks['train'] = np.array([b'demo_0'])
        masks['valid'] = np.array([b'demo_1'])
    return path


@pytest.mark.parametrize('preload', [False, True])
def test_native_hdf5_chunks_masks_language_and_serialization(tmp_path, preload):
    path = hdf5(tmp_path / 'arbitrary.h5')
    ds = RoboCasaDataset([str(path)], chunk_size=4, filter_by_attribute='train', preload_data=preload)
    assert len(ds) == 3 and ds.num_episodes == 1 and ds.get_freq() == 20
    sample = ds[2]
    assert sample['action'].shape == (4, 12)
    assert sample['is_pad'].tolist() == [False, True, True, True]
    assert torch.all(sample['action'][0] == 1) and torch.all(sample['action'][1:] == 0)
    assert sample['image'].shape == (3, 3, 4, 5)
    assert sample['image'][0, 0, 0, 0] == 99
    assert sample['state'].shape == (8,) and sample['raw_lang'] == 'Pick item 0'
    assert pickle.loads(pickle.dumps(ds))[0]['raw_lang'] == 'Pick item 0'
    assert ds.extract_from_episode(0, ['action', 'state'])['state'].shape == (3, 8)
    with pytest.raises(IndexError):
        ds[len(ds)]
    with pytest.raises(IndexError):
        ds[-1]


def test_multiple_files_and_demo_languages(tmp_path):
    hdf5(tmp_path / 'a.h5'); hdf5(tmp_path / 'b.hdf5')
    ds = RoboCasaDataset([str(tmp_path)], chunk_size=2)
    assert len(ds) == 10 and ds.num_episodes == 4
    assert ds[3]['raw_lang'] == 'Pick item 1'
    assert ds[5]['raw_lang'] == 'Pick item 0'
    assert ds[4]['is_pad'].tolist() == [False, True]
    assert torch.all(ds[4]['action'][0] == 2)


def test_invalid_masks_and_missing_data_fail_early(tmp_path):
    path = hdf5(tmp_path / 'x.h5')
    with pytest.raises(KeyError):
        RoboCasaDataset([str(path)], filter_by_attribute='absent')
    with pytest.raises(FileNotFoundError):
        RoboCasaDataset([str(tmp_path / 'missing')])
    with h5py.File(path, 'a') as f:
        del f['data/demo_0'].attrs['ep_meta']
    with pytest.raises(KeyError):
        RoboCasaDataset([str(path)])


class Simulator:
    action_spec = (np.full(12, -1.), np.ones(12))
    def __init__(self):
        self.calls = 0
        self.succeeded = False
    def reset(self):
        return {'robot0_eef_pos': np.zeros(3), 'robot0_eef_quat': np.array([0, 0, 0, 1.]),
                'robot0_gripper_qpos': np.zeros(2), **{c + '_image': np.arange(60, dtype=np.uint8).reshape(4, 5, 3) for c in CAMERAS}}
    def get_ep_meta(self):
        return {'lang': 'Pick item 0'}
    def step(self, action):
        self.calls += 1
        self.action = action
        return self.reset(), 0., False, {}
    def _check_success(self):
        return self.succeeded
    def close(self):
        pass


def wrapped(monkeypatch, **options):
    sim = Simulator()
    monkeypatch.setitem(sys.modules, 'robocasa', SimpleNamespace())
    def make(**kwargs):
        sim.kwargs = kwargs
        return sim
    monkeypatch.setitem(sys.modules, 'robosuite', SimpleNamespace(make=make))
    cfg = SimpleNamespace(task='RestockPantry', env_kwargs={'robots': 'PandaOmron'}, max_timesteps=2, **options)
    return RoboCasaEnv(cfg), sim


def test_env_native_actions_timeout_and_sticky_done(monkeypatch):
    env, sim = wrapped(monkeypatch, rollout_index=3)
    obs = env.reset()
    assert sim.seed == 10003
    assert obs.state.shape == (8,) and obs.image.shape == (3, 3, 4, 5)
    assert obs.image[0, 0, 0, 0] == sim.reset()[CAMERAS[0] + '_image'][-1, 0, 0]
    action = MetaAction(action=np.full(12, 2.))
    assert not env.step(action)[2]
    _, _, done, info = env.step(action)
    assert done and info['timeout'] and not info['success']
    assert np.all(sim.action == 1.)
    env.step(action)
    assert sim.calls == 2
    with pytest.raises(ValueError):
        env.meta2act(MetaAction(action=np.zeros(7)))
    with pytest.raises(ValueError):
        env.meta2act(MetaAction(ctrl_type='abs', action=np.zeros(12)))
    env.reset()
    assert sim.seed == 10004
    sim.succeeded = True
    assert env.step(MetaAction(action=np.zeros(12)))[3]['success']


def test_dataset_and_online_state_agree(tmp_path, monkeypatch):
    ds = RoboCasaDataset([str(hdf5(tmp_path / 'x.h5'))])
    env, _ = wrapped(monkeypatch)
    np.testing.assert_array_equal(ds[0]['state'].numpy(), env.reset().state)
    obs = {'robot0_eef_pos': np.zeros(3), 'robot0_eef_quat': [0., 0., 1., 0.], 'robot0_gripper_qpos': np.zeros(2)}
    assert abs(state_vector(obs)[5]) == pytest.approx(np.pi)


def test_metadata_preserves_controller_and_rejects_wrong_task(tmp_path, monkeypatch):
    path = hdf5(tmp_path / 'native.h5')
    controller = {'type': 'BASIC', 'body_parts': {'right': {'type': 'OSC_POSE'}}}
    with h5py.File(path, 'a') as f:
        f['data'].attrs['env_args'] = json.dumps({'env_name': 'RestockPantry', 'env_kwargs': {
            'robots': 'PandaOmron', 'controller_configs': controller}})
    env, sim = wrapped(monkeypatch, dataset_path=str(path))
    assert sim.kwargs['controller_configs'] == controller
    with pytest.raises(ValueError, match='differs'):
        RoboCasaEnv(SimpleNamespace(task='MicrowaveThawing', dataset_path=str(path)))


def test_packaged_training_config_instantiates_real_dataset(tmp_path):
    from vlastudio.configuration import read_config
    from vlastudio.data_utils.utils import _create_dataset_from_config
    config, _ = read_config('robocasa.restock_pantry', 'task')
    config['datasets'][0]['args']['dataset_path_list'] = [str(hdf5(tmp_path / 'sample.h5'))]
    ds = _create_dataset_from_config(config['datasets'][0], SimpleNamespace())
    assert ds[0]['action'].shape == (50, config['meta']['action_dim'])
    assert ds[0]['state'].shape == (config['meta']['state_dim'],)


def test_config_aliases_and_runtime():
    import vlastudio as vla
    from vlastudio.configuration import read_config
    from vlastudio.extensions import resolve
    from vlastudio.profiles import environment_for
    for name in ['robocasa.restock_pantry', 'robocasa.microwave_thawing']:
        config, path = read_config(name, 'env')
        assert callable(resolve(config['type'], kind='env'))
        assert environment_for(config, path, check_platform=False)['platforms'] == ['linux']
        assert vla.load_env(name, check_dependencies=False).config_path.is_file()
        task = vla.load_dataset(name, check_dependencies=False)
        assert task.config_path.is_file()


def test_rollout_uses_success_not_timeout_and_freezes_finished_slots():
    class Vector:
        closed = False
        calls = 0
        def __len__(self): return 2
        def reset(self): return [MetaObs(state=np.zeros(8)), MetaObs(state=np.zeros(8))]
        def step(self, actions):
            self.calls += 1
            return self.reset(), [0., 0.], [True, self.calls == 2], [{'success': self.calls > 1}, {'success': self.calls == 2}]
        def close(self): self.closed = True
    env = Vector()
    result = evaluate(SimpleNamespace(max_timesteps=3), env, SimpleNamespace(select_action=lambda: None))
    assert result['success'] == [False, True] and result['success_rate'] == .5
    assert result['horizon'] == [1, 2] and env.closed


def test_cpu_training_backward_and_reload(tmp_path):
    ds = RoboCasaDataset([str(hdf5(tmp_path / 'x.h5'))], chunk_size=2)
    model = torch.nn.Linear(8, 24)
    sample = ds[0]
    loss = (model(sample['state']).reshape(2, 12) - sample['action']).square().mean()
    loss.backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
    torch.save(model.state_dict(), tmp_path / 'model.pt')
    other = torch.nn.Linear(8, 24)
    other.load_state_dict(torch.load(tmp_path / 'model.pt', weights_only=True))
    torch.testing.assert_close(model(sample['state']), other(sample['state']))
