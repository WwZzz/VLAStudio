"""Standard dataset/train contracts and offline/lazy asset behavior."""
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from vlastudio.utils import navigation_assets as assets


def test_download_is_lazy_cached_and_checksum_verified(tmp_path, monkeypatch):
    import hashlib
    import requests
    payload = b'official fixture bytes'
    digest = hashlib.sha256(payload).hexdigest()
    calls = []

    class Response:
        headers = {'Content-Type': 'application/octet-stream'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size): yield payload

    def get(*args, **kwargs):
        calls.append(args)
        return Response()

    monkeypatch.setattr(requests, 'get', get)
    target = tmp_path/'asset'
    with pytest.raises(FileNotFoundError):
        assets.fetch('https://example.invalid/fixture', target, digest, download=False)
    assert not calls
    assert assets.fetch('https://example.invalid/fixture', target, digest).read_bytes() == payload
    assets.fetch('https://example.invalid/fixture', target, digest, download=False)
    assert len(calls) == 1
    target.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        assets.fetch('https://example.invalid/fixture', target, digest)
    assert len(calls) == 1


def test_replay_cache_does_not_change_training_rng(tmp_path):
    from vlastudio.data_utils.datasets.navigation import _NavigationDataset

    class Fixture(_NavigationDataset):
        def _replay(self, episode):
            random.seed(987)
            np.random.seed(987)
            torch.manual_seed(987)
            yield {'action': torch.tensor([[0]]), 'state': torch.empty(0),
                   'raw_lang': 'Stop.', 'navigation': {}}

    dataset = Fixture()
    dataset.episodes = [{'episode_id': '1'}]
    dataset.episode_lengths = [1]
    dataset._sampling('episode', 1, None)
    dataset._cache({'fixture': 1}, tmp_path)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    first = dataset[0]
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    torch.testing.assert_close(dataset[0]['action'], first['action'])


def test_r2r_verified_mirror_fallback_and_offline_reuse(tmp_path, monkeypatch):
    import hashlib
    import requests
    records = [{'scan': 'fixture', 'path_id': 1, 'path': ['a', 'b'], 'instructions': ['Go forward.']}]
    original = json.dumps(records, indent=4).encode()
    payload = ('\n'.join(json.dumps(x) for x in records)+'\n').encode()
    monkeypatch.setattr(assets, 'R2R_ASSETS', {'train': ('fixture', hashlib.sha256(original).hexdigest())})
    monkeypatch.setattr(assets, 'R2R_MIRROR', {'train': (hashlib.sha256(payload).hexdigest(), assets.canonical_json_sha256(records))})
    calls = []

    class Response:
        headers = {'Content-Type': 'application/octet-stream'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size): yield payload

    def get(url, **kwargs):
        calls.append(url)
        if 'dropbox' in url:
            raise requests.ConnectionError('fixture network failure')
        return Response()

    monkeypatch.setattr(requests, 'get', get)
    path = assets.r2r_annotations('train', tmp_path)
    assert json.loads(path.read_text()) == records
    assert assets.r2r_annotations('train', tmp_path, download=False) == path
    assert len(calls) == 2
    path.write_text('[]')
    with pytest.raises(ValueError, match='checksum'):
        assets.r2r_annotations('train', tmp_path, download=False)


def test_standard_dataset_handles_resolve_without_loading_simulators(tmp_path):
    import vlastudio as vla
    for name in ['navigation.r2r', 'navigation.vlnce']:
        dataset = vla.load_dataset(name, cache_dir=tmp_path, check_dependencies=False)
        assert dataset.config_path.is_file()
        assert dataset.cache_dir == tmp_path.resolve()
    assert not list(tmp_path.iterdir())


def test_replay_datasets_declare_native_runtime_dependencies():
    from vlastudio.core.configuration import read_config
    from vlastudio.deploy.interpreter.recipes import dataset_environment
    for alias in ['navigation.r2r', 'navigation.vlnce']:
        config, path = read_config(alias, 'task')
        profile = dataset_environment(config, path, check_platform=False)
        assert profile['platforms'] == ['linux']
        assert any(value.startswith('torch>=') for value in profile['requirements'])


def test_cma_supervised_loss_backward_and_save_round_trip(tmp_path, monkeypatch):
    pytest.importorskip('gym')
    pytest.importorskip('yacs')
    from vlastudio.policy.vlnce_cma import CMAPolicy, load_model
    import requests

    def reject(*args, **kwargs):
        pytest.fail('Random policy construction must not download pretrained weights')

    monkeypatch.setattr(requests, 'get', reject)
    torch.set_num_threads(1)
    model = load_model(SimpleNamespace(is_training=True, pretrained_weight_path=None))['model'].train()
    row = {'instruction': torch.tensor([2, 3, 0, 0]),
           'rgb_features': torch.randn(4, 2048, 4, 4), 'depth_features': torch.randn(4, 128, 4, 4),
           'action': torch.tensor([[1], [1], [2], [0]]), 'progress': torch.tensor([0., .2, .5, 1.])}
    result = model(episodes=[row])
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert model.action_distribution.linear.weight.grad.abs().sum() > 0
    assert model.net.state_encoder.rnn.weight_ih_l0.grad.abs().sum() > 0
    assert not model.net.rgb_encoder.cnn.training
    model.save_pretrained(tmp_path)
    loaded = CMAPolicy.from_pretrained(tmp_path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_vlnbert_training_isolates_inplace_feature_dropout():
    from vlastudio.policy.navigation_training import vlnbert_loss

    class Navigator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2176, 1)
            self.vln_bert = SimpleNamespace(config=SimpleNamespace(directions=0))
        def forward(self, mode, sentence, cand_feats, **kwargs):
            cand_feats[..., :-128].mul_(.5)
            return sentence[:, 0], self.projection(cand_feats).squeeze(-1)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.navigator = Navigator()
        def encode_instructions(self, instructions):
            return torch.zeros(1, 4), torch.zeros(1, 3, 4), torch.ones(1, 3, dtype=torch.bool)

    model = Model()
    features = torch.randn(3, 2, 2176)
    unchanged = features.clone()
    row = {'candidate_features': features, 'candidate_mask': torch.ones(3, 2, dtype=torch.bool),
           'angles': torch.zeros(3, 128), 'labels': torch.tensor([0, 1, 0]), 'raw_lang': 'Turn left.'}
    vlnbert_loss(model, [row])['loss'].backward()
    assert model.navigator.projection.weight.grad.abs().sum() > 0
    torch.testing.assert_close(features, unchanged)
