"""Driving contracts without downloading assets or starting a simulator."""
import gzip
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from vlastudio.benchmark.base import MetaAction
from vlastudio.benchmark.carla import Bench2DriveEnv, route_metrics
from vlastudio.utils.driving import COMMANDS, action_table, driving_observation, local_points


def test_official_success_excludes_only_min_speed():
    record = {'route_id': 'r', 'status': 'Completed', 'scores': {'score_composed': 100, 'score_route': 100},
              'infractions': {'min_speed_infractions': ['slow'], 'collisions_vehicle': []}}
    assert route_metrics(record)['success']
    record['infractions']['collisions_vehicle'] = ['collision']
    assert not route_metrics(record)['success']
    record['infractions']['collisions_vehicle'] = []
    record['status'] = 'Failed - Agent crashed'
    assert not route_metrics(record)['success']


def test_vehicle_control_tags_order_ranges_and_reverse():
    env = object.__new__(Bench2DriveEnv)
    action = MetaAction(ctrl_space='vehicle', ctrl_type='control', action=np.array([-.2, .5, 0, 1]))
    assert env.meta2act(action) == [-.2, .5, 0, 1]
    for invalid in [[0, 0, 0], [0, 0, 0, .5], [2, 0, 0, 0], [0, -.1, 0, 0], [0, 0, 1.1, 0], [0, np.nan, 0, 0]]:
        with pytest.raises(ValueError):
            env.meta2act(invalid)
    with pytest.raises(ValueError):
        env.meta2act({'ctrl_space': 'joint', 'action': [0, 0, 0, 0]})


def test_declared_action_order_is_enforced_before_startup():
    with pytest.raises(ValueError, match='action_order'):
        Bench2DriveEnv(SimpleNamespace(action_order=['throttle', 'steer', 'brake', 'reverse']))


def test_driving_coordinates_and_commands():
    np.testing.assert_allclose(local_points([[3, 2]], [1, 2], np.pi/2), [[2, 0]], atol=1e-6)
    np.testing.assert_allclose(local_points([[1, 0]], [0, 0], 0), [[0, 1]], atol=1e-6)
    observation = driving_observation(3, [2, 1], -1)
    np.testing.assert_equal(observation['state'], [3])
    assert observation['raw_lang'].startswith('follow the lane\n')
    assert observation['raw_lang'].endswith('[2.0, 1.0]')
    assert 'navigation' not in observation
    assert len(action_table()) == 39 and action_table()[-1][3]


def test_parallel_camera_conversion_preserves_exact_policy_pixels():
    pytest.importorskip('cv2')
    from vlastudio.utils.driving import compose_images
    generator = np.random.default_rng(7)
    images = [generator.integers(0, 256, (50, 1600, 3), dtype=np.uint8) for _ in range(3)]
    sequential = compose_images(images, jpeg=True, jpeg_workers=1)
    parallel = compose_images(images, jpeg=True, jpeg_workers=3)
    np.testing.assert_array_equal(sequential, parallel)


def test_text_adapter_reconstructs_all_six_original_tcp_inputs():
    from vlastudio.policy.tcp_b2d.inputs import network_state
    observations = [driving_observation(i+.5, [i, -i], i+1) for i in range(6)]
    state = torch.tensor(np.stack([row['state'] for row in observations]))
    packed = network_state(state, [row['raw_lang'] for row in observations])
    expected = torch.cat([state, torch.tensor([[i, -i] for i in range(6)]), torch.eye(6)], dim=1)
    torch.testing.assert_close(packed, expected)
    assert [row['raw_lang'].split('\n')[0] for row in observations] == COMMANDS
    # Route geometry stays per observation; it must not broadcast across a batch.
    torch.testing.assert_close(packed[:, 1:3], expected[:, 1:3])


@pytest.mark.parametrize('invalid', ['text', 'frame', 'unit', 'axes', 'missing_target', 'nonfinite'])
def test_tcp_rejects_ambiguous_or_missing_task_conditions(invalid):
    from vlastudio.policy.tcp_b2d.inputs import network_state
    sample = driving_observation(2, [10, -3], 1)
    if invalid == 'text': sample['raw_lang'] = 'Drive somewhere nice.'
    elif invalid == 'frame': sample['raw_lang'] = sample['raw_lang'].replace('ego frame', 'world frame')
    elif invalid == 'unit': sample['raw_lang'] = sample['raw_lang'].replace('meters', 'centimeters')
    elif invalid == 'axes': sample['raw_lang'] = sample['raw_lang'].replace('x=forward, y=right', 'x=right, y=forward')
    elif invalid == 'missing_target': sample['raw_lang'] = 'turn left'
    else: sample['raw_lang'] = sample['raw_lang'].replace('[10.0, -3.0]', '[NaN, -3.0]')
    with pytest.raises(ValueError):
        network_state(torch.tensor(sample['state'])[None], sample['raw_lang'])


def test_training_and_environment_share_text_and_geometry_contract():
    from vlastudio.policy.tcp_b2d import get_data_collator, get_data_processor
    from vlastudio.benchmark.base import MetaObs, MetaPolicy
    from dataclasses import asdict
    live = driving_observation(2, [10, -3], 1)
    live['image'] = np.zeros((1, 3, 2, 2), dtype=np.uint8)
    meta = MetaObs(**live)
    meta.to_batch()
    sample = MetaPolicy(None).normed_mobs_to_samples(meta)[0]
    sample['reasoning'] = {'supervision': dict(waypoints=np.zeros((4, 2)), action_index=0, future_action_index=np.zeros(4),
                  expert_features=np.zeros(1536), expert_value=0., future_features=np.zeros((4, 1536)))}
    batch = get_data_collator(SimpleNamespace(is_training=True), {})([sample])
    assert batch['raw_lang'] == [live['raw_lang']]
    policy = MetaPolicy(SimpleNamespace(data_processor=get_data_processor(None, {}), data_collator=None))
    inference = policy.meta2obs([sample])
    torch.testing.assert_close(inference['tcp_state'], batch['tcp_state'])
    assert tuple(batch['tcp_state'].shape) == (1, 9)
    assert tuple(batch['state'].shape) == (1, 1)
    assert 'action_index' not in asdict(meta)


def test_terminal_observation_and_negative_warmup_have_no_training_actions(tmp_path):
    from PIL import Image
    from vlastudio.data_utils.datasets.bench2drive import Bench2DriveDataset
    route = tmp_path/'Scenario_Town01_Route1_Weather0'
    (route/'anno').mkdir(parents=True)
    (route/'expert_assessment').mkdir()
    for index in range(32):
        row = dict(x=index, y=0, theta=np.pi/2, x_target=30, y_target=0, speed=3, next_command=4)
        (route/'anno'/f'{index:05d}.json.gz').write_bytes(gzip.compress(json.dumps(row).encode()))
    # The terminal frame (31) has no expert; -1 is a sensor warmup action.
    for index in range(-1, 31):
        features = np.full(1538, index, dtype=np.float32)
        features[-1] = 0
        np.savez(route/'expert_assessment'/f'{index:05d}.npz', features)
    for camera, value in [('rgb_front_left', 30), ('rgb_front', 60), ('rgb_front_right', 90)]:
        folder = route/'camera'/camera
        folder.mkdir(parents=True)
        Image.fromarray(np.full((30, 1600, 3), value, dtype=np.uint8)).save(folder/'00000.jpg')
    dataset = Bench2DriveDataset(dataset_dir=tmp_path, download=False)
    assert len(dataset) == 6
    sample = dataset[0]
    assert sample['reasoning']['frame_id'] == '00000'
    np.testing.assert_equal(sample['state'], [3])
    assert sample['raw_lang'].startswith('follow the lane\n')
    from vlastudio.policy.tcp_b2d.inputs import encode_task
    np.testing.assert_allclose(encode_task(sample['state'], sample['raw_lang'])[1:3], [30, 0], atol=1e-5)
    np.testing.assert_allclose(sample['reasoning']['supervision']['waypoints'], [[5, 0], [10, 0], [15, 0], [20, 0]], atol=1e-5)
    np.testing.assert_equal(sample['reasoning']['supervision']['future_features'][:, 0], [5, 10, 15, 20])
    np.testing.assert_equal(sample['action'], [[0, 0, 1, 0]])
    assert sample['image'].shape == (1, 3, 256, 900)
    assert sample['image'][0, 0, 100, 0] == 30 and sample['image'][0, 0, 100, -1] == 90
    assert set(sample) <= {'image', 'state', 'raw_lang', 'action', 'is_pad', 'reasoning', 'episode_id', 'timestamp', '__index__'}
    from vlastudio.policy.tcp_b2d import get_data_collator
    collate = get_data_collator(SimpleNamespace(is_training=True), {})
    legacy = {**sample, **sample['reasoning']['supervision']}
    legacy.pop('reasoning')
    old, current = collate([legacy]), collate([sample])
    for key in old:
        if isinstance(old[key], torch.Tensor):
            torch.testing.assert_close(current[key], old[key], rtol=0, atol=0)
        else:
            assert current[key] == old[key]


def test_batchnorm_freeze_keeps_training_gradients_and_checkpoint_contract(tmp_path, monkeypatch):
    from torch import nn
    from vlastudio.policy.tcp_b2d import TCPConfig, TCPPolicy, upstream
    def network(download):
        return nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 2, 1))
    monkeypatch.setattr(upstream, 'network', network)
    model = TCPPolicy(TCPConfig(download=False)).train()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model.navigator(torch.randn(2, 3, 4, 4)).square().mean().backward()
    optimizer.step()
    assert torch.equal(original['navigator.0.running_mean'], model.state_dict()['navigator.0.running_mean'])
    assert not torch.equal(original['navigator.1.weight'], model.state_dict()['navigator.1.weight'])
    model.save_pretrained(tmp_path)
    loaded = TCPPolicy.from_pretrained(tmp_path)
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
    config = json.loads((tmp_path/'config.json').read_text())
    assert (config['action_dim'], config['state_dim'], config['network_state_dim']) == (4, 1, 9)
    assert json.loads((tmp_path/'normalize.json').read_text())['action']['bench2drive'] == 'identity'


def test_official_fused_controls_and_braking_preserve_input_state(monkeypatch):
    from torch import nn
    from vlastudio.policy.tcp_b2d import TCPConfig, TCPPolicy, upstream
    class Navigator(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(**{name: 1 for name in ['turn_KP', 'turn_KI', 'turn_KD', 'turn_n',
                'speed_KP', 'speed_KI', 'speed_KD', 'speed_n']})
        def control_pid(self, waypoints, speed, target):
            target.zero_()  # The upstream PID mutates its target argument.
            return .4, .6, False, {}
    monkeypatch.setattr(upstream, 'network', lambda download: Navigator())
    module = SimpleNamespace(PIDController=lambda *args: object(), Discrete_Actions_DICT=action_table())
    monkeypatch.setattr(upstream, 'modules', lambda download: (module, None))
    model = TCPPolicy(TCPConfig(download=False)).eval()
    chosen = [1]
    def predict(image, state):
        logits = torch.zeros(1, 39)
        logits[:, chosen[0]] = 1
        return {'pred_wp': torch.zeros(1, 4, 2), 'action_index': logits}
    monkeypatch.setattr(model, '_predict', predict)
    state = torch.tensor([[.2]])
    original = state.clone()
    observation = {**driving_observation(.2, [3, 4], 4), 'state': state,
                   'image': torch.zeros(1), 'timestamp': torch.tensor([0])}
    np.testing.assert_allclose(model.select_action(observation).numpy(), [[[-.05, .5, 0, 0]]], atol=1e-6)
    assert torch.equal(state, original)
    chosen[0] = 0
    np.testing.assert_allclose(model.select_action(observation).numpy(), [[[.2, 0, 1, 0]]], atol=1e-6)
    chosen[0] = 38
    model.config.planner_type = 'only_ctrl'
    np.testing.assert_allclose(model.select_action(observation).numpy(), [[[.5, .5, 0, 1]]], atol=1e-6)


@pytest.mark.parametrize('name,link', [('../escape', None), ('link', '../../escape')])
def test_asset_extraction_rejects_escaping_members(tmp_path, name, link):
    from vlastudio.utils.driving_assets import extract
    archive = tmp_path/'fixture.tar'
    with tarfile.open(archive, 'w') as output:
        member = tarfile.TarInfo(name)
        if link:
            member.type, member.linkname = tarfile.SYMTYPE, link
            output.addfile(member)
        else:
            member.size = 1
            output.addfile(member, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='Unsafe'):
        extract(archive, tmp_path/'asset')
    assert not (tmp_path/'asset').exists()


def test_carla_recipe_declares_native_installation():
    from vlastudio.core.configuration import read_config
    from vlastudio.deploy.interpreter.recipes import environment_for
    from vlastudio.deploy.interpreter.installers import preflight
    config, path = read_config('carla.bench2drive', 'env')
    profile = environment_for(config, path, check_platform=False)
    assert profile['installers'] == ['carla'] and 'carla' in profile['checks']
    from packaging.specifiers import SpecifierSet
    assert '3.10.16' in SpecifierSet(profile['requires_python'])
    assert '3.11.0' not in SpecifierSet(profile['requires_python'])
    preflight(profile)


def test_native_installer_uses_target_interpreter_and_shared_cache(tmp_path):
    from vlastudio.deploy.interpreter.installers import finish
    commands, env = [], {'UV_OFFLINE': '1'}
    finish({'installers': ['carla']}, tmp_path, '/target/bin/python', env, commands.append, pip=lambda *args: None)
    root = str(tmp_path/'driving/runtimes/carla-0.9.15')
    assert commands == [['/target/bin/python', '-m', 'vlastudio.benchmark.carla.install', '--prefix', root, '--offline']]
    assert env['VLASTUDIO_RUNTIME_CARLA_ROOT'] == root


def test_unwrapped_runtime_archive_remains_traversable(tmp_path):
    from vlastudio.utils.driving_assets import extract
    archive = tmp_path/'fixture.tar'
    with tarfile.open(archive, 'w') as output:
        member = tarfile.TarInfo('CarlaUE4.sh')
        member.mode, member.size = 0o755, 1
        output.addfile(member, io.BytesIO(b'x'))
    target = extract(archive, tmp_path/'runtime')
    if __import__('os').name != 'nt':
        assert target.stat().st_mode & 0o777 == 0o755


def test_interrupted_hub_transfers_resume_without_changing_revision(tmp_path, monkeypatch):
    import huggingface_hub
    import requests
    import time
    from vlastudio.utils import driving_assets
    monkeypatch.setattr(driving_assets, 'cache_root', lambda: tmp_path)
    calls, delays = [], []
    def download(repository, filename, **options):
        calls.append((repository, filename, options))
        if len(calls) < 3:
            raise requests.exceptions.ChunkedEncodingError('interrupted fixture transfer')
        return str(tmp_path/'cached-shard')
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    monkeypatch.setattr(time, 'sleep', delays.append)
    assert driving_assets.hf_asset('fixture', 'route.tar.gz', 'pinned-revision', dataset=True) == tmp_path/'cached-shard'
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2] and delays == [1, 2]
    calls.clear()
    delays.clear()
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        driving_assets.hf_asset('fixture', 'route.tar.gz', 'pinned-revision', dataset=True, download=False)
    assert len(calls) == 1 and not delays


def test_parked_catalog_is_fresh_for_every_route():
    from vlastudio.benchmark.carla.parking import ParkingCatalog
    module = SimpleNamespace(Town15=[{'location': [1, 2, 3]}, {'location': [4, 5, 6]}])
    catalog = ParkingCatalog(module)
    module.Town15.pop()
    module.Town15[0]['location'][0] = 99
    catalog.reset('Town15')
    assert module.Town15 == [{'location': [1, 2, 3]}, {'location': [4, 5, 6]}]
    module.Town15.clear()
    catalog.reset('Town15')
    assert len(module.Town15) == 2


def test_only_duplicate_stock_parking_is_disabled():
    from vlastudio.benchmark.carla.parking import configure_parking
    class World:
        def __init__(self): self.disabled, self.ticks, self.queries = None, 0, 0
        def get_environment_objects(self, label):
            self.queries += 1
            return [SimpleNamespace(id=label, name='BP_ParkCars70_SM_0'),
                    SimpleNamespace(id=99, name='Unrelated_static_car')]
        def enable_environment_objects(self, identifiers, enabled):
            assert enabled is False
            self.disabled = identifiers
        def tick(self): self.ticks += 1
    world = World()
    report = configure_parking(world, labels=[1, 2, 1])
    assert world.disabled == {1, 2} and world.ticks == 1
    assert report['disabled_stock_parked_objects'] == 2
    original = World()
    assert configure_parking(original, False, labels=[1])['disabled_stock_parked_objects'] == 0
    assert original.queries == 0 and original.ticks == 0


def test_scenario_builder_rejects_swallowed_initialization_failures():
    from vlastudio.benchmark.carla.parking import audited_builder
    requested = SimpleNamespace(name='ParkingExit_1')
    scene = SimpleNamespace(scenario_configurations=[requested], missing_scenario_configurations=[requested],
                            list_scenarios=[], _parked_ids=[])
    def failed(value, ego):
        value.missing_scenario_configurations.clear()  # Upstream swallows the exception and removes it.
    trace = {}
    with pytest.raises(RuntimeError, match='ParkingExit_1'):
        audited_builder(failed, trace)(scene, None)
    assert trace['errors'] == ['ParkingExit_1']
    def successful(value, ego):
        value.list_scenarios = [SimpleNamespace(config=requested, other_actors=[1, 2, 3])]
        value.missing_scenario_configurations.clear()
        return 'original result'
    scene.missing_scenario_configurations = [requested]
    trace = {}
    assert audited_builder(successful, trace)(scene, None) == 'original result'
    assert trace['initialized'] == {'ParkingExit_1'} and trace['actors']['ParkingExit_1'] == 3


def test_continuous_bicycle_flow_keeps_master_plan_for_each_actor():
    from vlastudio.benchmark.carla.bicycle import nonmutating_spawn
    flow = SimpleNamespace(_plan=[1, 2, 3, 4])
    master = flow._plan
    def spawn(value):
        value._plan.pop(0)
        return value._plan
    first = nonmutating_spawn(spawn)(flow)
    first.pop(0)
    second = nonmutating_spawn(spawn)(flow)
    assert flow._plan is master and master == [1, 2, 3, 4] and second == [2, 3, 4]
    def failed(value):
        value._plan.clear()
        raise RuntimeError('actor setup failed')
    with pytest.raises(RuntimeError):
        nonmutating_spawn(failed)(flow)
    assert flow._plan is master and master == [1, 2, 3, 4]


def test_bicycle_repair_does_not_mask_unrelated_or_unconfigured_errors():
    from vlastudio.benchmark.carla.bicycle import repair_initializer
    def unrelated(scenario, config): raise ValueError('unrelated setup error')
    with pytest.raises(ValueError, match='unrelated setup error'):
        repair_initializer(unrelated, {})(None, None)
    def disconnected(scenario, config): raise ValueError("Couldn't find a proper plan for the bicycle flow")
    with pytest.raises(ValueError, match='explicit end_actor_flow'):
        repair_initializer(disconnected, {})(None, SimpleNamespace(other_parameters={}))
    original = lambda scenario, config: 'unchanged result'
    reports = {}
    assert repair_initializer(original, reports)(None, None) == 'unchanged result'
    assert reports == {}
