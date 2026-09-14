import json
from pathlib import Path
import pytest
import vlastudio as vla


def configs(tmp_path):
    result = []
    for name in ('policy', 'task', 'training', 'env'):
        p = tmp_path / (name + '.yaml')
        p.write_text('name: example\n')
        result.append(p)
    return result


def test_train_evaluate_in_real_workers(tmp_path):
    """Exercise dispatch, config paths with spaces, artifacts and custom entrypoints."""
    policy_cfg, task, training, env = configs(tmp_path)
    worker = tmp_path / 'custom worker.py'
    worker.write_text('''
def run(command, argv):
    import argparse, json
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument('-o', required=True)
    p.add_argument('-m')
    args, _ = p.parse_known_args(argv)
    output = Path(args.o)
    output.mkdir(parents=True, exist_ok=True)
    if command == 'train':
        (output / 'policy_metadata.json').write_text('{}')
    else:
        assert (Path(args.m) / 'policy_metadata.json').is_file()
        (output / 'metrics.json').write_text(json.dumps({'total': 2, 'success_rate': 0.5}))
    return 0
''')
    # Reuse this interpreter while exercising real subprocess dispatch.
    import sys
    import yaml
    from unittest.mock import patch
    manifest = tmp_path / 'runtime.yaml'
    manifest.write_text(yaml.safe_dump({'python': '3.11', 'requirements': [],
                                      'entrypoint': str(worker) + ':run'}))
    import tempfile
    with tempfile.TemporaryDirectory(prefix='vla-') as cache, patch('vlastudio.cli.prepare', return_value=Path(sys.executable)):
        policy = vla.load_policy(policy_cfg, runtime_manifest=manifest, cache_dir=cache,
                                 plugin_path=[Path(yaml.__file__).parents[1]])
        result = vla.train(policy, vla.load_dataset(task), training,
                           output_dir=tmp_path / 'my checkpoints')
        assert result.policy is policy
        assert policy.checkpoint == result.checkpoint
        bench = vla.load_env(env, runtime_manifest=manifest)
        evaluation = bench.evaluate(policy, output_dir=tmp_path / 'evaluation')
    assert evaluation.metrics['metrics.json']['success_rate'] == 0.5
    with pytest.raises(ValueError, match='empty'):
        bench.evaluate(policy, output_dir=evaluation.output_dir)


def test_failed_train_preserves_checkpoint(tmp_path, monkeypatch):
    p, t, c, e = configs(tmp_path)
    policy = vla.load_policy(p, checkpoint=tmp_path / 'old')
    monkeypatch.setattr(vla, 'run', lambda *a, **kw: 7)
    with pytest.raises(vla.TaskError, match='7'):
        vla.train(policy, vla.load_dataset(t), c, output_dir=tmp_path / 'new')
    assert policy.checkpoint == tmp_path / 'old'


def test_api_arguments_and_validation(tmp_path, monkeypatch):
    p, t, c, e = configs(tmp_path)
    policy = vla.load_policy(p)
    dataset = vla.load_dataset(t)
    calls = []
    def run(command, args, **options):
        calls.append((command, args, options))
        Path(args[args.index('-o') + 1]).mkdir()
        return 0
    monkeypatch.setattr(vla, 'run', run)
    vla.train(policy, dataset, c, output_dir=tmp_path / 'out',
              overrides={'training.use_cpu': True, 'policy.args.chunk_size': 3}, offline=True)
    assert calls[0][1][-4:] == ['--training.use_cpu', 'true', '--policy.args.chunk_size', '3']
    assert calls[0][2]['offline'] is True
    with pytest.raises(ValueError, match='simulation.yaml'):
        vla.load_env(e).evaluate(policy, output_dir=tmp_path / 'eval')
    with pytest.raises(ValueError, match='function parameters'):
        vla.train(policy, dataset, c, output_dir='unused', overrides={'output_dir': 'other'})
    with pytest.raises(TypeError, match='Unknown runtime'):
        vla.load_policy(p, typo=True)
    with pytest.raises(ValueError, match='checkpoint'):
        vla.load_env(e).evaluate(vla.load_policy(p), output_dir='unused')
