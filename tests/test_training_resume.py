"""Resume the same persisted run without changing config, data or checkpoints."""
import json
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip('fcntl')
from vlastudio.webui import training
from test_webui_training import storage
from test_webui_training import log_server
import urllib.request


@pytest.fixture
def failed_run(storage, monkeypatch):
    calls = []
    monkeypatch.setattr(training.subprocess, 'Popen', lambda *a, **kw: (calls.append((a, kw)) or SimpleNamespace(wait=lambda: 0)))
    run = training.start(dict(name='Resume test', policy='sample', task='sample', training='sample'))
    folder = training.directory(run['id'])
    run.update(status='failed', exit_code=1, error='old error')
    training.write(folder / 'run.json', run)
    (folder / 'training.log').write_text('original failure\n')
    (folder / 'metrics.jsonl').write_text(''.join(json.dumps(dict(step=i, loss=1/i))+'\n' for i in (1, 2, 3)))
    return run, folder, calls


def checkpoint(folder, step, complete=True):
    path = folder / 'model' / f'checkpoint-{step}'
    path.mkdir(parents=True)
    (path / 'trainer_state.json').write_text(json.dumps(dict(global_step=step)))
    if complete:
        (path / 'model.safetensors').write_bytes(b'original weights')
    return path


def test_resume_latest_complete_checkpoint_preserves_identity_and_history(failed_run, storage):
    run, folder, calls = failed_run
    path = checkpoint(folder, 2)
    checkpoint(folder, 3, complete=False)
    # Catalog changes must not change an existing run's resolved configuration.
    (storage.root / 'training/sample.yaml').write_text('max_steps: 999\n')
    result = training.resume(run['id'])
    assert result['id'] == run['id'] and result['output_dir'] == run['output_dir']
    assert result['data_dir'] == str(storage.data) and result['resume_checkpoint'] == str(path)
    assert result['status'] == 'starting' and 'error' not in result and 'exit_code' not in result
    config = yaml.safe_load((folder / 'training.yaml').read_text())
    assert config['max_steps'] == 3 and config['resume_from_checkpoint'] == str(path)
    assert [row['step'] for row in training.read_metrics(folder)] == [1, 2]
    assert (path / 'model.safetensors').read_bytes() == b'original weights'
    assert 'original failure' in (folder / 'training.log').read_text()
    archive = next((folder / 'attempts').iterdir())
    assert len((archive / 'metrics.jsonl').read_text().splitlines()) == 3
    assert json.loads((archive / 'run.json').read_text())['status'] == 'failed'
    assert calls[-1][1]['env']['VLASTUDIO_DATA_CACHE_DIR'] == str(storage.data)
    with pytest.raises(ValueError):
        training.resume(run['id'])
    assert len(calls) == 2


def test_no_checkpoint_retries_original_config_without_stale_metrics(failed_run):
    run, folder, calls = failed_run
    result = training.resume(run['id'])
    assert result['resume_checkpoint'] is None
    assert yaml.safe_load((folder / 'training.yaml').read_text())['resume_from_checkpoint'] is False
    assert training.read_metrics(folder) == []
    assert training.detail(run['id'])['resume']['available'] is False


def test_resume_rejects_completed_and_other_active_runs(failed_run):
    run, folder, calls = failed_run
    run['status'] = 'completed'
    training.write(folder / 'run.json', run)
    with pytest.raises(ValueError, match='Only failed'):
        training.resume(run['id'])
    run['status'] = 'failed'
    training.write(folder / 'run.json', run)
    training.start(dict(name='Other', policy='sample', task='sample', training='sample'))
    with pytest.raises(ValueError, match='already active'):
        training.resume(run['id'])


def test_resume_launch_failure_keeps_run_recoverable(failed_run, monkeypatch):
    run, folder, calls = failed_run
    def fail(*a, **kw):
        raise OSError('Launch unavailable')
    monkeypatch.setattr(training.subprocess, 'Popen', fail)
    result = training.resume(run['id'])
    assert result['status'] == 'failed' and result['error'] == 'Launch unavailable'
    assert training.detail(run['id'])['resume']['available'] is True


def test_resume_http_endpoint(failed_run):
    run, folder, calls = failed_run
    with log_server() as url:
        request = urllib.request.Request(url.split('/api/')[0] + '/api/training/resume',
                                         data=json.dumps({'id': run['id']}).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
        assert result['id'] == run['id'] and result['status'] == 'starting'
