"""Training storage and log API regressions; no GPU or model downloads required."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import urllib.error
import urllib.request

import pytest

pytest.importorskip('fcntl')
from vlastudio.webui import settings, training
from vlastudio.webui.__main__ import Handler
from http.server import ThreadingHTTPServer


@pytest.fixture
def storage(tmp_path, monkeypatch):
    data, checkpoints = tmp_path / 'selected-data', tmp_path / 'weights'
    preferences = tmp_path / 'settings.json'
    preferences.write_text(json.dumps({'data_dir': str(data), 'checkpoint_dir': str(checkpoints)}))
    monkeypatch.setattr(settings, 'SETTINGS_FILE', preferences)
    root = tmp_path / 'configs'
    documents = {'policy': 'name: tiny\ntype: example.policy\nargs: {}\n',
                 'task': 'name: sample\ndatasets:\n- name: sample\n  type: example.Dataset\n  args:\n    dataset_path_list: [data/sample]\n',
                 'training': 'max_steps: 3\nlogging_steps: 1\n'}
    for kind, content in documents.items():
        folder = root / kind
        folder.mkdir(parents=True)
        (folder / 'sample.yaml').write_text(content)
    monkeypatch.setattr(training, 'config_roots', lambda: [root])
    return SimpleNamespace(data=data, checkpoints=checkpoints, root=root)


def test_training_worker_uses_selected_data_directory(storage, monkeypatch):
    calls = []
    monkeypatch.setenv('VLASTUDIO_DATA_CACHE_DIR', '/wrong/previous-cache')
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(wait=lambda: 0)
    monkeypatch.setattr(training.subprocess, 'Popen', launch)
    run = training.start(dict(name='Storage check', policy='sample', task='sample', training='sample'))
    assert run['data_dir'] == str(storage.data)
    assert calls[0][1]['env']['VLASTUDIO_DATA_CACHE_DIR'] == str(storage.data)
    assert calls[0][1]['env']['TQDM_POSITION'] == '-1'
    assert Path(run['output_dir']).is_relative_to(storage.checkpoints)
    assert training.detail(run['id'])['run']['data_dir'] == str(storage.data)
    with pytest.raises(ValueError, match='already active'):
        training.start(dict(name='Duplicate', policy='sample', task='sample', training='sample'))


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    folder = tmp_path / 'run'
    folder.mkdir()
    (folder / 'run.json').write_text(json.dumps({'id': 'sample', 'name': 'test', 'status': 'failed',
                                              'created_at': '2026-11-13T00:00:00+00:00'}))
    payload = ('[INFO] first\rprogress 1\rprogress 2\n关键词 café 🧪\n' + 'x' * 70000).encode()
    (folder / 'training.log').write_bytes(payload)
    monkeypatch.setattr(training, 'entries', lambda: {'sample': str(folder)})
    return folder, payload


def test_log_preview_reports_freshness_and_preserves_complete_download(saved_run):
    folder, payload = saved_run
    detail = training.detail('sample')
    assert detail['log_bytes'] == len(payload)
    assert detail['log_updated_at'] and detail['checked_at']
    assert '\r' not in detail['log']
    assert training.artifact('sample', 'training.log').read_bytes() == payload
    with pytest.raises(ValueError):
        training.artifact('sample', '../run.json')


@contextmanager
def log_server():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.session = SimpleNamespace()
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/api/training/log?id=sample'
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_full_log_and_incremental_byte_ranges(saved_run):
    _, payload = saved_run
    with log_server() as url:
        with urllib.request.urlopen(url) as response:
            assert response.headers['Content-Type'].startswith('text/plain')
            assert response.headers.get('Content-Disposition') is None
            assert response.read() == payload
        request = urllib.request.Request(url, headers={'Range': 'bytes=7-37'})
        with urllib.request.urlopen(request) as response:
            assert response.status == 206
            assert response.headers['Content-Range'] == f'bytes 7-37/{len(payload)}'
            assert response.read() == payload[7:38]
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url, headers={'Range': f'bytes={len(payload)}-'}))
        assert error.value.code == 416


def test_empty_log_range(saved_run):
    folder, _ = saved_run
    (folder / 'training.log').write_bytes(b'')
    with log_server() as url:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-1023'}))
        assert error.value.code == 416
        assert error.value.headers['Content-Range'] == 'bytes */0'
