"""Persistent training requests; workers outlive HTTP connections and WebUI restarts."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import uuid

import yaml
from . import settings
from .profiles import ROOT, config_roots

LOCK = threading.RLock()
ACTIVE = ('starting', 'running', 'stopping')


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def registry():
    return settings.SETTINGS_FILE.with_name('training-runs.json')


def entries():
    path = registry()
    return json.loads(path.read_text()) if path.exists() else {}


def paths(kind):
    result = {}
    for root in config_roots():
        for path in sorted((root / kind).rglob('*.yaml')):
            if path.is_file() and not path.is_symlink():
                result.setdefault(path.relative_to(root / kind).with_suffix('').as_posix(), path)
    return result


def catalog():
    return {kind: list(paths(kind)) for kind in ('policy', 'task', 'training')}


def directory(identifier):
    value = entries().get(identifier)
    if not value:
        raise ValueError('Unknown training run')
    return Path(value)


def alive(item):
    try:
        stat = Path(f"/proc/{item['pid']}/stat").read_text().split()
        return stat[2] != "Z" and stat[21] == item["process_start"]
    except (OSError, KeyError, IndexError):
        return False


def read(folder):
    item = json.loads((folder / 'run.json').read_text())
    if item['status'] in ACTIVE and not alive(item) and (item.get('pid') or (datetime.now(timezone.utc) - datetime.fromisoformat(item['created_at'])).total_seconds() > 30):
        # Readers never overwrite a worker that may be committing its final result.
        item.update(status='interrupted')
    return item


def runs():
    with LOCK:
        result = []
        for value in entries().values():
            try:
                result.append(read(Path(value)))
            except (OSError, ValueError):
                continue
        return sorted(result, key=lambda row: row['created_at'], reverse=True)


def start(payload):
    with LOCK:
        if any(row['status'] in ACTIVE for row in runs()):
            raise ValueError('A training run is already active')
        if not isinstance(payload, dict) or not isinstance(payload.get('name'), str):
            raise ValueError('Enter a run name')
        name = payload['name'].strip()
        if not 1 <= len(name) <= 100:
            raise ValueError('Run name must contain 1–100 characters')
        configs = {}
        for kind in ('policy', 'task', 'training'):
            path = paths(kind).get(payload.get(kind))
            if path is None:
                raise ValueError(f'Select a valid {kind} configuration')
            try:
                doc = yaml.safe_load(path.read_text())
            except yaml.YAMLError as exc:
                raise ValueError(f'Invalid {kind} YAML: {exc}') from None
            if not isinstance(doc, dict):
                raise ValueError(f'Invalid {kind} configuration')
            configs[kind] = doc
        overrides = payload.get('overrides', {})
        if not isinstance(overrides, dict):
            raise ValueError('Overrides must be a JSON object')
        limits = {'max_steps': (1, 100000000), 'per_device_train_batch_size': (1, 65536),
                  'gradient_accumulation_steps': (1, 65536), 'logging_steps': (1, 1000000),
                  'save_steps': (1, 100000000), 'dataloader_num_workers': (0, 256), 'seed': (0, 2147483647)}
        for key, value in overrides.items():
            if key == 'learning_rate':
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                    raise ValueError('Invalid learning rate')
            elif key in limits:
                low, high = limits[key]
                if type(value) is not int or not low <= value <= high:
                    raise ValueError(f'Invalid {key}')
            else:
                raise ValueError(f'Unsupported override: {key}; edit the training YAML for advanced options')
        # Resolve the full configuration here, before accepting a request.
        configs['training'].update(overrides)
        configs['training'].update(report_to='none', resume_from_checkpoint=False)
        identifier = uuid.uuid4().hex
        folder = settings.checkpoint_directory() / identifier
        folder.mkdir(parents=True)
        for kind, doc in configs.items():
            (folder / f'{kind}.yaml').write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False))
        item = dict(id=identifier, name=name, status='starting', created_at=now(),
                    policy=payload['policy'], task=payload['task'], training=payload['training'],
                    overrides=overrides, data_dir=str(settings.data_directory()), max_steps=configs['training'].get('max_steps'), output_dir=str(folder / 'model'), python=sys.executable)
        write(folder / 'run.json', item)
        index = entries()
        index[identifier] = str(folder)
        registry().parent.mkdir(parents=True, exist_ok=True)
        write(registry(), index)
        env = settings.runtime_environment(os.environ)
        env["VLASTUDIO_DATA_CACHE_DIR"] = item["data_dir"]
        env.setdefault("TQDM_POSITION", "-1")
        env.setdefault("TQDM_MININTERVAL", "2")
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), env.get('PYTHONPATH', '')])
        try:
            with (folder / 'training.log').open('ab') as log:
                proc = subprocess.Popen([sys.executable, '-u', '-m', 'vlastudio.webui.training_worker', str(folder)],
                                        cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
            # The worker owns run.json after launch; no parent writes may race it.
            threading.Thread(target=proc.wait, daemon=True).start()
        except OSError as exc:
            item.update(status='failed', error=str(exc), finished_at=now())
            write(folder / 'run.json', item)
        return item


def detail(identifier):
    with LOCK:
        folder = directory(identifier)
        item = read(folder)
    metrics = []
    path = folder / 'metrics.jsonl'
    if path.exists():
        with path.open('rb') as stream:
            offset = max(0, path.stat().st_size - 4 * 1024 * 1024)
            stream.seek(offset)
            if offset:
                stream.readline()
            metric_text = stream.read().decode('utf-8', errors='replace')
        for line in metric_text.splitlines():
            try:
                metrics.append(json.loads(line))
            except ValueError:
                pass  # Worker may be appending the last line.
    log = folder / 'training.log'
    log_stat = log.stat() if log.exists() else None
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 65536))
            tail = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', stream.read().decode('utf-8', errors='replace'))
    else:
        tail = ''
    files = [str(p.relative_to(folder)) for p in sorted(folder.rglob('*'))
             if p.is_file() and not p.is_symlink() and p.name != 'run.json']
    return dict(run=item, metrics=metrics[-10000:], log=tail.replace('\r\n', '\n').replace('\r', '\n'), files=files,
                checked_at=now(), log_bytes=log_stat.st_size if log_stat else 0,
                log_updated_at=datetime.fromtimestamp(log_stat.st_mtime, timezone.utc).isoformat() if log_stat else None)


def stop(identifier):
    with LOCK:
        folder = directory(identifier)
        item = read(folder)
        if item['status'] not in ACTIVE:
            raise ValueError('Training is no longer running')
        if not alive(item):
            raise ValueError('Training worker is not ready; retry shortly')
        os.kill(item['pid'], signal.SIGTERM)
        return {'ok': True}


def artifact(identifier, relative):
    folder = directory(identifier).resolve()
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder) or not path.is_file() or path.name == 'run.json':
        raise ValueError('Invalid training artifact')
    return path
