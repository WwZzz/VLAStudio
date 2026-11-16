"""Bindings produced by validated environment installs, separate from UI drafts."""
import json
import time
from pathlib import Path

from ..paths import cache_root


def location():
    return cache_root() / 'webui' / 'component-interpreters.json'


def read():
    path = location()
    if not path.exists():
        return {'interpreters': {}, 'bindings': {}}
    return json.loads(path.read_text(encoding='utf-8'))


def entries():
    return list(read()['interpreters'].values())


def choose(role, key):
    data = read()
    keys = [key]
    if role == 'inference':
        if key.startswith(('http://', 'https://', 'ws://', 'wss://')):
            keys.append('vlastudio.policy.remote')
        else:
            # Training writes this metadata beside final model weights.
            path = Path(key).expanduser() / 'policy_metadata.json'
            if path.is_file():
                try:
                    doc = json.loads(path.read_text(encoding='utf-8'))
                    keys.extend(str(doc.get(k, '')) for k in ('policy_module', 'policy_name'))
                except (OSError, ValueError):
                    pass
    for candidate in keys:
        identifier = data['bindings'].get(role + ':' + candidate)
        if identifier in data['interpreters']:
            return data['interpreters'][identifier]


def register(component, environment, python, variables):
    from filelock import FileLock
    from ..env_manager import atomic_json
    path = location()
    path.parent.mkdir(parents=True, exist_ok=True)
    identifier = 'managed-' + environment
    entry = dict(id=identifier, name=component['label'], python=python, env=variables,
                 managed=True, environment=environment, component=component['id'], verified_at=time.time())
    with FileLock(str(path) + '.lock', timeout=20):
        data = read()
        data['interpreters'][identifier] = entry
        for role, names in component['bindings'].items():
            for name in names:
                data['bindings'][role + ':' + name] = identifier
        atomic_json(path, data)
    return entry
