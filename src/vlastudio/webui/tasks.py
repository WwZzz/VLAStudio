"""User-created real-robot tasks, shared by browsers and retained across restarts."""
import json
import fcntl
import os
from pathlib import Path
import tempfile
import threading
import uuid

TASKS_FILE = Path.home() / '.config/vlastudio/webui-tasks.json'
_lock = threading.RLock()


def list_tasks():
    with _lock:
        if not TASKS_FILE.exists():
            return []
        return json.loads(TASKS_FILE.read_text(encoding='utf-8'))['tasks']


def get_task(identifier):
    for task in list_tasks():
        if task['id'] == identifier:
            return task
    raise ValueError('Unknown real-robot task')


def create_task(name, instruction):
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
        raise ValueError('Task name must contain 1–120 characters')
    if not isinstance(instruction, str) or len(instruction.strip()) > 2000:
        raise ValueError('Task instruction must contain at most 2000 characters')
    name, instruction = name.strip(), instruction.strip()
    TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _lock, TASKS_FILE.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        tasks = list_tasks()
        for task in tasks:
            if task['name'].casefold() == name.casefold():
                if task['instruction'] == instruction:
                    return task
                raise ValueError('A task with this name already exists')
        task = dict(id=uuid.uuid4().hex, name=name, instruction=instruction)
        tasks.append(task)
        path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=TASKS_FILE.parent,
                                             prefix='.webui-tasks-', delete=False) as output:
                path = Path(output.name)
                json.dump(dict(version=1, tasks=tasks), output, ensure_ascii=False, indent=2)
                output.flush()
                os.fsync(output.fileno())
            os.replace(path, TASKS_FILE)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
        return task
