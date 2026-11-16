"""Identity and lifecycle metadata for collections created by the WebUI."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

from .settings import data_directory
from .tasks import get_task


def scope(robot, task="", real_task="", *, recording_source="collection"):
    if not robot:
        raise ValueError("Select a robot")
    real = get_task(real_task) if real_task and not task else {}
    return dict(robot=robot, task=task or real.get("name", ""),
                task_id=task or real.get("id", ""), task_type="benchmark" if task else "real",
                recording_source=recording_source)


def recording_source(directory, item=None):
    item = read_manifest(directory) if item is None else item
    source = (item or {}).get('recording_source')
    if source in ('collection', 'inference'):
        return source
    # Old inference recordings already carry an explicit inference manifest.
    manifest = directory / 'eval_real_manifest.json'
    if manifest.is_file() and not manifest.is_symlink():
        try:
            if json.loads(manifest.read_text(encoding='utf-8')).get('script') == 'vlastudio infer':
                return 'inference'
        except (ValueError, OSError):
            pass
    return 'collection' if item else 'unknown'


def identity(context, suffix):
    if not isinstance(suffix, str) or (suffix and not re.fullmatch(r"[\w-][\w.-]{0,39}", suffix)):
        raise ValueError("Invalid dataset suffix")
    key = dict(context, recording_source=context.get('recording_source', 'collection'), suffix=suffix)
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    label = "_".join(filter(None, ['inference' if key['recording_source'] == 'inference' else '', key["robot"], key["task"] or "untasked", suffix]))
    label = re.sub(r"[^\w.-]+", "-", label).strip(".-") or "dataset"
    canonical = dict(key, dataset=f"{label[:60]}-{digest}")
    root = data_directory()
    matches = []
    for directory in sorted(root.iterdir()) if root.is_dir() else ():
        item = read_manifest(directory) if directory.is_dir() else None
        normalized = dict(item, recording_source=recording_source(directory, item)) if item else {}
        if item and item.get("dataset") == directory.name and re.fullmatch(r"[\w-][\w.-]{0,79}", directory.name) and all(normalized.get(k) == v for k, v in key.items()):
            if directory.name == canonical["dataset"]:
                return canonical
            matches.append(directory.name)
    return dict(key, dataset=matches[0]) if matches else canonical


def now():
    return datetime.now(timezone.utc).isoformat()


def read_manifest(directory):
    path = directory / "dataset.json"
    if directory.is_symlink() or path.is_symlink() or not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as stream:
            item = json.load(stream)
    except (ValueError, FileNotFoundError):
        return None
    return item if isinstance(item, dict) and "suffix" in item else None


def write_manifest(directory, item):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(item, stream, ensure_ascii=False, indent=2)
            stream.flush()
            tmp.replace(directory / "dataset.json")
        finally:
            tmp.unlink(missing_ok=True)


def ensure(item):
    directory = data_directory() / item["dataset"]
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists() or directory.is_symlink():
        existing = read_manifest(directory)
        normalized = dict(existing, recording_source=recording_source(directory, existing)) if existing else {}
        if existing and all(normalized.get(k) == v for k, v in item.items()):
            if normalized != existing:
                write_manifest(directory, normalized)
            return normalized
        raise ValueError("Dataset directory already exists with a different identity")
    directory.mkdir()
    stamp = now()
    result = dict(item, created_at=stamp, modified_at=stamp)
    write_manifest(directory, result)
    return result


def modified(directory, episode=None):
    import fcntl
    lock = directory / ".dataset.lock"
    if lock.is_symlink():
        raise ValueError("Invalid dataset lock")
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        item = read_manifest(directory)
        if item:
            item["modified_at"] = now()
            if episode is not None:
                item.setdefault("episodes", {})[str(episode)] = {"saved_at": item["modified_at"],
                    "recording_source": recording_source(directory, item)}
            write_manifest(directory, item)


def choices(context):
    default = identity(context, "")
    found = {}
    root = data_directory()
    for directory in root.iterdir() if root.is_dir() else ():
        item = read_manifest(directory) if directory.is_dir() else None
        normalized = dict(item, recording_source=recording_source(directory, item)) if item else {}
        if item and item.get("dataset") == directory.name and all(normalized.get(k) == v for k, v in context.items()):
            found[item["suffix"]] = normalized
    found.setdefault("", default)
    return [found[k] for k in sorted(found)]
