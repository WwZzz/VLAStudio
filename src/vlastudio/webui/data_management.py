"""Dataset catalog and versioned interval annotations, independent of device sessions."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from . import datasets, episodes


def catalog():
    root = datasets.data_directory()
    result = []
    from .profiles import profile_paths
    robot_profiles = profile_paths("robot")
    for directory in sorted(root.iterdir()) if root.is_dir() else ():
        if not directory.is_dir() or directory.is_symlink() or not re.fullmatch(r"[\w-][\w.-]{0,79}", directory.name):
            continue
        item = datasets.read_manifest(directory)
        if not item:
            item = dict(dataset=directory.name, robot="", task="", suffix="")
            manifest = directory / "eval_real_manifest.json"
            if manifest.is_file() and not manifest.is_symlink():
                try:
                    source = json.loads(manifest.read_text())
                    item.update(robot=Path(source.get("robot", "")).stem, task=source.get("task", ""))
                except (ValueError, OSError):
                    pass
            if not item["robot"] and directory.name in robot_profiles:
                item["robot"] = directory.name
            if not item["task"]:
                episode = next(iter(sorted(directory.glob("episode_*.hdf5"))), None)
                if episode and not episode.is_symlink():
                    import h5py
                    try:
                        with h5py.File(episode, "r") as recording:
                            item["task"] = str(recording.attrs.get("task", ""))
                    except OSError:
                        pass
            item["legacy"] = True
        stats = directory.stat()
        result.append(dict(item, created_at=item.get("created_at"), modified_at=item.get("modified_at") or
                           datetime.fromtimestamp(stats.st_mtime, timezone.utc).isoformat()))
    return result


def annotation_path(dataset, index, version):
    source = episodes.checked(dataset, index, version)[0]
    directory = source.parent if source.suffix == ".hdf5" else source.parent.parent
    folder = directory / "annotations"
    if folder.is_symlink():
        raise ValueError("Invalid annotation directory")
    path = folder / f"episode_{index:04d}.json"
    if path.is_symlink():
        raise ValueError("Invalid annotation path")
    return directory, path


def load_versions(path):
    return json.loads(path.read_text(encoding="utf-8"))["versions"] if path.exists() else []


def annotations(dataset, index, version):
    directory, path = annotation_path(dataset, index, version)
    meta = episodes.read(dataset, index, version)
    versions = load_versions(path)
    matching = [doc for doc in versions if doc["source_version"] == version]
    doc = matching[-1] if matching else dict(
        schema="interval-annotations-v1", dataset=dataset, episode_id=index, source_version=version,
        length=meta["count"], revision=None, records=[dict(id="episode", scope="episode",
        start=0, end=meta["count"], language=(datasets.read_manifest(directory) or {}).get("task", ""),
        outcome="unknown", reviewed=False)])
    return dict(document=doc, history=list(reversed(matching)), source_changed=bool(versions and not matching))


def validate_records(records, length):
    if not isinstance(records, list) or not 1 <= len(records) <= 2000:
        raise ValueError("Invalid annotation records")
    ids = set()
    whole = []
    segments = []
    for row in records:
        if not isinstance(row, dict) or set(row) != {"id", "scope", "start", "end", "language", "outcome", "reviewed"}:
            raise ValueError("Invalid annotation fields")
        if not isinstance(row["id"], str) or not 1 <= len(row["id"]) <= 80 or row["id"] in ids:
            raise ValueError("Invalid annotation id")
        ids.add(row["id"])
        if type(row["start"]) is not int or type(row["end"]) is not int or not 0 <= row["start"] < row["end"] <= length:
            raise ValueError("Invalid frame interval")
        if not isinstance(row["language"], str) or len(row["language"]) > 8192 or row["outcome"] not in ("unknown", "success", "failure") or type(row["reviewed"]) is not bool:
            raise ValueError("Invalid annotation value")
        if row["scope"] == "episode":
            whole.append(row)
        elif row["scope"] == "segment":
            segments.append(row)
        else:
            raise ValueError("Invalid annotation scope")
    if len(whole) != 1 or (whole[0]["start"], whole[0]["end"]) != (0, length):
        raise ValueError("Whole episode annotation must span all frames")
    parts = sorted(segments, key=lambda row: row["start"])
    if parts and ([r["start"] for r in parts] != [0] + [r["end"] for r in parts[:-1]] or parts[-1]["end"] != length):
        raise ValueError("Segments must cover the episode without gaps or overlaps")
    return [whole[0], *parts]


def save(data):
    dataset, index, version = (data[k] for k in ("dataset", "index", "version"))
    directory, path = annotation_path(dataset, index, version)
    path.parent.mkdir(exist_ok=True)
    lock = path.with_suffix(".lock")
    if lock.is_symlink():
        raise ValueError("Invalid annotation lock")
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        current = annotations(dataset, index, version)
        doc = current["document"]
        if data.get("revision") != doc["revision"]:
            raise ValueError("Annotations changed in another window. Reload before saving.")
        records = validate_records(data["records"], doc["length"])
        stamp = datasets.now()
        updated = dict(doc, records=records, parent_revision=doc["revision"], producer=dict(kind="human"),
                       updated_at=stamp, revision=uuid.uuid4().hex)
        versions = load_versions(path)
        versions.append(updated)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            try:
                json.dump(dict(versions=versions), tmp, ensure_ascii=False, indent=2)
                tmp.flush()
                os.fsync(tmp.fileno())
                episodes.checked(dataset, index, version)
                tmp_path.replace(path)
            finally:
                tmp_path.unlink(missing_ok=True)
        datasets.modified(directory)
        return dict(document=updated, history=list(reversed([v for v in versions if v["source_version"] == version])), source_changed=False)


def remove_dataset(dataset):
    if not isinstance(dataset, str) or not re.fullmatch(r"[\w-][\w.-]{0,79}", dataset):
        raise ValueError("Invalid dataset")
    root = datasets.data_directory()
    directory = root / dataset
    if not directory.is_dir() or directory.is_symlink() or directory.resolve().parent != root:
        raise ValueError("Invalid dataset directory")
    trash = root / ".trash"
    if trash.is_symlink():
        raise ValueError("Invalid trash directory")
    destination = trash / uuid.uuid4().hex
    destination.mkdir(parents=True)
    directory.rename(destination / dataset)
