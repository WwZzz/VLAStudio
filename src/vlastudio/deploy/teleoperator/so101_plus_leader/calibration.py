"""Load the landmarks associated with this exact seven-servo calibration."""
import fcntl
import hashlib
import json
from pathlib import Path


def lock_port(port):
    device = Path(port).resolve(strict=True)
    path = Path.home()/".cache"/("vlastudio-leader-" + device.name + ".lock")
    path.parent.mkdir(exist_ok=True)
    lock = path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError(f"Leader port is busy: {device}") from None
    return lock


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(directory, robot_id):
    path = directory / (robot_id + ".json")
    reference = path.with_suffix(".reference.json")
    if not reference.is_file() or not path.is_file():
        raise ValueError("Calibrate the SO101 Plus leader and capture its Rebot reference first")
    ref = json.loads(reference.read_text())
    if ref["calibration"] != digest(path):
        raise ValueError("Leader calibration changed; capture the Rebot reference again")
    return ref
