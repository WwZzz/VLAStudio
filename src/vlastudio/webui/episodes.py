"""Browse completed recordings under the WebUI output directory."""
import base64
import hashlib
from pathlib import Path
import re
import uuid

from .profiles import ROOT

ROOT_DIR = ROOT / "data/webui"


def files(dataset, index):
    if not isinstance(dataset, str) or not re.fullmatch(r"[\w-][\w.-]{0,79}", dataset):
        raise ValueError("Invalid dataset")
    if type(index) is not int or not 0 <= index < 10**12:
        raise ValueError("Invalid episode")
    directory = ROOT_DIR / dataset
    if directory.is_symlink() or directory.resolve().parent != ROOT_DIR.resolve():
        raise ValueError("Invalid dataset directory")
    paths = [directory / f"episode_{index:04d}.hdf5",
             directory / "video" / f"real_eval_episode_{index:04d}.mp4"]
    return [p for p in paths if p.is_file() and p.resolve() == directory.resolve() / p.relative_to(directory)]


def revision(paths):
    stats = [p.stat() for p in paths]
    return hashlib.sha256(str([(s.st_ino, s.st_mtime_ns, s.st_size) for s in stats]).encode()).hexdigest()[:24]


def catalog(active_video=None):
    rows = []
    if not ROOT_DIR.exists():
        return rows
    for directory in sorted(ROOT_DIR.iterdir()):
        if not directory.is_dir() or directory.is_symlink() or not re.fullmatch(r"[\w-][\w.-]{0,79}", directory.name):
            continue
        indices = set()
        for folder, pattern in [(directory, r"episode_(\d{4,12})\.hdf5"), (directory / "video", r"real_eval_episode_(\d{4,12})\.mp4")]:
            if not folder.is_dir() or folder.is_symlink():
                continue
            for item in folder.iterdir():
                match = re.fullmatch(pattern, item.name)
                if match:
                    indices.add(int(match[1]))
        for index in sorted(indices):
            paths = files(directory.name, index)
            if paths and not (directory.name == active_video and paths[0].suffix == ".mp4"):
                rows.append(dict(dataset=directory.name, index=index, version=revision(paths),
                                 rerecordable=len(paths) == 1 and paths[0].suffix == ".hdf5"))
    return rows


def checked(dataset, index, version):
    paths = files(dataset, index)
    if not paths or revision(paths) != version:
        raise ValueError("Recording changed. Reopen the episode library.")
    return paths


def replacement_task(dataset, index, version):
    import h5py
    paths = checked(dataset, index, version)
    if len(paths) != 1 or paths[0].suffix != ".hdf5":
        raise ValueError("Only standalone collection episodes can be re-recorded")
    with h5py.File(paths[0], "r") as recording:
        return str(recording.attrs.get("task", ""))


def remove(dataset, index, version):
    paths = checked(dataset, index, version)
    # Keep deletion reversible. Roll back a partial move of a video/dataset pair.
    trash = ROOT_DIR / ".trash"
    if trash.is_symlink():
        raise ValueError("Invalid trash directory")
    target = trash / uuid.uuid4().hex / dataset
    target.mkdir(parents=True)
    moved = []
    try:
        for p in paths:
            dst = target / p.name
            p.rename(dst)
            moved.append((dst, p))
    except OSError:
        for dst, p in reversed(moved):
            dst.rename(p)
        raise


def jpeg(rgb, *, bgr=False):
    import cv2
    if max(rgb.shape[:2]) > 800:
        scale = 800 / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    image = rgb if bgr or rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        raise ValueError("Failed to decode camera frame")
    return base64.b64encode(buf).decode()


def read(dataset, index, version, frame=None):
    path = checked(dataset, index, version)[0]
    if path.suffix == ".mp4":
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or count < 1:
                raise ValueError("Video is unavailable")
            if frame is None:
                return dict(count=count, fps=cap.get(cv2.CAP_PROP_FPS) or 30)
            if not 0 <= frame < count:
                raise ValueError("Invalid frame")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError("Failed to read video frame")
            return dict(images={"Video": jpeg(image, bgr=True)}, values={})
        finally:
            cap.release()
    import h5py
    import numpy as np
    with h5py.File(path, "r") as recording:
        stamps = recording["timestamps"]
        count = len(stamps)
        if count == 0:
            raise ValueError("Empty recording")
        if frame is None:
            times = np.asarray(stamps, dtype=float).reshape(-1)
            timeline = times - times[0]
            valid = len(times) == count and np.isfinite(timeline).all() and (np.diff(timeline) >= 0).all()
            return dict(count=count, fps=float(recording.attrs.get("fps", 30)) or 30,
                        times=timeline.tolist() if valid else None)
        if not 0 <= frame < count:
            raise ValueError("Invalid frame")
        images, values = {}, {}
        def extract(name, item):
            if not isinstance(item, h5py.Dataset) or not item.shape or item.shape[0] != count:
                return
            if len(item.shape) == 4 and item.shape[-1] == 3 and item.dtype == np.uint8:
                images[name.removeprefix("observation/images/")] = jpeg(item[frame])
            elif len(item.shape) <= 2 and item.size // count <= 256 and item.dtype.kind in "biuf" and name != "timestamps":
                vals = np.asarray(item[frame]).reshape(-1)
                values[name.removeprefix("observation/")] = [round(float(v), 5) if np.isfinite(v) else None for v in vals]
        recording.visititems(extract)
        return dict(images=images, values=values)
