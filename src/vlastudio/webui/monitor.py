"""Best-effort SHM reader. This process owns no device handle or control channel."""
import base64
import json
import os
from pathlib import Path
import select
import sys
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .profiles import channels, model_path
from vlastudio.deploy.shm_utils import SharedMemoryChannel


class Skeleton:
    def __init__(self, path):
        self.name = path.stem
        self.joints = []
        for joint in ET.parse(path).getroot().findall("joint"):
            origin = joint.find("origin")
            def vector(key):
                return np.fromstring(origin.get(key, "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            transform = np.eye(4)
            transform[:3, 3] = vector("xyz")
            transform[:3, :3] = Rotation.from_euler("xyz", vector("rpy")).as_matrix()
            axis = joint.find("axis")
            axis = np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.array([1., 0., 0.])
            self.joints.append((joint.find("parent").get("link"), joint.find("child").get("link"),
                                transform, axis, joint.get("type")))
        self.roots = {j[0] for j in self.joints} - {j[1] for j in self.joints}

    def links(self, angles):
        poses = {root: np.eye(4) for root in self.roots}
        steps = []
        index = 0
        for parent, child, origin, axis, kind in self.joints:
            motion = np.eye(4)
            if kind in ("revolute", "continuous", "prismatic"):
                if index < len(angles):
                    if kind == "prismatic":
                        motion[:3, 3] = axis * angles[index]
                    else:
                        motion[:3, :3] = Rotation.from_rotvec(axis * angles[index]).as_matrix()
                index += 1
            steps.append((parent, child, origin @ motion))
        segments = []
        while steps:
            remaining = []
            for parent, child, transform in steps:
                if parent not in poses:
                    remaining.append((parent, child, transform))
                    continue
                poses[child] = poses[parent] @ transform
                segments.append([poses[parent][:3, 3].tolist(), poses[child][:3, 3].tolist()])
            if len(steps) == len(remaining):
                break
            steps = remaining
        return segments


def images(data, name, bgr=False):
    for key, value in data.items():
        if isinstance(value, dict):
            yield from images(value, name if key == "image" else f"{name}/{key}", bgr)
        elif isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim in (2, 3) and min(value.shape[:2]) > 8:
            if value.ndim == 3 and value.shape[2] not in (1, 3, 4):
                continue
            scale = min(1., 400 / max(value.shape[:2]))
            small = cv2.resize(value, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if small.ndim == 3 and not bgr:
                small = cv2.cvtColor(small, cv2.COLOR_RGB2BGR if small.shape[2] == 3 else cv2.COLOR_RGBA2BGR)
            ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ok:
                yield name if key == "image" else f"{name}/{key}", base64.b64encode(jpeg).decode("ascii")


def run(profile):
    os.nice(12)
    cv2.setNumThreads(1)
    rows = channels(profile)
    readers, models, retry, inodes = {}, {}, {}, {}
    for name, typ, args in rows:
        path = model_path(typ, args)
        if path and path.is_file():
            models[name] = Skeleton(path)
            keys = ["left_qpos", "right_qpos"] if typ.endswith(".BiAliciaD") else [name]
            rest = {k: {"name":path.stem, "links":models[name].links([0.]*6), "rest":True} for k in keys}
            print(json.dumps({"images":{}, "axes":{}, "models":rest, "stamps":{}}), flush=True)
    tick = 0
    last_view = time.monotonic()
    try:
        while True:
            start = time.monotonic()
            if select.select([sys.stdin], [], [], 0)[0]:
                if not os.read(sys.stdin.fileno(), 256):
                    return
                last_view = start
            if start - last_view > 4:
                time.sleep(.2)
                continue
            packet = {"images": {}, "axes": {}, "models": {}, "stamps": {}}
            for name, typ, args in rows:
                # A CLI session can restart independently and create a new SHM inode.
                if name in readers and tick % 8 == 0:
                    try:
                        replaced = Path("/dev/shm", name).stat().st_ino != inodes[name]
                    except FileNotFoundError:
                        replaced = True
                    if replaced:
                        readers.pop(name).destroy()
                if name not in readers:
                    if start < retry.get(name, 0):
                        continue
                    retry[name] = start + 1
                    try:
                        inodes[name] = Path("/dev/shm", name).stat().st_ino
                        readers[name] = SharedMemoryChannel(name, is_writer=False, timeout=0)
                    except (OSError, ValueError, TimeoutError):
                        continue
                data = readers[name].read(blocking=False, skip_unchanged=True, copy=False)
                if not data:
                    continue
                stamp = data.get("__timestamp__", start)
                packet["stamps"][name] = stamp
                if tick % 2 == 0:
                    packet["images"].update(images(data, name, args.get("bgr_to_rgb") is False))
                arms = {k: data[k] for k in ("left_qpos", "right_qpos") if k in data}
                if not arms:
                    q = data.get("qpos", data.get("joint_angles"))
                    arms = {name: q} if q is not None else {}
                for arm, values in arms.items():
                    q = np.asarray(values, dtype=float).flatten()[:64]
                    if not len(q) or not np.all(np.isfinite(q)):
                        continue
                    packet["axes"][arm] = [stamp, q.tolist()]
                    if name in models:
                        model = models[name]
                        packet["models"][arm] = {"name": model.name, "links": model.links(q[:6])}
            print(json.dumps(packet, separators=(",", ":"), allow_nan=False), flush=True)
            tick += 1
            time.sleep(max(0, .125 - (time.monotonic()-start)))
    finally:
        for reader in readers.values():
            reader.destroy()


if __name__ == "__main__":
    run(sys.argv[1])
