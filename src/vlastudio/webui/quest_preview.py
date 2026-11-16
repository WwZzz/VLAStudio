"""Bounded, read-only Quest visualization, independent of robot action sampling."""
import json
import os
import sys
import time

import numpy as np


def controller_sample(goal):
    if goal is None:
        return None
    metadata = goal.metadata or {}
    position = metadata.get("vr_position")
    quaternion = metadata.get("quaternion")
    if position is None or not isinstance(quaternion, dict):
        return None
    pos = np.asarray(position, dtype=float)
    quat = np.asarray([quaternion.get(k, 0) for k in ("x", "y", "z", "w")], dtype=float)
    if pos.shape != (3,) or not np.isfinite(pos).all() or not np.isfinite(quat).all() or np.linalg.norm(quat)<.5:
        return None
    buttons = metadata.get("buttons") or {}
    def pressed(key):
        return bool(buttons.get(key) or buttons.get("button_"+key) or buttons.get(key.upper()))
    trigger = float(metadata.get("trigger", 0))
    if not np.isfinite(trigger):
        return None
    return dict(position=pos.tolist(), quaternion=(quat/np.linalg.norm(quat)).tolist(),
                grip=bool(metadata.get("grip_active") or buttons.get("squeeze") or buttons.get("unsqueeze")),
                trigger=float(np.clip(trigger, 0, 1)),
                buttons={k:pressed(k) for k in ("a", "b", "x", "y")})


def publish(device, data):
    """Called at most 20 Hz. Preview failures never interrupt control."""
    now = time.monotonic()
    if now < getattr(device, "_next_preview", 0):
        return
    device._next_preview = now + .05
    try:
        if getattr(device, "_preview_shm", None) is None:
            device._preview_shm = device.create_shm(name=device.preview_channel_name, max_size_mb=1)
            device._preview_goals = {}
        hands = {}
        for hand in ("left", "right"):
            goal = data.get(hand)
            if goal is device._preview_goals.get(hand):
                continue
            device._preview_goals[hand] = goal
            sample = controller_sample(goal)
            if sample is not None:
                hands[hand] = sample
        if hands:
            device._preview_shm.write(dict(hands=hands))
    except Exception:
        pass


def run(channel):
    from vlastudio.deploy.shm_utils import SharedMemoryChannel
    os.nice(12)
    reader = None
    try:
        while True:
            if os.getppid() == 1:
                return
            if reader is None:
                try:
                    reader = SharedMemoryChannel(channel, timeout=0)
                except (OSError, ValueError, TimeoutError):
                    time.sleep(.2)
                    continue
            message = reader.read(blocking=False, skip_unchanged=True, copy=True)
            if message:
                print(json.dumps(dict(stamp=message["__timestamp__"], hands=message["hands"]),
                                 allow_nan=False, separators=(",", ":")), flush=True)
            time.sleep(.05)
    finally:
        if reader:
            reader.destroy()


if __name__ == "__main__":
    run(sys.argv[1])
