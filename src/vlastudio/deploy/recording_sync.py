"""Frame synchronization with zero-filled optional inputs on the recording side."""
from collections import deque

import numpy as np
import yaml

from vlastudio.deploy.shm_utils import SharedMemoryDataSynchronizer


class OptionalInput:
    def __init__(self, fields, events=()):
        self.events = tuple(events)
        self.zeros = {}
        for name, field in fields.items():
            spec = {"dtype": field} if isinstance(field, str) else field
            dtype = np.dtype(spec["dtype"])
            if dtype.kind not in "buif":
                raise ValueError(f"Unsupported recording dtype for {name}: {dtype}")
            self.zeros[name] = np.zeros(tuple(spec.get("shape", ())), dtype=dtype)
        if not self.zeros or set(self.events) & self.zeros.keys():
            raise ValueError("Declare value fields and edge events separately")
        self.pending = deque(maxlen=200)

    def sample(self, ref):
        latest = None
        result = dict.fromkeys(self.events, False)
        while self.pending and self.pending[0]["__timestamp__"] <= ref:
            latest = self.pending.popleft()
            for key in self.events:
                result[key] |= bool(latest.get(key, False))
        for key, zero in self.zeros.items():
            value = zero if latest is None else np.asarray(latest[key], dtype=zero.dtype)
            if value.shape != zero.shape:
                raise ValueError(f"Recording field {key}: expected {zero.shape}, got {value.shape}")
            result[key] = value.reshape(1) if value.ndim == 0 else value
        stamp = latest["__timestamp__"] if latest is not None else 0.
        return result, latest is not None, stamp


def load_optional_inputs(path, channel_names, *, single_quest_name=None):
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or set(config) - {"required", "optional"}:
        raise ValueError("Expected a recording YAML with required and/or optional channels")
    optional = config.get("optional", {})
    names = set(channel_names)
    if single_quest_name is not None:
        optional = {single_quest_name: optional["quest3_rebot_b601rs"]}
        required = names - set(optional)
    else:
        required = set(config.get("required", names - set(optional)))
    if not required or required & optional.keys() or required | optional.keys() != names:
        raise ValueError(f"Recording groups must partition {sorted(names)} with at least one required input")
    return {name: OptionalInput(**spec) for name, spec in optional.items()}


class RecordingSynchronizer(SharedMemoryDataSynchronizer):
    def __init__(self, *, shm_channels, optional, teleop_name, **kwargs):
        super().__init__([(n, ch) for n, ch in shm_channels if n not in optional], **kwargs)
        self.optional = optional
        self.optional_channels = [(n, ch) for n, ch in shm_channels if n in optional]
        self.teleop_name = teleop_name
        self.episode_start = float("inf")

    def begin_episode(self, start_time):
        super().begin_episode(start_time)
        self.episode_start = start_time
        for stream in self.optional.values():
            stream.pending.clear()

    def read_and_buffer(self):
        super().read_and_buffer()
        for name, channel in self.optional_channels:
            packet = channel.read(blocking=False, skip_unchanged=True, copy=True)
            if packet is not None and packet["__timestamp__"] >= self.episode_start:
                self.optional[name].pending.append(packet)

    def get_synced_frame_blocking(self, *args, **kwargs):
        frame = super().get_synced_frame_blocking(*args, **kwargs)
        if frame is None:
            return None
        ref = self.last_sync_stats["ref_ts"]
        for name, stream in self.optional.items():
            data, valid, stamp = stream.sample(ref)
            # Sensor keys are flattened by the saver. Scope their metadata per device.
            prefix = "" if name == self.teleop_name else name + "_"
            data[prefix + "has_sample"] = valid
            data[prefix + "source_timestamp_s"] = np.array([stamp], dtype=np.float64)
            data[prefix + "age_s"] = ref - stamp if valid else -1.
            if name == self.teleop_name:
                data.update(has_action=valid, new_action=valid)
            frame[name] = data
        return frame
