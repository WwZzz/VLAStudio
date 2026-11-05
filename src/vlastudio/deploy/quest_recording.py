"""Record sparse Quest actions alongside continuously sampled robot observations."""
from collections import deque

import numpy as np

from vlastudio.deploy.shm_utils import SharedMemoryDataSynchronizer


class QuestRecordingSynchronizer(SharedMemoryDataSynchronizer):
    def __init__(self, *, quest_name, shm_channels, **kwargs):
        sensors = [(name, ch) for name, ch in shm_channels if name != quest_name]
        if not sensors:
            raise ValueError("A sensor stream is required for Quest action sampling")
        super().__init__(sensors, **kwargs)
        self.quest_name = quest_name
        self.quest = dict(shm_channels)[quest_name]
        self.pending = deque(maxlen=self.buffer_maxlen)
        self.episode_start = float("inf")
        self.latest = None

    def begin_episode(self, start_time):
        super().begin_episode(start_time)
        self.pending.clear()
        self.latest = None
        self.episode_start = start_time

    def read_and_buffer(self):
        super().read_and_buffer()
        packet = self.quest.read(blocking=False, skip_unchanged=True, copy=True)
        if packet is not None and packet["__timestamp__"] >= self.episode_start:
            self.pending.append(packet)

    def get_synced_frame_blocking(self, *args, **kwargs):
        frame = super().get_synced_frame_blocking(*args, **kwargs)
        if frame is None:
            return None
        ref = self.last_sync_stats["ref_ts"]
        new_action = go_home = anchored = False
        while self.pending and self.pending[0]["__timestamp__"] <= ref:
            self.latest = self.pending.popleft()
            new_action = True
            # Preserve observed button edges even when two packets land in one frame.
            go_home |= bool(self.latest.get("go_home"))
            anchored |= bool(self.latest.get("anchor_just_set"))

        latest = self.latest or {}
        stamp = latest.get("__timestamp__", 0.)
        frame[self.quest_name] = {
            "action": latest["action"] if latest else np.zeros(7),
            "unsqueeze_active": bool(latest.get("unsqueeze_active")),
            "left_unsqueeze_active": bool(latest.get("left_unsqueeze_active")),
            "right_unsqueeze_active": bool(latest.get("right_unsqueeze_active")),
            "go_home": go_home,
            "anchor_just_set": anchored,
            "has_action": bool(latest),
            "new_action": new_action,
            "source_timestamp_s": np.array([stamp], dtype=np.float64),
            "age_s": ref - stamp if latest else -1.,
        }
        return frame
