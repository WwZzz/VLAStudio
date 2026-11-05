"""Application-side thermal policy; never changes servo/firmware settings."""
import json
import math
import os
from pathlib import Path
import threading
import time
import tempfile
import uuid


class ThermalPolicy:
    PROTECTION_COUNT = 10
    RECOVERY_COUNT = 10
    FRESH_SECONDS = 0.5
    LEGACY_E1 = b"Firmware reported thermal status 0xe1\n"
    LEGACY_E2 = b"Firmware reported thermal status 0xe2\n"
    LEGACY_RECORDED_E2 = (
        b"Firmware thermal protection (0xE2) on this body. See left_health_frames_0129.json. "
        b"Keep motion inhibited pending service.\n"
    )
    LEGACY_RECORDS = (LEGACY_E1, LEGACY_E2, LEGACY_RECORDED_E2)

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.path = None
        self.last_rx = None
        self.consecutive_e1 = 0
        self.consecutive_e2 = 0
        self.normal_count = 0
        self.needs_ack = False
        self.hard_fault = None
        self.legacy_record = None

    def bind(self, path):
        with self.lock:
            self.path = Path(path)
            self.path.parent.mkdir(exist_ok=True)
            if self.path.exists():
                old = self.path.read_bytes()
                if old in self.LEGACY_RECORDS:
                    self.legacy_record = old
                    self.needs_ack = True
            if self.hard_fault is not None:
                self._persist()

    def _persist(self):
        if self.path is not None:
            record = {"code": self.hard_fault, "consecutive_e1": self.consecutive_e1,
                      "consecutive_e2": self.consecutive_e2,
                      "policy": "e1_warning_only_e2_10"}
            try:
                with self.path.open("x") as out:
                    json.dump(record, out)
            except FileExistsError:
                # Confirmed ten-frame faults replace recoverable old markers.
                if self.path.read_bytes() in self.LEGACY_RECORDS:
                    with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent,
                                                     delete=False) as out:
                        json.dump(record, out)
                        out.flush()
                        os.fsync(out.fileno())
                        tmp = out.name
                    os.replace(tmp, self.path)
                    self.legacy_record = None

    def observe(self, status):
        """Call once per NEW CRC-valid joint frame, not per cache read."""
        with self.lock:
            now = self.clock()
            if self.last_rx is not None and now - self.last_rx > self.FRESH_SECONDS:
                self.consecutive_e1 = self.consecutive_e2 = self.normal_count = 0
            self.last_rx = now
            if status == 0xE2:
                self.consecutive_e2 += 1
                self.consecutive_e1 = 0
                self.normal_count = 0
                if self.consecutive_e2 >= self.PROTECTION_COUNT:
                    self.hard_fault = "0xe2_persistent"
                    self.needs_ack = True
                    self._persist()
            elif status == 0xE1:
                # E1 is diagnostic only: never latch, persist, or inhibit a target.
                # It cannot clear an existing E2/unknown fault either.
                self.consecutive_e1 += 1
                self.consecutive_e2 = 0
                self.normal_count = 0
            else:
                self.consecutive_e1 = self.consecutive_e2 = 0
                self.normal_count = self.normal_count + 1 if status == 0 else 0

    def reason(self):
        with self.lock:
            if self.path is None:
                return "Arm identity not verified"
            if self.hard_fault is not None:
                return "Confirmed firmware thermal fault: " + self.hard_fault
            if self.path.exists() and self.legacy_record is None:
                return "Stored thermal fault requires service"
            if self.last_rx is None or self.clock() - self.last_rx > self.FRESH_SECONDS:
                return "No fresh joint-status frame"
            if self.needs_ack:
                return "Stored warning: healthy recheck and acknowledgement required"
            return None

    def acknowledge_recovery(self, temperatures):
        """Explicit recovery only for old one-frame thermal records.

        Does not clear confirmed ten-frame E1/E2 or an unknown record.
        Caller must supply a freshly queried ten-servo temperature report.
        Never sends or retries a movement command.
        """
        with self.lock:
            if self.hard_fault is not None or self.path is None:
                raise RuntimeError("Confirmed fault cannot be acknowledged as a transient warning")
            if self.last_rx is None or self.clock() - self.last_rx > self.FRESH_SECONDS:
                raise RuntimeError("Fresh status required")
            if self.normal_count < self.RECOVERY_COUNT or self.consecutive_e1 or self.consecutive_e2:
                raise RuntimeError("Ten consecutive new idle frames required")
            if (len(temperatures) != 10 or any(not math.isfinite(t) or not 0 <= t < 55
                                             for t in temperatures)):
                raise RuntimeError("Fresh, plausible cool temperature report required")
            if self.path.exists():
                if self.legacy_record is None or self.path.read_bytes() != self.legacy_record:
                    raise RuntimeError("Only an old single-frame record can be migrated")
                archive = self.path.parent / "archive"
                archive.mkdir(exist_ok=True)
                self.path.rename(archive / (self.path.name + "." + uuid.uuid4().hex + ".thermal-history"))
            self.legacy_record = None
            self.needs_ack = False
