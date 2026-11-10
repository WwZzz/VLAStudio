"""Switch a robot's input reader at a control-loop boundary."""
import json
import mmap
import time
from pathlib import Path

from .bridge import InputGate


def mapping(path):
    with path.open("r+b") as file:
        return mmap.mmap(file.fileno(), 0)


class SourceSelector:
    """One supervisor writes requests; one robot acknowledges completion."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.control = mapping(self.directory / "selector")

    def select(self, descriptor):
        path = self.directory / "source.json"
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(descriptor))
        pending.replace(path)
        token = (self.control[0] + 1) % 256
        self.control[0] = token
        deadline = time.monotonic() + 4
        while self.control[1] != token:
            if time.monotonic() >= deadline:
                raise ValueError("Robot did not acknowledge source selection")
            time.sleep(.005)
        if self.control[2]:
            raise ValueError((self.directory / "source-error").read_text())


class SourceGate(SourceSelector):
    def __init__(self, directory):
        super().__init__(directory)
        self.reader = self.gate = None

    def _unbind(self, robot):
        if self.reader is not None:
            self.reader.destroy()
            self.reader = None
        if self.gate is not None:
            self.gate.box.close()
            self.gate = None
        bus = getattr(robot, "bus", None)
        if getattr(bus, "armed", False):
            bus.cancel_pending_target()

    def _bind(self, robot, source):
        from vlastudio.deploy.shm_utils import SharedMemoryChannel
        self._unbind(robot)
        hook = getattr(robot, "prepare_action_source", None)
        contract = hook(source) if hook else None
        if source is None:
            return
        if not hook and source.get("control_mode"):
            if hasattr(robot, "control_mode") and source["control_mode"] != robot.control_mode:
                raise ValueError("Input mode does not match the connected robot profile")
            if robot.get_action_dim() != 7 * source["arms"]:
                raise ValueError("Cartesian input dimension does not match the connected robot")
        if source["kind"] == "infer":
            spec = dict(contract or {}, size=int(robot.get_action_dim()))
            Path(source["mailbox"]).with_name("action-contract").write_text(json.dumps(spec))
        if source["channel"]:
            self.reader = SharedMemoryChannel(source["channel"], timeout=0.)
            if self.reader.shm is None:
                raise ValueError("Input device has no live action channel")
            self.reader.read(skip_unchanged=True)
        box = mapping(Path(source["mailbox"]))
        self.gate = InputGate(box, source["kind"] == "infer")
        self.gate.filter(robot, None)

    def filter(self, robot, unused):
        token = self.control[0]
        if token != self.control[1]:
            try:
                self._bind(robot, json.loads((self.directory / "source.json").read_text()))
                self.control[2] = 0
            except Exception as exc:
                self._unbind(robot)
                (self.directory / "source-error").write_text(str(exc))
                self.control[2] = 1
            self.control[1] = token
            return None
        if self.gate is None:
            return None
        # Read while paused too: old frames are consumed before the RUN edge.
        msg = self.reader.read(skip_unchanged=True, copy=True) if self.reader else None
        return self.gate.filter(robot, msg)
