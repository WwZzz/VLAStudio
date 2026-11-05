"""Private, session-scoped mailbox inherited only by the launched Linux workers."""
import mmap
import os
import select
import sys

PHASES = ("loading", "ready", "recording", "save", "infer_ready", "running", "paused", "busy")
PAUSE, RUN, RESET_A, RESET_B, RESETTING = range(5)


def mailbox():
    fd = os.environ.get("VLASTUDIO_WEB_FD")
    return mmap.mmap(int(fd), 5) if fd else None


_box = None


def phase(name):
    global _box
    if _box is None and os.environ.get("VLASTUDIO_WEB_FD"):
        _box = mailbox()
    if _box is not None:
        if name == "running":
            _box[0] = RUN
        elif name == "paused" and _box[0] == RUN:
            _box[0] = PAUSE
        _box[2] = PHASES.index(name)
        _box[3] = (_box[3] + 1) % 256


def keyboard(native):
    return WebKeys() if os.environ.get("VLASTUDIO_WEB_FD") else native()


def request_home():
    if _box is None or os.environ.get("VLASTUDIO_WEB_GATE") != "infer" or _box[4] == 255:
        return False
    _box[0] = RESET_B if _box[0] == RESET_A else RESET_A
    return True


class WebKeys:
    """Line-based equivalent of KBHit on the supervisor's private stdin pipe."""
    def get_input(self):
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        line = sys.stdin.readline()
        if not line:
            raise KeyboardInterrupt
        return line.strip()

    get_line = get_input

    def clear_line_buffer(self):
        pass

    set_curses_term = clear_line_buffer
    set_normal_term = clear_line_buffer


class InputGate:
    """Runs inside the robot process. No HTTP, socket or file reads per action."""
    def __init__(self):
        self.box = mailbox()
        self.policy = os.environ.get("VLASTUDIO_WEB_GATE") == "infer"
        self.previous = None
        self.reanchor = False

    def filter(self, robot, message):
        command = self.box[0]
        if command != self.previous:
            self.previous = command
            self.box[4] = command
            if command in (RESET_A, RESET_B):
                self.box[1] = RESETTING
                return {"cmd": "reset", "go_home": True}
            if command == PAUSE:
                bus = getattr(robot, "bus", None)
                if getattr(bus, "armed", False) and hasattr(bus, "cancel_pending_target"):
                    bus.cancel_pending_target()
            self.reanchor = command == RUN
            self.box[1] = command
            return None  # discard input already queued across a transition
        if command != RUN:
            self.box[1] = PAUSE
            if self.policy and message and message.get("cmd") == "reset":
                self.box[1] = RESETTING
                return message
            return None
        if self.reanchor and message:
            if message.get("unsqueeze_active") and not message.get("anchor_just_set"):
                return None
            self.reanchor = False
        return message
