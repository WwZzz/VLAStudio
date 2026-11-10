"""Private, session-scoped mailbox inherited only by the launched Linux workers."""
import json
import mmap
import os
import select
import sys

PHASES = ("loading", "ready", "recording", "save", "infer_ready", "running", "paused", "busy", "record_paused", "waiting_robot")
PAUSE, RUN, RESET_A, RESET_B, RESETTING, RESET_AND_RESUME = range(6)


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


def episode_options(start_command):
    """Optional per-episode metadata carried by a web start command."""
    if os.environ.get("VLASTUDIO_WEB_FD") and start_command:
        return json.loads(start_command)
    return {}


def wait_for_robot():
    """A preloaded source may own its inputs before the robot opens its sensors."""
    path = os.environ.get("VLASTUDIO_ROBOT_READY_FILE")
    if path and not os.path.exists(path):
        import time
        phase("waiting_robot")
        while not os.path.exists(path):
            time.sleep(.1)


def request_home():
    if _box is None or os.environ.get("VLASTUDIO_WEB_GATE") != "infer" or _box[4] == 255:
        return False
    _box[0] = RESET_B if _box[0] == RESET_A else RESET_A
    return True


class WebKeys:
    """Line-based equivalent of KBHit on the supervisor's private stdin pipe."""
    resumable_recording = True

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


def configure_device(device):
    """Attach WebUI control only to robots launched by this session."""
    from vlastudio.deploy.robot.base import BaseRobot
    if isinstance(device, BaseRobot) and os.environ.get("VLASTUDIO_WEB_GATE"):
        directory = os.environ.get("VLASTUDIO_SOURCE_DIRECTORY")
        if directory:
            from .source import SourceGate
            device.control_shm_name = None
            device.set_action_filter(SourceGate(directory))
        else:
            device.set_action_filter(InputGate())
        from .markers import bind_marker_provider
        from .profiles import catalog
        settings = catalog()['robot_options'].get(os.environ.get('VLASTUDIO_WEB_PROFILE', ''), {})
        bind_marker_provider(device, settings.get('marker_provider'))


class InputGate:
    """Runs inside the robot process. No HTTP, socket or file reads per action."""
    def __init__(self, box=None, policy=None):
        self.box = mailbox() if box is None else box
        self.policy = os.environ.get("VLASTUDIO_WEB_GATE") == "infer" if policy is None else policy
        self.previous = None
        self.reanchor = False

    def filter(self, robot, message):
        command = self.box[0]
        if command != self.previous:
            self.previous = command
            self.box[4] = command
            if command == RESET_AND_RESUME:
                self.box[1] = RESETTING
                # A successful reset must be acknowledged by the device. Some
                # legacy drivers only submit a target and cannot confirm arrival.
                reset = getattr(robot, "reset_for_teleoperation", None)
                if reset is None:
                    return {"cmd": "reset", "go_home": True}
                confirmed = reset()
                if self.box[0] == command:
                    self.box[0] = RUN if confirmed else PAUSE
                return None
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
        if command == RESET_AND_RESUME:
            self.box[0] = PAUSE
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
