"""Shared device connection and independently loadable collect/inference sessions."""
from collections import deque
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import threading
import time

from .bridge import PAUSE
from .profiles import ROOT, catalog, channels, config
from .runtimes import resolve_runtime
from .session import Preview, Session, check_channels, lock_devices
from .source import SourceSelector


class Studio:
    def __init__(self):
        self.lock = threading.RLock()
        self.preview = Preview()
        self.runners = {mode: Session(self.preview, self._source_exited) for mode in ("collect", "infer")}
        self.host = self.directory = self.selector = None
        self.connected = {}
        self.robot_exit = None
        self.locks = []
        self.runtime_dir = tempfile.TemporaryDirectory(prefix="vla-studio-")
        self.ready_file = Path(self.runtime_dir.name) / "ready"
        self.logs = deque(maxlen=100)
        self.selected = None

    def _release_idle_locks(self):
        if not self.host and not any(r.proc for r in self.runners.values()):
            for file in self.locks:
                file.close()
            self.locks.clear()

    def _source_exited(self, sid):
        with self.lock:
            if self.selected == sid:
                self.selected = None
                if self.host and self.host.poll() is None and self.robot_type:
                    try:
                        self.selector.select(None)
                    except (OSError, ValueError) as exc:
                        self.logs.append(str(exc))
            self._release_idle_locks()

    def _compatible_profile(self, profile, opts):
        for runner in self.runners.values():
            if runner.proc and (runner.robot != profile or any(
                runner.options.get(k) and runner.options[k] != opts.get(k)
                for k in ("task", "on_success"))):
                raise ValueError("Use the robot and task selected for the loaded source")

    def connect(self, opts):
        profile = opts["robot"]
        rows = channels(profile)
        robots = [row for row in rows if ".deploy.robot." in row[1]]
        if len(robots) > 1:
            raise ValueError("Select a single robot or a bimanual robot profile")
        settings = catalog()["robot_options"].get(profile, {})
        tasks = settings.get("tasks", [])
        task = opts.get("task") or (tasks[0] if tasks else "")
        behavior = opts.get("on_success", "continue")
        if (tasks and task not in tasks) or behavior not in ("continue", "pause"):
            raise ValueError("Invalid task configuration")
        with self.lock:
            if self.host:
                raise ValueError("Robot is already connected")
            self._compatible_profile(profile, dict(task=task, on_success=behavior))
            runtime = resolve_runtime(profile)
            runtime.probe()
            self.preview.stop()
            try:
                lock_devices(profile, self.locks)
                check_channels(name for name, _, _ in rows)
                self.directory = tempfile.TemporaryDirectory(prefix="vla-robot-")
                (Path(self.directory.name) / "selector").write_bytes(bytes(3))
                self.selector = SourceSelector(self.directory.name)
                env = dict(runtime.env, VLASTUDIO_WEB_GATE="managed", VLASTUDIO_WEB_PROFILE=profile,
                           VLASTUDIO_SOURCE_DIRECTORY=self.directory.name, VLASTUDIO_ROBOT_READY_FILE=str(self.ready_file),
                           VLASTUDIO_DEVICE_SETUP="vlastudio.webui.bridge.configure_device")
                host = subprocess.Popen(runtime.module("vlastudio.deploy.device_host"), cwd=ROOT,
                    env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=tuple(f.fileno() for f in self.locks))
                self.host = host
                self.robot_exit = None
                self.logs.clear()
                threading.Thread(target=self._watch, args=(host,), daemon=True).start()
                from vlastudio.deploy.utils import load_config, coerce_finalize_deploy_list
                manifest = coerce_finalize_deploy_list(load_config(str(config("robot", profile))))
                manifest = [row for row in manifest if "visualizer" not in row["type"].lower()]
                for row in manifest:
                    if tasks and ".deploy.robot." in row["type"]:
                        row["args"].update(task=task, on_success=behavior)
                host.stdin.write((json.dumps(manifest) + "\n").encode())
                host.stdin.flush()
                self.connected = dict(robot=profile, task=task, on_success=behavior,
                                      id=os.path.basename(self.directory.name), pid=host.pid)
                self.robot_type = robots[0][1] if robots else ""
                self.preview.start(profile, runtime)
            except BaseException:
                self._close(stop_sources=False)
                raise

    def _watch(self, host):
        for line in host.stdout:
            self.logs.append(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line.decode(errors="replace").rstrip())[-2000:])
        code = host.wait()
        with self.lock:
            if self.host is host:
                self.logs.append(f"Device host exited ({code})")
                self._close()

    def _close(self, stop_sources=True):
        if stop_sources:
            for runner in self.runners.values():
                runner.shutdown()
        host, self.host = self.host, None
        code = None
        if host is not None:
            host.stdin.close()
            code = host.wait()
            self.robot_exit = code
            self.logs.append(f"Robot shutdown exit code {code}")
        self.preview.stop()
        if self.selector:
            self.selector.control.close()
            self.selector = None
        if self.directory:
            self.directory.cleanup()
            self.directory = None
        self._release_idle_locks()
        self.ready_file.unlink(missing_ok=True)
        self.connected = {}
        self.selected = None
        return code

    def shutdown(self):
        with self.lock:
            self._close()

    def ready(self):
        return bool(self.host and self.host.poll() is None and self.ready_file.exists())

    def _source(self, mode, opts):
        tele = opts.get("teleop") if mode == "collect" else None
        channel = "policy_control_shm" if mode == "infer" else None
        settings = {}
        if tele:
            from vlastudio.utils.extensions import resolve
            channel, typ, params = channels(tele, "teleop")[0]
            describe = getattr(resolve(typ), "action_source_settings", None)
            if describe:
                settings = describe(params)
        return dict(kind=mode, channel=channel, **settings)

    def start(self, mode, opts):
        if mode not in self.runners:
            raise ValueError("Unknown mode")
        runtime = resolve_runtime(opts["robot"])
        info = runtime.probe(check_cuda=mode == "infer")
        with self.lock:
            if opts.get("connection") != self.connected.get("id"):
                raise ValueError("Robot connection changed")
            if self.host and opts["robot"] != self.connected["robot"]:
                raise ValueError("Select the connected robot")
            runner = self.runners[mode]
            if self.host:
                opts = dict(opts, task=self.connected["task"], on_success=self.connected["on_success"])
            opts.setdefault("on_success", "continue")
            self._compatible_profile(opts["robot"], opts)
            descriptor = self._source(mode, opts)
            other = self.runners["infer" if mode == "collect" else "collect"]
            if opts.get("output") and other.proc and opts["output"] == other.output:
                raise ValueError("Use a different output dataset for each loaded source")
            try:
                lock_devices(opts["robot"], self.locks, quest=(opts.get("teleop") or "").startswith("quest"))
                runner.start(mode, opts, self.ready_file, runtime, info)
            except BaseException:
                self._release_idle_locks()
                raise
            runner.descriptor = dict(descriptor, mailbox=str(runner.mailbox_path))

    def _release_source(self, runner):
        if self.selected == runner.sid and self.robot_type:
            self.selector.select(None)
        if self.selected == runner.sid:
            self.selected = None

    def action(self, mode, sid, action, real_task=None):
        with self.lock:
            runner = self.runners[mode]
            with runner.lock:
                state = runner.state()
                if sid != runner.sid or action not in state["actions"]:
                    raise ValueError("Source state changed; refresh and retry")
                if action != "stop" and not self.ready():
                    raise ValueError("Robot is not ready")
                if action == "stop":
                    if state["phase"] in ("recording", "record_paused", "save"):
                        raise ValueError("Finish and save or discard the current episode first")
                    runner.box[0] = PAUSE
                    self._release_source(runner)
                if action in ("run", "tele_run", "reset", "home") and self.selected != sid:
                    other = self.runners["infer" if mode == "collect" else "collect"]
                    other_state = other.state()
                    if other_state["phase"] in ("recording", "record_paused", "save", "homing", "busy", "stopping"):
                        raise ValueError("Finish the current episode or pending operation before switching")
                    if other_state["controls_enabled"] or other_state["phase"] == "running":
                        pause = "pause" if mode == "collect" else "tele_pause"
                        other.action(other.sid, pause)
                        deadline = time.monotonic() + 4
                        while other.proc and not (other.box[1] == PAUSE and
                                (pause == "tele_pause" or other.state()["phase"] == "paused")):
                            if time.monotonic() >= deadline:
                                raise ValueError("Waiting for the previous source to pause; retry after it pauses")
                            time.sleep(.01)
                    self.selected = None
                    if self.robot_type:
                        self.selector.select(runner.descriptor)
                    elif mode == "infer":
                        runner.mailbox_path.with_name("action-contract").write_text("{}")
                    self.selected = sid
                runner.action(sid, action, real_task=real_task)

    def disconnect(self, token, all_sources=False):
        with self.lock:
            if not self.host or self.connected.get("id") != token:
                raise ValueError("Robot connection changed")
            if all_sources:
                collect = self.runners["collect"].state()
                if collect["phase"] in ("recording", "record_paused", "save"):
                    raise ValueError("Finish and save or discard the current episode first")
            elif any(r.proc is not None for r in self.runners.values()):
                raise ValueError("Unload teleoperation and policy before disconnecting the robot")
            code = self._close()
            if code:
                raise ValueError(f"Robot shutdown was not confirmed (exit {code}); check device logs")

    def state(self, mode):
        with self.lock:
            state = self.runners[mode].state()
            state["robot_exit"] = self.robot_exit
            state["prepared"] = next((r.options for r in self.runners.values() if r.proc), None)
            state["connection"] = dict(self.connected, ready=self.ready()) if self.host else None
            state["loaded"] = [k for k,r in self.runners.items() if r.proc]
            state["logs"] = list(self.logs)[-45:] + state["logs"]
            return state
