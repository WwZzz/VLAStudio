import base64
import collections
import fcntl
import json
import mmap
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time

from .bridge import PHASES, PAUSE, RUN, RESET_A, RESET_B, RESETTING, RESET_AND_RESUME
from .profiles import ROOT, channels, config, catalog
from .tasks import get_task
from .runtimes import resolve_runtime


def module(name):
    return [sys.executable, "-u", "-m", name]


def command_line(mode, options, runtime=None):
    robot = options["robot"]
    base = "vlastudio.entrypoints."
    command_module = base + ("collect_data" if mode == "collect" else "eval_real")
    cmd = runtime.module(command_module) if runtime is not None else module(command_module)
    cmd += ["--robot", str(config("robot", robot))]
    settings = catalog()['robot_options'].get(robot, {})
    if mode not in settings.get('modes', ['collect', 'infer']):
        raise ValueError('Selected robot profile does not support this mode')
    if settings.get('tasks'):
        task = options.get('task') or settings['tasks'][0]
        if task not in settings['tasks']:
            raise ValueError('Unknown simulation task')
        cmd += ['--robot.args.task', task]
        behavior = options.get('on_success', 'continue')
        if behavior not in ('continue', 'pause'):
            raise ValueError('Invalid success behavior')
        cmd += ['--robot.args.on_success', behavior]
        cmd += ['--task', task]
        if mode == 'infer':
            cmd += ['--robot.args.input_mode', 'native']
    elif options.get('real_task'):
        task = get_task(options['real_task'])
        cmd += ['--task=' + (task['instruction'] or task['name'])]
    rate = int(options.get("rate", 30))
    if not 1 <= rate <= 120:
        raise ValueError("频率范围为 1–120 Hz")
    output = options.get("output", "").strip()
    if output and not re.fullmatch(r"[\w-][\w.-]{0,79}", output):
        raise ValueError("数据集名只支持字母、数字、中文、点、横线和下划线")
    if mode == "collect":
        if not output:
            raise ValueError("请填写数据集名")
        cmd += ["--frequency", str(rate), "--dataset-format", "hdf5", "--hdf5-compression", "lzf"]
        if options.get("teleop"):
            cmd += ["--teleop", str(config("teleop", options["teleop"]))]
        if options.get("sync"):
            cmd += ["--sync-config", str(config("recording", options["sync"]))]
    elif mode == "infer":
        device = options.get('device', 'auto')
        if device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('Policy device must be auto, cpu or cuda')
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cmd += ['--device', device]
        policy = options.get("policy", "").strip()
        if not policy or policy.startswith("-") or len(policy) > 1024:
            raise ValueError("请填写策略服务器地址或服务端的模型路径")
        if policy.startswith("__dummy") and robot != "dummy":
            raise ValueError("模拟策略仅允许搭配 dummy 设备")
        cmd += ["-m", policy, "-pr", str(rate), "-am", str(config("action_manager", options.get("manager", "basic")))]
    else:
        raise ValueError("Unknown mode")
    if output:
        dest = (ROOT / "data/webui" / output).resolve()
        if dest.parent != (ROOT / "data/webui").resolve():
            raise ValueError("Invalid dataset path")
        cmd += ["-o", str(dest)]
    return cmd


class Preview:
    def __init__(self):
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.proc = None
        self.clear()

    def clear(self):
        self.images, self.axes, self.models, self.stamps = {}, {}, {}, {}
        self.devices = {}
        self.profile = ""
        self.error = ""
        self.last_view = 0.

    def start(self, profile, runtime=None):
        channels(profile)
        if runtime is None:
            runtime = resolve_runtime(profile)
            runtime.probe()
        self.stop()
        with self.lock:
            env = dict(runtime.env, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            proc = subprocess.Popen(runtime.module("vlastudio.webui.monitor") + [profile], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env, start_new_session=True)
            os.set_blocking(proc.stdin.fileno(), False)
            self.proc, self.profile = proc, profile
            threading.Thread(target=self._read, args=(proc,), daemon=True).start()
            threading.Thread(target=self._errors, args=(proc,), daemon=True).start()

    def _errors(self, proc):
        for line in proc.stderr:
            with self.lock:
                if proc is self.proc:
                    self.error = line.decode(errors="replace").strip()[-300:]

    def _read(self, proc):
        for line in proc.stdout:
            try:
                p = json.loads(line)
                with self.lock:
                    if proc is not self.proc:
                        return
                    for name, image in p["images"].items():
                        stamp = p["stamps"][name.split("/", 1)[0]]
                        self.images[name] = (str(stamp), base64.b64decode(image))
                    if p["images"]:
                        self.changed.notify_all()
                    self.models.update(p["models"])
                    self.stamps.update(p["stamps"])
                    self.devices.update(p.get('devices', {}))
                    for key, row in p["axes"].items():
                        self.axes.setdefault(key, collections.deque(maxlen=240)).append(row)
            except (ValueError, KeyError):
                continue

    def latest_image(self, name, after):
        with self.changed:
            self._touch()
            if name not in self.images:
                return None
            if after is not None:
                self.changed.wait_for(
                    lambda: name not in self.images or self.images[name][0] != after,
                    timeout=.8,
                )
            return self.images.get(name)

    def _touch(self):
        now = time.monotonic()
        if self.proc is not None and now - self.last_view >= 1.:
            try:
                os.write(self.proc.stdin.fileno(), b".")
            except (BlockingIOError, BrokenPipeError):
                pass
            self.last_view = now

    def state(self):
        with self.lock:
            now = time.monotonic()
            self._touch()
            return dict(profile=self.profile, now=now, images=list(self.images), models=self.models.copy(),
                        devices={k:dict(v) for k,v in self.devices.items()},
                        axes={k:list(v) for k,v in self.axes.items()}, error=self.error,
                        age={k:round(time.monotonic()-v,2) for k,v in self.stamps.items()})

    def stop(self):
        with self.lock:
            proc, self.proc = self.proc, None
            self.clear()
            self.changed.notify_all()
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()  # preview owns only read-only SHM handles
                proc.wait()


KEYS = {
    "ready": {"record": ""}, "recording": {"record_pause": "", "finish": "finish"},
    "record_paused": {"record_resume": "", "finish": "finish"}, "save": {"save": "", "discard": "x"},
    "infer_ready": {"run": ""}, "running": {"pause": ""}, "paused": {"run": "", "reset": "r", "save": "s"},
}


def lock_devices(robot, locks, quest=False):
    cache = Path.home()/".cache"
    cache.mkdir(exist_ok=True)
    names = ["vlastudio-webui-session.lock"]
    if "rebot" in robot:
        names.append("vlastudio-rebot-session.lock")
    if "alicia" in robot or "head" in robot or quest:
        names.append("vlastudio-bi-alicia-teleop.lock")
    for name in sorted(names):
        if any(Path(file.name).name == name for file in locks):
            continue
        f = (cache/name).open("a")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            raise ValueError("设备正在被其他会话使用") from None
        locks.append(f)


def check_channels(names):
    for name in names:
        shm = Path("/dev/shm")/name
        if shm.exists() and subprocess.run(["fuser", str(shm)], capture_output=True).returncode == 0:
            raise ValueError(f"共享内存 {name} 正在使用")



class Session:
    def __init__(self, preview, on_exit):
        self.lock = threading.RLock()
        self.preview = preview
        self.on_exit = on_exit
        self.proc = self.box = None
        self.mode = self.robot = self.sid = ""
        self.logs = collections.deque(maxlen=230)
        self.locks = []
        self.pending = None
        self.stopping = False
        self.code = None
        self.quest_proc = None
        self.quest_hands = {}
        self.quest_samples = collections.deque(maxlen=64)
        self.quest_config = None

    def start(self, mode, opts, ready_file, runtime, info):
        opts = dict(opts)
        if mode == 'infer' and opts.get('device', 'auto') == 'auto':
            opts['device'] = info['device']
        cmd = command_line(mode, opts, runtime)
        with self.lock:
            if self.proc is not None:
                raise ValueError("已有会话运行中")
            self.mode, self.robot = mode, opts["robot"]
            self.options = {k:opts[k] for k in ("robot","teleop","sync","policy","manager","rate","output","task","on_success","real_task") if k in opts}
            rows = channels(self.robot)
            if sum(".deploy.robot." in typ for _, typ, _ in rows) > 1:
                raise ValueError("请选择单机器人或双臂组合配置，以便逐次确认暂停与回零")
            self.teleop = opts.get("teleop", "") if mode == "collect" else ""
            self.reset_ok = any(t.rsplit(".",1)[-1] in ("AliciaD","BiAliciaD","RebotB601RS") and
                                not a.get("read_only") for _,t,a in rows)
            from vlastudio.utils.extensions import resolve
            self.reset_ok = self.reset_ok or bool(catalog()["robot_options"].get(self.robot, {}).get("tasks")) or any(getattr(resolve(t), 'supports_reset', False) for _, t, _ in rows)
            keyboard_rows = channels(self.teleop, 'teleop') if self.teleop else []
            quest_rows = [(n, a) for n, t, a in keyboard_rows
                          if t == "vlastudio.deploy.teleoperator.quest3.Quest3Teleop"]
            self.quest_config = dict(quest_rows[0][1]) if quest_rows else None
            self.quest_hands = {}
            self.quest_samples.clear()
            self.keyboard_arms = next((int(a.get('arms', 1)) for _, t, a in keyboard_rows
                                       if getattr(resolve(t), 'browser_keyboard', False)), 0)
            self.keyboard_bindings = next((resolve(t).bindings_for_arms(int(a.get('arms', 1)), a.get('key_bindings'))
                                           for _, t, a in keyboard_rows if getattr(resolve(t), 'browser_keyboard', False)), [])
            self.keyboard_base_bindings = next((resolve(t).bindings_for_base(a.get('base_bindings'))
                                               for _, t, a in keyboard_rows if hasattr(resolve(t), 'bindings_for_base')), [])
            self.keyboard_axes = catalog()['robot_options'].get(self.robot, {}).get('keyboard_axes', list(range(7)))
            self.keyboard_keys = {key for arm in self.keyboard_bindings for axis, pair in enumerate(arm)
                                  if axis in self.keyboard_axes for key in pair}
            self.keyboard_keys.update(key for pair in self.keyboard_base_bindings for key in pair)
            self.keyboard_owner = None
            self.keyboard_last_seen = 0
            if self.keyboard_arms:
                expected = next((a.get('action', {}).get('arms') for _, _, a in rows
                                 if a.get('action', {}).get('arms')), None)
                if expected and self.keyboard_arms != expected:
                    raise ValueError('Keyboard arm count does not match robot configuration')
            self.output = opts.get("output", "").strip()
            self.observation_controls = False
            self.gated = mode == "collect" and (self.teleop or self.reset_ok)
            try:
                source_names = [n for n, _, _ in keyboard_rows] if mode == "collect" else ["policy_control_shm"]
                if set(source_names).intersection(n for n, _, _ in rows):
                    raise ValueError("Source and robot channel names must be distinct")
                check_channels(source_names)
                self.keyboard_dir = tempfile.TemporaryDirectory(prefix='vla-source-')
                self.mailbox_path = Path(self.keyboard_dir.name) / 'mailbox'
                with self.mailbox_path.open("w+b") as mailbox:
                    fd = mailbox.fileno()
                    os.ftruncate(fd, 5)
                    self.box = mmap.mmap(fd, 5)
                    self.box[:] = bytes([PAUSE,PAUSE,0,0,PAUSE])
                    env = dict(runtime.env, VLASTUDIO_WEB_FD=str(fd), PYTHONUNBUFFERED="1",
                               VLASTUDIO_WEB_PROFILE=self.robot,
                               PYTHONDONTWRITEBYTECODE="1",
                               VLASTUDIO_DEVICE_SETUP="vlastudio.webui.bridge.configure_device")
                    env["VLASTUDIO_ATTACHED_ROBOT"] = "1"
                    env["VLASTUDIO_ACTION_CONTRACT"] = str(self.mailbox_path.with_name("action-contract"))
                    env["VLASTUDIO_ROBOT_READY_FILE"] = str(ready_file)
                    self.keyboard_path = str(Path(self.keyboard_dir.name) / 'input.sock')
                    env['VLASTUDIO_KEYBOARD_SOCKET'] = self.keyboard_path
                    if self.gated:
                        env["VLASTUDIO_WEB_GATE"] = "collect"
                    elif mode == "infer":
                        env["VLASTUDIO_WEB_GATE"] = "infer"
                        if not self.output:
                            env["EVAL_REAL_NO_RECORD"] = "1"
                    self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                                pass_fds=(fd, *(f.fileno() for f in self.locks)))
            except BaseException:
                self._release()
                raise
            self.sid = secrets.token_hex(8)
            self.stopping, self.pending, self.code = False, None, None
            self.reset_after_pause = False
            self.episode = None
            self.retaking = False
            self.logs.clear()
            self.runtime = dict(name=runtime.name, python=info['python'])
            self.logs.append(f"Runtime: {runtime.name} ({info['python']})")
            if quest_rows:
                try:
                    self.quest_proc = subprocess.Popen(runtime.module("vlastudio.webui.quest_preview") +
                        [quest_rows[0][0] + "_preview"], cwd=ROOT, env=runtime.env,
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        start_new_session=True)
                    threading.Thread(target=self._read_quest, args=(self.quest_proc,), daemon=True).start()
                except OSError as error:
                    self.logs.append(f"Quest preview unavailable: {error}")
            threading.Thread(target=self._watch, args=(self.proc,), daemon=True).start()

    def _read_quest(self, proc):
        for line in proc.stdout:
            try:
                packet = json.loads(line)
                with self.lock:
                    if proc is not self.quest_proc:
                        return
                    for hand, sample in packet["hands"].items():
                        if hand in ("left", "right"):
                            sample = dict(sample, stamp=packet["stamp"], hand=hand)
                            self.quest_hands[hand] = sample
                            self.quest_samples.append(sample)
            except (KeyError, TypeError, ValueError):
                continue

    def _release(self):
        proc, self.quest_proc = self.quest_proc, None
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self.quest_hands = {}
        self.quest_samples.clear()
        directory = getattr(self, 'keyboard_dir', None)
        if directory is not None:
            directory.cleanup()
            self.keyboard_dir = None
        if self.box is not None:
            self.box.close()
            self.box = None
        for f in self.locks:
            f.close()
        self.locks.clear()

    def _watch(self, proc):
        for line in proc.stdout:
            clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line.decode(errors="replace").rstrip())
            with self.lock:
                match = re.search(r"(?:Press Enter to START episode |Recording episode |\[SessionRecorder\] (?:started (?:video-only )?episode |episode ))(\d+)", clean)
                if match:
                    self.episode = int(match[1])
                elif re.search(r"\[SessionRecorder\] (?:saved|discarded) episode ", clean):
                    self.episode = None
                self.logs.append(clean[-2000:])
        code = proc.wait()
        with self.lock:
            self.code = code
            self.logs.append(f"进程已退出 · code {code}")
            self.box[0] = PAUSE
            self._release()
            self.proc = None
            sid = self.sid
        self.on_exit(sid)

    def state(self):
        with self.lock:
            live = self.proc is not None
            phase = PHASES[self.box[2]] if live else "idle"
            if (live and self.reset_after_pause and phase == "paused"
                    and self.pending != self.box[3]):
                self.reset_after_pause = False
                self._key("reset", "paused")
            if live and self.pending == self.box[3]:
                phase = "busy"
            allowed = list(KEYS.get(phase, {})) if live else []
            collect_controls = live and self.mode == "collect" and phase in ("ready", "recording", "record_paused", "save")
            controls_enabled = False
            if live and self.mode == "collect":
                if self.gated:
                    controls_enabled = self.box[0] == self.box[1] == RUN or self.box[0] == RESET_AND_RESUME
                else:
                    controls_enabled = self.observation_controls
            if collect_controls and self.gated:
                ack, cmd = self.box[1], self.box[0]
                if ack == RUN and cmd == RUN:
                    allowed.append("tele_pause")
                    if self.reset_ok:
                        allowed.append("home")
                elif ack == PAUSE and self.box[4] == cmd and cmd != RUN:
                    allowed.append("tele_run")
                    if self.reset_ok:
                        allowed.append("home")
            if collect_controls:
                if not self.gated:
                    allowed.append("tele_pause" if controls_enabled else "tele_run")
                if not controls_enabled and not (self.retaking and phase in ("recording", "record_paused", "save")):
                    allowed = [a for a in allowed if a in ("tele_run", "home", "save", "discard")]
            if live and self.gated and self.box[1] == RESETTING:
                phase, allowed = "homing", []
            if live and self.mode == "infer" and phase == "running" and self.reset_ok:
                allowed.append("reset")
            if self.mode == "infer" and not self.output and "save" in allowed:
                allowed.remove("save")
            if (live and self.mode == "infer" and phase == "paused" and self.box[4] != 255
                    and (self.box[4] != self.box[0] or self.box[1] == RESETTING)):
                phase = "homing" if self.box[0] in (RESET_A, RESET_B) else "busy"
                allowed = []
            if live:
                allowed.append("stop")
            if self.stopping and live:
                phase, allowed = "stopping", []
            with self.preview.lock:
                blocked = [row for row in self.preview.devices.values() if row.get('paused')]
            simulation = dict(paused=bool(live and blocked), reason='')
            if simulation['paused']:
                simulation['reason'] = ('limit' if any(row.get('truncated') for row in blocked)
                                        else 'terminated' if any(row.get('terminated') for row in blocked)
                                        else 'success')
                allowed = [action for action in allowed if action not in ('record', 'record_resume', 'tele_run')]
            paired_leader = getattr(self, "descriptor", {}).get("align_leader", False)
            if not controls_enabled and not simulation["paused"] and not paired_leader and "home" in allowed:
                allowed.remove("home")
            tele = ""
            if live and self.mode == "collect":
                tele = "running" if controls_enabled else "paused"
                if simulation["paused"]:
                    tele = "blocked"
                elif self.gated and self.box[1] == RESETTING:
                    tele = "homing"
            return dict(id=self.sid, active=live, mode=self.mode, robot=self.robot, phase=phase,
                        simulation=simulation,
                        options=self.options if live else {}, episode=self.episode if live else None,
                        quest=dict(config=self.quest_config) if live and self.quest_config else None,
                        keyboard_arms=self.keyboard_arms if live else 0,
                        keyboard_bindings=self.keyboard_bindings if live else [],
                        keyboard_base_bindings=self.keyboard_base_bindings if live else [],
                        keyboard_axes=self.keyboard_axes if live else [],
                        tele=tele, controls_enabled=controls_enabled, actions=allowed, logs=list(self.logs), exit=self.code)

    def quest_preview(self, sid):
        with self.lock:
            if not self.proc or sid != self.sid or not self.quest_config:
                raise ValueError("Quest session changed")
            return dict(now=time.monotonic(), samples=list(self.quest_samples))

    def keyboard(self, data):
        with self.lock:
            if not self.proc or data.get('id') != self.sid or not self.keyboard_arms:
                raise ValueError('No matching keyboard session')
            keys = data.get('keys')
            if not isinstance(keys, list) or len(keys) > len(self.keyboard_keys) or any(not isinstance(k, str) or k not in self.keyboard_keys for k in keys):
                raise ValueError('Invalid keys')
            owner = data.get('owner', '')
            if not isinstance(owner, str) or not 1 <= len(owner) <= 80:
                raise ValueError('Missing keyboard controller identifier')
            now = time.monotonic()
            if self.keyboard_owner != owner and now - self.keyboard_last_seen < .35:
                return
            self.keyboard_owner = owner
            self.keyboard_last_seen = now
            if self.box[0] != RUN:
                keys = []
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as connection:
                connection.setblocking(False)
                try:
                    connection.sendto(json.dumps(dict(keys=keys)).encode(), self.keyboard_path)
                except (FileNotFoundError, BlockingIOError, ConnectionRefusedError):
                    pass

    def action(self, sid, action, real_task=None):
        with self.lock:
            state = self.state()
            if sid != self.sid or action not in state["actions"]:
                raise ValueError("会话状态已改变，请重试")
            if action == "stop":
                self.box[0] = PAUSE
                self.stopping = True
                self.proc.terminate()
            elif action in ("tele_pause", "tele_run", "home"):
                if action == "home":
                    self.box[1] = RESETTING
                    if state["controls_enabled"]:
                        self.box[0] = RESET_AND_RESUME
                    else:
                        self.box[0] = RESET_B if self.box[0] == RESET_A else RESET_A
                    if state["phase"] in ("recording", "record_paused"):
                        self._key("finish", state["phase"])
                elif self.gated:
                    self.box[0] = RUN if action == "tele_run" else PAUSE
                else:
                    self.observation_controls = action == "tele_run"
                if action == "tele_pause" and state["phase"] == "recording":
                    self._key("record_pause", "recording")
            else:
                if self.mode == "infer" and action == "reset" and state["phase"] == "running":
                    self.reset_after_pause = True
                    self.box[0] = PAUSE
                    self._key("pause", "running")
                    return
                if self.mode == "infer" and action == "pause":
                    self.box[0] = PAUSE  # ignore late policy output while inference returns
                if action == "record":
                    self.retaking = False
                command = None
                if action == "record" and self.mode == "collect" and real_task is not None:
                    if catalog()["robot_options"].get(self.robot, {}).get("tasks"):
                        raise ValueError("Simulation task is bound to the environment")
                    instruction = ""
                    if real_task != "":
                        task = get_task(real_task)
                        instruction = task["instruction"] or task["name"]
                    command = json.dumps({"task": instruction}, ensure_ascii=False)
                self._key(action, state["phase"], command)

    def start_retake(self, index, version, task):
        self._key("record", "ready", json.dumps(dict(index=index, version=version, task=task)))
        self.retaking = True

    def _key(self, action, phase, command=None):
        self.pending = self.box[3]
        if command is None:
            command = KEYS[phase][action]
        self.proc.stdin.write((command + "\n").encode())
        self.proc.stdin.flush()

    def request_shutdown(self):
        with self.lock:
            proc = self.proc
            if proc is not None and not self.stopping:
                self.box[0] = PAUSE
                self.stopping = True
                proc.terminate()
            return proc

    def shutdown(self):
        proc = self.request_shutdown()
        if proc is not None:
            proc.wait()
