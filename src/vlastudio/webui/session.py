import collections
import fcntl
import json
import mmap
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time

from .bridge import PHASES, PAUSE, RUN, RESET_A, RESET_B, RESETTING
from .profiles import ROOT, channels, config


def module(name):
    return [sys.executable, "-u", "-m", name]


def command_line(mode, options):
    robot = options["robot"]
    base = "vlastudio.entrypoints."
    cmd = module(base + ("collect_data" if mode == "collect" else "eval_real"))
    cmd += ["--robot", str(config("robot", robot))]
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
        policy = options.get("policy", "").strip()
        if not policy or policy.startswith("-") or len(policy) > 1024:
            raise ValueError("请填写策略服务器地址或 Jetson 上的模型路径")
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
        self.proc = None
        self.clear()

    def clear(self):
        self.images, self.axes, self.models, self.stamps = {}, {}, {}, {}
        self.profile = ""
        self.error = ""
        self.last_view = 0.

    def start(self, profile):
        channels(profile)
        self.stop()
        with self.lock:
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            proc = subprocess.Popen(module("vlastudio.webui.monitor") + [profile], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
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
                    self.images.update(p["images"])
                    self.models.update(p["models"])
                    self.stamps.update(p["stamps"])
                    for key, row in p["axes"].items():
                        self.axes.setdefault(key, collections.deque(maxlen=240)).append(row)
            except (ValueError, KeyError):
                continue

    def state(self):
        with self.lock:
            now = time.monotonic()
            if self.proc is not None and now - self.last_view >= 1.:
                try:
                    os.write(self.proc.stdin.fileno(), b".")
                except (BlockingIOError, BrokenPipeError):
                    pass
                self.last_view = now
            return dict(profile=self.profile, now=now, images=list(self.images), models=self.models.copy(),
                        axes={k:list(v) for k,v in self.axes.items()}, error=self.error,
                        age={k:round(time.monotonic()-v,2) for k,v in self.stamps.items()})

    def stop(self):
        with self.lock:
            proc, self.proc = self.proc, None
            self.clear()
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()  # preview owns only read-only SHM handles
                proc.wait()


KEYS = {
    "ready": {"record": ""}, "recording": {"finish": ""}, "save": {"save": "", "discard": "x"},
    "infer_ready": {"run": ""}, "running": {"pause": ""}, "paused": {"run": "", "reset": "r", "save": "s"},
}


class Session:
    def __init__(self):
        self.lock = threading.RLock()
        self.preview = Preview()
        self.proc = self.box = None
        self.mode = self.robot = self.sid = ""
        self.logs = collections.deque(maxlen=230)
        self.locks = []
        self.pending = None
        self.stopping = False
        self.code = None

    def _resources(self, robot, teleop):
        cache = Path.home()/".cache"
        cache.mkdir(exist_ok=True)
        names = ["vlastudio-webui-session.lock"]
        if "rebot" in robot:
            names.append("vlastudio-rebot-session.lock")
        if "alicia" in robot or "head" in robot or teleop.startswith("quest"):
            names.append("vlastudio-bi-alicia-teleop.lock")
        for name in sorted(names):
            f = (cache/name).open("a")
            self.locks.append(f)
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("设备正在被其他会话使用") from None
        # Native CLI sessions may predate the session locks. Never unlink a live device channel.
        names = [n for n, _, _ in channels(robot)]
        if teleop:
            names.extend(n for n, _, _ in channels(teleop, "teleop"))
        if self.mode == "infer":
            names.append("policy_control_shm")
        for name in names:
            shm = Path("/dev/shm")/name
            if shm.exists() and subprocess.run(["fuser", str(shm)], capture_output=True).returncode == 0:
                raise ValueError(f"共享内存 {name} 正在使用")

    def start(self, mode, opts):
        cmd = command_line(mode, opts)
        with self.lock:
            if self.proc is not None:
                raise ValueError("已有会话运行中")
            self.preview.stop()
            self.mode, self.robot = mode, opts["robot"]
            self.options = {k:opts[k] for k in ("robot","teleop","sync","policy","manager","rate","output") if k in opts}
            rows = channels(self.robot)
            if sum(".deploy.robot." in typ for _, typ, _ in rows) > 1:
                raise ValueError("请选择单机器人或双臂组合配置，以便逐次确认暂停与回零")
            self.teleop = opts.get("teleop", "") if mode == "collect" else ""
            self.reset_ok = any(t.rsplit(".",1)[-1] in ("AliciaD","BiAliciaD","RebotB601RS") and
                                not a.get("read_only") for _,t,a in rows)
            self.output = opts.get("output", "")
            self.gated = mode == "collect" and (self.teleop or self.reset_ok)
            try:
                self._resources(self.robot, self.teleop)
                with tempfile.TemporaryFile(dir="/dev/shm") as mailbox:
                    fd = mailbox.fileno()
                    os.ftruncate(fd, 5)
                    self.box = mmap.mmap(fd, 5)
                    self.box[:] = bytes([PAUSE,255,0,0,255])
                    env = dict(os.environ, VLASTUDIO_WEB_FD=str(fd), PYTHONUNBUFFERED="1",
                               PYTHONDONTWRITEBYTECODE="1")
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
            self.logs.clear()
            threading.Thread(target=self._watch, args=(self.proc,), daemon=True).start()
            self.preview.start(self.robot)

    def _release(self):
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
                self.logs.append(clean[-2000:])
        code = proc.wait()
        with self.lock:
            self.code = code
            self.logs.append(f"进程已退出 · code {code}")
            self._release()
            self.proc = None
            self.preview.stop()

    def state(self):
        with self.lock:
            live = self.proc is not None
            phase = PHASES[self.box[2]] if live else "idle"
            if live and self.pending == self.box[3]:
                phase = "busy"
            allowed = list(KEYS.get(phase, {})) if live else []
            if live and self.gated and phase in ("ready","recording","save"):
                ack, cmd = self.box[1], self.box[0]
                if ack == RUN and cmd == RUN:
                    allowed.append("tele_pause")
                elif ack == PAUSE and self.box[4] == cmd and cmd != RUN:
                    allowed.append("tele_run")
                    if self.reset_ok:
                        allowed.append("home")
                if self.teleop and ack != RUN and "record" in allowed:
                    allowed.remove("record")
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
            return dict(id=self.sid, active=live, mode=self.mode, robot=self.robot, phase=phase,
                        options=self.options if live else {},
                        tele=("running" if self.box[1]==RUN else "homing" if self.box[1]==RESETTING else "paused")
                        if live and self.gated else "", actions=allowed, logs=list(self.logs), exit=self.code)

    def action(self, sid, action):
        with self.lock:
            state = self.state()
            if sid != self.sid or action not in state["actions"]:
                raise ValueError("会话状态已改变，请重试")
            if action == "stop":
                self.stopping = True
                self.proc.terminate()
            elif action in ("tele_pause", "tele_run", "home"):
                if action == "home":
                    self.box[0] = RESET_B if self.box[0] == RESET_A else RESET_A
                else:
                    self.box[0] = RUN if action == "tele_run" else PAUSE
                if action == "tele_pause" and state["phase"] == "recording":
                    self._key("finish", "recording")
            else:
                if self.mode == "infer" and action == "pause":
                    self.box[0] = PAUSE  # ignore late policy output while inference returns
                self._key(action, state["phase"])

    def _key(self, action, phase):
        self.pending = self.box[3]
        self.proc.stdin.write((KEYS[phase][action]+"\n").encode())
        self.proc.stdin.flush()

    def shutdown(self):
        with self.lock:
            proc = self.proc
            if proc is not None and not self.stopping:
                self.stopping = True
                proc.terminate()
        if proc is not None:
            proc.wait()  # the device owns parking and verified disable
        self.preview.stop()
