"""Launch workers using the caller's installed Python environment."""
import os
from pathlib import Path
import platform
import re
import signal
import threading
import subprocess
import sys


def _glibcxx_version(path):
    try:
        versions = re.findall(rb"GLIBCXX_(\d+)\.(\d+)\.(\d+)", Path(path).read_bytes())
    except OSError:
        return None
    return max((tuple(map(int, version)) for version in versions), default=None)



def _configure_headless_graphics(command, python, env):
    if (command != "evaluate" or not sys.platform.startswith("linux")
            or env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        return
    env.setdefault("MUJOCO_GL", "egl")
    if env.get("LD_PRELOAD"):
        return

    # Conda Python can prefer its own older C++ runtime via RPATH. Mesa's EGL
    # driver then fails to load even though the host provides a compatible one.
    bundled = Path(python).resolve().parent.parent / "lib" / "libstdc++.so.6"
    machine = {"AMD64": "x86_64", "arm64": "aarch64"}.get(platform.machine(), platform.machine())
    candidates = [Path(root) / f"{machine}-linux-gnu" / "libstdc++.so.6"
                  for root in ("/usr/lib", "/lib")]
    bundled_version = _glibcxx_version(bundled)
    for candidate in candidates:
        system_version = _glibcxx_version(candidate)
        if bundled_version and system_version and system_version > bundled_version:
            env["LD_PRELOAD"] = str(candidate)
            break



def execute(command, args, python, app, env, entrypoint=None, isolated=True):
    # Preserve cwd: task data, configuration and checkpoint relative paths remain user-relative.
    paths = [str(app), os.getcwd()]
    paths += [x for x in env.get("VLASTUDIO_PLUGIN_PATH", "").split(os.pathsep) if x]
    child_env = dict(env)
    _configure_headless_graphics(command, python, child_env)
    child_env["PYTHONPATH"] = os.pathsep.join(paths)
    if isolated:
        child_env["PYTHONNOUSERSITE"] = "1"
    elif env.get("PYTHONPATH"):
        child_env["PYTHONPATH"] += os.pathsep + env["PYTHONPATH"]
    worker_args = [command, *args]
    if entrypoint:
        worker_args = ["--entrypoint", entrypoint, *worker_args]
    kwargs = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    child = subprocess.Popen([str(python), "-m", "vlastudio.deploy.interpreter.worker", *worker_args], env=child_env, **kwargs)
    def forward(signum, frame=None):
        if child.poll() is None:
            if os.name != "nt":
                os.killpg(child.pid, signum)
            elif signum == signal.SIGINT:
                child.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                child.terminate()
    old_term = None
    if threading.current_thread() is threading.main_thread():
        old_term = signal.signal(signal.SIGTERM, forward)
    try:
        return child.wait()
    except KeyboardInterrupt:
        forward(signal.SIGINT)
        # Allow the worker to close/checkpoint before terminating it.
        try:
            child.wait(timeout=10)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        return 130
    finally:
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)
