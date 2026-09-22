"""Persistent per-profile environments, prepared only on explicit invocation."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import threading
import subprocess
import sys
from filelock import FileLock

CORE = ["PyYAML==6.0.2", "platformdirs==4.3.6", "filelock==3.18.0", "loguru==0.7.3"]


def project_root():
    from .paths import package_source_root
    package = package_source_root()
    repo = package.parent.parent
    if (repo / "pyproject.toml").is_file():
        return repo
    return None


def install_application(python, env):
    """Put the live VLAStudio package into a managed interpreter (no extra deps)."""
    root = project_root()
    if root is None:
        return
    run([*uv_command(), "pip", "install", "--python", str(python), "--no-deps", "-e", str(root)], env)


def _libero_git_spec(requirements):
    for requirement in requirements or []:
        match = re.search(r"git\+(https?://\S+LIBERO\.git)(?:@([0-9a-fA-F]{7,40}))?", requirement, re.I)
        if match:
            return match.group(1), match.group(2)
    return "https://github.com/Lifelong-Robot-Learning/LIBERO.git", "8f1084e3132a39270c3a13ebe37270a43ece2a01"


def _looks_like_libero_checkout(path):
    path = Path(path)
    return (path / "libero" / "libero").is_dir() and ((path / "libero" / "libero" / "__init__.py").is_file())


def find_libero_source(env=None, requirements=None):
    """LIBERO's setup.py emits an empty wheel; import needs the git checkout on sys.path."""
    env = env or os.environ
    from .paths import cache_root, package_source_root
    candidates = [package_source_root() / "third_party" / "libero"]
    if env.get("LIBERO_ROOT"):
        candidates.append(Path(env["LIBERO_ROOT"]).expanduser())
    uv_cache = Path(env.get("UV_CACHE_DIR") or (cache_root(env.get("VLASTUDIO_CACHE")) / "uv"))
    if uv_cache.is_dir():
        candidates.extend(sorted(uv_cache.glob("git-v0/checkouts/*/*")))
    src = cache_root(env.get("VLASTUDIO_CACHE")) / "src" / "LIBERO"
    candidates.append(src)
    for candidate in candidates:
        if _looks_like_libero_checkout(candidate):
            return candidate.resolve()
    return None


def _clone_libero_source(env, requirements):
    from .paths import cache_root
    dest = cache_root(env.get("VLASTUDIO_CACHE")) / "src" / "LIBERO"
    if _looks_like_libero_checkout(dest):
        return dest.resolve()
    url, sha = _libero_git_spec(requirements)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    run(["git", "clone", "--filter=blob:none", url, str(dest)], env)
    if sha:
        run(["git", "-C", str(dest), "checkout", sha], env)
    if not _looks_like_libero_checkout(dest):
        raise RuntimeError(f"Failed to checkout an importable LIBERO tree at {dest}")
    return dest.resolve()


def _seed_libero_config(source):
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")).expanduser()
    config_file = config_dir / "config.yaml"
    if config_file.is_file():
        return
    root = Path(source) / "libero" / "libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join([
            f"assets: {root / 'assets'}",
            f"bddl_files: {root / 'bddl_files'}",
            f"benchmark_root: {root}",
            f"datasets: {root.parent / 'datasets'}",
            f"init_states: {root / 'init_files'}",
            "",
        ]),
        encoding="utf-8",
    )


def ensure_source_layout_packages(python, profile, env):
    """Repair packages whose published/build metadata does not ship importable modules."""
    requirements = list(profile.get("requirements") or [])
    if profile.get("lock"):
        requirements.extend(profile["lock"].splitlines())
    if not any(re.search(r"(?i)(^|[\s/@=])libero($|[\s/@=<>!~;])", item) or "LIBERO.git" in item for item in requirements):
        return
    probe = subprocess.run([str(python), "-c", "from libero.libero import benchmark"], env=env)
    if probe.returncode == 0:
        return
    source = find_libero_source(env, requirements) or _clone_libero_source(env, requirements)
    _seed_libero_config(source)
    script = (
        "import pathlib, site, sys, sysconfig\n"
        "source = pathlib.Path(sys.argv[1]).resolve()\n"
        "bases = list(site.getsitepackages()) or [sysconfig.get_path('purelib')]\n"
        "path = pathlib.Path(bases[0]) / 'libero-src.pth'\n"
        "path.write_text(str(source) + '\\n', encoding='utf-8')\n"
        "print(path)\n"
    )
    run([python, "-c", script, str(source)], env)
    run([python, "-c", "from libero.libero import benchmark"], env)


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


def uv_command():
    executable = shutil.which("uv")
    if executable:
        return [executable]
    import uv
    return [uv.find_uv_bin()]


def run(command, env):
    subprocess.run([str(x) for x in command], env=env, check=True)


def files_under(root):
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in (".git", ".venv", "__pycache__", ".ipynb_checkpoints") and not (Path(directory) / d / ".git").exists())
        for name in sorted(files):
            if not name.endswith((".pyc", ".pyo")) and name not in (".git", ".package_origin"):
                yield Path(directory) / name


def snapshot(cache):
    """Copy only application files, never the parent environment's site-packages."""
    from .paths import package_source_root
    package = package_source_root()
    entries = [(f, Path("vlastudio") / f.relative_to(package)) for f in files_under(package)]
    digest = hashlib.sha256()
    for source, target in entries:
        digest.update(str(target).encode())
        digest.update(source.read_bytes())
    destination = cache / "apps" / digest.hexdigest()[:24]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if not (destination / ".ready").exists():
            destination.mkdir(parents=True, exist_ok=True)
            for source, target in entries:
                out = destination / target
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, out)
            (destination / ".ready").write_text("ready")
        origin = destination / "vlastudio" / ".package_origin"
        origin.parent.mkdir(parents=True, exist_ok=True)
        origin.write_text(str(package.resolve()), encoding="utf-8")
    return destination


def environment_identity(profile):
    identity = {k: v for k, v in profile.items() if k != "entrypoint"}
    identity["platform"] = [sys.platform, platform.machine()]
    identity["core"] = CORE
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return key, identity


def environment_target(profile, cache, name=None):
    if name is not None:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name):
            raise ValueError('Environment names must start with a letter or digit and contain only letters, digits, dots, underscores or hyphens (max 128 characters)')
        return cache / 'envs' / ('named-' + name)
    key, _ = environment_identity(profile)
    return cache / 'envs' / key


def build_environment(profile, env):
    result = dict(env)
    requirements = profile.get('requirements', []) + profile.get('lock', '').splitlines()
    if any(re.match(r'(?i)\s*(robomimic|egl[-_]probe)(?:\s|[=<>!~;\[]|$)', requirement) for requirement in requirements):
        # egl-probe's legacy CMake declaration needs a policy floor on CMake 4.
        result.setdefault('CMAKE_POLICY_VERSION_MINIMUM', '3.5')
    return result


def prepare(profile, cache, env, offline=False, name=None):
    env = build_environment(profile, env)
    key, identity = environment_identity(profile)
    target = environment_target(profile, cache, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    with FileLock(str(target) + ".lock"):
        # Named environments keep the existing interpreter. Do not re-sync.
        if python.is_file() and (name is not None or (target / ".ready").is_file()):
            return python
        if offline:
            raise RuntimeError(f"Environment {key} is not prepared. Run prepare online first.")
        print(f"Preparing environment {key} (Python {profile['python']})", flush=True)
        uv = uv_command()
        # An interrupted download leaves a valid but incomplete environment behind.
        run([*uv, "venv", "--allow-existing", "--python", profile["python"], target], env)
        requirements = target / "requirements.in"
        requirements.write_text("\n".join(CORE + profile.get("requirements", [])) + "\n", encoding="utf-8")
        lock = target / "requirements.lock"
        if profile.get("lock"):
            lock.write_text(profile["lock"], encoding="utf-8")
            run([*uv, "pip", "sync", "--python", python, lock], env)
            # Supplied locks must include the lightweight runtime dependencies.
            run([python, "-c", "import yaml, platformdirs, filelock, loguru"], env)
        else:
            extra = []
            if profile.get("overrides"):
                overrides = target / "overrides.txt"
                overrides.write_text("\n".join(profile["overrides"]) + "\n", encoding="utf-8")
                extra = ["--override", overrides]
            run([*uv, "pip", "compile", "--quiet", requirements, "--python", python, "--output-file", lock, *extra], env)
            run([*uv, "pip", "sync", "--python", python, lock], env)
        install_application(python, env)
        ensure_source_layout_packages(python, profile, env)
        for overlay in profile.get("overlays", []):
            # Patches remain confined to this managed environment. They are part of its cache key.
            script = "import importlib.util,json; print(json.dumps([list(importlib.util.find_spec(n).submodule_search_locations)[0] for n in " + repr([overlay["source_module"], overlay["target_module"]]) + "]))"
            output = subprocess.check_output([str(python), "-c", script], env=env, text=True)
            source_root, destination_root = map(Path, json.loads(output))
            source = (source_root / overlay["source"]).resolve()
            source.relative_to(source_root.resolve())
            destination_root.resolve().relative_to(target.resolve())
            if not source.is_dir():
                raise RuntimeError(f"Missing runtime overlay: {source}")
            shutil.copytree(source, destination_root, dirs_exist_ok=True)
        (target / "profile.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
        (target / ".ready").write_text("ready")
    return python


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
    child = subprocess.Popen([str(python), "-m", "vlastudio.worker", *worker_args], env=child_env, **kwargs)
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
