"""Locations shared by source checkouts and installed wheels."""
import os
import json
from pathlib import Path
from platformdirs import user_cache_dir, user_config_dir


def package_root():
    return Path(__file__).resolve().parent


def package_source_root():
    """The installed or checkout package, not a cached application snapshot."""
    here = Path(__file__).resolve().parent
    origin = here / ".package_origin"
    if origin.is_file():
        path = Path(origin.read_text(encoding="utf-8").strip())
        if (path / "__init__.py").is_file():
            return path
    try:
        here.relative_to(cache_root() / "apps")
    except ValueError:
        return here
    live = _installed_package_root()
    return live if live is not None else here


def _installed_package_root():
    try:
        from importlib.metadata import distribution
        from urllib.parse import unquote, urlparse
        dist = distribution("vlastudio")
    except Exception:
        return None
    try:
        info = json.loads(dist.read_text("direct_url.json") or "")
        url = info.get("url", "")
        if url.startswith("file:"):
            root = Path(unquote(urlparse(url).path))
            for candidate in (root / "src" / "vlastudio", root / "vlastudio"):
                if (candidate / "__init__.py").is_file():
                    return candidate
    except Exception:
        pass
    try:
        locate = Path(dist.locate_file("vlastudio/__init__.py"))
        if locate.is_file():
            return locate.parent
    except Exception:
        pass
    return None


def drop_app_snapshots(pythonpath, cache=None):
    """Remove cached application snapshots from a PYTHONPATH value."""
    apps = cache_root(cache) / "apps"
    kept = []
    for part in (pythonpath or "").split(os.pathsep):
        if not part:
            continue
        try:
            Path(part).expanduser().resolve().relative_to(apps)
        except ValueError:
            kept.append(part)
    return os.pathsep.join(kept)


def legacy_root():
    """Compatibility name for tools written before the source-layout migration."""
    return package_root()


def user_settings():
    path = Path(os.environ.get("VLASTUDIO_SETTINGS", str(Path(user_config_dir("vlastudio")) / "settings.json")))
    if not path.is_file():
        return {}
    settings = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("VLAStudio settings.json must contain an object")
    return settings


def cache_root(value=None):
    return Path(
        value
        or os.environ.get("VLASTUDIO_CACHE")
        or os.environ.get("VLASTUDIO_CACHE_DIR")
        or os.environ.get("ILSTD_CACHE")
        or user_settings().get("cache_dir")
        or Path.home() / '.cache' / 'vlastudio'
    ).expanduser().resolve()


def runtime_env(cache, model_cache=None, data_cache=None):
    env = os.environ.copy()
    env["VLASTUDIO_CACHE"] = str(cache)
    env["VLASTUDIO_CACHE_DIR"] = str(cache)
    explicit_data = data_cache or env.get("VLASTUDIO_DATA_CACHE_DIR")
    data = Path(explicit_data).expanduser().resolve() if explicit_data else cache / "data"
    env["ILSTD_CACHE"] = str(data)
    env["VLASTUDIO_DATA_CACHE_DIR"] = str(data)
    for variable, suffix in (("HF_DATASETS_CACHE", "huggingface"), ("HF_LEROBOT_HOME", "lerobot")):
        if explicit_data:
            env[variable] = str(data / suffix)
        else:
            env.setdefault(variable, str(data / suffix))
    env.setdefault("UV_CACHE_DIR", str(cache / "uv"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(cache / "python"))
    # CUDA wheels can take longer than uv's default cache-lock timeout to download.
    env.setdefault("UV_LOCK_TIMEOUT", "3600")
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    if model_cache:
        env["HF_HOME"] = str(Path(model_cache).expanduser().resolve())
    else:
        env.setdefault("HF_HOME", str(cache / "models" / "huggingface"))
    env.setdefault("TORCH_HOME", str(cache / "models" / "torch"))
    env.setdefault("OPENPI_DATA_HOME", str(cache / "models" / "openpi"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "torchinductor"))
    env.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
    return env
