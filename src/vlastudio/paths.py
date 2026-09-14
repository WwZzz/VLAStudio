"""Locations shared by source checkouts and installed wheels."""
import os
import json
from pathlib import Path
from platformdirs import user_cache_dir, user_config_dir


def package_root():
    return Path(__file__).resolve().parent


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
        or user_cache_dir("vlastudio")
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
