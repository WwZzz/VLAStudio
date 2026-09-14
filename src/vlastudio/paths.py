"""Locations shared by source checkouts and installed wheels."""
import os
import json
from pathlib import Path
from platformdirs import user_cache_dir, user_config_dir


def legacy_root():
    bundled = Path(__file__).parent / "_legacy"
    return bundled if bundled.is_dir() else Path(__file__).resolve().parents[2]


def user_settings():
    path = Path(os.environ.get("VLASTUDIO_SETTINGS", str(Path(user_config_dir("vlastudio")) / "settings.json")))
    if not path.is_file():
        return {}
    settings = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("VLAStudio settings.json must contain an object")
    return settings


def cache_root(value=None):
    return Path(value or os.environ.get("VLASTUDIO_CACHE_DIR") or os.environ.get("ILSTD_CACHE") or user_settings().get("cache_dir") or user_cache_dir("vlastudio")).expanduser().resolve()


def runtime_env(cache, model_cache=None):
    env = os.environ.copy()
    env["VLASTUDIO_CACHE_DIR"] = str(cache)
    env["ILSTD_CACHE"] = str(cache / "data")
    env["UV_CACHE_DIR"] = str(cache / "uv")
    env["UV_PYTHON_INSTALL_DIR"] = str(cache / "python")
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    if model_cache:
        env["HF_HOME"] = str(Path(model_cache).expanduser().resolve())
    else:
        env.setdefault("HF_HOME", str(cache / "models" / "huggingface"))
    env.setdefault("TORCH_HOME", str(cache / "models" / "torch"))
    return env
