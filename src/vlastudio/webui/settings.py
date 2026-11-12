"""Server-local preferences and private Hugging Face credentials."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

SETTINGS_FILE = Path(os.environ.get("XDG_CONFIG_HOME", Path.home()/".config")) / "vlastudio/webui-settings.json"
DEFAULT_DATA = Path(__file__).resolve().parents[3] / "data/webui"


def stored():
    if not SETTINGS_FILE.exists():
        return {}, "missing"
    try:
        raw = SETTINGS_FILE.read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError()
        return data, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError):
        raise ValueError("Cannot read WebUI settings") from None


def public():
    data, version = stored()
    return dict(data_dir=str(Path(data.get("data_dir", DEFAULT_DATA)).expanduser()),
                config_root=data.get("config_root", ""), hf_username=data.get("hf_username", ""),
                token_set=bool(data.get("hf_token", os.environ.get("HF_TOKEN", ""))), version=version)


def data_directory():
    return Path(public()["data_dir"]).expanduser().resolve()


def extra_configs():
    value = public()["config_root"]
    return Path(value).expanduser().resolve() if value else None


def runtime_environment(environ):
    env = dict(environ)
    data, _ = stored()
    if "hf_token" in data:
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            env.pop(key, None)
        if data["hf_token"]:
            env["HF_TOKEN"] = data["hf_token"]
    if data.get("hf_username"):
        env["HF_USERNAME"] = data["hf_username"]
    return env


def absolute_directory(value, required=False):
    if not isinstance(value, str):
        raise ValueError("Directory must be a server-local path")
    if not value.strip() and not required:
        return ""
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise ValueError("Use an absolute path on the server")
    return str(path.resolve())


def save(studio, payload):
    with studio.lock:
        data, version = stored()
        if payload.get("version") != version:
            raise ValueError("Settings changed. Reopen settings before saving.")
        data_dir = absolute_directory(payload.get("data_dir", ""), required=True)
        config_root = absolute_directory(payload.get("config_root", ""))
        old = public()
        changed = data_dir != str(data_directory()) or config_root != old["config_root"]
        if changed and (studio.host or any(r.proc for r in studio.runners.values())):
            raise ValueError("Disconnect devices before changing storage or configuration directories")
        username = payload.get("hf_username", "")
        if not isinstance(username, str) or type(payload.get("clear_token", False)) is not bool:
            raise ValueError("Invalid account settings")
        username = username.strip()
        if username and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", username):
            raise ValueError("Invalid Hugging Face username")
        token = payload.get("hf_token")
        if token is not None and (not isinstance(token, str) or len(token)>4096 or any(c.isspace() for c in token)):
            raise ValueError("Invalid Hugging Face token")
        if config_root and not Path(config_root).is_dir():
            raise ValueError("Extra configuration root must be an existing directory")
        target = Path(data_dir)
        target.mkdir(parents=True, exist_ok=True)
        # Check actual write access, including on network-mounted storage.
        with tempfile.TemporaryFile(dir=target):
            pass
        data.update(data_dir=data_dir, config_root=config_root, hf_username=username)
        if payload.get("clear_token"):
            data["hf_token"] = ""
        elif token:
            data["hf_token"] = token
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(prefix=".webui-settings-", dir=SETTINGS_FILE.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, SETTINGS_FILE)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return public()
