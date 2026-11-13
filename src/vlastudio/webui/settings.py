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


def configured_roots(data):
    if "config_roots" in data:
        return data["config_roots"]
    legacy = data.get("config_root")
    return [legacy] if legacy else []


def public():
    data, version = stored()
    return dict(data_dir=str(Path(data.get("data_dir", DEFAULT_DATA)).expanduser()),
                checkpoint_dir=str(Path(data.get("checkpoint_dir", DEFAULT_DATA.parent / "checkpoints")).expanduser()),
                config_roots=configured_roots(data), hf_username=data.get("hf_username", ""),
                token_set=bool(data.get("hf_token", os.environ.get("HF_TOKEN", ""))), version=version)


def data_directory():
    return Path(public()["data_dir"]).expanduser().resolve()


def checkpoint_directory():
    return Path(public()["checkpoint_dir"]).expanduser().resolve()


def extra_configs():
    data, _ = stored()
    return [Path(root).expanduser().resolve() for root in configured_roots(data)]


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


def browse_directory(value, initial=False):
    path = Path(absolute_directory(value, required=True)) if value else Path.home()
    if initial:
        # The editable field may already contain a new directory to be created on save.
        while not path.is_dir() and path != path.parent:
            path = path.parent
    with os.scandir(path) as entries:
        folders = []
        for entry in entries:
            try:
                if entry.is_dir():
                    folders.append({"name": entry.name, "path": entry.path})
            except OSError:
                continue
    folders.sort(key=lambda folder: (folder["name"].casefold(), folder["name"]))
    return {"path": str(path), "parent": str(path.parent) if path != path.parent else None,
            "folders": folders}


def create_directory(value, name):
    if (not isinstance(name, str) or not name.strip() or name.strip() in (".", "..")
            or len(name.strip().encode("utf-8")) > 255
            or any(char in name for char in ("/", "\\"))
            or any(ord(char) < 32 or ord(char) == 127 for char in name)):
        raise ValueError("Enter a folder name without path separators")
    parent = Path(absolute_directory(value, required=True))
    if not parent.is_dir():
        raise ValueError("The parent directory no longer exists")
    child = parent / name.strip()
    try:
        child.mkdir()
    except FileExistsError:
        raise ValueError("A file or folder with this name already exists") from None
    except PermissionError:
        raise ValueError("You do not have permission to create a folder here") from None
    return {"path": str(child)}


def save(studio, payload):
    with studio.lock:
        data, version = stored()
        if payload.get("version") != version:
            raise ValueError("Settings changed. Reopen settings before saving.")
        data_dir = absolute_directory(payload.get("data_dir", ""), required=True)
        checkpoint_dir = absolute_directory(payload.get("checkpoint_dir", str(checkpoint_directory())), required=True)
        previous = configured_roots(data)
        if "config_roots" in payload:
            requested = payload["config_roots"]
        elif "config_root" in payload:
            if len(previous) > 1:
                raise ValueError("Refresh the page to edit multiple configuration directories")
            requested = configured_roots(payload)
        else:
            requested = previous
        if not isinstance(requested, list):
            raise ValueError("Configuration directories must be a list")
        roots = list(dict.fromkeys(absolute_directory(root, required=True) for root in requested))
        changed = data_dir != str(data_directory()) or roots != previous
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
        for root in roots:
            if not Path(root).is_dir():
                raise ValueError(f"Configuration directory does not exist: {root}")
        target = Path(data_dir)
        target.mkdir(parents=True, exist_ok=True)
        # Check actual write access, including on network-mounted storage.
        with tempfile.TemporaryFile(dir=target):
            pass
        checkpoint_target = Path(checkpoint_dir)
        checkpoint_target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=checkpoint_target):
            pass
        data.pop("config_root", None)
        data.update(data_dir=data_dir, checkpoint_dir=checkpoint_dir, config_roots=roots, hf_username=username)
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
