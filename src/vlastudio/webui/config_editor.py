"""Read and atomically update named deployment profiles."""
import hashlib
import inspect
import ast
import textwrap
import json
import os
from pathlib import Path
import re
import tempfile

import yaml

from .profiles import CONFIGS

KINDS = ("robot", "teleop", "recording", "action_manager")


def profile_path(kind, name):
    if kind not in KINDS or not isinstance(name, str) or not re.fullmatch(r"[\w][\w.-]{0,79}", name):
        raise ValueError("Invalid configuration name")
    folder = (CONFIGS / kind).resolve()
    path = folder / (name + ".yaml")
    if path.resolve().parent != folder or path.is_symlink():
        raise ValueError("Invalid configuration path")
    return path


def read(kind, name):
    path = profile_path(kind, name)
    raw = path.read_bytes()
    content = raw.decode("utf-8")
    doc = yaml.safe_load(content)
    return dict(kind=kind, name=name, content=content, document=doc,
                version=hashlib.sha256(raw).hexdigest())


class ProfileLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate YAML keys")
        return super().construct_mapping(node, deep=deep)


def validate(kind, content):
    if not isinstance(content, str) or len(content.encode("utf-8")) > 65536:
        raise ValueError("Configuration must be at most 64 KiB")
    try:
        doc = yaml.load(content, Loader=ProfileLoader)
    except (yaml.YAMLError, TypeError) as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc
    try:
        json.dumps(doc, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("Configuration values must be finite JSON-compatible values") from exc
    if kind == "recording":
        if not isinstance(doc, dict) or not isinstance(doc.get("required"), list) or not doc["required"]:
            raise ValueError("Recording configuration needs a nonempty required channel list")
        if not all(isinstance(n, str) and n for n in doc["required"]) or not isinstance(doc.get("optional", {}), dict):
            raise ValueError("Invalid required or optional recording channels")
        if len(doc["required"]) != len(set(doc["required"])) or set(doc["required"]) & doc.get("optional", {}).keys():
            raise ValueError("Required and optional channels must be distinct")
        for spec in doc.get("optional", {}).values():
            if not isinstance(spec, dict) or not isinstance(spec.get("fields"), dict) or not spec["fields"]:
                raise ValueError("Optional inputs need nonempty fields")
        return
    rows = doc if isinstance(doc, list) else [doc]
    if not rows or (kind == "action_manager" and len(rows) != 1):
        raise ValueError("Invalid device configuration list")
    names = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("type"), str) or not row["type"]:
            raise ValueError("Each device needs a type")
        args = row.get("args", {})
        if not isinstance(args, dict) or not isinstance(row.get("webui", {}), dict):
            raise ValueError("args and webui must be mappings")
        name = args.get("name") or args.get("robot_id") or row.get("name")
        if kind != "action_manager" and (not isinstance(name, str) or not name or "/" in name):
            raise ValueError("Each device needs a named shared-memory channel")
        if name in names:
            raise ValueError("Device channel names must be unique")
        names.add(name)
        if "fps" in args and (not isinstance(args["fps"], (float, int)) or not 0 < args["fps"] <= 10000):
            raise ValueError("fps must be positive and at most 10000")


def in_use(studio, kind, name):
    if kind == "robot" and (studio.connected.get("robot") == name or studio.preview.profile == name):
        return True
    key = dict(robot="robot", teleop="teleop", recording="sync", action_manager="manager")[kind]
    return any(r.proc and r.options.get(key) == name for r in studio.runners.values())


def save(studio, kind, name, content, version, target):
    source = profile_path(kind, name)
    dest = profile_path(kind, target)
    if isinstance(content, dict) and "devices" in content:
        content = device_content(content["devices"], content.get("as_list", True))
    validate(kind, content)
    with studio.lock:
        if read(kind, name)["version"] != version:
            raise ValueError("Configuration changed. Reopen it before saving.")
        if dest == source:
            if in_use(studio, kind, name):
                raise ValueError("Disconnect this configuration before overwriting it, or save a copy.")
        elif dest.exists():
            raise ValueError("A configuration with this name already exists")
        fd, temporary = tempfile.mkstemp(prefix="."+target+"-", suffix=".tmp", dir=dest.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.chmod(temporary, source.stat().st_mode & 0o777)
            os.replace(temporary, dest)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return read(kind, target)


def device_types():
    return sorted({row["type"] for row in device_library()})


def parameters(reference):
    from vlastudio.utils.extensions import resolve
    if reference not in device_types():
        raise ValueError("Save this device type in a configuration before requesting its defaults")
    try:
        cls = resolve(reference)
        result = {}
        overrides = {}
        # Only inherit parameters where the constructor forwards keyword arguments.
        for base in cls.__mro__:
            signature = inspect.signature(base.__init__)
            forwarded = {name for name, p in signature.parameters.items() if p.kind == p.VAR_KEYWORD}
            try:
                tree = ast.parse(textwrap.dedent(inspect.getsource(base.__init__)))
                for call in ast.walk(tree):
                    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                            and isinstance(call.func.value, ast.Name) and call.func.value.id in forwarded
                            and call.func.attr == "setdefault" and len(call.args) == 2):
                        try:
                            key, value = (ast.literal_eval(arg) for arg in call.args)
                            overrides.setdefault(key, value)
                        except (ValueError, TypeError):
                            pass
            except (OSError, TypeError, IndentationError):
                pass
            for key, parameter in signature.parameters.items():
                if key == "self" or parameter.kind in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL):
                    continue
                if key in result:
                    continue
                row = dict(required=parameter.default is inspect.Parameter.empty)
                if not row["required"]:
                    try:
                        default = overrides.get(key, parameter.default)
                        row["default"] = json.loads(json.dumps(default, allow_nan=False))
                    except (ValueError, TypeError):
                        continue
                result[key] = row
            if not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
                break
        return result
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Device defaults unavailable in this runtime: {exc}") from exc


def device_content(devices, as_list):
    if not isinstance(devices, list) or not devices or len(devices)>64:
        raise ValueError("Configuration needs between 1 and 64 devices")
    return yaml.safe_dump(devices if as_list or len(devices)>1 else devices[0],
                          sort_keys=False, allow_unicode=True)


def device_library():
    """Scan current saved profiles, keeping distinct device parameter sets."""
    groups = {}
    for kind in ("robot", "teleop", "action_manager"):
        folder = (CONFIGS / kind).resolve()
        for path in sorted(folder.rglob("*.yaml")):
            if path.is_symlink() or not path.resolve().is_relative_to(folder):
                continue
            try:
                doc = yaml.safe_load(path.read_text())
                for index, row in enumerate(doc if isinstance(doc, list) else [doc]):
                    if not isinstance(row, dict) or not isinstance(row.get("type"), str):
                        continue
                    signature = json.dumps(row, sort_keys=True, allow_nan=False)
                    reference = row["type"]
                    if ".bench_robot." in reference:
                        category = "Bench"
                    elif ".sensor.camera." in reference:
                        category = "Cameras"
                    elif ".teleoperator." in reference:
                        category = "Teleoperation"
                    elif ".robot." in reference:
                        category = "Robots"
                    elif ".action_manager." in reference:
                        category = "Action managers"
                    else:
                        category = "Other devices"
                    if signature not in groups:
                        args = row.get("args") or {}
                        if not isinstance(args, dict):
                            continue
                        groups[signature] = dict(category=category, type=reference,
                            name=args.get("name") or row.get("name") or reference.rsplit(".",1)[-1],
                            device=row, sources=[])
                    groups[signature]["sources"].append(dict(kind=kind,
                        config=path.relative_to(folder).with_suffix("").as_posix(), index=index))
            except (OSError, yaml.YAMLError, TypeError, ValueError, RecursionError):
                continue
    return sorted(groups.values(), key=lambda row:(row["category"], row["type"], str(row["name"])))
