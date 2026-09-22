"""Open component loading without importing built-in policy/device dependencies.

References: dotted.path.Class, module:Class, /path/component.py:Class,
/path/package/ (loads __init__.py), /path/component.py, module (policy
factories), or @kind/name (entry points/registration).
"""
import hashlib
import os
import json
import importlib.abc
import importlib
import importlib.util
from importlib.metadata import entry_points
from pathlib import Path
import sys

KINDS = {"policy", "robot", "device", "action_manager", "dataset", "config"}
_registry = {}


class _FileExtensionFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("_vlastudio_ext_"):
            return None
        filename = json.loads(os.environ.get("VLASTUDIO_FILE_MODULES", "{}")).get(fullname)
        return importlib.util.spec_from_file_location(fullname, filename) if filename else None


# Spawned data-loader/device processes import the same extension name from its file.
if not any(isinstance(finder, _FileExtensionFinder) for finder in sys.meta_path):
    sys.meta_path.append(_FileExtensionFinder())


def register(kind, name, target):
    """Register an object or import reference in this process.

    Use an installed entry point or a module/file reference across processes.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown extension kind: {kind}")
    key = (kind, name)
    if key in _registry:
        raise ValueError(f"Extension already registered: {kind}/{name}")
    _registry[key] = target


def _is_path_like(value):
    if not value:
        return False
    path = Path(value).expanduser()
    return path.exists() or value.endswith(".py") or value.startswith(".") or "/" in value or "\\" in value


def _filesystem_reference(reference):
    """Return (python_file, attribute) for a local file or package directory."""
    attribute = ""
    raw = reference
    if ".py:" in reference:
        raw, attribute = reference.rsplit(":", 1)
    elif ":" in reference:
        maybe, attr = reference.rsplit(":", 1)
        if not _is_path_like(maybe):
            return None
        raw, attribute = maybe, attr
    elif not _is_path_like(reference):
        return None
    path = Path(raw).expanduser()
    if path.is_dir() or raw.endswith(("/", "\\")):
        package = path
        path = package / "__init__.py"
        if not path.is_file():
            raise ImportError(f"Package directory has no __init__.py: {package}")
    return path, attribute


def _load_file_module(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ImportError(f"Cannot load extension: {path}")
    name = "_vlastudio_ext_" + hashlib.sha256(str(path).encode()).hexdigest()[:20]
    known = json.loads(os.environ.get("VLASTUDIO_FILE_MODULES", "{}"))
    known[name] = str(path)
    os.environ["VLASTUDIO_FILE_MODULES"] = json.dumps(known)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load extension: {path}")
        obj = importlib.util.module_from_spec(spec)
        sys.modules[name] = obj
        try:
            spec.loader.exec_module(obj)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def resolve(reference, kind=None, module=False):
    if not isinstance(reference, str):
        return reference
    from .compat import canonical
    reference = canonical(reference)
    if reference.startswith("@"):
        group, name = reference[1:].split("/", 1)
        if group not in KINDS or (kind and group != kind):
            raise ValueError(f"Invalid extension category: {reference}")
        value = _registry.get((group, name))
        if value is None:
            matches = list(entry_points(group=f"vlastudio.{group}", name=name))
            if len(matches) != 1:
                raise LookupError(f"Expected one plugin for {reference}, found {len(matches)}")
            value = matches[0].load()
        if value == reference:
            raise ValueError(f"Recursive extension reference: {reference}")
        return resolve(value, kind=group, module=module)
    file_ref = _filesystem_reference(reference)
    if file_ref is not None:
        path, attribute = file_ref
        obj = _load_file_module(path)
        return getattr(obj, attribute) if attribute else obj
    if ":" in reference:
        name, attribute = reference.split(":", 1)
        imported = importlib.import_module(name)
        return getattr(imported, attribute) if attribute else imported
    if module:
        return importlib.import_module(reference)
    name, attribute = reference.rsplit(".", 1)
    return getattr(importlib.import_module(name), attribute)


def create(config, kind=None):
    """Instantiate a configured component; the component owns its interface."""
    return resolve(config["type"], kind=kind)(**config.get("args", {}))
