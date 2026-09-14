"""Resolve user configs without importing training libraries."""
import os
from pathlib import Path
import yaml
from .paths import legacy_root
from .extensions import resolve


def resolve_config(value, category, base_dir=None):
    if value.startswith("@config/"):
        result = resolve(value, kind="config")
        return Path(result() if callable(result) else result).expanduser().resolve()
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.suffix and path.with_suffix(".yaml").is_file():
        return path.with_suffix(".yaml").resolve()
    roots = [Path.cwd() / "configs"]
    roots += [Path(x).expanduser() for x in os.environ.get("VLASTUDIO_CONFIG_PATH", "").split(os.pathsep) if x]
    roots.append(legacy_root() / "configs")
    candidates = [r / category for r in roots[:-1]]
    if base_dir:
        candidates.append(Path(base_dir))
    candidates.append(roots[-1] / category)
    # Explicit configs/foo/bar.yaml also works from outside a checkout.
    relative = Path(value)
    if relative.parts and relative.parts[0] == "configs":
        for root in roots:
            candidate = root.joinpath(*relative.parts[1:])
            if candidate.is_file():
                return candidate.resolve()
    names = [value] if path.suffix in (".yaml", ".yml") else [value.replace(".", "/") + ".yaml", value + ".yaml"]
    for base in candidates:
        for name in names:
            candidate = base / name
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"{category} config not found: {value}")


def read_config(value, category):
    path = resolve_config(value, category)
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}, path
