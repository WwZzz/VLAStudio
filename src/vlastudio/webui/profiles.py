from pathlib import Path

import yaml

from .settings import extra_configs

PACKAGE = Path(__file__).resolve().parents[1]
CONFIGS = PACKAGE / "configs"
ROOT = PACKAGE.parents[1]


KINDS = ("robot", "teleop", "recording", "action_manager")


def config_roots():
    extra = extra_configs()
    return [CONFIGS, extra] if extra and extra != CONFIGS.resolve() else [CONFIGS]


def profile_paths(kind):
    if kind not in KINDS:
        raise ValueError("Unknown configuration category")
    paths = {}
    for root in config_roots():
        for path in sorted((root/kind).glob("*.yaml")):
            if not path.is_symlink() and path.resolve().parent == (root/kind).resolve():
                paths[path.stem] = path
    return paths


def catalog():
    paths = {kind:profile_paths(kind) for kind in KINDS}
    result = {kind:sorted(items) for kind,items in paths.items()}
    result["robot_options"] = {}
    for name, path in paths["robot"].items():
        rows = yaml.safe_load(path.read_text())
        for row in rows if isinstance(rows, list) else [rows]:
            if isinstance(row, dict) and row.get("webui"):
                result["robot_options"][name] = row["webui"]
    return result


def config(kind, name):
    path = profile_paths(kind).get(name)
    if path is None:
        raise ValueError(f"Unknown {kind} profile: {name}")
    return path


def channels(profile, kind="robot"):
    doc = yaml.safe_load(config(kind, profile).read_text())
    result = []
    for row in doc if isinstance(doc, list) else [doc]:
        typ = row.get("type", "")
        if "visualizer" in typ.lower():
            continue
        args = row.get("args") or {}
        name = args.get("name") or args.get("robot_id") or row.get("name")
        if not name or "/" in name:
            raise ValueError("Profile requires a named shared-memory channel")
        result.append((name, typ, args))
    return result


def model_path(typ, args):
    root = PACKAGE / "deploy/robot"
    known = {"RebotB601RS": "rebot_b601rs/assets/ReBot_Arm_RS.urdf",
             "AliciaD": "alicia_d/urdf/Alicia_D_v5_6_gripper_50mm.urdf",
             "BiAliciaD": "alicia_d/urdf/Alicia_D_v5_6_gripper_50mm.urdf"}
    candidate = args.get("urdf_path")
    if candidate:
        for base in (ROOT, PACKAGE, root):
            path = (base / candidate).resolve()
            if path.is_file() and path.suffix == ".urdf":
                return path
    candidate = known.get(typ.rsplit(".", 1)[-1])
    return root / candidate if candidate else None
