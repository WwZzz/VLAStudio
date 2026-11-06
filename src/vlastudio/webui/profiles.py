from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parents[1]
CONFIGS = PACKAGE / "configs"
ROOT = PACKAGE.parents[1]


def catalog():
    result = {kind: sorted(p.stem for p in (CONFIGS / kind).glob("*.yaml"))
              for kind in ("robot", "teleop", "recording", "action_manager")}
    result['robot_options'] = {}
    for path in (CONFIGS / 'robot').glob('*.yaml'):
        rows = yaml.safe_load(path.read_text())
        rows = rows if isinstance(rows, list) else [rows]
        for row in rows:
            if row.get('webui'):
                result['robot_options'][path.stem] = row['webui']
    return result


def config(kind, name):
    if name not in catalog()[kind]:
        raise ValueError(f"Unknown {kind} profile: {name}")
    return CONFIGS / kind / (name + ".yaml")


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
