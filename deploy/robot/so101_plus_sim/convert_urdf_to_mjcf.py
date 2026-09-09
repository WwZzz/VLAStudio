#!/usr/bin/env python3
"""One-shot: convert so101_plus.urdf → mujoco_model/so101_plus.xml."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
URDF_PATH = HERE.parent / "so101_plus" / "so101_plus_model" / "so101_plus.urdf"
OUT_PATH = HERE / "mujoco_model" / "so101_plus.xml"


def _rpy_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_euler("xyz", rpy).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def _parse_xyz_rpy(el) -> tuple[np.ndarray, np.ndarray]:
    if el is None:
        return np.zeros(3), np.zeros(3)
    xyz = np.fromstring(el.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(el.get("rpy", "0 0 0"), sep=" ")
    return xyz, rpy


def _fmt(arr: np.ndarray, nd: int = 6) -> str:
    return " ".join(f"{float(x):.{nd}g}" for x in np.asarray(arr).reshape(-1))


def main() -> None:
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()

    links: dict = {}
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        mass = 0.01
        com = np.zeros(3)
        if inertial is not None:
            mass = float(inertial.find("mass").get("value", "0.01"))
            com, _ = _parse_xyz_rpy(inertial.find("origin"))
        visuals = []
        for vis in link.findall("visual"):
            xyz, rpy = _parse_xyz_rpy(vis.find("origin"))
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is None:
                continue
            mesh_file = Path(mesh_el.get("filename", "")).name
            if not mesh_file:
                continue
            color = (0.75, 0.75, 0.78, 1.0)
            rgba = vis.find("material/color")
            if rgba is not None:
                color = tuple(float(x) for x in rgba.get("rgba", "0.75 0.75 0.78 1").split())
            visuals.append((xyz, rpy, mesh_file, color))
        links[name] = {"mass": mass, "com": com, "visuals": visuals}

    children: dict = defaultdict(list)
    parent_of: dict = {}
    for j in root.findall("joint"):
        info = {
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
        }
        xyz, rpy = _parse_xyz_rpy(j.find("origin"))
        axis_el = j.find("axis")
        axis = np.fromstring(
            axis_el.get("xyz", "0 0 1") if axis_el is not None else "0 0 1", sep=" "
        )
        lim = j.find("limit")
        lo = float(lim.get("lower", "-3.14")) if lim is not None else -3.14
        hi = float(lim.get("upper", "3.14")) if lim is not None else 3.14
        info.update({"xyz": xyz, "rpy": rpy, "axis": axis, "lo": lo, "hi": hi})
        children[info["parent"]].append(info)
        parent_of[info["child"]] = info["parent"]

    roots = list(set(links) - set(parent_of))
    if len(roots) != 1:
        raise RuntimeError(f"Expected 1 URDF root, got {roots}")
    root_link = roots[0]

    mesh_names = sorted({v[2] for L in links.values() for v in L["visuals"]})
    lines: list[str] = []
    revolute_names: list[str] = []

    def emit_link_contents(link_name: str, indent: int) -> None:
        pad = "  " * indent
        L = links[link_name]
        lines.append(
            f'{pad}<inertial pos="{_fmt(L["com"])}" mass="{L["mass"]:.6g}" '
            f'diaginertia="1e-4 1e-4 1e-4"/>'
        )
        for xyz, rpy, mesh_file, color in L["visuals"]:
            mname = Path(mesh_file).stem
            quat = _rpy_to_quat_wxyz(rpy)
            rgba = _fmt(np.array(color), 4)
            lines.append(
                f'{pad}<geom type="mesh" mesh="{mname}" pos="{_fmt(xyz)}" '
                f'quat="{_fmt(quat)}" rgba="{rgba}" contype="0" conaffinity="0" '
                f'group="1" density="0"/>'
            )
            lines.append(
                f'{pad}<geom type="mesh" mesh="{mname}" pos="{_fmt(xyz)}" '
                f'quat="{_fmt(quat)}" group="3"/>'
            )
        if link_name == "gripper_body":
            lines.append(
                f'{pad}<site name="tcp" pos="0 0 0" size="0.008" rgba="1 0.2 0.2 0.4"/>'
            )

    def emit_child_joint(j: dict, indent: int) -> None:
        pad = "  " * indent
        child = j["child"]
        quat = _rpy_to_quat_wxyz(j["rpy"])
        lines.append(
            f'{pad}<body name="{child}" pos="{_fmt(j["xyz"])}" quat="{_fmt(quat)}">'
        )
        if j["type"] == "revolute":
            revolute_names.append(j["name"])
            lines.append(
                f'{pad}  <joint name="{j["name"]}" type="hinge" axis="{_fmt(j["axis"])}" '
                f'range="{j["lo"]:.6g} {j["hi"]:.6g}"/>'
            )
        elif j["type"] != "fixed":
            raise ValueError(f"Unsupported joint type {j['type']}")
        emit_link_contents(child, indent + 1)
        for jj in children.get(child, []):
            emit_child_joint(jj, indent + 1)
        lines.append(f"{pad}</body>")

    lines.append('<mujoco model="so101_plus">')
    lines.append('  <compiler angle="radian" meshdir="assets/" autolimits="true"/>')
    lines.append('  <option timestep="0.002" gravity="0 0 -9.81"/>')
    lines.append("  <default>")
    lines.append('    <joint damping="0.5" armature="0.01" frictionloss="0.05"/>')
    lines.append('    <position kp="80" forcerange="-20 20"/>')
    lines.append('    <geom condim="3" friction="1 0.05 0.01" margin="0.001"/>')
    lines.append("  </default>")
    lines.append("  <asset>")
    for mf in mesh_names:
        lines.append(f'    <mesh name="{Path(mf).stem}" file="{mf}"/>')
    lines.append("  </asset>")
    lines.append("  <worldbody>")
    lines.append('    <light pos="0.4 -0.2 0.8" dir="-0.3 0.2 -1" diffuse="0.8 0.8 0.8"/>')
    lines.append(
        '    <camera name="frontview" pos="0.45 -0.55 0.35" '
        'xyaxes="1 0.2 0 -0.1 0.3 1" fovy="55"/>'
    )
    lines.append(f'    <body name="{root_link}" pos="0 0 0">')
    emit_link_contents(root_link, 3)
    for j in children.get(root_link, []):
        emit_child_joint(j, 3)
    lines.append("    </body>")
    lines.append("  </worldbody>")
    lines.append("  <actuator>")
    for jn in revolute_names:
        lines.append(
            f'    <position name="{jn}" joint="{jn}" kp="80" ctrlrange="-3.2 3.2"/>'
        )
    lines.append("  </actuator>")
    lines.append("</mujoco>")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_PATH}")
    print(f"Revolute joints ({len(revolute_names)}): {revolute_names}")


if __name__ == "__main__":
    main()
