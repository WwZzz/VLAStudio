"""
MuJoCo passive viewer for BessicaSimRobot (14-DOF dual arm + 2 gripper).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from vlastudio.deploy.simulation.mujoco.base import prepare_mujoco_xml_path
from vlastudio.deploy.visualizer.base import BaseVisualizer

from .robot import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES, GRIPPER_MIRROR_JOINT_NAMES


def _draw_frame(viewer, pos: np.ndarray, axis_length: float = 0.05, axis_radius: float = 0.002):
    axes = [
        (np.array([1.0, 0.0, 0.0]), (1.0, 0.0, 0.0, 0.6)),
        (np.array([0.0, 1.0, 0.0]), (0.0, 1.0, 0.0, 0.6)),
        (np.array([0.0, 0.0, 1.0]), (0.0, 0.0, 1.0, 0.6)),
    ]
    with viewer.lock():
        for axis, color in axes:
            start = pos
            end = pos + axis * axis_length
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).flatten(),
                np.array(color, dtype=np.float32),
            )
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, axis_radius, start, end)
            viewer.user_scn.ngeom += 1


class Visualizer(BaseVisualizer):
    def __init__(
        self,
        shm_name: str,
        fps: float = 60.0,
        xml_path: Optional[str] = None,
        scene_name: Optional[str] = None,
        scene_xml_path: Optional[str] = None,
        show_tcp_frame: bool = True,
        **kwargs,
    ):
        super().__init__(shm_name=shm_name, fps=fps, **kwargs)
        self.xml_path = xml_path
        self.scene_name = scene_name
        self.scene_xml_path = scene_xml_path
        self.mjmodel = None
        self.mjdata = None
        self.viewer = None
        self.arm_qpos_indices = None
        self.gripper_qpos_indices = None
        self.gripper_mirror_qpos_indices = None
        self.gripper_ctrl_indices = None
        self._tcp_r = None
        self._tcp_l = None
        self.show_tcp_frame = show_tcp_frame
        self._generated_xml_path = None

    def setup(self) -> bool:
        try:
            robot_xml_path = str(Path(__file__).parent / "mujoco_model" / "bessica_d.xml")
            self.xml_path, self._generated_xml_path = prepare_mujoco_xml_path(
                robot_xml_path,
                xml_path=self.xml_path,
                scene_name=self.scene_name,
                scene_xml_path=self.scene_xml_path,
                generated_name="bessica_visualizer",
            )
            self.mjmodel = mujoco.MjModel.from_xml_path(self.xml_path)
            self.mjdata = mujoco.MjData(self.mjmodel)
            self.arm_qpos_indices = np.array(
                [self.mjmodel.jnt_qposadr[self.mjmodel.joint(n).id] for n in ARM_JOINT_NAMES],
                dtype=np.int32,
            )
            self.gripper_qpos_indices = np.array(
                [self.mjmodel.jnt_qposadr[self.mjmodel.joint(n).id] for n in GRIPPER_JOINT_NAMES],
                dtype=np.int32,
            )
            self.gripper_mirror_qpos_indices = np.array(
                [self.mjmodel.jnt_qposadr[self.mjmodel.joint(n).id] for n in GRIPPER_MIRROR_JOINT_NAMES],
                dtype=np.int32,
            )
            self.gripper_ctrl_indices = np.array(
                [self.mjmodel.actuator(n).id for n in GRIPPER_JOINT_NAMES],
                dtype=np.int32,
            )
            try:
                self._tcp_r = self.mjmodel.site("tcp_right").id
                self._tcp_l = self.mjmodel.site("tcp_left").id
            except KeyError:
                self._tcp_r = self._tcp_l = None
            self.viewer = mujoco.viewer.launch_passive(self.mjmodel, self.mjdata)
            return True
        except Exception as e:
            print(f"[Bessica Visualizer] setup failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def visualize(self, data: dict) -> bool:
        if self.viewer is None or not self.viewer.is_running():
            return False
        qpos = data.get("qpos", None)
        if qpos is None:
            return True
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.size != 16:
            return True
        arm_q = np.concatenate([qpos[:7], qpos[8:15]])
        grip_q = np.array([qpos[7], qpos[15]], dtype=np.float64)
        grip_slide_q = grip_q / 2.0
        self.mjdata.qpos[self.arm_qpos_indices] = arm_q
        self.mjdata.qpos[self.gripper_qpos_indices] = grip_slide_q
        self.mjdata.qpos[self.gripper_mirror_qpos_indices] = grip_slide_q
        self.mjdata.ctrl[self.gripper_ctrl_indices] = grip_slide_q
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        if self.show_tcp_frame and hasattr(self.viewer, 'user_scn'):
            self.viewer.user_scn.ngeom = 0
            if self._tcp_r is not None:
                _draw_frame(self.viewer, self.mjdata.site_xpos[self._tcp_r].copy(), 0.07, 0.003)
            if self._tcp_l is not None:
                _draw_frame(self.viewer, self.mjdata.site_xpos[self._tcp_l].copy(), 0.07, 0.003)
        self.viewer.sync()
        return True

    def cleanup(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        self.mjmodel = None
        self.mjdata = None
        if self._generated_xml_path is not None:
            try:
                Path(self._generated_xml_path).unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
            self._generated_xml_path = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shm", "-s", default="bessica_sim_test")
    args = parser.parse_args()
    Visualizer(shm_name=args.shm).start()
