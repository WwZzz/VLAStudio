"""MuJoCo visualizer for SO101 Plus sim (optional; in-process viewer is preferred)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from deploy.simulation.mujoco.base import prepare_mujoco_xml_path
from deploy.visualizer.base import BaseVisualizer

MJ_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
# Obs qpos is hardware order; remap to MuJoCo/URDF order for display.
_HW_TO_MJ = np.array([0, 1, 2, 3, 5, 4, 6], dtype=np.int64)


class Visualizer(BaseVisualizer):
    def __init__(
        self,
        shm_name: str,
        fps: float = 60.0,
        xml_path: Optional[str] = None,
        scene_name: Optional[str] = None,
        scene_xml_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(shm_name=shm_name, fps=fps, **kwargs)
        module_dir = Path(__file__).parent
        robot_xml = str(module_dir / "mujoco_model" / "so101_plus.xml")
        loaded, _ = prepare_mujoco_xml_path(
            robot_xml,
            xml_path=xml_path,
            scene_name=scene_name,
            scene_xml_path=scene_xml_path or str(module_dir / "mujoco_model" / "scene.xml"),
            generated_name="so101_plus_visualizer",
        )
        self.model = mujoco.MjModel.from_xml_path(loaded)
        self.data = mujoco.MjData(self.model)
        self.qpos_indices = np.array(
            [self.model.jnt_qposadr[self.model.joint(n).id] for n in MJ_JOINT_NAMES],
            dtype=np.int32,
        )

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                msg = self.read_data()
                if msg is not None and "qpos" in msg:
                    q_hw = np.asarray(msg["qpos"], dtype=np.float64).reshape(-1)
                    if q_hw.size >= 7:
                        q_mj = q_hw[_HW_TO_MJ]
                        self.data.qpos[self.qpos_indices] = q_mj
                        mujoco.mj_forward(self.model, self.data)
                viewer.sync()
                self.rate_sleep()
