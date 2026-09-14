"""
Bimanual SO101++ - two independent So101PP arms + Quest3 dual rel_ee.

Action layout (14D, same as bi_so101_plus):
  [L_xyz(3), L_rpy(3), L_grip, R_xyz(3), R_rpy(3), R_grip]
"""

from __future__ import annotations

from loguru import logger

from deploy.robot.bi_so101_plus.robot import BiSo101Plus
from deploy.robot.so101_pp.robot import So101PP


class BiSo101PP(BiSo101Plus):
    """Compose left + right ``So101PP`` for 14D dual VR teleop."""

    ARM_CLS = So101PP

    def __init__(
        self,
        *args,
        left_arm_id: str = "so101_pp_left",
        right_arm_id: str = "so101_pp_right",
        **kwargs,
    ):
        super().__init__(
            *args,
            left_arm_id=left_arm_id,
            right_arm_id=right_arm_id,
            **kwargs,
        )
        logger.info("[BiSo101PP] using SO101++ parallel-gripper backend for both arms")
