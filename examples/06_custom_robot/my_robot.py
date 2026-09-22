"""
External custom robot example.

Demonstrates how to add a *new robot externally* without modifying VLAStudio
source: define your own robot class, reference it from a YAML config via a local
file path, and drive it with a keyboard teleoperator through ``vlastudio collect``.

The example reuses the package's MuJoCo SO101 simulation under the hood
(``So101SimRobot``), so it runs on any desktop with ``mujoco`` installed. A
hardware robot would implement ``AbstractRobotInterface`` the same way — see the
method table in ``README.md``.
"""

from typing import Dict

import numpy as np

from vlastudio.deploy.robot.base import AbstractRobotInterface
from vlastudio.deploy.robot.so101_sim.robot import So101SimRobot


class MySimSo101(So101SimRobot):
    """A custom SO101 simulated robot built externally.

    Subclassing ``So101SimRobot`` reuses the MuJoCo model and the IK/teleop logic;
    the subclass demonstrates the extension points:

    - constructor: extra hardware parameters arrive from the YAML ``args``
    - ``connect()``: one-time hardware bring-up hook
    - ``get_action_dim()``: the action space the teleop/policy must produce
    - ``_get_robot_observation_core()``: what the robot publishes to its SHM
    - ``obs2meta()``: how that SHM observation maps into the policy ``state``

    For a from-scratch robot, implement the abstract members of
    ``AbstractRobotInterface`` instead of subclassing (see ``README.md``).
    """

    def __init__(self, custom_serial: str = "SO101-DEMO", **kwargs):
        super().__init__(**kwargs)
        self.custom_serial = custom_serial
        print(f"[MySimSo101] serial={self.custom_serial}, mode={self.control_mode}")

    def connect(self):
        ok = super().connect()
        if ok:
            print(f"[MySimSo101] '{self.custom_serial}' connected")
        return ok

    def get_action_dim(self) -> int:
        # delta_ee is 7D (dx,dy,dz,d_roll,d_pitch,d_yaw,d_gripper); qpos is 6D.
        return 7 if self.control_mode == "delta_ee" else 6

    def _get_robot_observation_core(self) -> Dict[str, np.ndarray]:
        obs = super()._get_robot_observation_core()
        obs["custom_serial_tag"] = np.asarray([len(self.custom_serial)], dtype=np.float32)
        return obs

    def obs2meta(self, device_data):
        if device_data is None:
            return {}
        qpos = device_data.get("qpos", np.zeros(6, dtype=np.float32))
        return {"state": np.asarray(qpos, dtype=np.float32)}


def check_contract():
    """Documentation aid: enumerate the abstract robot interface members."""
    members = [
        m for m in AbstractRobotInterface.__abstractmethods__
    ]
    return sorted(members)


if __name__ == "__main__":
    print("AbstractRobotInterface must implement:", check_contract())
