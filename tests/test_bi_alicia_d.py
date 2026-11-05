"""Routing regressions: no hardware is opened by these tests."""
import unittest
from unittest.mock import patch
import numpy as np
from vlastudio.deploy.robot.bi_alicia_d.robot import BiAliciaD


class FakeArm:
    def __init__(self, name, body_serial, **kwargs):
        self.name, self.body_serial = name, body_serial
        self.port = body_serial
        self.gripper_scale, self.gripper_absolute = 1., True
        self._target_gripper = 500.
        self._smooth_qpos = np.ones(6) * .1
        self._target_qpos = self._smooth_qpos.copy()
        self.inputs, self.writes = [], []
        self._connected = False

    def connect(self):
        self._connected = True
        return True

    def shutdown(self):
        self._connected = False

    def get_observation(self):
        return dict(qpos=np.r_[self._smooth_qpos, .5])

    def process_action(self, message):
        self.inputs.append(message)
        return dict(action=np.r_[self._smooth_qpos, .5] if message["unsqueeze_active"] else None)

    def publish_action(self, cmd, **kwargs):
        self.writes.append((np.array(cmd), kwargs))


class RoutingTests(unittest.TestCase):
    def setUp(self):
        with patch.object(BiAliciaD, "ARM_CLS", FakeArm):
            self.robot = BiAliciaD(gripper_absolute=True)
        self.robot.connect()

    def message(self, left, right):
        return dict(action=np.r_[np.arange(6) * .001, .5, np.arange(6) * .002, .5],
                    left_unsqueeze_active=left, right_unsqueeze_active=right)

    def test_left_only_does_not_write_right(self):
        self.robot.dispatch(self.message(True, False))
        self.assertEqual(len(self.robot.left.writes), 1)
        self.assertEqual(len(self.robot.right.writes), 0)
        self.assertTrue(self.robot.left.inputs[-1]["anchor_just_set"])
        self.assertFalse(self.robot.right.inputs[-1]["anchor_just_set"])

    def test_edges_are_per_hand(self):
        self.robot.dispatch(self.message(True, False))
        self.robot.dispatch(self.message(True, True))
        self.assertFalse(self.robot.left.inputs[-1]["anchor_just_set"])
        self.assertTrue(self.robot.right.inputs[-1]["anchor_just_set"])
        self.robot.dispatch(self.message(False, True))
        self.assertFalse(self.robot.left.inputs[-1]["unsqueeze_active"])

    def test_bad_vector_is_not_published(self):
        for a in (np.zeros(7), np.full(14, np.nan)):
            with self.assertRaises(ValueError):
                self.robot.dispatch(dict(action=a))
        self.assertFalse(self.robot.left.writes or self.robot.right.writes)

    def test_measured_observation_order(self):
        obs = self.robot.get_observation()
        self.assertEqual(obs["qpos"].shape, (14,))
        np.testing.assert_array_equal(self.robot._home, obs["qpos"])

    def test_gripper_isolated_and_normalized(self):
        m = self.message(False, True)
        m["action"][13] = .8
        self.robot.dispatch(m)
        self.assertFalse(self.robot.left.writes)
        self.assertEqual(self.robot.right._target_gripper, 800.)
        self.assertEqual(self.robot.right.writes[-1][0][6], .8)
        self.assertTrue(self.robot.right.writes[-1][1]["from_ik_teleop"])

    def test_qpos_routes_both(self):
        self.robot.control_mode = "qpos"
        self.robot.dispatch(self.message(False, False))
        self.assertEqual(len(self.robot.left.writes), 1)
        self.assertEqual(len(self.robot.right.writes), 1)

    def test_second_arm_failure_closes_first(self):
        self.robot.right.connect = lambda: False
        with self.assertRaises(ConnectionError):
            self.robot.connect()
        self.assertFalse(self.robot.left._connected)


if __name__ == "__main__":
    unittest.main()
