"""Homing regression with the real freshness/E1/E2 policy, no serial hardware."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import numpy as np
from vlastudio.deploy.robot.bi_alicia_d.robot import BiAliciaD
from vlastudio.deploy.robot.bi_alicia_d.thermal_policy import ThermalPolicy


class Clock:
    now = 0.

    def sleep(self, seconds):
        self.now += seconds


class HomeArm:
    def __init__(self, name, body_serial, **kwargs):
        self.name, self.body_serial, self.port = name, body_serial, body_serial
        self.q = np.r_[np.full(6, .1), .5]
        self._last_sent_qpos = self.q[:6].copy()
        self._last_sent_gripper = 500
        self._ik_solver = None
        self._connected = False
        self.writes, self.queries = [], 0
        self.fail_at = None
        self.status = 0

    def attach(self, clock, directory):
        self.clock = clock
        policy = ThermalPolicy(clock=lambda: clock.now)
        policy.bind(Path(directory) / self.body_serial)
        self._driver = SimpleNamespace(acquire_info=self.query,
                                      serial_comm=SimpleNamespace(thermal_policy=policy))

    def query(self, *args, **kwargs):
        self.queries += 1
        if self.fail_at and self.queries >= self.fail_at:
            return False
        self._driver.serial_comm.thermal_policy.observe(self.status)
        return True

    def connect(self):
        self._connected = True
        return True

    def shutdown(self):
        self._connected = False

    def get_observation(self):
        return dict(qpos=self.q.copy())

    def publish_action(self, cmd):
        # Old set_home stops querying, so this fails after clock.now > 0.5.
        reason = self._driver.serial_comm.thermal_policy.reason()
        if reason:
            raise RuntimeError(reason)
        self.writes.append((self.clock.now, np.asarray(cmd).copy()))
        self.q = np.asarray(cmd).copy()
        self._last_sent_qpos = self.q[:6].copy()
        self._last_sent_gripper = int(self.q[6] * 1000)


class HomeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.clock = Clock()
        with patch.object(BiAliciaD, 'ARM_CLS', HomeArm):
            self.robot = BiAliciaD(home_duration_s=3.)
        for arm in (self.robot.left, self.robot.right):
            arm.attach(self.clock, self.directory.name)
        self.robot.connect()
        self.saved_home = self.robot._home.copy()
        self.robot.left.q[:6] += .3
        self.robot.right.q[:6] -= .2
        self.robot.shm = object()
        self.robot.write_data = Mock()
        self.addCleanup(patch.stopall)
        patch('vlastudio.deploy.robot.bi_alicia_d.robot.time.sleep', self.clock.sleep).start()
        patch('vlastudio.deploy.robot.bi_alicia_d.robot.time.monotonic', lambda: self.clock.now).start()

    def test_three_second_home_keeps_feedback_fresh_and_recording_alive(self):
        self.assertTrue(self.robot.set_home())
        self.assertGreater(self.clock.now, 2.99)
        for i, arm in enumerate((self.robot.left, self.robot.right)):
            self.assertGreater(arm.queries, 90)
            self.assertGreater(arm.writes[-1][0], .5)
            np.testing.assert_allclose(arm.q, self.saved_home[i*7:(i+1)*7])
            self.assertFalse(arm._rel_ee_vr_unsqueeze_active)
        self.assertGreater(self.robot.write_data.call_count, 90)

    def test_missing_right_feedback_stops_both_target_streams(self):
        self.robot.right.fail_at = 15
        self.assertFalse(self.robot.set_home())
        self.assertLess(self.clock.now, 1.)
        self.assertEqual(len(self.robot.left.writes), len(self.robot.right.writes))
        self.assertLess(len(self.robot.left.writes), 15)
        self.assertFalse(np.allclose(self.robot.left.q, self.saved_home[:7]))

    def test_e1_warnings_do_not_interrupt_home(self):
        self.robot.right.status = 0xE1
        self.assertTrue(self.robot.set_home())
        self.assertIsNone(self.robot.right._driver.serial_comm.thermal_policy.hard_fault)

    def test_tenth_e2_still_stops_home(self):
        self.robot.right.status = 0xE2
        self.assertFalse(self.robot.set_home())
        p = self.robot.right._driver.serial_comm.thermal_policy
        self.assertEqual(p.hard_fault, '0xe2_persistent')
        self.assertEqual(p.consecutive_e2, 10)
        self.assertLess(len(self.robot.left.writes), 10)

    def test_failed_send_does_not_advance_home(self):
        self.robot.right.publish_action = lambda cmd: None
        self.assertFalse(self.robot.set_home())
        self.assertEqual(len(self.robot.left.writes), 1)

    def test_final_send_without_physical_arrival_is_not_success(self):
        arm = self.robot.right
        original = arm.publish_action
        def stalled(cmd):
            measured = arm.q.copy()
            original(cmd)
            arm.q = measured  # Transport accepts commands, encoder does not follow.
        arm.publish_action = stalled
        self.assertFalse(self.robot.set_home())
        self.assertGreater(self.clock.now, 4.99)


if __name__ == '__main__':
    unittest.main()
