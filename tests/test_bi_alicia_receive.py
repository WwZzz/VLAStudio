"""Transient thermal frames must not call a console sink under the SDK lock."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from vlastudio.deploy.robot.bi_alicia_d.body_binding import _ExactPort, SerialComm


class ReceiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.comm = _ExactPort(port='/dev/not-opened')
        self.comm.thermal_policy.bind(Path(self.directory.name) / 'ARM')

    def frame(self, status):
        # SerialComm.read_frame normally verifies CRC and framing before returning.
        frame = bytearray(21)
        frame[:4] = b'\xaa\x06\x00\x0f'
        frame[18] = status
        frame[-1] = 0xff
        return frame

    def test_alternating_single_e2_never_logs_or_inhibits(self):
        frames = [self.frame(s) for _ in range(100) for s in (0xE2, 0)]
        with patch.object(SerialComm, 'read_frame', side_effect=frames), patch(
                'vlastudio.deploy.robot.bi_alicia_d.body_binding.logger.warning',
                side_effect=AssertionError('Console called while communication lock is held')):
            for _ in frames:
                with self.comm._lock:
                    self.comm.read_frame()
                self.comm._check_packet(b'\xaa\x06\x03')
        self.assertIsNone(self.comm.thermal_policy.hard_fault)

    def test_tenth_e2_still_inhibits_without_console_io(self):
        with patch.object(SerialComm, 'read_frame', return_value=self.frame(0xE2)), patch(
                'vlastudio.deploy.robot.bi_alicia_d.body_binding.logger.warning',
                side_effect=AssertionError('Console called in receiver')):
            for _ in range(9):
                self.comm.read_frame()
                self.comm._check_packet(b'\xaa\x06\x03')
            self.comm.read_frame()
            with self.assertRaisesRegex(RuntimeError, '0xe2_persistent'):
                self.comm._check_packet(b'\xaa\x06\x03')
        self.assertTrue(self.comm.thermal_policy.path.exists())

    def test_e1_frames_are_silent_and_do_not_inhibit(self):
        with patch.object(SerialComm, 'read_frame', return_value=self.frame(0xE1)), patch(
                'vlastudio.deploy.robot.bi_alicia_d.body_binding.logger.warning',
                side_effect=AssertionError('Console called in receiver')):
            for _ in range(30):
                self.comm.read_frame()
                self.comm._check_packet(b'\xaa\x06\x03')


if __name__ == '__main__':
    unittest.main()
