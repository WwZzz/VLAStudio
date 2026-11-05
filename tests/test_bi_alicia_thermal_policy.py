import json
from pathlib import Path
import tempfile
import unittest

from vlastudio.deploy.robot.bi_alicia_d.thermal_policy import ThermalPolicy


class ThermalPolicyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "ARM"
        self.now = 0.0
        self.p = ThermalPolicy(clock=lambda: self.now)
        self.p.bind(self.path)

    def frames(self, code, count):
        for _ in range(count):
            self.now += .05
            self.p.observe(code)

    def test_nine_warnings_do_not_pause_or_persist(self):
        self.frames(0xE1, 9)
        self.assertFalse(self.path.exists())
        self.assertIsNone(self.p.reason())
        self.frames(0, 10)
        self.assertIsNone(self.p.reason())

    def test_persistent_e1_never_locks_or_persists(self):
        for _ in range(1000):
            self.frames(0xE1, 1)
            self.assertIsNone(self.p.reason())
        self.assertIsNone(self.p.hard_fault)
        self.assertFalse(self.p.needs_ack)
        self.assertFalse(self.path.exists())
        self.frames(0, 10)
        self.assertIsNone(self.p.reason())

    def test_idle_resets_counter(self):
        self.frames(0xE1, 9)
        self.frames(0, 1)
        self.frames(0xE1, 9)
        self.assertFalse(self.path.exists())
        self.assertEqual(self.p.consecutive_e1, 9)
        self.assertIsNone(self.p.reason())

    def test_missing_packets_cannot_join_two_warning_bursts(self):
        self.frames(0xE1, 9)
        self.now += 1
        self.frames(0xE1, 1)
        self.assertEqual(self.p.consecutive_e1, 1)
        self.assertFalse(self.path.exists())

    def test_tenth_e2_is_first_pause_and_survives_restart(self):
        for _ in range(9):
            self.frames(0xE2, 1)
            self.assertIsNone(self.p.reason())
            self.assertFalse(self.path.exists())
        self.frames(0xE2, 1)
        self.assertIsNotNone(self.p.reason())
        self.assertEqual(json.loads(self.path.read_text())["consecutive_e2"], 10)
        self.p = ThermalPolicy(clock=lambda: self.now)
        self.p.bind(self.path)
        self.frames(0, 12)
        with self.assertRaises(RuntimeError):
            self.p.acknowledge_recovery([40] * 10)
        self.assertIsNotNone(self.p.reason())

    def test_confirmed_e2_upgrades_legacy_marker(self):
        self.path.write_bytes(ThermalPolicy.LEGACY_E1)
        self.p.bind(self.path)
        self.frames(0xE2, 10)
        self.assertEqual(json.loads(self.path.read_text())["code"], "0xe2_persistent")
        self.p = ThermalPolicy(clock=lambda: self.now)
        self.p.bind(self.path)
        self.frames(0, 12)
        with self.assertRaises(RuntimeError):
            self.p.acknowledge_recovery([40] * 10)

    def test_legacy_single_warning_recovery_is_archived(self):
        self.path.write_bytes(ThermalPolicy.LEGACY_E1)
        self.p.bind(self.path)
        self.frames(0, 9)
        with self.assertRaises(RuntimeError):
            self.p.acknowledge_recovery([40] * 10)
        self.frames(0, 1)
        with self.assertRaises(RuntimeError):
            self.p.acknowledge_recovery([106] + [40] * 9)
        self.p.acknowledge_recovery([40] * 10)
        self.assertFalse(self.path.exists())
        self.assertEqual(len(list((self.path.parent / "archive").iterdir())), 1)

    def test_caching_and_stale_data_do_not_clear_warning(self):
        self.frames(0xE1, 1)
        for _ in range(50):
            self.p.reason()
        self.assertEqual(self.p.consecutive_e1, 1)
        self.frames(0, 10)
        self.now += 1
        with self.assertRaises(RuntimeError):
            self.p.acknowledge_recovery([40] * 10)

    def test_e1_cannot_clear_confirmed_e2(self):
        self.frames(0xE2, 10)
        self.frames(0xE1, 50)
        self.assertEqual(self.p.hard_fault, "0xe2_persistent")
        self.assertEqual(json.loads(self.path.read_text())["code"], "0xe2_persistent")
        self.assertIsNotNone(self.p.reason())

    def test_recorded_left_single_e2_is_not_confirmed(self):
        data = json.loads(Path(__file__).with_name("left_health_frames_0129.json").read_text())
        for t, hx, *_ in data["frames"]:
            self.now = t
            b = bytes.fromhex(hx)
            self.p.observe(b[18])
        self.assertIsNone(self.p.hard_fault)
        self.assertIsNone(self.p.reason())
        self.assertFalse(self.path.exists())

    def test_idle_or_e1_resets_e2(self):
        for interruption in (0, 0xE1):
            self.frames(0xE2, 9)
            self.frames(interruption, 1)
            self.frames(0xE2, 9)
            self.assertEqual(self.p.consecutive_e2, 9)
            self.assertIsNone(self.p.reason())
            self.frames(0, 1)

    def test_e2_resets_e1(self):
        self.frames(0xE1, 9)
        self.frames(0xE2, 1)
        self.frames(0xE1, 9)
        self.assertEqual(self.p.consecutive_e1, 9)
        self.assertIsNone(self.p.reason())

    def test_e2_cache_reads_do_not_count(self):
        self.frames(0xE2, 1)
        for _ in range(50):
            self.assertIsNone(self.p.reason())
        self.assertEqual(self.p.consecutive_e2, 1)

    def test_e2_gap_resets_consecutive_run(self):
        self.frames(0xE2, 9)
        self.now += 1
        self.frames(0xE2, 1)
        self.assertEqual(self.p.consecutive_e2, 1)
        self.assertIsNone(self.p.reason())

    def test_legacy_e2_requires_healthy_acknowledgement(self):
        self.path.write_bytes(ThermalPolicy.LEGACY_RECORDED_E2)
        self.p.bind(self.path)
        self.frames(0, 9)
        self.assertIsNotNone(self.p.reason())
        with self.assertRaises(RuntimeError):
            self.p.acknowledge_recovery([40] * 10)
        self.frames(0, 1)
        self.p.acknowledge_recovery([40] * 10)
        self.assertFalse(self.path.exists())
        self.assertIsNone(self.p.reason())
        self.assertEqual(len(list((self.path.parent / "archive").iterdir())), 1)

    def test_unknown_record_not_cleared(self):
        self.path.write_text("Unclassified fault")
        self.p.bind(self.path)
        self.frames(0, 10)
        with self.assertRaises(RuntimeError):
            self.p.acknowledge_recovery([40] * 10)
        self.assertIsNotNone(self.p.reason())

    def test_packet_guard_allows_e1_and_blocks_tenth_e2(self):
        from vlastudio.deploy.robot.bi_alicia_d.body_binding import _ExactPort
        comm = _ExactPort(port="/dev/not-opened")
        comm.thermal_policy = self.p
        self.frames(0xE1, 100)
        # Only exercise the write guard; no serial port is opened or written.
        comm._check_packet(b"\xaa\x06\x03")
        self.frames(0xE2, 9)
        comm._check_packet(b"\xaa\x06\x03")
        self.frames(0xE2, 1)
        with self.assertRaisesRegex(RuntimeError, "0xe2_persistent"):
            comm._check_packet(b"\xaa\x06\x03")


if __name__ == "__main__":
    unittest.main()
