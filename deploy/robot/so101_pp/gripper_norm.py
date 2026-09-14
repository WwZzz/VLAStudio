"""
Parallel-gripper normalized pose helpers for so101_pp.

Hardware (this PP gripper): higher raw encoder ≈ more closed (e.g. open≈1505, closed≈2700).
Public API convention (same as stock SO101): normalized 0=closed, 100=open.

We store in calibration JSON:
  - range_min / range_max = min/max raw encoder (open..closed or closed..open)
  - drive_mode = 1 if closed_raw > open_raw (needs invert), else 0

lerobot's bus drive_mode invert is unreliable for our path, so at runtime we force
bus drive_mode=0 and apply the invert ourselves when the flag is set.
"""

from __future__ import annotations

from lerobot.motors import MotorCalibration

GRIPPER = "gripper"
GRIPPER_POS = "gripper.pos"


def gripper_needs_invert(calibration: dict | None) -> bool:
    if not calibration or GRIPPER not in calibration:
        return False
    return bool(calibration[GRIPPER].drive_mode)


def disarm_bus_gripper_drive_mode(bus) -> bool:
    """
    Clear bus-level drive_mode for gripper (avoid double invert) and return
    whether so101_pp should invert normalized gripper values.
    """
    cal = bus.calibration.get(GRIPPER) if getattr(bus, "calibration", None) else None
    if cal is None:
        return False
    needs_invert = bool(cal.drive_mode)
    if needs_invert:
        bus.calibration[GRIPPER] = MotorCalibration(
            id=cal.id,
            drive_mode=0,
            homing_offset=cal.homing_offset,
            range_min=cal.range_min,
            range_max=cal.range_max,
        )
    return needs_invert


def flip_gripper_norm(values: dict, enabled: bool) -> dict:
    """In-place: gripper.pos <- 100 - gripper.pos when enabled."""
    if enabled and GRIPPER_POS in values:
        values[GRIPPER_POS] = 100.0 - float(values[GRIPPER_POS])
    return values
