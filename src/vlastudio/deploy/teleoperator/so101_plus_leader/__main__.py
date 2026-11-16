"""Interactive calibration for the active Rebot leader."""
import argparse
import json
import time

import numpy as np
import yaml
from pathlib import Path

from .controller import _default_robots_calib_dir
from . import SO101PlusLeader, SO101PlusLeaderConfig
from .mapping import JointMap, ORDER
from .calibration import digest, load_reference, lock_port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", default="so101_leader_plus")
    parser.add_argument("--calibration-dir", type=Path, default=_default_robots_calib_dir())
    parser.add_argument("--preview", action="store_true", help="Print mapped Rebot targets while moving the unpowered leader by hand")
    args = parser.parse_args()
    dev = SO101PlusLeader(SO101PlusLeaderConfig(port=args.port, id=args.id,
                                                calibration_dir=args.calibration_dir))
    port_lock = lock_port(args.port)
    try:
        dev.bus.connect()
        if args.preview:
            if not dev.is_calibrated:
                raise ValueError("Motor calibration differs from the saved file")
            ref = load_reference(args.calibration_dir, args.id)
            config = Path(__file__).resolve().parents[3] / "configs/teleop/so101_leader_plus.yaml"
            mapping = JointMap.from_reference(yaml.safe_load(config.read_text())["args"]["joints"], ref)
            q = np.zeros(7)
            while True:
                raw = dev.bus.sync_read("Present_Position")
                q = mapping.target([raw[k] for k in ORDER], q)
                print("Rebot J1..J6 degrees", np.rad2deg(q[:6]).round(1), "opening", round(q[6], 2), flush=True)
                time.sleep(.2)
        dev.bus.disable_torque()
        dev.calibrate()
        input("With torque OFF, place the upper arm horizontal and elbow at the pose matching "
              "Rebot J2/J3 zero. Press Enter to measure. ")
        values = dev.bus.sync_read("Present_Position")
        ref = {k: float(values[k]) for k in ORDER}
        input("Hold the leader trigger CLOSED, then press Enter. ")
        closed = dev.bus.read("Present_Position", "gripper")
        input("Hold the leader trigger OPEN, then press Enter. ")
        opened = dev.bus.read("Present_Position", "gripper")
        if abs(closed-opened)<40:
            raise ValueError("Trigger travel too small; verify open/closed and calibration")
        path = dev.calibration_fpath.with_suffix(".reference.json")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(joints=ref, closed=closed, opened=opened,
                                       port=args.port, calibration=digest(dev.calibration_fpath)), indent=2))
        # Validate the configured landmarks before publishing the reference.
        config = Path(__file__).resolve().parents[3] / "configs/teleop/so101_leader_plus.yaml"
        JointMap.from_reference(yaml.safe_load(config.read_text())["args"]["joints"],
                                json.loads(tmp.read_text()))
        tmp.replace(path)
        print("Saved", path)
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        try:
            if dev.is_connected:
                dev.bus.disconnect(disable_torque=not args.preview)
        finally:
            port_lock.close()


if __name__ == "__main__":
    main()
