"""Quest dual Alicia-D launcher; --check never sends motor/torque commands."""
import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def check(config, cameras=True):
    from vlastudio.deploy.robot.bi_alicia_d import BiAliciaD
    from vlastudio.deploy.robot.bi_alicia_d.body_binding import SerialComm, ServoDriver
    from vlastudio.utils.extensions import resolve
    allowed = {bytes(ServoDriver.INFO_COMMAND_MAP[k]) for k in ("version", "joint_gripper")}
    original = SerialComm.send_data
    def query_only(comm, data):
        if bytes(data) not in allowed:
            raise RuntimeError("Read-only check blocked a non-query packet")
        return original(comm, data)
    with patch.object(SerialComm, "send_data", query_only):
        robot = BiAliciaD(**config[0]["args"])
        try:
            robot.connect()
            for side, arm in (("left", robot.left), ("right", robot.right)):
                assert arm._driver.acquire_info("joint_gripper", wait=True, timeout=1)
                q = arm.get_observation()["qpos"][:6]
                solver = arm.ik_solver
                T = solver.compute_fk(q)
                solver.reset(q)
                # IK computation only: no result is sent to the arm.
                result = solver.solve(T)
                if not result[0]:
                    raise RuntimeError(side + " IK self-check failed")
                print(side, arm.body_serial, arm.port, "FK", T[:3, 3].tolist(), "IK", str(result))
        finally:
            robot.shutdown()
    if cameras:
        import concurrent.futures
        cams = []
        try:
            for row in config[1:]:
                cams.append(resolve(row["type"])(**row["args"]))
            def capture(cam):
                start = time.monotonic()
                count, first = 0, None
                while time.monotonic() - start < 3:
                    frame = cam.get_data()
                    if frame is not None:
                        count += 1
                        if first is None:
                            first = frame["image"]
                if first is None:
                    raise RuntimeError(cam.name + " has no frames")
                return cam.name, count / (time.monotonic() - start), first
            with concurrent.futures.ThreadPoolExecutor(3) as pool:
                frames = list(pool.map(capture, cams))
            import cv2
            tiles = []
            for name, fps, rgb in frames:
                print(name, "capture_fps", round(fps, 2))
                tile = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(tile, name, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,0), 2)
                tiles.append(tile)
            output = ROOT / "logs/bi_alicia_camera_check.jpg"
            output.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(output), np.concatenate(tiles, axis=1))
            print("Camera comparison:", output)
        finally:
            for cam in cams:
                cam.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="read arms, compute IK, test cameras; no motion")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--exposure-sweep-ms", type=float, nargs='+',
                        help="camera-only test through teleop camera processes; never connects arms/Quest")
    parser.add_argument("--measure-seconds", type=float, default=8.)
    parser.add_argument("--wrist-gain", type=int, default=None, help="temporary gain override in exposure test")
    parser.add_argument("--wrist-gamma", type=int, default=None, help="temporary gamma override in exposure test")
    parser.add_argument("--auto-exposure-test", action="store_true", help="camera-only automatic exposure test")
    parser.add_argument("--exposure-output", default=str(ROOT / 'logs/wrist_exposure'))
    parser.add_argument("--visualize", action="store_true", help="show camera windows on Jetson desktop")
    parser.add_argument("--smoke-seconds", type=float, default=0,
                        help="start full stack with arm writes restricted to queries, then exit")
    parser.add_argument("--robot", default=str(ROOT / "src/vlastudio/configs/robot/bi_alicia_d_rel_ee.yaml"))
    parser.add_argument("--teleop", default=str(ROOT / "src/vlastudio/configs/teleop/quest3_bi_alicia_rel_ee.yaml"))
    args = parser.parse_args()
    if args.smoke_seconds > 0:
        os.environ["VLASTUDIO_ALICIA_READ_ONLY"] = "1"
    config = yaml.safe_load(Path(args.robot).read_text())
    if args.exposure_sweep_ms or args.auto_exposure_test:
        import fcntl
        from benchmark_wrist_exposure import run
        with (Path.home() / '.cache/vlastudio-bi-alicia-teleop.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(config, args.exposure_sweep_ms, args.measure_seconds, args.exposure_output,
                gain=args.wrist_gain, gamma=args.wrist_gamma, auto_exposure=args.auto_exposure_test)
        return
    if args.check:
        check(config, not args.no_cameras)
        return
    from vlastudio.deploy.utils import load_device_configs, start_devices, get_shm_name
    from vlastudio.deploy.base import is_camera_config
    from vlastudio.deploy.shm_utils import cleanup_all_shm
    from vlastudio.deploy.visualizer.base import start_all_visualizers, stop_all_visualizers
    configs, names, _, visualizers = load_device_configs(args, [])
    if args.no_cameras:
        configs = [c for c in configs if not is_camera_config(c)]
        names = [get_shm_name(c) for c in configs]
    # Prevent a second launch from cleaning SHM used by the first session.
    import fcntl
    (Path.home() / ".cache").mkdir(exist_ok=True)
    with (Path.home() / ".cache/vlastudio-bi-alicia-teleop.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cleanup_all_shm(names)
        processes, viz = [], []
        try:
            processes = start_devices(configs)
            started = time.monotonic()
            if args.visualize:
                viz = start_all_visualizers(device_configs=configs, get_shm_name_func=get_shm_name,
                                           is_camera_config_func=is_camera_config,
                                           visualizer_configs=visualizers)
            print("Quest browser: https://<JETSON-IP>:8443 (accept certificate, enter VR). Ctrl+C exits.", flush=True)
            while True:
                for process in processes:
                    if not process.is_alive():
                        raise RuntimeError("Device process exited: " + process.name)
                if args.smoke_seconds and time.monotonic() - started >= args.smoke_seconds:
                    print("Read-only startup smoke completed; all device processes remained alive.", flush=True)
                    break
                time.sleep(.5)
        except KeyboardInterrupt:
            pass
        finally:
            stop_all_visualizers(viz)
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join()
            cleanup_all_shm(names)


if __name__ == "__main__":
    main()
