"""Display the simulated SO101 camera feed in real time.

Two modes:

1. Standalone (default): starts the custom external robot
   (examples/06_custom_robot/my_robot.py) and shows its ``frontview`` camera.

2. ``--connect``: attaches to an already-running robot's shared memory
   (e.g. one started by ``vlastudio collect`` for keyboard teleop) and shows the
   camera feed alongside it, without spawning a second robot.

Cross-platform behaviour:
- Windows: opens an OpenCV window.
- Linux/macOS: opens a window only when DISPLAY is set (physical X server or
  Xvfb, e.g. ``xvfb-run -a python view_camera.py``). Headless sessions read
  frames and print progress without opening a window.
"""

import argparse

import vlastudio as vla
from vlastudio.deploy.camera_viewer import LiveCameraViewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connect", action="store_true",
        help="attach to an already-running robot (e.g. via vlastudio collect) "
             "instead of starting one",
    )
    parser.add_argument(
        "--shm", default="my_so101_sim",
        help="robot shared-memory name to attach to when --connect is set",
    )
    parser.add_argument(
        "--robot", default="examples/06_custom_robot/my_robot.yaml",
        help="robot config for standalone mode",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--title", default="SO101 Sim - frontview")
    args = parser.parse_args()

    if args.connect:
        viewer = LiveCameraViewer(
            args.shm, camera_names=["frontview"],
            fps=args.fps, title=args.title,
        )
        n = viewer.run()
    else:
        n = vla.show_camera(
            args.robot,
            camera_names=["frontview"],
            fps=args.fps,
            title=args.title,
        )
    print(f"viewed {n} frames")


if __name__ == "__main__":
    main()
