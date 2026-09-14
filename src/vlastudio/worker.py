"""Runs legacy entry points only after the selected environment is ready."""
import runpy
import sys
from .paths import legacy_root

ENTRIES = {"device": None, "train": "train.py", "serve": "start_policy_server.py", "eval-real": "eval_real.py", "eval-sim": "eval_sim.py", "collect": "collect_data.py"}


def main():
    argv = sys.argv[1:]
    entrypoint = None
    if argv[0] == "--entrypoint":
        entrypoint, argv = argv[1], argv[2:]
    command, *args = argv
    root = legacy_root()
    # Compatibility imports stay confined to the explicitly launched worker.
    sys.path.insert(0, str(root))
    if entrypoint:
        from .extensions import resolve
        result = resolve(entrypoint)(command, args)
        raise SystemExit(result or 0)
    if command == "device":
        import argparse
        import signal
        from .configuration import read_config
        from .extensions import create
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", required=True)
        options = parser.parse_args(args)
        config, _ = read_config(options.config, "device")
        device = create(config)
        def stop(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, stop)
        try:
            device.start()
        finally:
            device.close()
        return
    sys.argv = [str(root / ENTRIES[command]), *args]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
