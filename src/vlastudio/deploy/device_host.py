"""Own robot and sensor workers until the controlling connection closes."""
from pathlib import Path
import os
import json
import select
import signal
import sys

from vlastudio.deploy.shm_utils import SharedMemoryChannel, cleanup_all_shm
from vlastudio.deploy.utils import get_shm_name, start_devices, stop_device_processes
from vlastudio.utils.extensions import resolve


def main():
    rows = json.loads(sys.stdin.readline())
    workers = []
    waiting = {}
    ready_file = Path(os.environ["VLASTUDIO_ROBOT_READY_FILE"])
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        cleanup_all_shm([get_shm_name(row) for row in rows])
        for cfg in rows:
            if stopping:
                return
            timeout = getattr(resolve(cfg["type"]), "process_shutdown_timeout_s", 8.)
            process, = start_devices([cfg])
            workers.append((get_shm_name(cfg), process, timeout))
        waiting = {name: SharedMemoryChannel(name, timeout=0.) for name, _, _ in workers}
        while not stopping:
            if waiting:
                for name, reader in list(waiting.items()):
                    if reader.read() is not None:
                        reader.destroy()
                        del waiting[name]
                if not waiting:
                    ready_file.touch()
            for name, process, _ in workers:
                if process.exitcode is not None:
                    raise RuntimeError(f"{name} exited unexpectedly ({process.exitcode})")
            if select.select([sys.stdin], [], [], .1)[0] and not sys.stdin.read(1):
                break
    finally:
        for reader in waiting.values():
            reader.destroy()
        stop_device_processes(workers)


if __name__ == "__main__":
    main()
