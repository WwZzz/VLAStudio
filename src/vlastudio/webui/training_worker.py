"""Detached supervisor and live Transformers logging for the existing training entrypoint."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .training import now, write


def execute(folder):
    print("[INFO] Training data cache: " + os.environ.get("VLASTUDIO_DATA_CACHE_DIR", "runtime default"), flush=True)
    from transformers import Trainer, TrainerCallback

    class Metrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero:
                values = {k: v for k, v in (logs or {}).items()
                          if type(v) in (int, float) and math.isfinite(v)}
                values.update(step=state.global_step, time=now())
                with (folder / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(values, allow_nan=False) + '\n')

    original = Trainer.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.add_callback(Metrics())

    Trainer.__init__ = initialize
    from vlastudio.entrypoints.train import main, parse_param
    sys.argv = ['train', '-p', str(folder / 'policy.yaml'), '-t', str(folder / 'task.yaml'),
                '-c', str(folder / 'training.yaml'), '-o', str(folder / 'model')]
    main(parse_param())


def supervise(folder):
    item = json.loads((folder / 'run.json').read_text())
    cancelled = False

    def cancel(*_):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, cancel)
    item.update(pid=os.getpid(), process_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
                status='running', started_at=now())
    write(folder / 'run.json', item)
    child = None
    try:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'vlastudio.webui.training_worker', str(folder), '--execute'], start_new_session=True)
        while child.poll() is None:
            if cancelled:
                item['status'] = 'stopping'
                write(folder / 'run.json', item)
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            time.sleep(.3)
        item.update(status='cancelled' if cancelled else 'completed' if child.returncode == 0 else 'failed',
                    exit_code=child.returncode)
    except Exception as exc:
        item.update(status='failed', error=str(exc))
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise
    finally:
        item['finished_at'] = now()
        write(folder / 'run.json', item)


if __name__ == '__main__':
    folder = Path(sys.argv[1]).resolve()
    if '--execute' in sys.argv:
        execute(folder)
    else:
        supervise(folder)
