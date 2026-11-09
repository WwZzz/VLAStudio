"""Stop OmniGibson after its current native frame, before interpreter teardown."""
import signal
import sys

from .benchmark_robot import BenchmarkRobot


class BehaviorRobot(BenchmarkRobot):
    process_shutdown_timeout_s = 30.0

    def _request_stop(self, signum, frame):
        # Closing Kit from a signal callback can interrupt its own native update.
        self._stop_requested = True
        self.is_running = False

    def _install_stop_handlers(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, self._request_stop)

    def connect(self):
        connected = super().connect()
        # OmniGibson installs its own SIGINT handler during import.
        self._install_stop_handlers()
        if getattr(self, '_stop_requested', False):
            raise KeyboardInterrupt
        return connected

    def start(self):
        self._stop_requested = False
        self._install_stop_handlers()
        return super().start()

    def close(self):
        if getattr(self, '_native_closed', False):
            return
        self._native_closed = True
        try:
            super().close()
        finally:
            # This device owns its dedicated process, so Kit can close here.
            og = sys.modules.get('omnigibson')
            if og is not None and og.app is not None:
                og.shutdown()
