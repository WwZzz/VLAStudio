"""Small synchronous client for a persistent native execution subprocess."""
from __future__ import annotations

import copy
import multiprocessing as mp
import os
from pathlib import Path
import signal
import threading
import time


class IntentSupersededError(RuntimeError):
    """The requested intent was replaced before its result was collected."""


def _session_entry(config, output_dir, connection):
    # Keep runtime/model imports in the subprocess. The client owns no control loop.
    from .worker import run_session
    run_session(config, output_dir, connection)


def _error_text(error):
    if isinstance(error, dict):
        return str(error.get('message', error.get('error', error)))
    return str(error)


class ExecutionSession:
    """Submit intents; VLAStudio owns observation, inference and actuation timing.

    Pause/cancel acknowledge that the runtime has stopped new publications. They
    are not a hardware emergency stop or a claim that physical motion has ended.
    Use a context manager or call close() to release the native processes and SHM.
    """

    def __init__(self, config, *, output_dir, cancel_event=None):
        self._cancel_event = cancel_event
        self.config = copy.deepcopy(config)
        self.output_dir = Path(output_dir).resolve()
        self._lock = threading.RLock()
        self._request_id = 0
        self._closed = False
        self._status = {'state': 'idle', 'version': 0, 'control_cycles': 0,
                        'stop_reason': None, 'video': None,
                        'output_dir': str(self.output_dir), 'observation_seq': 0}
        self._process_group = None
        self._observation_reader = None
        context = mp.get_context('spawn')
        self._connection, child = context.Pipe(duplex=True)
        self._process = context.Process(target=_session_entry,
                                        args=(self.config, str(self.output_dir), child),
                                        daemon=False, name='vlastudio-execution-session')
        try:
            self._process.start()
            child.close()
            deadline = time.monotonic() + float(self.config.get('startup_timeout_s', 60))
            message = self._receive(deadline, 'starting execution session')
            if 'ready' not in message:
                raise RuntimeError(_error_text(message.get('error', message)))
            self._status = dict(message['ready'])
            if self._status.get('observation_shm'):
                from vlastudio.deploy.shm_utils import SharedMemoryChannel
                self._observation_reader = SharedMemoryChannel(
                    name=self._status['observation_shm'], is_writer=False, timeout=2)
                if self._observation_reader.shm is None:
                    raise RuntimeError('Cannot attach execution observation snapshot')
            if os.name == 'posix':
                try:
                    if os.getpgid(self._process.pid) == self._process.pid:
                        self._process_group = self._process.pid
                except ProcessLookupError:
                    pass
        except BaseException:
            child.close()
            self._closed = True
            self._shutdown_processes()
            if self._observation_reader is not None:
                self._observation_reader.destroy()
            self._connection.close()
            raise

    def _receive(self, deadline, activity):
        while True:
            if activity == 'starting execution session' and self._cancel_event is not None and self._cancel_event.is_set():
                raise RuntimeError('Execution startup cancelled')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'Timed out {activity}')
            if self._connection.poll(min(.05, remaining)):
                try:
                    return self._connection.recv()
                except EOFError as error:
                    raise RuntimeError(f'Execution process closed its connection while {activity}') from error
            if not self._process.is_alive():
                # A reply may have arrived immediately before process exit.
                if self._connection.poll(0):
                    continue
                raise RuntimeError(f'Execution process exited while {activity} '
                                   f'(exit code {self._process.exitcode})')

    def _rpc(self, command, *, timeout_s=10, **fields):
        with self._lock:
            if self._closed:
                if command == 'status':
                    return copy.deepcopy(self._status)
                raise RuntimeError('Execution session is closed')
            self._request_id += 1
            request_id = self._request_id
            try:
                self._connection.send({'id': request_id, 'cmd': command, **fields})
                deadline = time.monotonic() + float(timeout_s)
                while True:
                    reply = self._receive(deadline, f'waiting for {command} acknowledgement')
                    # A previously timed-out request may still have a late reply.
                    if reply.get('id') != request_id:
                        continue
                    if 'error' in reply:
                        raise RuntimeError(_error_text(reply['error']))
                    result = reply.get('result')
                    if isinstance(result, dict) and 'state' in result:
                        self._status = dict(result)
                    return result
            except (EOFError, BrokenPipeError, OSError) as error:
                self._status.update(state='failed', stop_reason=str(error))
                raise RuntimeError(f'Execution transport failed during {command}: {error}') from error

    def update_context(self, context, limits=None, *, timeout_s=10):
        """Replace the latest policy context and invalidate preceding actions.

        Context payload fields are interpreted by the policy's
        set_execution_context hook; runtime version/epoch are assigned internally.
        """
        return self._rpc('set_intent', timeout_s=timeout_s,
                         context=copy.deepcopy(context), limits=copy.deepcopy(limits or {}))

    def set_intent(self, context, limits=None, *, timeout_s=10):
        """Compatibility alias for update_context."""
        return self.update_context(context, limits, timeout_s=timeout_s)

    def complete(self, version, epoch, *, evidence=None, reason='external_completion', timeout_s=10):
        return self._rpc('complete', version=version, epoch=epoch,
                         evidence=copy.deepcopy(evidence or {}), reason=reason, timeout_s=timeout_s)

    def status(self):
        return self._rpc('status')

    def observe(self):
        # Images never travel over the control pipe or block the control actor.
        with self._lock:
            if self._closed:
                raise RuntimeError('Execution session is closed')
            if self._observation_reader is None:
                raise RuntimeError('Execution observation channel is unavailable')
            return self._observation_reader.read(blocking=False)

    def pause(self, *, timeout_s=10):
        return self._rpc('pause', timeout_s=timeout_s)

    def resume(self, *, timeout_s=10):
        return self._rpc('resume', timeout_s=timeout_s)

    def cancel(self, *, timeout_s=10):
        return self._rpc('cancel', timeout_s=timeout_s)

    def wait(self, version, *, timeout_s=None):
        """Wait without holding the RPC lock between polls, allowing control calls."""
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f'Timed out waiting for intent {version}')
            status = self._rpc('status', timeout_s=10 if remaining is None else min(10, remaining))
            current = status.get('version', 0)
            if current > version:
                raise IntentSupersededError(f'Intent {version} was superseded by intent {current}')
            if current == version and status.get('state') in ('completed', 'exhausted', 'cancelled', 'failed', 'closed'):
                return status
            if status.get('state') == 'closed':
                return status
            time.sleep(.02 if remaining is None else min(.02, max(0, remaining)))

    def _signal_group(self, signum):
        # A group is owned only after the child has become its own session leader.
        # Never signal the inherited group of this client or unrelated processes.
        group = self._process_group
        if group is None and os.name == 'posix' and self._process.pid:
            try:
                if os.getpgid(self._process.pid) == self._process.pid:
                    group = self._process.pid
                    self._process_group = group
            except ProcessLookupError:
                pass
        if group is not None:
            try:
                os.killpg(group, signum)
            except ProcessLookupError:
                pass
            return True
        return False

    def _shutdown_processes(self):
        if self._process.pid is None:
            return
        self._process.join(timeout=float(self.config.get('close_timeout_s', 3)))
        if self._process.is_alive():
            # Let the session leader unwind its native SHM/process cleanup first.
            self._process.terminate()
            self._process.join(timeout=5)
        # A dead group leader can leave a blocked inference/device child behind.
        if os.name == 'posix':
            self._signal_group(signal.SIGKILL)
        elif self._process.is_alive():
            self._process.kill()
        self._process.join(timeout=1)

    def close(self, *, timeout_s=10):
        with self._lock:
            if self._closed:
                return copy.deepcopy(self._status)
            try:
                result = self._rpc('close', timeout_s=timeout_s)
                if isinstance(result, dict):
                    self._status = dict(result)
            except (RuntimeError, TimeoutError) as error:
                # Cleanup must still work after a crashed/unresponsive runtime.
                self._status.setdefault('close_errors', []).append(str(error))
                if not self._status.get('stop_reason'):
                    self._status['stop_reason'] = str(error)
            finally:
                self._closed = True
                self._shutdown_processes()
                if self._observation_reader is not None:
                    self._observation_reader.destroy()
                    self._observation_reader = None
                self._connection.close()
                self._status['state'] = 'closed'
            return copy.deepcopy(self._status)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
