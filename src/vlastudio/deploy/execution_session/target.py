"""A connected native target outlives the inference sessions borrowing it."""
from contextlib import contextmanager
from dataclasses import asdict
import multiprocessing as mp
import os
import signal
import threading
import time
import uuid


class TaskDeadlineExpired(TimeoutError):
    pass


def _request(transport, command, timeout_s=10, **fields):
    token = uuid.uuid4().hex
    transport.send({'id': token, 'cmd': command, **fields})
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if transport.poll(min(.05, max(0, deadline-time.monotonic()))):
            reply = transport.recv()
            if reply.get('id') != token:
                continue
            if 'error' in reply:
                raise RuntimeError(reply['error'])
            return reply['result']
    raise TimeoutError(f'Target {command} timed out')


def _serve(config, control):
    from .backends import create_backend
    from vlastudio.deploy.shm_utils import SharedMemoryChannel
    if os.name == 'posix':
        os.setsid()
    def interrupted(*_):
        raise RuntimeError('Target terminated')
    signal.signal(signal.SIGTERM, interrupted)
    backend = writer = lease = None
    last_capture = None
    last_result = {}
    failure = None
    def sample():
        nonlocal last_capture
        obs, captured = backend.observe()
        if obs is not None and captured != last_capture:
            writer.write({'observation': asdict(obs), 'captured_at': captured})
            last_capture = captured
    try:
        backend = create_backend(config, uuid.uuid4().hex[:12])
        backend.open()
        name = 'target_' + uuid.uuid4().hex
        writer = SharedMemoryChannel(name=name, max_size_mb=64, is_writer=True)
        sample()
        info = {'mode': backend.mode, 'observation_shm': name,
                'device_shm_names': backend.device_shm_names,
                'device_configs': backend.device_configs, 'pid': os.getpid()}
        control.send({'ready': info})
        while True:
            if control.poll():
                message = control.recv()
                command = message['cmd']
                if command == 'close':
                    break
                try:
                    if command == 'acquire':
                        incoming = message['connection']
                        if lease is not None or failure:
                            incoming.close()
                            raise RuntimeError(failure or 'Target already in use')
                        lease = incoming
                    elif command == 'release':
                        if lease is not None:
                            lease.close()
                            lease = None
                    else:
                        raise ValueError('Unknown target command')
                    control.send({'id': message['id'], 'result': {**info, 'environment_result': last_result}})
                except Exception as exc:
                    control.send({'id': message['id'], 'error': str(exc)})
            if lease is not None and lease.poll():
                try:
                    message = lease.recv()
                    if failure:
                        raise RuntimeError(failure)
                    if message['cmd'] != 'publish':
                        raise ValueError('Unknown execution target command')
                    deadline = message.get('deadline')
                    if deadline is not None and time.perf_counter() >= deadline:
                        raise TaskDeadlineExpired('Task deadline reached before publication')
                    last_result = backend.publish(message['action'])
                    sample()
                    lease.send({'id': message['id'], 'result': last_result})
                except (EOFError, BrokenPipeError):
                    lease.close()
                    lease = None
                except TaskDeadlineExpired as exc:
                    lease.send({'id': message['id'], 'error': str(exc)})
                except Exception as exc:
                    failure = f'{type(exc).__name__}: {exc}'
                    lease.send({'id': message['id'], 'error': failure})
            sample()
            time.sleep(.002)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            control.send({'error': f'{type(exc).__name__}: {exc}'})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if lease is not None:
            lease.close()
        if backend is not None:
            backend.close()
        if writer is not None:
            writer.destroy()
        control.close()


class Endpoint:
    """Borrowed pipe, transferred by multiprocessing when a session starts."""
    def __init__(self, connection):
        self.connection = connection

    def __deepcopy__(self, memo):
        return self


class ConnectedTarget:
    def __init__(self, config, *, startup_timeout_s=60):
        self._busy = threading.Lock()
        self._closed = False
        context = mp.get_context('spawn')
        self._control, child = context.Pipe()
        self._process = context.Process(target=_serve, args=(config, child),
                                        name='vlastudio-connected-target')
        try:
            self._process.start()
            child.close()
            if not self._control.poll(startup_timeout_s):
                raise TimeoutError('Connecting target timed out')
            reply = self._control.recv()
            if 'ready' not in reply:
                raise RuntimeError(reply.get('error', 'Target connection failed'))
            self.info = reply['ready']
        except BaseException:
            child.close()
            self.close()
            raise

    @contextmanager
    def execution_backend(self):
        if not self._busy.acquire(blocking=False):
            raise RuntimeError('Target is already executing a task')
        client = server = None
        acquired = False
        try:
            if self._closed or not self._process.is_alive():
                raise RuntimeError('Target is closed')
            client, server = mp.get_context('spawn').Pipe()
            info = _request(self._control, 'acquire', connection=server)
            acquired = True
            server.close()
            yield {'type': 'connected', 'endpoint': Endpoint(client), **info}
        finally:
            try:
                if acquired:
                    _request(self._control, 'release')
            finally:
                if client is not None:
                    client.close()
                if server is not None:
                    server.close()
                self._busy.release()

    def close(self):
        if self._closed:
            return
        if not self._busy.acquire(blocking=False):
            raise RuntimeError('Stop the active task before closing its target')
        try:
            self._closed = True
            if self._process.pid is not None:
                try:
                    self._control.send({'cmd': 'close'})
                except (EOFError, BrokenPipeError, OSError):
                    pass
                self._process.join(5)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(5)
                if os.name == 'posix':
                    try:
                        os.killpg(self._process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif self._process.is_alive():
                    self._process.kill()
                self._process.join(1)
            self._control.close()
        finally:
            self._busy.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class AttachedBackend:
    """Native runtime transport; closing it never disconnects the target."""
    def __init__(self, config):
        self.config = config
        self.mode = config['mode']
        self.device_shm_names = config['device_shm_names']
        self.device_configs = config['device_configs']
        self.initial_result = config.get('environment_result', {})
        self.reader = None

    def open(self):
        from vlastudio.deploy.shm_utils import SharedMemoryChannel
        self.reader = SharedMemoryChannel(name=self.config['observation_shm'], is_writer=False, timeout=2)
        if self.reader.shm is None:
            raise RuntimeError('Connected target observation channel is unavailable')

    def observe(self):
        from .backends import _meta_observation
        snapshot = self.reader.read(blocking=False)
        if snapshot is None:
            return None, 0
        return _meta_observation(snapshot['observation']), snapshot['captured_at']

    def publish(self, action):
        return _request(self.config['endpoint'].connection, 'publish',
                        action=action, deadline=self.config.get('task_deadline'))

    def close(self):
        if self.reader is not None:
            self.reader.destroy()
            self.reader = None
        self.config['endpoint'].connection.close()
