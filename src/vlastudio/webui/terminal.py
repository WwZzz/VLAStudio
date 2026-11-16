"""Bounded, server-local interactive shells for the WebUI terminal dialog."""
import base64
import os
from pathlib import Path
import secrets
import select
import signal
import struct
import subprocess
import sys
import threading
import time

from .profiles import ROOT

_SHELL_START = """
import fcntl, termios, os
from pathlib import Path
fcntl.ioctl(0, termios.TIOCSCTTY, 0)
shell = '/bin/bash' if Path('/bin/bash').is_file() else '/bin/sh'
args = [shell, '--noprofile', '--norc', '-i'] if shell.endswith('bash') else [shell, '-i']
os.execve(shell, args, os.environ)
"""


class Shell:
    def __init__(self, cwd, cols, rows):
        if os.name != 'posix':
            raise ValueError('Web terminal requires a Linux or macOS server')
        import pty
        self.lock = threading.RLock()
        self.buffer = bytearray()
        self.offset = 0
        self.touched = time.monotonic()
        self.closed = False
        self.done = False
        self.master, slave = pty.openpty()
        env = dict(os.environ, TERM='xterm-256color', COLORTERM='truecolor',
                   PS1=r'\u@\h:\w\$ ', PROMPT_COMMAND='',
                   PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))
        try:
            self.resize(cols, rows)
            # A fresh Python process acquires the controlling TTY before exec.
            # Avoid preexec_fn/fork callbacks in the threaded HTTP server.
            self.proc = subprocess.Popen(
                [sys.executable, '-c', _SHELL_START], cwd=cwd, env=env,
                stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        except BaseException:
            os.close(self.master)
            raise
        finally:
            os.close(slave)
        threading.Thread(target=self._read, daemon=True).start()

    def resize(self, cols, rows):
        import fcntl
        import termios
        if type(cols) is not int or type(rows) is not int or not (10 <= cols <= 500 and 2 <= rows <= 200):
            raise ValueError('Invalid terminal dimensions')
        fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack('HHHH', rows, cols, 0, 0))

    def _read(self):
        try:
            self._read_output()
        finally:
            self.done = True

    def _read_output(self):
        while not self.closed:
            if time.monotonic() - self.touched > 1800:
                self.close()
                return
            try:
                if not select.select([self.master], [], [], 1)[0]:
                    continue
                data = os.read(self.master, 65536)
                if not data:
                    return
                with self.lock:
                    self.buffer.extend(data)
                    overflow = max(0, len(self.buffer) - 1024 * 1024)
                    if overflow:
                        del self.buffer[:overflow]
                        self.offset += overflow
            except (OSError, ValueError):
                return

    def read(self, cursor):
        if type(cursor) is not int or cursor < 0:
            raise ValueError('Invalid terminal cursor')
        with self.lock:
            self.touched = time.monotonic()
            end = self.offset + len(self.buffer)
            if cursor > end:
                raise ValueError('Invalid terminal cursor')
            start = max(cursor, self.offset)
            chunk = bytes(self.buffer[start - self.offset:start - self.offset + 65536])
            return dict(data=base64.b64encode(chunk).decode('ascii'), cursor=start + len(chunk),
                        truncated=cursor < self.offset, exited=self.done and self.proc.poll() is not None,
                        exit_code=self.proc.poll())

    def write(self, data):
        if not isinstance(data, str) or len(data.encode('utf-8')) > 8192:
            raise ValueError('Terminal input is too large')
        with self.lock:
            if self.closed or self.proc.poll() is not None:
                raise ValueError('Terminal session has ended')
            self.touched = time.monotonic()
            payload = data.encode('utf-8')
            while payload:
                payload = payload[os.write(self.master, payload):]

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                os.killpg(self.proc.pid, signal.SIGHUP)
            except ProcessLookupError:
                pass
            os.close(self.master)
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=2)


class Terminals:
    def __init__(self, cwd=ROOT):
        self.cwd = Path(cwd).resolve()
        self.sessions = {}
        self.lock = threading.RLock()

    def dispatch(self, action, data):
        with self.lock:
            if action == 'open':
                for token, shell in list(self.sessions.items()):
                    if shell.closed or shell.proc.poll() is not None:
                        shell.close()
                        del self.sessions[token]
                if len(self.sessions) >= 4:
                    raise ValueError('Close another terminal before opening a new one')
                shell = Shell(self.cwd, data.get('cols', 100), data.get('rows', 28))
                token = secrets.token_urlsafe(32)
                self.sessions[token] = shell
                return dict(id=token, cwd=str(self.cwd))
            token = data.get('id')
            shell = self.sessions.get(token) if isinstance(token, str) else None
            if shell is None:
                if action == 'close':
                    return dict(ok=True)
                raise ValueError('Terminal session has ended; reopen the terminal')
            if action == 'read':
                return shell.read(data.get('cursor', 0))
            if action == 'write':
                shell.write(data.get('data'))
            elif action == 'resize':
                shell.resize(data.get('cols'), data.get('rows'))
            elif action == 'close':
                shell.close()
                del self.sessions[token]
            else:
                raise ValueError('Unknown terminal action')
            return dict(ok=True)

    def shutdown(self):
        with self.lock:
            for shell in self.sessions.values():
                shell.close()
            self.sessions.clear()
