"""Exercise actual PTY behavior, not shell mocks."""
import base64
import os
import time

import pytest

from vlastudio.webui.terminal import Terminals

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX PTY server')


def output_until(manager, token, needle, cursor=0):
    output = b''
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        packet = manager.dispatch('read', {'id': token, 'cursor': cursor})
        cursor = packet['cursor']
        output += base64.b64decode(packet['data'])
        if needle in output:
            return output, cursor
        time.sleep(.03)
    raise AssertionError(output.decode(errors='replace'))


def test_shell_cwd_python_resize_interrupt_and_cleanup(tmp_path):
    manager = Terminals(tmp_path)
    try:
        opened = manager.dispatch('open', {'cols': 90, 'rows': 24})
        token = opened['id']
        assert opened['cwd'] == str(tmp_path)
        manager.dispatch('write', {'id': token, 'data': 'pwd\r'})
        _, cursor = output_until(manager, token, str(tmp_path).encode())
        manager.dispatch('resize', {'id': token, 'cols': 110, 'rows': 35})
        manager.dispatch('write', {'id': token, 'data': "stty size; printf '\\x52\\x45\\x41\\x44\\x59\\n'\r"})
        output, cursor = output_until(manager, token, b'READY\r\n', cursor)
        assert b'35 110' in output
        manager.dispatch('write', {'id': token, 'data': 'sleep 30\r'})
        time.sleep(.2)
        manager.dispatch('write', {'id': token, 'data': '\x03'})
        manager.dispatch('write', {'id': token, 'data': "printf '\\x49\\x4e\\x54\\x45\\x52\\x52\\x55\\x50\\x54\\x45\\x44\\n'\r"})
        output_until(manager, token, b'INTERRUPTED\r\n', cursor)
        shell = manager.sessions[token]
        manager.dispatch('close', {'id': token})
        assert shell.proc.poll() is not None
        assert token not in manager.sessions
    finally:
        manager.shutdown()


def test_exit_drains_final_output_and_tokens_are_isolated(tmp_path):
    manager = Terminals(tmp_path)
    try:
        a = manager.dispatch('open', {})['id']
        b = manager.dispatch('open', {})['id']
        assert a != b
        with pytest.raises(ValueError):
            manager.dispatch('write', {'id': 'not-a-session', 'data': 'echo nope\r'})
        manager.dispatch('write', {'id': a, 'data': "printf '\\x46\\x49\\x4e\\x41\\x4c\\n'; exit\r"})
        output_until(manager, a, b'FINAL\r\n')
        assert manager.sessions[b].proc.poll() is None
    finally:
        manager.shutdown()


def test_dimensions_and_capacity_are_bounded(tmp_path):
    manager = Terminals(tmp_path)
    try:
        with pytest.raises(ValueError):
            manager.dispatch('open', {'cols': 0})
        for _ in range(4):
            manager.dispatch('open', {})
        with pytest.raises(ValueError):
            manager.dispatch('open', {})
    finally:
        manager.shutdown()
