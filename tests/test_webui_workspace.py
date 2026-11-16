"""Persistent workspace routing and upgrades that preserve the running backend."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import urllib.error
import urllib.request

import pytest

pytest.importorskip('fcntl')
from vlastudio.webui.__main__ import Handler


@contextmanager
def serve(handler, upstream=None):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    server.api_upstream_port = upstream
    server.session = server.terminals = None
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server, f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_workspace_routes_and_embedding_boundary():
    with serve(Handler) as (_, base):
        for path in ('/', '/collect', '/infer', '/data', '/train'):
            with urllib.request.urlopen(base + path) as response:
                assert b'id="workspaceFrame"' in response.read()
                assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
            with urllib.request.urlopen(base + path + '?workspace_frame=1') as response:
                content = response.read()
                assert b'id="workspaceFrame"' not in content
                assert b'id="terminalOpen"' in content
                assert "frame-ancestors 'self'" in response.headers['Content-Security-Policy']


class Backend(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.respond({'path': self.path, 'backend': 'existing'})

    def do_POST(self):
        self.respond({'body': json.loads(self.rfile.read(int(self.headers['Content-Length'])))})

    def respond(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_ui_upgrade_reuses_backend_and_enforces_request_boundary():
    with serve(Backend) as (backend, backend_url):
        with serve(Handler, backend.server_port) as (_, base):
            with urllib.request.urlopen(base + '/api/state?mode=infer') as response:
                assert json.load(response) == {'path': '/api/state?mode=infer', 'backend': 'existing'}
            request = urllib.request.Request(base + '/api/terminal/write', data=b'{"data":"echo test"}',
                                             headers={'Content-Type': 'application/json', 'Origin': base})
            with urllib.request.urlopen(request) as response:
                assert json.load(response)['body'] == {'data': 'echo test'}
            request.add_header('Origin', 'https://unrelated.example')
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            assert error.value.code == 403
            oversized = urllib.request.Request(base + '/api/terminal/write', data=b'x' * 8193)
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(oversized)
            assert error.value.code == 400
        with urllib.request.urlopen(backend_url + '/api/health') as response:
            assert json.load(response)['backend'] == 'existing'


def test_device_backend_upgrade_keeps_existing_terminal_service():
    with serve(Backend) as (backend, _):
        with serve(Handler) as (server, base):
            server.terminal_upstream_port = backend.server_port
            request = urllib.request.Request(base+'/api/terminal/read', data=b'{"id":"existing","cursor":42}',
                                             headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request) as response:
                assert json.load(response)['body']=={'id':'existing','cursor':42}
            with urllib.request.urlopen(base+'/api/health') as response:
                assert json.load(response)=={'ok':True}
