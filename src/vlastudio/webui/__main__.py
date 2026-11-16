"""Launch the LAN dashboard: python -m vlastudio.webui [--port 8765]."""
import argparse
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import config_editor, data_management, datasets, episodes, settings, training
from .terminal import Terminals
from .profiles import catalog
from .studio import Studio
from .tasks import list_tasks, create_task

ASSETS = Path(__file__).with_name("static")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, data, mime="application/json", status=200, headers=()):
        if mime == "application/json":
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        embedded = parse_qs(urlparse(self.path).query).get('workspace_frame') == ['1']
        ancestors = "'self'" if embedded else "'none'"
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; frame-ancestors " + ancestors)
        self.send_header("Content-Length", str(len(data)))
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        if data:
            self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/api/data/annotations':
            try:
                query = parse_qs(url.query)
                return self.reply(data_management.annotations(query['dataset'][0], int(query['index'][0]), query['version'][0]))
            except (KeyError, ValueError, OSError) as exc:
                return self.reply({'error':str(exc)}, status=409)
        if url.path == '/api/episode/outcomes':
            try:
                return self.reply(data_management.outcomes(parse_qs(url.query)['dataset'][0]))
            except (KeyError, ValueError, OSError) as exc:
                return self.reply({'error':str(exc)}, status=409)
        if url.path.startswith('/api/') and not url.path.startswith('/api/training/') and getattr(self.server, 'api_upstream_port', None):
            return self.forward_api()
        if url.path in ('/', '/collect', '/infer', '/data', '/train') and parse_qs(url.query).get('workspace_frame') != ['1']:
            return self.reply((ASSETS/'workspace.html').read_bytes(), 'text/html; charset=utf-8')
        files = {"/": ("index.html","text/html"), "/collect":("index.html","text/html"),
                 "/infer":("index.html","text/html"), "/app.js":("app.js","text/javascript"),
                 "/style.css":("style.css","text/css"),
                 "/vlastudio-mark.svg":("vlastudio-mark.svg","image/svg+xml")}
        files["/log-search.js"] = ("log-search.js", "text/javascript")
        files["/theme.js"] = ("theme.js", "text/javascript")
        files["/training-config.js"] = ("training-config.js", "text/javascript")
        files["/training-charts.js"] = ("training-charts.js", "text/javascript")
        files["/training-artifacts.js"] = ("training-artifacts.js", "text/javascript")
        files["/settings.js"] = ("settings.js", "text/javascript")
        files['/interpreters.js'] = ('interpreters.js', 'text/javascript')
        files['/environment-installer.js'] = ('environment-installer.js', 'text/javascript')
        files["/config-devices.js"] = ("config-devices.js", "text/javascript")
        files["/config-editor.js"] = ("config-editor.js", "text/javascript")
        files["/quest.js"] = ("quest.js", "text/javascript")
        files["/episodes.js"] = ("episodes.js", "text/javascript")
        files["/i18n.js"] = ("i18n.js", "text/javascript")
        files['/workspace.js'] = ('workspace.js', 'text/javascript')
        files['/workspace.css'] = ('workspace.css', 'text/css')
        files['/outcomes.js'] = ('outcomes.js', 'text/javascript')
        files['/outcomes.css'] = ('outcomes.css', 'text/css')
        for name, mime in (("terminal.js", "text/javascript"), ("terminal.css", "text/css"),
                           ("vendor/xterm.js", "text/javascript"), ("vendor/addon-fit.js", "text/javascript"),
                           ("vendor/xterm.css", "text/css")):
            files['/' + name] = (name, mime)
        if url.path in files:
            file, mime = files[url.path]
            return self.reply((ASSETS/file).read_bytes(), mime + "; charset=utf-8")
        s = self.server.session
        if url.path in ("/train", "/train.js", "/train.css"):
            name = "train.html" if url.path == "/train" else url.path[1:]
            mime = "text/html" if name.endswith(".html") else "text/javascript" if name.endswith(".js") else "text/css"
            return self.reply((ASSETS / name).read_bytes(), mime + "; charset=utf-8")
        if url.path.startswith("/api/training/"):
            try:
                query = parse_qs(url.query)
                if url.path == "/api/training/catalog":
                    return self.reply(training.catalog())
                if url.path == "/api/training/runs":
                    return self.reply(training.runs())
                if url.path == "/api/training/run":
                    return self.reply(training.detail(query["id"][0]))
                if url.path == "/api/training/metrics":
                    return self.reply(training.metrics(query["id"][0]))
                if url.path == "/api/training/log":
                    path = training.artifact(query["id"][0], "training.log")
                    with path.open("rb") as stream:
                        total = os.fstat(stream.fileno()).st_size
                        start, end, status = 0, total - 1, 200
                        byte_range = self.headers.get("Range")
                        if byte_range:
                            import re
                            match = re.fullmatch(r"bytes=(\d+)-(\d*)", byte_range)
                            if not match:
                                return self.reply({"error":"Invalid log byte range"}, status=416)
                            start = int(match[1])
                            end = min(int(match[2]), total - 1) if match[2] else total - 1
                            if start >= total or end < start:
                                return self.reply(b"", "text/plain", status=416,
                                                  headers=(("Content-Range", f"bytes */{total}"),))
                            status = 206
                        remaining = max(0, end - start + 1)
                        stream.seek(start)
                        self.send_response(status)
                        self.send_header("Content-Type", "text/plain; charset=utf-8")
                        self.send_header("Content-Length", str(remaining))
                        self.send_header("Accept-Ranges", "bytes")
                        if status == 206:
                            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.end_headers()
                        while remaining:
                            block = stream.read(min(65536, remaining))
                            if not block:
                                break
                            self.wfile.write(block)
                            remaining -= len(block)
                    return
                if url.path == "/api/training/model":
                    identifier = query["id"][0]
                    run = training.detail(identifier)["run"]
                    if run["status"] != "completed":
                        raise ValueError("The final model is available after training completes")
                    model = training.directory(identifier) / "model"
                    if not model.is_dir():
                        raise ValueError("No saved model")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/gzip")
                    self.send_header("Content-Disposition", 'attachment; filename="model-' + identifier + '.tar.gz"')
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    import tarfile
                    with tarfile.open(fileobj=self.wfile, mode="w|gz", dereference=False) as archive:
                        for path in sorted(model.rglob("*")):
                            relative = path.relative_to(model)
                            if (path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(model.resolve())
                                    and not any(part.startswith("checkpoint-") for part in relative.parts)):
                                archive.add(path, arcname="model/" + relative.as_posix(), recursive=False)
                    return
                if url.path == "/api/training/artifact":
                    path = training.artifact(query["id"][0], query["file"][0])
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(path.stat().st_size))
                    from urllib.parse import quote
                    self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(path.name))
                    self.end_headers()
                    import shutil
                    with path.open("rb") as stream:
                        shutil.copyfileobj(stream, self.wfile)
                    return
                return self.reply({"error":"Not found"}, status=404)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                return self.reply({"error":str(exc)}, status=409)
        if url.path == "/data":
            return self.reply((ASSETS / "data.html").read_bytes(), "text/html; charset=utf-8")
        if url.path in ("/data.js", "/data.css"):
            name = url.path[1:]
            return self.reply((ASSETS / name).read_bytes(), "text/javascript" if name.endswith(".js") else "text/css")
        if url.path == "/api/health":
            return self.reply({"ok": True})
        if url.path == "/api/data/catalog":
            return self.reply(data_management.catalog())
        if url.path == "/api/data/annotations":
            try:
                query = {k:v[0] for k,v in parse_qs(url.query).items()}
                return self.reply(data_management.annotations(query["dataset"], int(query["index"]), query["version"]))
            except (ValueError, KeyError, OSError) as exc:
                return self.reply({"error":str(exc)}, status=409)
        if url.path == "/api/settings":
            return self.reply(settings.public())
        if url.path == '/api/settings/environments':
            from .environment_installation import catalog as environment_catalog
            return self.reply(environment_catalog())
        if url.path == '/api/settings/environments/log':
            from .environment_installation import logs
            try:
                query = parse_qs(url.query)
                return self.reply(logs(query['id'][0], query['component'][0]))
            except (KeyError, ValueError, OSError) as exc:
                return self.reply({'error': str(exc)}, status=409)
        if url.path == "/api/device-types":
            return self.reply(config_editor.device_types())
        if url.path == "/api/device-library":
            return self.reply(config_editor.device_library())
        if url.path == "/api/device-parameters":
            try:
                reference = parse_qs(url.query)["type"][0]
                return self.reply(config_editor.parameters(reference))
            except (ValueError, KeyError) as exc:
                return self.reply({"error":str(exc)}, status=409)
        if url.path == "/api/config":
            try:
                query = parse_qs(url.query)
                kind, name = query["kind"][0], query["name"][0]
                item = config_editor.read(kind, name)
                item["in_use"] = bool(config_editor.in_use(s, kind, name))
                return self.reply(item)
            except (ValueError, KeyError, OSError) as exc:
                return self.reply({"error":str(exc)}, status=409)
        if url.path == "/api/catalog":
            return self.reply(catalog(quest_setup=True))
        if url.path in ("/api/episodes", "/api/episode"):
            try:
                with s.lock:
                    infer = s.runners["infer"]
                    active_episode = infer.episode if infer.proc else None
                    active_video = infer.output if infer.proc and active_episode is not None else None
                if url.path == "/api/episodes":
                    return self.reply(episodes.catalog(active_video, parse_qs(url.query).get("dataset", [None])[0], active_episode))
                query = {k:v[0] for k,v in parse_qs(url.query).items()}
                dataset, index = query["dataset"], int(query["index"])
                if dataset == active_video and index == active_episode and all(p.suffix == ".mp4" for p in episodes.files(dataset, index)):
                    raise ValueError("Finish this inference trial before opening its video")
                return self.reply(episodes.read(dataset, index, query["version"],
                                      int(query["frame"]) if "frame" in query else None))
            except (KeyError, ValueError, OSError, RuntimeError) as exc:
                return self.reply({"error": str(exc)}, status=409)
        if url.path == "/api/tasks":
            return self.reply(list_tasks())
        if url.path == "/api/quest-preview":
            try:
                sid = parse_qs(url.query).get("id", [""])[0]
                return self.reply(s.runners["collect"].quest_preview(sid))
            except ValueError as exc:
                return self.reply({"error":str(exc)}, status=409)
        if url.path == "/api/state":
            mode = parse_qs(url.query).get("mode", ["collect"])[0]
            if mode not in s.runners:
                return self.reply({"error": "Unknown mode"}, status=400)
            return self.reply(dict(session=s.state(mode), preview=s.preview.state()))
        if url.path == "/api/camera-stream":
            name = parse_qs(url.query).get("name", [""])[0]
            if s.preview.latest_image(name, None) is None:
                return self.reply({"error":"Camera unavailable"}, status=404)
            self.connection.settimeout(2)
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32768)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            stamp = None
            try:
                while True:
                    frame = s.preview.latest_image(name, stamp)
                    if frame is None:
                        return
                    next_stamp, image = frame
                    if next_stamp == stamp:
                        continue
                    header = (f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(image)}"
                              f"\r\nX-Frame-Stamp: {next_stamp}\r\n\r\n").encode("ascii")
                    self.wfile.write(header + image + b"\r\n")
                    self.wfile.flush()
                    stamp = next_stamp
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return
        if url.path == "/api/image":
            query = parse_qs(url.query)
            after = query.get("after", [None])[0]
            frame = s.preview.latest_image(query.get("name", [""])[0], after)
            if frame:
                stamp, image = frame
                if stamp == after:
                    return self.reply(b"", "image/jpeg", status=204)
                age_ms = max(0, (time.monotonic() - float(stamp)) * 1000)
                return self.reply(image, "image/jpeg", headers=(
                    ("X-Frame-Stamp", stamp), ("X-Frame-Age-Ms", f"{age_ms:.1f}"),
                ))
        self.reply({"error":"Not found"},status=404)

    def save_episode_outcome(self, data):
        mode, sid, outcome = data['mode'], data['id'], data['outcome']
        if mode not in ('collect', 'infer') or outcome not in ('unknown', 'success', 'failure'):
            raise ValueError('Invalid episode outcome')
        def backend(action=False):
            port = getattr(self.server, 'api_upstream_port', None)
            if not port:
                if action:
                    self.server.session.action(mode, sid, 'save')
                return self.server.session.state(mode)
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=20)
            try:
                connection.request('POST' if action else 'GET', '/api/action' if action else '/api/state?mode='+mode,
                                   json.dumps(dict(mode=mode, id=sid, action='save')) if action else None,
                                   {'Content-Type':'application/json'})
                response = connection.getresponse(); result = json.loads(response.read())
                if response.status != 200:
                    raise ValueError(result.get('error', 'Save failed'))
                return result.get('session', result)
            finally:
                connection.close()
        before = backend()
        if before.get('id') != sid or 'save' not in before.get('actions', []):
            raise ValueError('This episode is no longer awaiting save')
        dataset, index = before['options']['output'], before['episode']
        previous = episodes.files(dataset, index)
        previous_version = episodes.revision(previous) if previous else None
        backend(True)
        for _ in range(100):
            current = backend()
            paths = episodes.files(dataset, index)
            if paths and episodes.revision(paths) != previous_version and current.get('phase') not in ('save', 'infer_save', 'busy', 'stopping'):
                version = episodes.revision(paths)
                try:
                    data_management.save_outcome(dict(dataset=dataset, index=index, version=version, outcome=outcome))
                except (ValueError, OSError) as exc:
                    raise ValueError('Recording saved; label was not saved. Update its result in Data management.') from exc
                return dict(ok=True, dataset=dataset, index=index, outcome=outcome)
            time.sleep(.15)
        raise ValueError('Save confirmation timed out. Check this episode in Data management before retrying.')

    def configure_inference_recording(self, data):
        self.server.session.configure_recording(data['id'], data['record_inference'])
        return {'ok':True}

    def forward_api(self, port=None):
        """Keep an existing device/terminal backend alive during a UI upgrade."""
        connection = http.client.HTTPConnection('127.0.0.1', port or self.server.api_upstream_port, timeout=65)
        started = False
        try:
            size = int(self.headers.get('Content-Length', '0'))
            limit = 1048576 if self.path == '/api/data/annotations' else 262144 if self.path == '/api/config' else 8192
            if size < 0 or size > limit or (self.command == 'POST' and not size):
                return self.reply({'error':'Invalid request size'}, status=400)
            body = self.rfile.read(size) if size else None
            headers = {k:v for k,v in self.headers.items() if k.lower() not in ('host','origin','connection','content-length','accept-encoding')}
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in ('server','date','connection','transfer-encoding'):
                    self.send_header(key, value)
            self.end_headers()
            started = True
            while True:
                chunk = response.read1(16384)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, ValueError, http.client.HTTPException):
            if not started:
                self.reply({'error':'The existing service is unavailable'}, status=502)
        finally:
            connection.close()

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self.reply({"error":"Origin mismatch"},status=403)
        if self.path.startswith('/api/terminal/') and getattr(self.server, 'terminal_upstream_port', None):
            return self.forward_api(self.server.terminal_upstream_port)
        if self.path == '/api/inference/recording' and getattr(self.server, 'api_upstream_port', None):
            return self.forward_api()
        if self.path in ('/api/episode/outcome', '/api/episode/save-outcome', '/api/data/annotations', '/api/inference/recording'):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= (1048576 if self.path == '/api/data/annotations' else 8192) or self.headers.get('Sec-Fetch-Site') == 'cross-site':
                    raise ValueError('Invalid request')
                data = json.loads(self.rfile.read(size))
                if self.path == '/api/inference/recording':
                    return self.reply(self.configure_inference_recording(data))
                if self.path == '/api/data/annotations':
                    return self.reply(data_management.save(data))
                if self.path.endswith('/save-outcome'):
                    return self.reply(self.save_episode_outcome(data))
                return self.reply(data_management.save_outcome(data))
            except (KeyError, ValueError, OSError, RuntimeError) as exc:
                return self.reply({'error':str(exc)}, status=409)
        if self.path.startswith('/api/') and not self.path.startswith('/api/training/') and getattr(self.server, 'api_upstream_port', None):
            return self.forward_api()
        s = self.server.session
        try:
            size = int(self.headers.get("Content-Length","0"))
            limit = 1048576 if self.path == "/api/data/annotations" else 262144 if self.path == "/api/config" else 8192
            if not 1 <= size <= limit:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(size))
            if self.path.startswith('/api/terminal/'):
                if (self.headers.get('Content-Type', '').split(';')[0] != 'application/json'
                        or self.headers.get('Sec-Fetch-Site') == 'cross-site'):
                    return self.reply({'error': 'Same-origin JSON request required'}, status=403)
                if not isinstance(data, dict):
                    raise ValueError('Invalid terminal request')
                return self.reply(self.server.terminals.dispatch(self.path.rsplit('/', 1)[-1], data))
            if self.path == "/api/training/start":
                return self.reply(training.start(data))
            if self.path == "/api/training/stop":
                return self.reply(training.stop(data["id"]))
            if self.path == "/api/training/resume":
                return self.reply(training.resume(data["id"], interpreter_id=data.get('interpreter')))
            if self.path == "/api/data/annotations":
                with s.lock:
                    return self.reply(data_management.save(data))
            if self.path == "/api/data/delete":
                with s.lock:
                    if any(r.proc and r.output == data["dataset"] for r in s.runners.values()):
                        raise ValueError("Disconnect before deleting from this dataset")
                    data_management.remove_dataset(data["dataset"])
                return self.reply({"ok":True})
            if self.path in ("/api/datasets", "/api/datasets/create"):
                with s.lock:
                    context = s.collection(data)
                    if self.path.endswith("/create"):
                        return self.reply(datasets.ensure(datasets.identity(context, data["suffix"])))
                    return self.reply(datasets.choices(context))
            if self.path == "/api/settings/directories/create":
                return self.reply(settings.create_directory(data.get("path"), data.get("name")))
            if self.path == "/api/settings/interpreter/probe":
                from .interpreters import probe
                return self.reply(probe(data))
            if self.path == '/api/settings/environments/start':
                from .environment_installation import submit
                return self.reply(submit(data))
            if self.path == '/api/settings/environments/cancel':
                from .environment_installation import cancel
                return self.reply(cancel(data))
            if self.path == "/api/settings/directories":
                return self.reply(settings.browse_directory(data.get("path"), data.get("initial") is True))
            if self.path == "/api/settings":
                return self.reply(settings.save(s, data))
            if self.path == "/api/config":
                return self.reply(config_editor.save(s, data["kind"], data["name"], data["content"],
                                                    data["version"], data["target"]))
            if self.path == "/api/episode/rerecord":
                s.rerecord(data)
                return self.reply({"ok": True})
            if self.path == "/api/episode/delete":
                with s.lock:
                    if any(r.proc and r.output == data["dataset"] for r in s.runners.values()):
                        raise ValueError("Disconnect before deleting from this dataset")
                    episodes.remove(data["dataset"], data["index"], data["version"])
                return self.reply({"ok": True})
            if self.path == "/api/tasks":
                return self.reply(create_task(data['name'], data.get('instruction', '')))
            if self.path == "/api/robot/connect":
                s.connect(data)
            elif self.path == "/api/disconnect":
                s.disconnect(data["connection"], all_sources=True)
            elif self.path == "/api/robot/disconnect":
                s.disconnect(data["connection"])
            elif self.path == "/api/start":
                s.start(data["mode"], data)
            elif self.path == "/api/action":
                s.action(data["mode"], data["id"], data["action"], real_task=data.get("real_task"), suffix=data.get("suffix"))
            elif self.path == "/api/keyboard":
                s.runners["collect"].keyboard(data)
            elif self.path == "/api/preview":
                with s.lock:
                    if s.host is not None:
                        raise ValueError("运行期间使用当前会话的数据源")
                    if data["robot"]:
                        s.preview.start(data["robot"])
                    else:
                        s.preview.stop()
            else:
                return self.reply({"error":"Not found"},status=404)
            self.reply({"ok":True})
        except (ValueError, KeyError, TypeError, OSError) as e:
            self.reply({"error":str(e)},status=409)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runtimes", type=Path, help="Site JSON mapping simulator names to Python launchers")
    parser.add_argument('--api-upstream-port', type=int, help='Reuse an existing backend on the local machine during a UI upgrade')
    parser.add_argument('--terminal-upstream-port', type=int, help='Retain existing local shell sessions during a backend upgrade')
    args = parser.parse_args()
    if args.api_upstream_port is not None and (not 1 <= args.api_upstream_port <= 65535 or args.api_upstream_port == args.port):
        parser.error('The upstream must use a different local port')
    if args.terminal_upstream_port is not None and (not 1 <= args.terminal_upstream_port <= 65535 or args.terminal_upstream_port == args.port):
        parser.error('The terminal upstream must use a different local port')
    if args.runtimes is not None:
        os.environ['VLASTUDIO_WEBUI_RUNTIMES'] = str(args.runtimes.expanduser().resolve())
    settings.register_interpreter_alias()
    server = ThreadingHTTPServer((args.host,args.port), Handler)
    server.api_upstream_port = args.api_upstream_port
    server.terminal_upstream_port = args.terminal_upstream_port
    server.session = None if args.api_upstream_port else Studio()
    server.terminals = None if args.api_upstream_port or args.terminal_upstream_port else Terminals()
    print(f"VLAStudio WebUI http://{socket.gethostname()}:{args.port}", flush=True)
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if server.terminals:
            server.terminals.shutdown()
        if server.session:
            server.session.shutdown()


if __name__ == "__main__":
    main()
