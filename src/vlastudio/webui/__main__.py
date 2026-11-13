"""Launch the LAN dashboard: python -m vlastudio.webui [--port 8765]."""
import argparse
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
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        self.send_header("Content-Length", str(len(data)))
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        if data:
            self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        files = {"/": ("index.html","text/html"), "/collect":("index.html","text/html"),
                 "/infer":("index.html","text/html"), "/app.js":("app.js","text/javascript"),
                 "/style.css":("style.css","text/css"),
                 "/vlastudio-mark.svg":("vlastudio-mark.svg","image/svg+xml")}
        files["/log-search.js"] = ("log-search.js", "text/javascript")
        files["/theme.js"] = ("theme.js", "text/javascript")
        files["/training-config.js"] = ("training-config.js", "text/javascript")
        files["/settings.js"] = ("settings.js", "text/javascript")
        files["/config-devices.js"] = ("config-devices.js", "text/javascript")
        files["/config-editor.js"] = ("config-editor.js", "text/javascript")
        files["/quest.js"] = ("quest.js", "text/javascript")
        files["/episodes.js"] = ("episodes.js", "text/javascript")
        files["/i18n.js"] = ("i18n.js", "text/javascript")
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
                    active_video = infer.output if infer.proc else None
                if url.path == "/api/episodes":
                    return self.reply(episodes.catalog(active_video, parse_qs(url.query).get("dataset", [None])[0]))
                query = {k:v[0] for k,v in parse_qs(url.query).items()}
                dataset, index = query["dataset"], int(query["index"])
                if dataset == active_video and all(p.suffix == ".mp4" for p in episodes.files(dataset, index)):
                    raise ValueError("Disconnect before opening this video")
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

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self.reply({"error":"Origin mismatch"},status=403)
        s = self.server.session
        try:
            size = int(self.headers.get("Content-Length","0"))
            limit = 1048576 if self.path == "/api/data/annotations" else 262144 if self.path == "/api/config" else 8192
            if not 1 <= size <= limit:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(size))
            if self.path == "/api/training/start":
                return self.reply(training.start(data))
            if self.path == "/api/training/stop":
                return self.reply(training.stop(data["id"]))
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
    args = parser.parse_args()
    if args.runtimes is not None:
        os.environ['VLASTUDIO_WEBUI_RUNTIMES'] = str(args.runtimes.expanduser().resolve())
    server = ThreadingHTTPServer((args.host,args.port), Handler)
    server.session = Studio()
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
        server.session.shutdown()


if __name__ == "__main__":
    main()
