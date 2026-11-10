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

from . import episodes
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
        files["/episodes.js"] = ("episodes.js", "text/javascript")
        files["/i18n.js"] = ("i18n.js", "text/javascript")
        if url.path in files:
            file, mime = files[url.path]
            return self.reply((ASSETS/file).read_bytes(), mime + "; charset=utf-8")
        s = self.server.session
        if url.path == "/api/catalog":
            return self.reply(catalog())
        if url.path in ("/api/episodes", "/api/episode"):
            try:
                with s.lock:
                    infer = s.runners["infer"]
                    active_video = infer.output if infer.proc else None
                if url.path == "/api/episodes":
                    return self.reply(episodes.catalog(active_video))
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
        if url.path == "/api/state":
            mode = parse_qs(url.query).get("mode", ["collect"])[0]
            if mode not in s.runners:
                return self.reply({"error": "Unknown mode"}, status=400)
            return self.reply(dict(session=s.state(mode), preview=s.preview.state()))
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
            if not 1 <= size <= 8192:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(size))
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
            elif self.path == "/api/robot/disconnect":
                s.disconnect(data["connection"])
            elif self.path == "/api/start":
                s.start(data["mode"], data)
            elif self.path == "/api/action":
                s.action(data["mode"],data["id"],data["action"])
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
