"""Launch the LAN dashboard: python -m vlastudio.webui [--port 8765]."""
import argparse
import base64
import json
from pathlib import Path
import signal
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .profiles import catalog
from .session import Session

ASSETS = Path(__file__).with_name("static")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, data, mime="application/json", status=200):
        if mime == "application/json":
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        files = {"/": ("index.html","text/html"), "/collect":("index.html","text/html"),
                 "/infer":("index.html","text/html"), "/app.js":("app.js","text/javascript"),
                 "/style.css":("style.css","text/css"),
                 "/vlastudio-mark.svg":("vlastudio-mark.svg","image/svg+xml")}
        if url.path in files:
            file, mime = files[url.path]
            return self.reply((ASSETS/file).read_bytes(), mime + "; charset=utf-8")
        s = self.server.session
        if url.path == "/api/catalog":
            return self.reply(catalog())
        if url.path == "/api/state":
            return self.reply(dict(session=s.state(), preview=s.preview.state()))
        if url.path == "/api/image":
            with s.preview.lock:
                image = s.preview.images.get(parse_qs(url.query).get("name",[""])[0])
            if image:
                return self.reply(base64.b64decode(image), "image/jpeg")
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
            if self.path == "/api/start":
                s.start(data["mode"], data)
            elif self.path == "/api/action":
                s.action(data["id"],data["action"])
            elif self.path == "/api/keyboard":
                s.keyboard(data)
            elif self.path == "/api/preview":
                with s.lock:
                    if s.proc is not None:
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
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host,args.port), Handler)
    server.session = Session()
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
