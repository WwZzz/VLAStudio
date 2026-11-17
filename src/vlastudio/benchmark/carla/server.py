"""Own one CARLA server process; cleanup never targets unrelated simulator jobs."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def _process_start(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def close_owned_server(run_dir):
    """Recover an owned server after its worker exits, without matching other jobs."""
    receipt = Path(run_dir)/'process.json'
    if not receipt.is_file():
        return
    value = json.loads(receipt.read_text())
    pid = value['pid']
    if value.get('start_ticks') is None or _process_start(pid) != value['start_ticks']:
        return
    try:
        command = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        if value['binary'].encode() != command[0] or f'-userdir={Path(run_dir)}'.encode() not in command:
            raise RuntimeError('Owned CARLA process identity no longer matches its receipt')
        os.killpg(pid, signal.SIGTERM)
        for _ in range(50):
            if _process_start(pid) != value['start_ticks']:
                return
            time.sleep(.1)
        os.killpg(pid, signal.SIGKILL)
    except (FileNotFoundError, ProcessLookupError):
        pass


def graphics_environment(runtime=None):
    env = os.environ.copy()
    runtime = runtime or env.get('VLASTUDIO_CARLA_GRAPHICS_RUNTIME')
    if runtime:
        value = json.loads(Path(runtime).read_text())
        env['LD_LIBRARY_PATH'] = value['library_path']+':'+env.get('LD_LIBRARY_PATH', '')
        env['VK_ICD_FILENAMES'] = value['icd']
        if value.get('egl_vendor'):
            env['__EGL_VENDOR_LIBRARY_FILENAMES'] = value['egl_vendor']
    return env


def free_ports():
    for _ in range(100):
        first, second = socket.socket(), socket.socket()
        try:
            first.bind(('127.0.0.1', 0))
            port = first.getsockname()[1]
            second.bind(('127.0.0.1', port+1))
            return port
        except OSError:
            pass
        finally:
            first.close()
            second.close()
    raise RuntimeError('Could not allocate CARLA RPC and streaming ports')


class CarlaServer:
    def __init__(self, root=None, *, gpu=0, graphics_runtime=None, run_dir=None, timeout=180.):
        from vlastudio.benchmark.carla.install import cache_root
        self.root = Path(root or os.environ.get('VLASTUDIO_RUNTIME_CARLA_ROOT',
                          str(cache_root()/'runtimes/carla-0.9.15'))).expanduser().resolve()
        if not self.root.joinpath('CarlaUE4.sh').is_file():
            raise FileNotFoundError('Install the CARLA server first: python -m vlastudio.benchmark.carla.install')
        import tempfile
        cache_root().mkdir(parents=True, exist_ok=True)
        self.run_dir = Path(run_dir) if run_dir else Path(tempfile.mkdtemp(prefix='carla-', dir=cache_root()))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.port, self.process = free_ports(), None
        self.timeout = timeout
        self.log_path = self.run_dir/'server.log'
        env = graphics_environment(graphics_runtime)
        xdg = self.run_dir/'xdg'
        xdg.mkdir(mode=0o700, exist_ok=True)
        env['XDG_RUNTIME_DIR'] = str(xdg)
        options = {}
        if hasattr(os, 'geteuid') and os.geteuid() == 0:
            # UE4 refuses root execution. Drop privileges only for this child.
            uid = int(os.environ.get('VLASTUDIO_CARLA_SERVER_UID', '65534'))
            gid = int(os.environ.get('VLASTUDIO_CARLA_SERVER_GID', '65534'))
            if uid == 0:
                raise ValueError('CARLA server UID must be nonzero')
            os.chown(self.run_dir, uid, gid)
            os.chown(xdg, uid, gid)
            options.update(user=uid, group=gid, extra_groups=[])
        binary = self.root/'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
        command = ([str(binary), 'CarlaUE4'] if binary.is_file() else [str(self.root/'CarlaUE4.sh')])+['-RenderOffScreen', '-nosound',
                   f'-carla-rpc-port={self.port}', f'-graphicsadapter={int(gpu)}',
                   '-quality-level=Epic', f'-userdir={self.run_dir}']
        with self.log_path.open('ab') as output:
            self.process = subprocess.Popen(command, cwd=self.root, env=env,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True, **options)
        (self.run_dir/'process.json').write_text(json.dumps({'pid': self.process.pid,
            'start_ticks': _process_start(self.process.pid), 'binary': command[0]}, indent=2)+'\n')

    def connect(self):
        import carla
        deadline = time.monotonic()+self.timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('CARLA exited before RPC startup: '+self.log_path.read_text(errors='replace')[-3000:])
            client = carla.Client('127.0.0.1', self.port)
            client.set_timeout(2.)
            try:
                client.get_world()
                if client.get_server_version() != '0.9.15':
                    raise ValueError('Bench2Drive requires the CARLA 0.9.15 server')
                client.set_timeout(self.timeout)
                return client
            except RuntimeError:
                time.sleep(.5)
        raise TimeoutError(f'CARLA RPC startup timed out; see {self.log_path}')

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=10)
            except ProcessLookupError:
                self.process.wait(timeout=10)
        self.process = None

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
