"""Pinned RoboTwin runtime configuration, invoked by the explicit installer."""
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import zipfile

from filelock import FileLock

REVISION = '0f288c819a64c02accac42190e78af5022e618a1'
CUROBO = '0a50de1ba72db304195d59d9d0b1ed269696047f'
PYTORCH3D = '75ebeeaea0908c5527e7b1e305fbc7681382db47'
ASSETS_REVISION = '3dc3b798668feb99ac61cc9086d84cbcc3d79186'
RESOURCES = {
    'embodiments': (219847741, '85ffeff55a5066def5931224a85cfa3f8abaa1fbf779bd17789a7fb3f85bc789'),
    'objects': (3737778549, '6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27'),
    'background_texture': (10970687027, '54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070'),
}


def configure():
    """Apply the two upstream installation patches, checking their exact context."""
    def patch(module, relative, before, after):
        root = Path(next(iter(importlib.util.find_spec(module).submodule_search_locations)))
        path = root / relative
        text = path.read_text(encoding='utf-8')
        if after in text:
            return
        if before not in text:
            raise RuntimeError(f'Upstream patch context changed: {path}')
        path.write_text(text.replace(before, after), encoding='utf-8')
    for name in ('urdf_file', 'srdf_file'):
        patch('sapien', 'wrapper/urdf_loader.py', f'open({name}, "r")', f'open({name}, "r", encoding="utf-8")')
    patch('mplib', 'planner.py',
          'if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:',
          'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:')


def assets():
    from vlastudio.deploy.interpreter.downloads import download
    from vlastudio.core.paths import cache_root
    root = Path(os.environ['VLASTUDIO_RUNTIME_ROBOTWIN_ROOT'])
    destination = root / 'assets'
    destination.mkdir(exist_ok=True)
    with FileLock(str(root / '.assets.lock'), timeout=3600):
        for name, (size, digest) in RESOURCES.items():
            receipt = destination / ('.' + name + '.json')
            target = destination / name
            if receipt.exists() and target.is_dir():
                if json.loads(receipt.read_text())['revision'] != ASSETS_REVISION:
                    raise ValueError(f'Asset version changed: {name}')
                if name == 'embodiments':
                    configure_paths(root)
                continue
            if target.exists():
                raise ValueError(f'Unmanaged or incomplete assets: {target}')
            print(f'Downloading RoboTwin resource: {name}', flush=True)
            archive = download(f'https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{ASSETS_REVISION}/{name}.zip',
                               cache_root() / 'downloads' / 'robotwin' / (name + '.zip'), size, digest,
                               offline=os.environ.get('UV_OFFLINE') == '1')
            print(f'Extracting RoboTwin resource: {name}', flush=True)
            with tempfile.TemporaryDirectory(dir=destination) as temp:
                stage = Path(temp).resolve()
                with zipfile.ZipFile(archive) as source:
                    for member in source.infolist():
                        path = (stage / member.filename).resolve()
                        if not path.is_relative_to(stage) or stat.S_ISLNK(member.external_attr >> 16):
                            raise ValueError('Unsafe resource archive member')
                    source.extractall(stage)
                extracted = stage / name
                if not extracted.is_dir():
                    raise ValueError(f'Missing expected resource directory: {name}')
                extracted.rename(target)
            receipt.write_text(json.dumps({'revision': ASSETS_REVISION, 'file': name + '.zip', 'sha256': digest}), encoding='utf-8')
            if name == 'embodiments':
                configure_paths(root)


def configure_paths(root):
    templates = list((root / 'assets' / 'embodiments').rglob('*_tmp.yml'))
    if not templates:
        raise RuntimeError('No embodiment configuration templates found')
    for template in templates:
        text = template.read_text(encoding='utf-8')
        text = text.replace('${ASSETS_PATH}', str(root)).replace('$ASSETS_PATH', str(root))
        template.with_name(template.name.replace('_tmp.yml', '.yml')).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    import sys
    {'configure': configure, 'assets': assets}[sys.argv[1]]()
