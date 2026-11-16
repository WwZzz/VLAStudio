"""Pinned BEHAVIOR resources; invoked only after explicit license acceptance."""
import json
import os
from pathlib import Path
import stat
import tempfile
import zipfile

from filelock import FileLock

from vlastudio.deploy.interpreter.downloads import download
from vlastudio.core.paths import cache_root

REVISION = '9f0d57d465726976ed98138d3f8b8ca3e2186775'
RESOURCES = {
    'omnigibson-robot-assets': ('omnigibson-robot-assets.zip', 501401227, '5e9fe726172c8b4beb4ba4e12dbf5cf6c5791553a05c52be661eb2d6a3d241b8'),
    'behavior-1k-assets': ('behavior-1k-assets-3.7.2rc1.zip', 31280662307, '3bd7cdc1a8c5621768c2aaaf57c12a024560ab4f49d0162e540e769d8b165e8b'),
    '2025-challenge-task-instances': ('2025-challenge-task-instances.zip', 105293007, '6262a496c82a534f42166ca1569ef28dc4cd9e53ffd0f9f8b0e7ecb8c62acde3'),
}


def assets(*, accept_license=False):
    if not accept_license:
        raise ValueError('BEHAVIOR assets require explicit dataset license acceptance')
    root = Path(os.environ['OMNIGIBSON_DATA_PATH'])
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / '.assets.lock'), timeout=3600):
        for name, (filename, size, digest) in RESOURCES.items():
            target = root / name
            receipt = root / ('.' + name + '.json')
            if receipt.exists() and json.loads(receipt.read_text())['sha256'] == digest and target.is_dir():
                continue
            if target.exists():
                raise ValueError(f'Unmanaged or incomplete assets: {target}')
            print(f'Downloading BEHAVIOR resource: {name}', flush=True)
            source = download(f'https://huggingface.co/datasets/behavior-1k/zipped-datasets/resolve/{REVISION}/{filename}',
                              cache_root() / 'downloads' / 'behavior' / filename, size, digest,
                              offline=os.environ.get('UV_OFFLINE') == '1')
            print(f'Extracting BEHAVIOR resource: {name}', flush=True)
            with tempfile.TemporaryDirectory(dir=root) as temp:
                stage = Path(temp).resolve()
                unpacked = stage / name
                unpacked.mkdir()
                with zipfile.ZipFile(source) as archive:
                    for member in archive.infolist():
                        if not (unpacked / member.filename).resolve().is_relative_to(unpacked) or stat.S_ISLNK(member.external_attr >> 16):
                            raise ValueError('Unsafe resource archive member')
                    archive.extractall(unpacked)
                unpacked.rename(target)
            receipt.write_text(json.dumps({'revision': REVISION, 'sha256': digest}), encoding='utf-8')
    # Use the upstream license/key routine without exposing or redistributing its key.
    from omnigibson.utils.asset_utils import download_behavior_1k_assets
    download_behavior_1k_assets(accept_license=True)


if __name__ == '__main__':
    import sys
    import traceback
    try:
        assets(accept_license='--accept-license' in sys.argv)
    except Exception:
        # OmniGibson's exit handler can replace a failed download's exit status.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
