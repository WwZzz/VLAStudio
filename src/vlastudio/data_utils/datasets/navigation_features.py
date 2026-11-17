"""Convert the authors' Places365 TSV to an indexed, read-only panorama array."""
import base64
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np


class PanoramaFeatureStore:
    """Shared data asset reader; no model, training logic or writable policy cache."""
    signature = 'places365-resnet152-36x2048-v1'

    def __init__(self, path=None, download=True):
        self.path, self.download = path, download
        self._features = self._index = None

    def read(self, scan_id, viewpoint_id):
        if self._features is None:
            root = panorama_features(self.path, self.download)
            self._index = json.loads((root/'index.json').read_text())
            self._features = np.load(root/'features.npy', mmap_mode='r')
            if self._features.ndim != 3 or self._features.shape[1:] != (36, 2048) or self._features.dtype != np.float32:
                raise ValueError('Panorama data must be float32 [viewpoints,36,2048]')
        return self._features[self._index[scan_id+'_'+viewpoint_id]].copy()

    def __getstate__(self):
        return {**self.__dict__, '_features': None, '_index': None}


def convert_features(archive, output, expected_sha256):
    from types import SimpleNamespace
    a = SimpleNamespace(archive=str(archive), output=str(output), sha256=expected_sha256)
    path = Path(a.archive)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
    if digest is None:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(16*1024*1024), b''):
                h.update(chunk)
        digest = h.hexdigest()
    if digest != a.sha256:
        raise ValueError('Feature archive checksum mismatch')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'index.json').exists():
        if json.loads((out/'provenance.json').read_text())['archive_sha256'] != digest:
            raise ValueError('Feature store belongs to another archive')
        return
    csv.field_size_limit(sys.maxsize)
    import io
    with zipfile.ZipFile(path) as archive:
        name = 'ResNet-152-places365.tsv'
        # Count records without holding 4 GB of base64 text in memory.
        with archive.open(name) as stream:
            count = sum(1 for _ in stream)
        array = np.lib.format.open_memmap(out/'features.npy', mode='w+', dtype=np.float32, shape=(count, 36, 2048))
        index = {}
        with archive.open(name) as stream:
            rows = csv.DictReader(io.TextIOWrapper(stream), delimiter='\t',
                fieldnames=['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features'])
            for i, row in enumerate(rows):
                key = row['scanId']+'_'+row['viewpointId']
                if key in index:
                    raise ValueError('Duplicate panorama in feature archive')
                feature = np.frombuffer(base64.b64decode(row['features']), dtype=np.float32).reshape(36, 2048)
                if not np.isfinite(feature).all() or (int(row['image_w']), int(row['image_h']), int(row['vfov'])) != (640, 480, 60):
                    raise ValueError('Unexpected feature observation protocol')
                array[i] = feature
                index[key] = i
                if i % 1000 == 0:
                    print('Converted panoramas', i, '/', count, flush=True)
        array.flush()
    (out/'provenance.json').write_text(json.dumps({'archive_sha256': digest, 'panoramas': count,
        'shape': [count, 36, 2048], 'dtype': 'float32', 'source': 'Official Recurrent-VLN-BERT Places365 ResNet-152 features'}, indent=2)+'\n')
    (out/'index.json').write_text(json.dumps(index)+'\n')
    print('Complete feature store', count, flush=True)


FEATURE_SHA = '791344c4987de20ff792f80159a7f9e229310f421f59e4494c564b938c5bab8d'


def panorama_features(path=None, download=True):
    from filelock import FileLock
    from .navigation import cache_root, dataset_root
    from vlastudio.utils.assets import fetch
    root = Path(path) if path else dataset_root()/'places365'
    if (root/'index.json').is_file() and (root/'features.npy').is_file():
        return root
    archive = fetch('https://dl.dropboxusercontent.com/s/85tpa6tc3enl5ud/ResNet-152-places365.zip',
                    cache_root()/'downloads/ResNet-152-places365.zip', FEATURE_SHA, download)
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root)+'.lock'):
        convert_features(archive, root, FEATURE_SHA)
    return root
