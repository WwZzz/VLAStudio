"""Verify and extract the official navigator tensors with restricted loading."""

import json
from pathlib import Path
import shutil
import torch
from vlastudio.utils.assets import sha256 as digest
from vlastudio.utils.assets import fetch
from vlastudio.core.paths import cache_root as shared_cache
import io
import zipfile


BERT_SHA = '612fccc817d104ef09de4df70c081ddd692cbb85c219785c7bbfc627507c3ee0'
VOCAB_SHA = '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'


def convert_checkpoint(source, output, expected_sha256, vocabulary):
    from types import SimpleNamespace
    a = SimpleNamespace(source=str(source), output=str(output), sha256=expected_sha256, vocabulary=str(vocabulary))
    if digest(a.source) != a.sha256:
        raise ValueError('Source checkpoint checksum mismatch')
    if digest(a.vocabulary) != '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3':
        raise ValueError('BERT uncased vocabulary checksum mismatch')
    source = torch.load(a.source, weights_only=True, map_location='cpu')
    state = source['vln_bert']['state_dict']
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError('Navigator contains non-tensor state')
    from vlastudio.policy.recurrent_vlnbert.vendor.model import VLNBERT
    model = VLNBERT(feature_size=2176)
    model.load_state_dict(state, strict=True)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': state, 'source_sha256': a.sha256,
                'source_epoch': source['vln_bert'].get('epoch')}, out/'policy.pt')
    (out/'config.json').write_text(json.dumps({'ctrl_space': 'navigation', 'ctrl_type': 'candidate_viewpoint',
        'state_dim': 3, 'action_dim': 1, 'chunk_size': 1, 'feature_store': None}, indent=2)+'\n')
    (out/'policy_metadata.json').write_text(json.dumps({'policy_module': 'policy.recurrent_vlnbert', 'policy_name': 'recurrent_vlnbert'}))
    (out/'normalize.json').write_text(json.dumps({'datasets': [{'dataset_id': 'r2r',
        'ctrl_space': 'navigation', 'ctrl_type': 'candidate_viewpoint'}],
        'state': {'r2r': 'identity'}, 'action': {'r2r': 'identity'}}))
    shutil.copyfile(a.vocabulary, out/'vocab.txt')
    print('Strictly converted official navigator', len(state), 'tensors', flush=True)


class _RemoteZip(io.RawIOBase):
    """Read the published ZIP member without downloading the unused checkpoint."""
    def __init__(self, url):
        import requests
        self.url, self.pos = url, 0
        response = requests.get(url, headers={'Range': 'bytes=0-0'}, timeout=(20, 90))
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError('Official checkpoint server must support HTTP Range')
        self.size = int(response.headers['Content-Range'].split('/')[-1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        return self.pos

    def read(self, size=-1):
        import requests
        size = min(size if size >= 0 else self.size-self.pos, self.size-self.pos)
        if size <= 0:
            return b''
        response = requests.get(self.url, headers={'Range': f'bytes={self.pos}-{self.pos+size-1}'}, timeout=(20, 180))
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != size:
            raise RuntimeError('Invalid checkpoint archive range')
        self.pos += size
        return response.content


def official_checkpoint(download=True):
    from filelock import FileLock
    cache = shared_cache() / "navigation"
    root = cache / "models/recurrent_vlnbert"
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root) + ".lock"):
        required = ['policy.pt', 'config.json', 'policy_metadata.json', 'normalize.json', 'vocab.txt']
        if all((root / file).is_file() for file in required):
            return root
        source = cache/'downloads/vlnbert-prevalent.pt'
        if not source.is_file():
            if not download:
                raise FileNotFoundError(f'Checkpoint missing with download=false: {source}')
            source.parent.mkdir(parents=True, exist_ok=True)
            reader = _RemoteZip('https://zenodo.org/records/4437864/files/snap.zip?download=1')
            temporary = source.with_suffix('.partial')
            with zipfile.ZipFile(reader) as archive, archive.open('snap/VLNBERT-PREVALENT-final/state_dict/best_val_unseen') as incoming, temporary.open('wb') as output:
                import shutil
                shutil.copyfileobj(incoming, output, 8*1024*1024)
            if digest(temporary) != BERT_SHA:
                temporary.unlink()
                raise ValueError('Official VLN-BERT checkpoint checksum mismatch')
            temporary.replace(source)
        vocabulary = fetch('https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt',
                           cache/'downloads/vocab.txt', VOCAB_SHA, download)
        convert_checkpoint(source, root, BERT_SHA, vocabulary)
    return root
