"""Shared asset downloads, integrity checks, safe extraction and checkpoint lookup."""
import hashlib
import json
from pathlib import Path
import tarfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, path, checksum=None, download=True):
    """Atomically cache a public asset; local mode never makes a network request."""
    from filelock import FileLock
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path)+'.lock'):
        if path.is_file():
            if checksum and sha256(path) != checksum:
                raise ValueError(f'Asset checksum mismatch: {path}')
            return path
        if not download:
            raise FileNotFoundError(f'Asset missing with download=false: {path}')
        import requests
        temporary = path.with_name(path.name+'.partial')
        with requests.get(url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise ValueError(f'Expected an asset, received HTML: {path.name}')
            with temporary.open('wb') as output:
                for chunk in response.iter_content(8*1024*1024):
                    output.write(chunk)
        if checksum and sha256(temporary) != checksum:
            temporary.unlink()
            raise ValueError(f'Download checksum mismatch: {path.name}')
        temporary.replace(path)
    return path


def extract(archive, destination, *, strip_root=False):
    """Extract a data archive atomically; reject escapes, devices and unsafe links."""
    from filelock import FileLock
    import tempfile
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination/'.vlastudio-asset.json'
    with FileLock(str(destination)+'.lock'):
        if receipt.is_file():
            return destination
        if destination.exists():
            raise ValueError(f'Incomplete or unmanaged asset: {destination}')
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            root = Path(temporary).resolve()
            with tarfile.open(archive) as source:
                for member in source.getmembers():
                    target = (root/member.name).resolve()
                    if not target.is_relative_to(root) or member.isdev():
                        raise ValueError('Unsafe asset archive member')
                    if member.issym() or member.islnk():
                        link = ((target.parent if member.issym() else root)/member.linkname).resolve()
                        if not link.is_relative_to(root):
                            raise ValueError('Unsafe asset archive link')
                source.extractall(root, **({'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}))
            children = list(root.iterdir())
            extracted = children[0] if strip_root and len(children) == 1 and children[0].is_dir() else root
            if extracted == root:
                # TemporaryDirectory starts at 0700; native runtime children may
                # intentionally run with a different, unprivileged UID.
                extracted.chmod(0o755)
            # Rename within the same filesystem; temporary-directory cleanup is safe afterwards.
            extracted.rename(destination)
        receipt.write_text(json.dumps({'archive_sha256': sha256(archive)}, indent=2)+'\n')
    return destination


def hf_asset(repository, filename, revision, *, dataset=False, download=True, cache_dir=None):
    from huggingface_hub import hf_hub_download
    import requests
    import time
    for attempt in range(6):
        try:
            return Path(hf_hub_download(repository, filename, revision=revision,
                repo_type='dataset' if dataset else 'model',
                cache_dir=str(cache_dir) if cache_dir is not None else None, local_files_only=not download))
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError):
            if not download or attempt == 5:
                raise
            # The Hub resumes its existing partial file. Do not discard cached
            # bytes or retry authentication / missing-file errors.
            time.sleep(min(2**attempt, 30))


def resolve_checkpoint(reference, *, official, download=True):
    if reference == 'official':
        return official(download=download)
    path = Path(reference).expanduser()
    if path.exists():
        return path
    # A configured HF repository is also a valid pretrained-weight source.
    if '/' in str(reference) and not path.is_absolute() and not str(reference).startswith(('.', '~')):
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(str(reference), local_files_only=not download))
    raise FileNotFoundError(f'Checkpoint not found: {reference}')
