"""Pinned, lazy asset caching shared by driving datasets, policies and runtimes."""
import hashlib
import json
from pathlib import Path
import tarfile

B2D_REVISION = '7ec25d1c9f7522d923ce5f3420986cef1cb2d956'
TCP_REVISION = '8a08b07883f10b7d83f6bf5dd475bda91a91c50a'
TCP_WEIGHTS_REVISION = '140aea58c02185e787c3df39d7ad79e910967b8b'
B2D_DATA_REVISION = 'd84ae18cf20a88b9220df1bc15c073d8824b7ac1'


def cache_root():
    from vlastudio.core.paths import cache_root as shared_cache
    return shared_cache()/'driving'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, filename, *, checksum=None, download=True):
    from filelock import FileLock
    target = cache_root()/'downloads'/filename
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target)+'.lock'):
        if target.is_file():
            if checksum and sha256(target) != checksum:
                raise ValueError(f'Driving asset checksum mismatch: {target}')
            return target
        if not download:
            raise FileNotFoundError(f'Asset missing with download=false: {target}')
        import requests
        temporary = target.with_suffix(target.suffix+'.partial')
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise ValueError(f'Expected an archive, received HTML: {filename}')
            with temporary.open('wb') as output:
                for chunk in response.iter_content(8*1024*1024):
                    output.write(chunk)
        if checksum and sha256(temporary) != checksum:
            temporary.unlink()
            raise ValueError(f'Driving download checksum mismatch: {filename}')
        temporary.replace(target)
    return target


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
            raise ValueError(f'Incomplete or unmanaged driving asset: {destination}')
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            root = Path(temporary).resolve()
            with tarfile.open(archive) as source:
                for member in source.getmembers():
                    target = (root/member.name).resolve()
                    if not target.is_relative_to(root) or member.isdev():
                        raise ValueError('Unsafe driving archive member')
                    if member.issym() or member.islnk():
                        link = ((target.parent if member.issym() else root)/member.linkname).resolve()
                        if not link.is_relative_to(root):
                            raise ValueError('Unsafe driving archive link')
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


def source(name, download=True):
    repositories = {'bench2drive': ('Thinklab-SJTU/Bench2Drive', B2D_REVISION),
                    'tcp': ('Thinklab-SJTU/Bench2DriveZoo', TCP_REVISION)}
    repository, revision = repositories[name]
    root = cache_root()/'sources'/f'{name}-{revision[:12]}'
    if root.joinpath('.vlastudio-asset.json').is_file():
        return root
    archive = fetch(f'https://github.com/{repository}/archive/{revision}.tar.gz',
                    f'{name}-{revision}.tar.gz', download=download)
    return extract(archive, root, strip_root=True)


def hf_asset(repository, filename, revision, *, dataset=False, download=True):
    from huggingface_hub import hf_hub_download
    import requests
    import time
    for attempt in range(6):
        try:
            return Path(hf_hub_download(repository, filename, revision=revision,
                repo_type='dataset' if dataset else 'model',
                cache_dir=str(cache_root()/'huggingface'), local_files_only=not download))
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError):
            if not download or attempt == 5:
                raise
            # The Hub resumes its existing partial file. Do not discard cached
            # bytes or retry authentication / missing-file errors.
            time.sleep(min(2**attempt, 30))
