"""Resumable, hash-verified downloads for pinned public simulator resources."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import shutil
import time
import urllib.request

from filelock import FileLock


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _cleanup_parts(parts, count):
    """Remove only our cached parts; preserve notebook or user sidecars."""
    if not parts.is_dir():
        return
    for index in range(count):
        for suffix in ('', '.partial'):
            part = parts / (str(index) + suffix)
            if part.is_file():
                part.unlink()
    if not any(parts.iterdir()):
        parts.rmdir()


def download(url, path, size, digest, *, offline=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + '.lock', timeout=3600):
        parts = path.with_name(path.name + '.parts')
        chunk = 8 * 1024 * 1024
        count = (size + chunk - 1) // chunk
        if path.exists():
            if path.stat().st_size == size and sha256(path) == digest:
                _cleanup_parts(parts, count)
                return path
            raise ValueError(f'Cached resource failed SHA-256 verification: {path}')
        if offline:
            raise ValueError(f'Resource is not cached: {path.name}')
        parts.mkdir(exist_ok=True)

        def fetch(index):
            start, stop = index * chunk, min(size, (index + 1) * chunk)
            part = parts / str(index)
            if part.exists() and part.stat().st_size == stop - start:
                return
            for attempt in range(4):
                try:
                    # A resumed request must not reuse an expired cached CDN redirect.
                    query = f'vlastudio_range={start}-{stop - 1}&vlastudio_attempt={time.time_ns()}'
                    request = urllib.request.Request(
                        url + ('&' if '?' in url else '?') + query,
                        headers={'Range': f'bytes={start}-{stop - 1}'},
                    )
                    with urllib.request.urlopen(request, timeout=30) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{stop - 1}/{size}':
                            raise ValueError('Server did not return the requested resource range')
                        data = response.read(stop - start + 1)
                    if len(data) != stop - start:
                        raise ValueError('Incomplete resource range')
                    temp = part.with_suffix('.partial')
                    temp.write_bytes(data)
                    temp.replace(part)
                    return
                except Exception as error:
                    detail = type(error).__name__ + (f' ({error.code})' if hasattr(error, 'code') else '')
                    if attempt == 3:
                        raise RuntimeError(f'Resource download failed: {path.name}, part {index}: {detail}; retry to resume') from None
                    print(f'{path.name}: retry part {index} after {detail}', flush=True)
                    time.sleep(1 + attempt)

        workers = max(1, min(64, int(os.environ.get('VLASTUDIO_DOWNLOAD_WORKERS', '8'))))
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, i) for i in range(count)]
            try:
                for completed, future in enumerate(as_completed(futures), 1):
                    try:
                        future.result()
                    except RuntimeError as error:
                        failures.append(str(error))
                        print(str(error), flush=True)
                    if completed % 32 == 0 or completed == count:
                        print(f'{path.name}: checked {completed}/{count} parts, {len(failures)} failed', flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        if failures:
            raise RuntimeError(f'{path.name}: {len(failures)} parts failed; completed parts are cached. Retry to resume.')
        temporary = path.with_suffix(path.suffix + '.partial')
        with temporary.open('wb') as output:
            for index in range(count):
                with (parts / str(index)).open('rb') as source:
                    shutil.copyfileobj(source, output)
        if sha256(temporary) != digest:
            raise ValueError(f'Download failed SHA-256 verification: {path.name}; inspect {parts}')
        temporary.replace(path)
        _cleanup_parts(parts, count)
        return path
