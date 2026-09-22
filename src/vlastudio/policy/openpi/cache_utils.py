"""Auto-download missing OpenPI cache assets from the Hugging Face mirror.

The OpenPI upstream downloads checkpoints and tokenizer assets from Google Cloud
Storage (``gs://openpi-assets``, ``gs://big_vision``), which is unreachable from
many networks. VLAStudio mirrors the default OpenPI cache directory layout
(``~/.cache/openpi``) to a public Hugging Face repository. When a configured
OpenPI cache path is missing at startup, it is downloaded from that repository
into the expected cache location.

Environment variables:
- ``VLASTUDIO_OPENPI_CACHE_REPO``: mirror repo id (default ``WWZzz/openpi_cache``).
- ``VLASTUDIO_OPENPI_AUTO_DOWNLOAD``: set to ``0`` to disable auto-download.
"""

import os
import time
from pathlib import Path

from loguru import logger

DEFAULT_OPENPI_CACHE_REPO = "WWZzz/openpi_cache"
_MARKER = "openpi-assets"


def _auto_download_enabled() -> bool:
    value = os.environ.get("VLASTUDIO_OPENPI_AUTO_DOWNLOAD", "1")
    return value.strip().lower() not in ("0", "false", "no", "off")


def _cache_repo() -> str:
    return os.environ.get("VLASTUDIO_OPENPI_CACHE_REPO", DEFAULT_OPENPI_CACHE_REPO)


def _openpi_cache_root() -> Path:
    marker = os.environ.get("OPENPI_DATA_HOME")
    return Path(marker if marker else "~/.cache/openpi").expanduser()


def _download(local_dir: Path, allow_patterns: list[str]) -> None:
    from filelock import FileLock
    from huggingface_hub import snapshot_download

    repo = _cache_repo()
    message = (
        f"OpenPI cache miss: downloading {allow_patterns} from HF repo "
        f"'{repo}' into '{local_dir}'"
    )
    print(message, flush=True)
    logger.info(message)
    local_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(local_dir / ".vlastudio-openpi-download.lock"), timeout=3600)
    with lock:
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                snapshot_download(
                    repo_id=repo,
                    repo_type="model",
                    local_dir=str(local_dir),
                    allow_patterns=allow_patterns,
                    max_workers=8,
                )
                print(f"OpenPI cache download complete for {allow_patterns}", flush=True)
                logger.info("OpenPI cache download complete for {}", allow_patterns)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                delay = 30 * attempt
                print(
                    f"OpenPI cache download attempt {attempt} failed ({exc}); "
                    f"retrying in {delay}s",
                    flush=True,
                )
                logger.warning(
                    "OpenPI cache download attempt {} failed: {}", attempt, exc
                )
                time.sleep(delay)
        if last_exc is not None:
            raise RuntimeError(f"OpenPI cache download from '{repo}' failed: {last_exc}")


def _try_download(path: Path, local_dir: Path, allow_patterns: list[str]) -> None:
    if not _auto_download_enabled():
        logger.warning(
            "OpenPI cache path '{}' is missing and auto-download is disabled", path
        )
        return
    try:
        _download(local_dir, allow_patterns)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "OpenPI cache auto-download failed for '{}': {} "
            "(set VLASTUDIO_OPENPI_AUTO_DOWNLOAD=0 to disable, or "
            "VLASTUDIO_OPENPI_CACHE_REPO to use another mirror)",
            path,
            exc,
        )
        raise RuntimeError(
            f"OpenPI cache is missing at '{path}' and the automatic download "
            f"from '{_cache_repo()}' failed: {exc}. "
            "Set VLASTUDIO_OPENPI_AUTO_DOWNLOAD=0 to skip downloading, or "
            "VLASTUDIO_OPENPI_CACHE_REPO to use a different mirror."
        ) from exc


def ensure_openpi_checkpoint(path) -> Path:
    """Return ``path``, downloading a missing checkpoint dir from the mirror.

    A checkpoint directory is missing when it does not exist or is empty. Paths
    containing the mirrored ``openpi-assets`` layout are downloaded relative to
    that directory, so any cache root works. Other paths are left untouched:
    VLAStudio never downloads into arbitrary custom locations.
    """
    path = Path(os.path.expanduser(str(path)))
    if path.is_dir() and any(path.iterdir()):
        return path
    parts = path.resolve().parts
    if _MARKER in parts:
        index = parts.index(_MARKER)
        local_dir = Path(*parts[:index])
        repo_rel = "/".join(parts[index:])
        _try_download(path, local_dir, [f"{repo_rel}/**"])
    else:
        logger.warning(
            "OpenPI checkpoint directory '{}' is missing and does not follow "
            "the openpi-assets cache layout; skipping auto-download",
            path,
        )
    return path


def ensure_paligemma_tokenizer() -> None:
    """Ensure the PaliGemma sentencepiece tokenizer exists in the OpenPI cache.

    openpi's PaliGemmaTokenizer downloads ``gs://big_vision/paligemma_tokenizer.model``
    into the OpenPI data home. Pre-populating that file lets tokenizer
    construction succeed on networks without Google Cloud Storage access.
    """
    root = _openpi_cache_root()
    target = root / "big_vision" / "paligemma_tokenizer.model"
    if target.is_file():
        return
    _try_download(target, root, ["big_vision/paligemma_tokenizer.model"])
