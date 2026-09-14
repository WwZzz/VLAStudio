"""Persistent, policy-aware caches for fully processed task datasets.

The cache boundary is deliberately after the policy data processor and data
collator. Each source sample is collated as a batch of one and serialized to an
mmap-friendly NumPy shard (or the compatibility HDF5 backend). Training only
deserializes those ready-to-use samples and merges them with
:class:`CacheCollator`.
"""

from __future__ import annotations

import contextlib
import copy
import bisect
import fnmatch
import hashlib
import importlib.util
import json
import os
import pickle
import re
import shutil
import socket
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import h5py
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate


CACHE_FORMAT_VERSION = 1
DEFAULT_CACHE_FORMAT = "npy"
DEFAULT_CACHE_LEVEL = "collator"
SUPPORTED_CACHE_FORMATS = {"npy", "hdf5"}
SUPPORTED_CACHE_LEVELS = {"collator", "processor"}
DEFAULT_MAX_OPEN_SHARDS = 16
_CACHE_CONFIG_KEYS = {
    "level",
    "root",
    "policy_name",
    "format",
    "shard_size_mb",
    "max_open_shards",
    "local_dir",
    "image_storage",
    "num_workers",
    "prefetch_factor",
}
_LEGACY_CACHE_KEYS = {
    "enable_cache",
    "cache_root",
    "cache_policy_name",
    "cache_format",
    "cache_shard_size_mb",
    "cache_local_dir",
}
_HASHED_SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml"}
_IGNORED_SOURCE_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
_TASK_PIPELINE_KEYS = (
    "action_normalize",
    "state_normalize",
    "action_norm_mask",
    "state_norm_mask",
    "max_expansion",
    "transforms",
    "action_dim",
    "state_dim",
    "camera_names",
    "image_size",
    "use_reasoning",
    "use_prev_subtask",
    "meta",
)
_EFFECTIVE_PIPELINE_ARG_KEYS = (
    "action_normalize",
    "state_normalize",
    "action_norm_mask",
    "state_norm_mask",
    "max_expansion",
    "chunk_size",
    "action_dim",
    "state_dim",
    "camera_names",
    "image_size",
    "image_sizes",
    "use_reasoning",
    "use_prev_subtask",
)


def _jsonable(value: Any) -> Any:
    """Convert configuration-like values to a deterministic JSON form."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() <= 64:
            return value.detach().cpu().tolist()
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _slug(value: str, fallback: str = "cache") -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return (value or fallback)[:96]


def get_task_cache_config(
    task_config: Mapping[str, Any], *, required: bool = False
) -> Optional[Dict[str, Any]]:
    """Return the nested task-cache config; its presence enables caching."""
    legacy = sorted(key for key in _LEGACY_CACHE_KEYS if key in task_config)
    if legacy:
        raise ValueError(
            "Top-level task-cache keys are no longer supported: "
            f"{', '.join(legacy)}. Put these settings under task_config.cache."
        )
    if "cache" not in task_config or task_config.get("cache") is False:
        if required:
            raise ValueError(
                "Task cache is not enabled. Add a 'cache:' mapping to the task config."
            )
        return None
    configured = task_config.get("cache")
    if configured is None or configured is True:
        return {}
    if not isinstance(configured, Mapping):
        raise TypeError("task_config.cache must be a mapping, true, false, or null")
    unknown = sorted(set(configured) - _CACHE_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown task_config.cache settings: {', '.join(unknown)}")
    return copy.deepcopy(dict(configured))


def is_task_cache_enabled(task_config: Mapping[str, Any]) -> bool:
    return get_task_cache_config(task_config) is not None


def resolve_cache_root(task_config: Mapping[str, Any]) -> Path:
    """Resolve an explicit data cache, task setting, or runtime/legacy default."""
    cache_config = get_task_cache_config(task_config) or {}
    configured = cache_config.get("root")
    explicit_data = os.environ.get("VLASTUDIO_DATA_CACHE_DIR")
    if explicit_data:
        root = Path(os.path.expandvars(os.path.expanduser(explicit_data))) / "tasks"
    elif configured:
        root = Path(os.path.expandvars(os.path.expanduser(str(configured))))
    elif os.environ.get("VLASTUDIO_CACHE_DIR") and os.environ.get("ILSTD_CACHE"):
        root = Path(os.environ["ILSTD_CACHE"]) / "tasks"
    else:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        root = Path(os.path.expandvars(os.path.expanduser(hf_home))) / "ilstd_cache"
    return root.resolve()


def hash_policy_module(module_path: str) -> str:
    """Hash source/config files belonging to a policy module."""
    spec = importlib.util.find_spec(module_path)
    if spec is None:
        raise ImportError(f"Cannot resolve policy module for task cache: {module_path}")

    if spec.submodule_search_locations:
        roots = [Path(path) for path in spec.submodule_search_locations]
    elif spec.origin:
        roots = [Path(spec.origin)]
    else:
        raise ImportError(f"Policy module has no filesystem source: {module_path}")

    digest = hashlib.sha256()
    files: List[tuple[str, Path]] = []
    for root in roots:
        if root.is_file():
            files.append((root.name, root))
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _HASHED_SOURCE_SUFFIXES:
                continue
            if any(part in _IGNORED_SOURCE_PARTS for part in path.parts):
                continue
            files.append((path.relative_to(root).as_posix(), path))

    if not files:
        raise RuntimeError(f"No hashable source files found for policy module: {module_path}")

    for relative_path, path in sorted(files, key=lambda item: item[0]):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _object_signature(value: Any, depth: int = 0, seen: Optional[set[int]] = None) -> Any:
    """Extract cache-relevant, deterministic state from processors/collators."""
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (bool, int, float, str, Path, torch.dtype, torch.device)):
        return _jsonable(value)
    if isinstance(value, (np.generic, np.ndarray, torch.Tensor)):
        return _jsonable(value)
    if isinstance(value, Mapping):
        if len(value) > 64:
            return {"type": value.__class__.__name__, "length": len(value)}
        return {
            str(key): _object_signature(item, depth + 1, seen)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            return {"type": value.__class__.__name__, "length": len(value)}
        return [_object_signature(item, depth + 1, seen) for item in value]

    signature: Dict[str, Any] = {
        "class": f"{value.__class__.__module__}.{value.__class__.__qualname__}"
    }
    if depth >= 3 or id(value) in seen:
        return signature
    seen.add(id(value))

    common_attributes = (
        "name_or_path",
        "model_max_length",
        "pad_token_id",
        "padding_side",
        "max_token_len",
        "max_seq_len",
        "image_size",
        "chunk_size",
        "model_action_dim",
        "discrete_state_input",
        "pi05",
        "dtype",
        "video",
    )
    attributes: Dict[str, Any] = {}
    raw_attributes = getattr(value, "__dict__", {})
    for key, item in raw_attributes.items():
        if key.startswith("_"):
            continue
        if isinstance(item, (bool, int, float, str, Path, torch.dtype, torch.device, np.generic)):
            attributes[key] = _jsonable(item)
        elif isinstance(item, (list, tuple, Mapping)) and len(item) <= 64:
            try:
                attributes[key] = _object_signature(item, depth + 1, seen)
            except Exception:
                continue
    for key in common_attributes:
        if key in attributes or not hasattr(value, key):
            continue
        try:
            attributes[key] = _object_signature(getattr(value, key), depth + 1, seen)
        except Exception:
            continue

    for nested_name in ("tokenizer", "processor", "collator", "image_processor"):
        nested = getattr(value, nested_name, None)
        if nested is not None:
            attributes[nested_name] = _object_signature(nested, depth + 1, seen)
    if attributes:
        signature["attributes"] = attributes
    return signature


def infer_collation_spec(collator: Any) -> Dict[str, Any]:
    """Capture padding details needed by the generic cache collator."""
    queue: List[Any] = [collator]
    seen: set[int] = set()
    tokenizer = None
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        candidate = getattr(current, "tokenizer", None)
        if candidate is not None:
            tokenizer = candidate
            break
        for name in ("processor", "collator"):
            nested = getattr(current, name, None)
            if nested is not None:
                queue.append(nested)

    padding_side = getattr(tokenizer, "padding_side", "right") if tokenizer is not None else "right"
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
    if pad_token_id is None:
        pad_token_id = 0
    return {
        "padding_side": padding_side if padding_side in {"left", "right"} else "right",
        "pad_token_id": int(pad_token_id),
        "label_pad_token_id": -100,
    }


def _dataset_name(config: Mapping[str, Any], index: int, kind: str) -> str:
    if config.get("name"):
        return str(config["name"])
    for key, value in config.items():
        if isinstance(value, list):
            return str(key)
    return f"{kind}_{index:04d}"


@dataclass(frozen=True)
class CacheDescriptor:
    index: int
    kind: str
    name: str
    config: Dict[str, Any]
    config_hash: str
    dataset_hash: str
    cache_dir: Path
    cache_file: Path
    manifest_file: Path


class _FileLock:
    """Small cross-platform advisory lock used for first-build serialization."""

    def __init__(self, path: Path):
        self.path = path
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        if os.name == "nt":
            import msvcrt

            self._handle.seek(0)
            if self._handle.tell() == 0:
                self._handle.write(b"0")
                self._handle.flush()
            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is None:
            return
        if os.name == "nt":
            import msvcrt

            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_uint8_npy_from_raw_atomic(path: Path, raw_path: Path, size: int) -> None:
    """Wrap a raw byte stream in an mmap-compatible NPY file without buffering it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    destination = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint8, shape=(size,)
    )
    position = 0
    with raw_path.open("rb") as source:
        while True:
            chunk = source.read(16 * 1024 * 1024)
            if not chunk:
                break
            stop = position + len(chunk)
            destination[position:stop] = np.frombuffer(chunk, dtype=np.uint8)
            position = stop
    if position != size:
        raise IOError(f"Raw cache shard size changed while finalizing {raw_path}")
    destination.flush()
    del destination
    os.replace(temporary, path)


def _normalize_image_storage_config(
    cache_config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    configured = cache_config.get("image_storage")
    if configured in (None, False):
        return None
    if not isinstance(configured, Mapping):
        raise TypeError("task_config.cache.image_storage must be a mapping or false")
    unknown = sorted(
        set(configured)
        - {"dtype", "fields", "value_range", "restore_dtype", "strict_range"}
    )
    if unknown:
        raise ValueError(
            "Unknown cache.image_storage settings: " + ", ".join(unknown)
        )

    storage_dtype = str(configured.get("dtype", "uint8")).lower()
    if storage_dtype != "uint8":
        raise ValueError("cache.image_storage.dtype currently supports only 'uint8'")
    fields = configured.get("fields")
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, Sequence) or not fields or not all(
        isinstance(field, str) and field for field in fields
    ):
        raise ValueError("cache.image_storage.fields must be a non-empty list of paths")
    value_range = configured.get("value_range")
    if (
        not isinstance(value_range, Sequence)
        or isinstance(value_range, (str, bytes))
        or len(value_range) != 2
    ):
        raise ValueError("cache.image_storage.value_range must contain [minimum, maximum]")
    minimum, maximum = (float(value_range[0]), float(value_range[1]))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
        raise ValueError("cache.image_storage.value_range must be finite and increasing")
    restore_dtype = str(configured.get("restore_dtype", "float32")).lower()
    if restore_dtype not in {"float16", "float32", "float64", "bfloat16"}:
        raise ValueError(
            "cache.image_storage.restore_dtype must be float16, float32, float64, or bfloat16"
        )
    return {
        "dtype": storage_dtype,
        "fields": list(fields),
        "value_range": [minimum, maximum],
        "restore_dtype": restore_dtype,
        "strict_range": bool(configured.get("strict_range", True)),
    }


def _matches_image_storage_field(path: Sequence[str], config: Mapping[str, Any]) -> bool:
    dotted_path = ".".join(path)
    return any(fnmatch.fnmatchcase(dotted_path, pattern) for pattern in config["fields"])


def _map_image_storage_fields(
    value: Any,
    config: Optional[Mapping[str, Any]],
    transform: Any,
    path: Sequence[str] = (),
) -> Any:
    if not config:
        return value
    if isinstance(value, Mapping):
        return value.__class__(
            (key, _map_image_storage_fields(item, config, transform, (*path, str(key))))
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(
            _map_image_storage_fields(item, config, transform, (*path, str(index)))
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _map_image_storage_fields(item, config, transform, (*path, str(index)))
            for index, item in enumerate(value)
        ]
    if _matches_image_storage_field(path, config):
        return transform(value, config, ".".join(path))
    return value


def _validate_image_range(
    minimum_value: float,
    maximum_value: float,
    config: Mapping[str, Any],
    field_path: str,
) -> None:
    if not np.isfinite(minimum_value) or not np.isfinite(maximum_value):
        raise ValueError(f"Image field {field_path!r} contains non-finite values")
    if not config.get("strict_range", True):
        return
    expected_minimum, expected_maximum = config["value_range"]
    tolerance = max(1e-5, (expected_maximum - expected_minimum) * 1e-5)
    if (
        minimum_value < expected_minimum - tolerance
        or maximum_value > expected_maximum + tolerance
    ):
        raise ValueError(
            f"Image field {field_path!r} has range [{minimum_value}, {maximum_value}], "
            f"outside configured cache range [{expected_minimum}, {expected_maximum}]"
        )


def _encode_image_value(value: Any, config: Mapping[str, Any], field_path: str) -> Any:
    minimum, maximum = config["value_range"]
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise TypeError(
                f"Image field {field_path!r} must be floating point "
                "before uint8 encoding"
            )
        if value.numel():
            _validate_image_range(
                float(value.min().item()), float(value.max().item()), config, field_path
            )
        normalized = (value - minimum) / (maximum - minimum)
        return (normalized * 255.0).round().clamp(0, 255).to(torch.uint8)
    if isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(
                f"Image field {field_path!r} must be floating point "
                "before uint8 encoding"
            )
        if value.size:
            _validate_image_range(
                float(value.min()), float(value.max()), config, field_path
            )
        normalized = (value - minimum) / (maximum - minimum)
        return np.rint(normalized * 255.0).clip(0, 255).astype(np.uint8)
    raise TypeError(
        f"Image field {field_path!r} must be a torch.Tensor or numpy.ndarray, got {type(value)}"
    )


def _decode_image_value(value: Any, config: Mapping[str, Any], field_path: str) -> Any:
    minimum, maximum = config["value_range"]
    dtype_name = config["restore_dtype"]
    torch_dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    numpy_dtypes = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.uint8:
            raise TypeError(f"Encoded image field {field_path!r} is not uint8")
        restored = value.to(torch_dtypes[dtype_name]) / 255.0
        return restored * (maximum - minimum) + minimum
    if isinstance(value, np.ndarray):
        if value.dtype != np.uint8:
            raise TypeError(f"Encoded image field {field_path!r} is not uint8")
        if dtype_name == "bfloat16":
            raise ValueError("bfloat16 restoration requires torch.Tensor image fields")
        restored = value.astype(numpy_dtypes[dtype_name]) / 255.0
        return restored * (maximum - minimum) + minimum
    raise TypeError(
        f"Encoded image field {field_path!r} must be a tensor or array, got {type(value)}"
    )


def _encode_image_storage(value: Any, config: Optional[Mapping[str, Any]]) -> Any:
    if not config:
        return value
    matches = 0

    def encode(item: Any, item_config: Mapping[str, Any], field_path: str) -> Any:
        nonlocal matches
        matches += 1
        return _encode_image_value(item, item_config, field_path)

    encoded = _map_image_storage_fields(value, config, encode)
    if matches == 0:
        raise ValueError(
            "cache.image_storage.fields did not match any collated batch field: "
            + ", ".join(config["fields"])
        )
    return encoded


def _decode_image_storage(value: Any, config: Optional[Mapping[str, Any]]) -> Any:
    if not config:
        return value
    matches = 0

    def decode(item: Any, item_config: Mapping[str, Any], field_path: str) -> Any:
        nonlocal matches
        matches += 1
        return _decode_image_value(item, item_config, field_path)

    decoded = _map_image_storage_fields(value, config, decode)
    if matches == 0:
        raise ValueError(
            "Cached image fields are missing from a record: "
            + ", ".join(config["fields"])
        )
    return decoded


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, Mapping):
        return value.__class__((key, _to_cpu(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _episode_id(sample: Any, fallback: int) -> str:
    value: Any = fallback
    if isinstance(sample, Mapping):
        for key in ("episode_id", "episode_index", "trajectory_id", "traj_index"):
            if key in sample:
                value = sample[key]
                break
        else:
            metadata = sample.get("meta")
            if isinstance(metadata, Mapping):
                for key in ("episode_id", "episode_index", "trajectory_id", "traj_index"):
                    if key in metadata:
                        value = metadata[key]
                        break
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.item() if value.size == 1 else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)


class _IndexedCacheDataset(Dataset):
    """Attach the source index without changing the source dataset contract."""

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, Any]:
        return index, self.dataset[index]


class _CacheBuildCollator:
    """Run the expensive policy pipeline inside DataLoader workers."""

    def __init__(
        self,
        processor: Any,
        collator: Any,
        image_storage: Optional[Mapping[str, Any]],
        cache_level: str,
    ):
        self.processor = processor
        self.collator = collator
        self.image_storage = image_storage
        self.cache_level = cache_level
        if cache_level not in SUPPORTED_CACHE_LEVELS:
            raise ValueError(
                f"Unsupported task cache level: {cache_level!r}"
            )

    def __call__(self, indexed_samples: Sequence[tuple[int, Any]]) -> tuple[int, str, Any]:
        if len(indexed_samples) != 1:
            raise ValueError(
                "Task-cache build DataLoader must use batch_size=1 to preserve "
                "the one-source-sample-per-record cache contract"
            )
        index, sample = indexed_samples[0]
        processed = self.processor(sample) if self.processor is not None else sample
        if self.cache_level == "processor":
            payload = processed
        else:
            payload = (
                self.collator([processed])
                if self.collator is not None
                else default_collate([processed])
            )
        payload = _to_cpu(_encode_image_storage(payload, self.image_storage))
        return int(index), _episode_id(sample, fallback=index), payload


def _iter_cache_records(
    dataset: Any,
    processor: Any,
    collator: Any,
    image_storage: Optional[Mapping[str, Any]],
    num_workers: int,
    prefetch_factor: int,
    cache_level: str = DEFAULT_CACHE_LEVEL,
) -> Iterator[tuple[int, str, Any]]:
    """Yield processed records in source order using a non-shuffling DataLoader."""
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        if num_workers:
            logger.warning(
                "Task-cache source is not map-style; falling back to a sequential "
                "iterator because parallel workers cannot partition it safely"
            )
        build_collator = _CacheBuildCollator(
            processor, collator, image_storage, cache_level
        )
        for index, sample in enumerate(iter(dataset)):
            yield build_collator([(index, sample)])
        return

    loader_kwargs: Dict[str, Any] = {
        "dataset": _IndexedCacheDataset(dataset),
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": _CacheBuildCollator(
            processor, collator, image_storage, cache_level
        ),
        "drop_last": False,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        # Cache construction makes one pass, so persistent workers only delay
        # process cleanup and provide no next-epoch benefit.
        loader_kwargs["persistent_workers"] = False
    logger.info(
        f"Building {cache_level}-level task cache with DataLoader(shuffle=False, batch_size=1, "
        f"num_workers={num_workers}, prefetch_factor="
        f"{prefetch_factor if num_workers else None})"
    )
    expected_index = 0
    for index, episode_id, batch in DataLoader(**loader_kwargs):
        if index != expected_index:
            raise RuntimeError(
                "Task-cache DataLoader returned samples out of order: "
                f"expected {expected_index}, got {index}"
            )
        yield index, episode_id, batch
        expected_index += 1


class _PayloadWriter:
    """Append pickled records to one contiguous HDF5 byte stream."""

    def __init__(self, h5_file: h5py.File, flush_records: int = 128):
        self.payloads = h5_file.create_dataset(
            "payloads",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=(4 * 1024 * 1024,),
        )
        self.offsets = h5_file.create_dataset(
            "offsets", shape=(1,), maxshape=(None,), dtype=np.uint64, chunks=True
        )
        self.offsets[0] = 0
        self.episode_ids = h5_file.create_dataset(
            "episode_ids",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True,
        )
        self.flush_records = flush_records
        self._records: List[bytes] = []
        self._episodes: List[str] = []
        self.sample_count = 0
        self.byte_count = 0

    def append(self, payload: Any, episode_id: str) -> None:
        self._records.append(pickle.dumps(_to_cpu(payload), protocol=pickle.HIGHEST_PROTOCOL))
        self._episodes.append(episode_id)
        if len(self._records) >= self.flush_records:
            self.flush()

    def flush(self) -> None:
        if not self._records:
            return
        joined = b"".join(self._records)
        encoded = np.frombuffer(joined, dtype=np.uint8)
        old_bytes = self.byte_count
        self.byte_count += encoded.size
        self.payloads.resize((self.byte_count,))
        self.payloads[old_bytes : self.byte_count] = encoded

        lengths = np.asarray([len(record) for record in self._records], dtype=np.uint64)
        new_offsets = old_bytes + np.cumsum(lengths, dtype=np.uint64)
        old_samples = self.sample_count
        self.sample_count += len(self._records)
        self.offsets.resize((self.sample_count + 1,))
        self.offsets[old_samples + 1 : self.sample_count + 1] = new_offsets
        self.episode_ids.resize((self.sample_count,))
        self.episode_ids[old_samples : self.sample_count] = self._episodes
        self._records.clear()
        self._episodes.clear()


def _episode_runs(episode_ids: Sequence[str]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for index, episode_id in enumerate(episode_ids):
        if not runs or runs[-1]["episode_id"] != episode_id:
            runs.append({"episode_id": episode_id, "start": index, "stop": index + 1})
        else:
            runs[-1]["stop"] = index + 1
    return runs


def _build_hdf5_cache(
    dataset: Any,
    processor: Any,
    collator: Any,
    descriptor: CacheDescriptor,
    common_manifest: Mapping[str, Any],
    image_storage: Optional[Mapping[str, Any]] = None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    cache_level: str = DEFAULT_CACHE_LEVEL,
) -> Dict[str, Any]:
    descriptor.cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = descriptor.cache_file.with_name(
        f".{descriptor.cache_file.name}.tmp-{socket.gethostname()}-{os.getpid()}"
    )
    if temporary.exists():
        temporary.unlink()

    started = time.time()
    episode_ids: List[str] = []
    try:
        with h5py.File(temporary, "w", libver="latest") as h5_file:
            h5_file.attrs["format_version"] = CACHE_FORMAT_VERSION
            h5_file.attrs["complete"] = False
            writer = _PayloadWriter(h5_file)
            for index, episode_id, payload in _iter_cache_records(
                dataset,
                processor,
                collator,
                image_storage,
                num_workers,
                prefetch_factor,
                cache_level,
            ):
                episode_ids.append(episode_id)
                writer.append(payload, episode_id)
                if (index + 1) % 1000 == 0:
                    logger.info(
                        f"Task cache '{descriptor.name}': processed {index + 1:,} samples"
                    )
            writer.flush()
            if writer.sample_count == 0:
                raise ValueError(f"Cannot cache empty dataset: {descriptor.name}")
            h5_file.attrs["sample_count"] = writer.sample_count
            h5_file.attrs["payload_bytes"] = writer.byte_count
            h5_file.attrs["complete"] = True
            h5_file.flush()

        os.replace(temporary, descriptor.cache_file)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise

    manifest = {
        **common_manifest,
        "format_version": CACHE_FORMAT_VERSION,
        "cache_format": "hdf5",
        "cache_level": cache_level,
        "complete": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {
            "index": descriptor.index,
            "kind": descriptor.kind,
            "name": descriptor.name,
            "config": descriptor.config,
            "config_hash": descriptor.config_hash,
            "dataset_hash": descriptor.dataset_hash,
        },
        "sample_count": len(episode_ids),
        "episode_count": len({episode_id for episode_id in episode_ids}),
        "episode_runs": _episode_runs(episode_ids),
        "build_seconds": time.time() - started,
        "build_dataloader": {
            "shuffle": False,
            "batch_size": 1,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor if num_workers else None,
        },
    }
    _write_json_atomic(descriptor.manifest_file, manifest)
    logger.info(
        f"Built task cache '{descriptor.name}' with {len(episode_ids):,} samples at "
        f"{descriptor.cache_file}"
    )
    return manifest


def _build_npy_cache(
    dataset: Any,
    processor: Any,
    collator: Any,
    descriptor: CacheDescriptor,
    common_manifest: Mapping[str, Any],
    shard_size_bytes: int,
    image_storage: Optional[Mapping[str, Any]] = None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    cache_level: str = DEFAULT_CACHE_LEVEL,
) -> Dict[str, Any]:
    """Build mmap-friendly payload and offset ``.npy`` shard pairs.

    A shard is closed only at an episode boundary after reaching the target
    size.  Consequently, normal episodes are never split between shards and a
    distributed sampler can assign complete shards to ranks.
    """
    descriptor.cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_episode_ids: List[str] = []
    shards: List[Dict[str, Any]] = []
    record_offsets: List[int] = [0]
    record_count = 0
    record_bytes = 0
    shard_start = 0
    previous_episode: Optional[str] = None
    raw_path = descriptor.cache_dir / f".payload.raw-{os.getpid()}"
    raw_stream = raw_path.open("wb")

    def flush_shard() -> None:
        nonlocal record_offsets, record_count, record_bytes, shard_start, raw_stream
        if record_count == 0:
            return
        raw_stream.flush()
        os.fsync(raw_stream.fileno())
        raw_stream.close()
        shard_index = len(shards)
        payload_name = f"payload-{shard_index:05d}.npy"
        offsets_name = f"offsets-{shard_index:05d}.npy"
        _write_uint8_npy_from_raw_atomic(
            descriptor.cache_dir / payload_name, raw_path, record_bytes
        )
        _write_npy_atomic(
            descriptor.cache_dir / offsets_name,
            np.asarray(record_offsets, dtype=np.uint64),
        )
        shard_stop = shard_start + record_count
        shard_episode_ids = all_episode_ids[shard_start:shard_stop]
        shards.append(
            {
                "index": shard_index,
                "start": shard_start,
                "stop": shard_stop,
                "sample_count": record_count,
                "payload_bytes": record_bytes,
                "payload": payload_name,
                "offsets": offsets_name,
                "episode_ids": list(dict.fromkeys(shard_episode_ids)),
            }
        )
        shard_start = shard_stop
        record_offsets = [0]
        record_count = 0
        record_bytes = 0
        raw_path.unlink()
        raw_stream = raw_path.open("wb")

    try:
        for index, episode_id, payload in _iter_cache_records(
            dataset,
            processor,
            collator,
            image_storage,
            num_workers,
            prefetch_factor,
            cache_level,
        ):
            if (
                record_count
                and record_bytes >= shard_size_bytes
                and previous_episode is not None
                and episode_id != previous_episode
            ):
                flush_shard()
            record = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            raw_stream.write(record)
            record_bytes += len(record)
            record_count += 1
            record_offsets.append(record_bytes)
            all_episode_ids.append(episode_id)
            previous_episode = episode_id
            if (index + 1) % 1000 == 0:
                logger.info(f"Task cache '{descriptor.name}': processed {index + 1:,} samples")
        flush_shard()
    finally:
        if not raw_stream.closed:
            raw_stream.close()
        with contextlib.suppress(FileNotFoundError):
            raw_path.unlink()

    if not all_episode_ids:
        raise ValueError(f"Cannot cache empty dataset: {descriptor.name}")
    manifest = {
        **common_manifest,
        "format_version": CACHE_FORMAT_VERSION,
        "cache_format": "npy",
        "cache_level": cache_level,
        "complete": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {
            "index": descriptor.index,
            "kind": descriptor.kind,
            "name": descriptor.name,
            "config": descriptor.config,
            "config_hash": descriptor.config_hash,
            "dataset_hash": descriptor.dataset_hash,
        },
        "sample_count": len(all_episode_ids),
        "episode_count": len(set(all_episode_ids)),
        "episode_runs": _episode_runs(all_episode_ids),
        "shard_size_bytes": shard_size_bytes,
        "shards": shards,
        "build_seconds": time.time() - started,
        "build_dataloader": {
            "shuffle": False,
            "batch_size": 1,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor if num_workers else None,
        },
    }
    _write_json_atomic(descriptor.manifest_file, manifest)
    logger.info(
        f"Built mmap task cache '{descriptor.name}' with {len(all_episode_ids):,} samples "
        f"in {len(shards)} shards at {descriptor.cache_dir}"
    )
    return manifest


def _build_cache(
    dataset: Any,
    processor: Any,
    collator: Any,
    descriptor: CacheDescriptor,
    common_manifest: Mapping[str, Any],
    cache_format: str,
    shard_size_bytes: int,
    image_storage: Optional[Mapping[str, Any]] = None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    cache_level: str = DEFAULT_CACHE_LEVEL,
) -> Dict[str, Any]:
    if cache_format == "npy":
        return _build_npy_cache(
            dataset,
            processor,
            collator,
            descriptor,
            common_manifest,
            shard_size_bytes,
            image_storage,
            num_workers,
            prefetch_factor,
            cache_level,
        )
    if cache_format == "hdf5":
        return _build_hdf5_cache(
            dataset, processor, collator, descriptor, common_manifest, image_storage,
            num_workers, prefetch_factor, cache_level
        )
    raise ValueError(f"Unsupported task cache format: {cache_format}")


_NPY_SHARD_POOL_LOCK = threading.RLock()
_NPY_SHARD_POOL: "OrderedDict[tuple[str, int], tuple[np.ndarray, np.ndarray]]" = OrderedDict()


def _close_memmap(array: np.ndarray) -> None:
    memory_map = getattr(array, "_mmap", None)
    if memory_map is not None:
        with contextlib.suppress(Exception):
            memory_map.close()


def _evict_npy_shards(max_open_shards: int) -> None:
    while len(_NPY_SHARD_POOL) > max_open_shards:
        _, (payload, _) = _NPY_SHARD_POOL.popitem(last=False)
        _close_memmap(payload)


def _close_dataset_npy_shards(manifest_file: str) -> None:
    with _NPY_SHARD_POOL_LOCK:
        keys = [key for key in _NPY_SHARD_POOL if key[0] == manifest_file]
        for key in keys:
            payload, _ = _NPY_SHARD_POOL.pop(key)
            _close_memmap(payload)


class CacheDataset(Dataset):
    """Map-style dataset backed by HDF5 or mmap-friendly NumPy shards."""

    def __init__(
        self,
        cache_file: os.PathLike[str] | str,
        manifest_file: os.PathLike[str] | str,
        local_cache_dir: Optional[os.PathLike[str] | str] = None,
        max_open_shards: int = DEFAULT_MAX_OPEN_SHARDS,
    ):
        self.cache_file = str(cache_file)
        self.manifest_file = str(manifest_file)
        with open(self.manifest_file, "r", encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        if not self.manifest.get("complete"):
            raise RuntimeError(f"Incomplete task cache manifest: {self.manifest_file}")
        self.name = self.manifest["dataset"]["name"]
        self.dataset_id = self.name
        self.cache_level = self.manifest.get("cache_level", DEFAULT_CACHE_LEVEL)
        self.episode_runs = self.manifest.get("episode_runs", [])
        self.collation_spec = self.manifest.get("collation_spec", {})
        self.image_storage = self.manifest.get("image_storage")
        self.cache_format = self.manifest.get("cache_format", "hdf5")
        self._length = int(self.manifest["sample_count"])
        self.shards = self.manifest.get("shards", [])
        self.shard_ranges = [
            (int(shard["start"]), int(shard["stop"])) for shard in self.shards
        ]
        self._shard_stops = [stop for _, stop in self.shard_ranges]
        self.local_cache_dir = str(local_cache_dir) if local_cache_dir else None
        self.max_open_shards = int(max_open_shards)
        if self.max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")
        self._h5: Optional[h5py.File] = None
        self._offsets: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self._length

    def _ensure_hdf5_open(self) -> None:
        if self._h5 is not None:
            return
        self._h5 = h5py.File(self.cache_file, "r", swmr=True, libver="latest")
        if not bool(self._h5.attrs.get("complete", False)):
            raise RuntimeError(f"Incomplete task cache file: {self.cache_file}")
        self._offsets = self._h5["offsets"][:]

    def _stage_file(self, source: Path, shard_index: int) -> Path:
        if self.local_cache_dir is None:
            return source
        dataset_hash = self.manifest["dataset"]["dataset_hash"]
        destination_dir = Path(self.local_cache_dir) / dataset_hash[:16]
        destination = destination_dir / source.name
        destination_dir.mkdir(parents=True, exist_ok=True)
        lock_path = destination_dir / f".{source.name}.lock"
        with _FileLock(lock_path):
            if destination.is_file() and destination.stat().st_size == source.stat().st_size:
                return destination
            temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        logger.info(
            f"Staged task-cache shard {shard_index} for '{self.name}' to {destination}"
        )
        return destination

    def _npy_pool_key(self, shard_index: int) -> tuple[str, int]:
        return (self.manifest_file, shard_index)

    def _npy_record(self, shard_index: int, local_index: int) -> Any:
        pool_key = self._npy_pool_key(shard_index)
        with _NPY_SHARD_POOL_LOCK:
            cached = _NPY_SHARD_POOL.get(pool_key)
            if cached is not None:
                _NPY_SHARD_POOL.move_to_end(pool_key)
                payload, offsets = cached
                start = int(offsets[local_index])
                stop = int(offsets[local_index + 1])
                return pickle.loads(memoryview(payload[start:stop]))

        shard = self.shards[shard_index]
        cache_dir = Path(self.manifest_file).parent
        payload_path = self._stage_file(cache_dir / shard["payload"], shard_index)
        offsets_path = self._stage_file(cache_dir / shard["offsets"], shard_index)
        with _NPY_SHARD_POOL_LOCK:
            cached = _NPY_SHARD_POOL.get(pool_key)
            if cached is None:
                payload = np.load(payload_path, mmap_mode="r", allow_pickle=False)
                # Offsets are small and frequently accessed. Keeping them in RAM
                # avoids a second mmap/file descriptor per shard.
                offsets = np.load(offsets_path, allow_pickle=False)
                _NPY_SHARD_POOL[pool_key] = (payload, offsets)
                _evict_npy_shards(self.max_open_shards)
            else:
                payload, offsets = cached
                _NPY_SHARD_POOL.move_to_end(pool_key)
            start = int(offsets[local_index])
            stop = int(offsets[local_index + 1])
            return pickle.loads(memoryview(payload[start:stop]))

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        if self.cache_format == "npy":
            shard_index = bisect.bisect_right(self._shard_stops, index)
            shard = self.shards[shard_index]
            local_index = index - int(shard["start"])
            return self._npy_record(shard_index, local_index)
        self._ensure_hdf5_open()
        assert self._h5 is not None and self._offsets is not None
        start = int(self._offsets[index])
        stop = int(self._offsets[index + 1])
        payload = self._h5["payloads"][start:stop].tobytes()
        return pickle.loads(payload)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._offsets = None
        _close_dataset_npy_shards(self.manifest_file)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_offsets"] = None
        return state

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()


class CacheBeforeCollatorDataset(CacheDataset):
    """Load processor-level records for the original policy collator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cache_level != "processor":
            raise ValueError(
                "CacheBeforeCollatorDataset requires a processor-level cache, "
                f"got {self.cache_level!r}"
            )

    def __getitem__(self, index: int) -> Any:
        payload = super().__getitem__(index)
        return _decode_image_storage(payload, self.image_storage)


def _padding_value(path: Sequence[str], dtype: Any, spec: Mapping[str, Any]) -> Any:
    key = path[-1].lower() if path else ""
    if "label" in key:
        return spec.get("label_pad_token_id", -100)
    if "input_id" in key:
        return spec.get("pad_token_id", 0)
    if "mask" in key or dtype in (torch.bool, np.bool_):
        return False
    return 0


def _pad_tensor(value: torch.Tensor, shape: Sequence[int], path: Sequence[str], spec: Mapping[str, Any]) -> torch.Tensor:
    if list(value.shape) == list(shape):
        return value
    fill = _padding_value(path, value.dtype, spec)
    output = torch.full(tuple(shape), fill, dtype=value.dtype, device=value.device)
    slices = [slice(0, size) for size in value.shape]
    key = path[-1].lower() if path else ""
    if spec.get("padding_side") == "left" and value.ndim >= 2 and any(
        marker in key for marker in ("input_id", "label", "attention_mask", "position_id")
    ):
        slices[1] = slice(shape[1] - value.shape[1], shape[1])
    output[tuple(slices)] = value
    return output


def _merge_tensors(values: Sequence[torch.Tensor], path: Sequence[str], spec: Mapping[str, Any]) -> torch.Tensor:
    if values[0].ndim == 0:
        return torch.stack(list(values))
    rank = values[0].ndim
    if any(value.ndim != rank for value in values):
        raise ValueError(f"Cached tensor ranks differ at {'.'.join(path)}")
    target_shape = [max(value.shape[dimension] for value in values) for dimension in range(rank)]
    # Dimension zero is the already-collated batch/row dimension and is concatenated.
    padded = [
        _pad_tensor(value, [value.shape[0], *target_shape[1:]], path, spec)
        for value in values
    ]
    return torch.cat(padded, dim=0)


def _pad_array(value: np.ndarray, shape: Sequence[int], path: Sequence[str], spec: Mapping[str, Any]) -> np.ndarray:
    if list(value.shape) == list(shape):
        return value
    fill = _padding_value(path, value.dtype.type, spec)
    output = np.full(tuple(shape), fill, dtype=value.dtype)
    slices = [slice(0, size) for size in value.shape]
    key = path[-1].lower() if path else ""
    if spec.get("padding_side") == "left" and value.ndim >= 2 and any(
        marker in key for marker in ("input_id", "label", "attention_mask", "position_id")
    ):
        slices[1] = slice(shape[1] - value.shape[1], shape[1])
    output[tuple(slices)] = value
    return output


def _merge_cached(values: Sequence[Any], path: Sequence[str], spec: Mapping[str, Any]) -> Any:
    first = values[0]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Cached values mix None and non-None at {'.'.join(path)}")
    if isinstance(first, torch.Tensor):
        return _merge_tensors(values, path, spec)
    if isinstance(first, np.ndarray):
        if first.ndim == 0:
            return np.stack(values)
        rank = first.ndim
        if any(value.ndim != rank for value in values):
            raise ValueError(f"Cached array ranks differ at {'.'.join(path)}")
        target_shape = [max(value.shape[dimension] for value in values) for dimension in range(rank)]
        padded = [
            _pad_array(value, [value.shape[0], *target_shape[1:]], path, spec)
            for value in values
        ]
        return np.concatenate(padded, axis=0)
    if isinstance(first, Mapping):
        keys = list(first.keys())
        for value in values[1:]:
            if list(value.keys()) != keys:
                raise ValueError(f"Cached mapping keys differ at {'.'.join(path)}")
        return first.__class__(
            (key, _merge_cached([value[key] for value in values], [*path, str(key)], spec))
            for key in keys
        )
    if isinstance(first, tuple):
        if any(len(value) != len(first) for value in values):
            raise ValueError(f"Cached tuple lengths differ at {'.'.join(path)}")
        return tuple(
            _merge_cached([value[index] for value in values], [*path, str(index)], spec)
            for index in range(len(first))
        )
    if isinstance(first, list):
        if all(len(value) == len(first) for value in values):
            return [
                _merge_cached([value[index] for value in values], [*path, str(index)], spec)
                for index in range(len(first))
            ]
        return [item for value in values for item in value]
    if isinstance(first, (bool, int, float, np.generic)):
        return torch.as_tensor(values)
    if isinstance(first, (str, bytes)):
        return list(values)
    return list(values)


class CacheCollator:
    """Merge cached, already-collated single samples into a training batch."""

    def __init__(
        self,
        spec: Optional[Mapping[str, Any]] = None,
        image_storage: Optional[Mapping[str, Any]] = None,
    ):
        self.spec = dict(spec or {})
        self.spec.setdefault("padding_side", "right")
        self.spec.setdefault("pad_token_id", 0)
        self.spec.setdefault("label_pad_token_id", -100)
        self.image_storage = copy.deepcopy(image_storage) if image_storage else None

    @classmethod
    def from_datasets(cls, datasets: Sequence[CacheDataset]) -> "CacheCollator":
        invalid_levels = sorted(
            {dataset.cache_level for dataset in datasets if dataset.cache_level != "collator"}
        )
        if invalid_levels:
            raise ValueError(
                f"CacheCollator requires collator-level caches, got {invalid_levels}"
            )
        specs = [dataset.collation_spec for dataset in datasets if dataset.collation_spec]
        if specs and any(spec != specs[0] for spec in specs[1:]):
            raise ValueError("Task cache datasets were built with incompatible collation specs")
        image_specs = [dataset.image_storage for dataset in datasets]
        if image_specs and any(spec != image_specs[0] for spec in image_specs[1:]):
            raise ValueError("Task cache datasets were built with incompatible image storage specs")
        return cls(specs[0] if specs else None, image_specs[0] if image_specs else None)

    def __call__(self, instances: Sequence[Any]) -> Any:
        if not instances:
            raise ValueError("CacheCollator received an empty batch")
        decoded = [
            _decode_image_storage(instance, self.image_storage) for instance in instances
        ]
        return _merge_cached(decoded, [], self.spec)


class TaskCacheManager:
    """Resolve, build, validate, and load all caches for a policy/task pair."""

    def __init__(
        self,
        args: Any,
        task_config: Mapping[str, Any],
        policy_config: Mapping[str, Any],
        processor: Any,
        collator: Any,
    ):
        self.args = args
        self.task_config = copy.deepcopy(dict(task_config))
        self.policy_config = copy.deepcopy(dict(policy_config))
        self.processor = processor
        self.collator = collator
        self.cache_config = get_task_cache_config(task_config, required=True)
        assert self.cache_config is not None
        self.cache_root = resolve_cache_root(task_config)
        self.cache_level = str(
            self.cache_config.get("level", DEFAULT_CACHE_LEVEL)
        ).lower()
        if self.cache_level not in SUPPORTED_CACHE_LEVELS:
            raise ValueError(
                f"cache.level must be one of {sorted(SUPPORTED_CACHE_LEVELS)}, "
                f"got {self.cache_level!r}"
            )
        self.cache_format = str(
            self.cache_config.get("format", DEFAULT_CACHE_FORMAT)
        ).lower()
        if self.cache_format not in SUPPORTED_CACHE_FORMATS:
            raise ValueError(
                f"cache.format must be one of {sorted(SUPPORTED_CACHE_FORMATS)}, "
                f"got {self.cache_format!r}"
            )
        shard_size_mb = int(self.cache_config.get("shard_size_mb", 512))
        if shard_size_mb <= 0:
            raise ValueError("cache.shard_size_mb must be positive")
        self.shard_size_bytes = shard_size_mb * 1024 * 1024
        self.max_open_shards = int(
            self.cache_config.get("max_open_shards", DEFAULT_MAX_OPEN_SHARDS)
        )
        if self.max_open_shards <= 0:
            raise ValueError("cache.max_open_shards must be positive")
        default_num_workers = getattr(args, "dataloader_num_workers", 0) or 0
        self.num_workers = int(
            self.cache_config.get("num_workers", default_num_workers)
        )
        if self.num_workers < 0:
            raise ValueError("cache.num_workers must be non-negative")
        default_prefetch_factor = getattr(args, "dataloader_prefetch_factor", 2) or 2
        self.prefetch_factor = int(
            self.cache_config.get("prefetch_factor", default_prefetch_factor)
        )
        if self.prefetch_factor <= 0:
            raise ValueError("cache.prefetch_factor must be positive")
        self.image_storage = _normalize_image_storage_config(self.cache_config)
        configured_local_dir = self.cache_config.get("local_dir") or os.environ.get(
            "ILSTD_CACHE_LOCAL"
        )
        self.local_cache_dir = (
            Path(os.path.expandvars(os.path.expanduser(str(configured_local_dir)))).resolve()
            if configured_local_dir
            else None
        )
        self.module_path = str(
            policy_config.get("module_path") or policy_config.get("type") or "unknown_policy"
        )
        self.policy_source_hash = hash_policy_module(self.module_path)
        self.configured_policy_name = self.cache_config.get("policy_name")
        if self.configured_policy_name:
            self.policy_key = _slug(str(self.configured_policy_name), "shared-policy")
        else:
            self.policy_key = (
                f"{_slug(self.module_path, 'policy')}--{self.policy_source_hash[:16]}"
            )
        self.policy_dir = self.cache_root / self.policy_key
        self.pipeline_signature = {
            "processor": _object_signature(processor)
        }
        if self.cache_level == "collator":
            self.pipeline_signature["collator"] = _object_signature(collator)
            self.collation_spec = infer_collation_spec(collator)
        else:
            self.collation_spec = {}
        self.task_pipeline = {
            key: copy.deepcopy(task_config[key])
            for key in _TASK_PIPELINE_KEYS
            if key in task_config
        }
        self.effective_pipeline_args = {
            key: copy.deepcopy(getattr(args, key))
            for key in _EFFECTIVE_PIPELINE_ARG_KEYS
            if hasattr(args, key)
        }
        self.descriptors = self._make_descriptors()
        task_identity = {
            "policy_key": self.policy_key,
            "datasets": [descriptor.dataset_hash for descriptor in self.descriptors],
        }
        self.task_hash = _stable_hash(task_identity)
        task_name = _slug(getattr(args, "task", "task"), "task")
        self.task_dir = self.policy_dir / "tasks" / f"{task_name}--{self.task_hash[:16]}"
        self.task_manifest_file = self.task_dir / "manifest.json"
        self.artifacts_dir = self.task_dir / "normalizer"
        self.lock_file = self.task_dir / ".build.lock"

    def _make_descriptors(self) -> List[CacheDescriptor]:
        configs: List[tuple[str, Dict[str, Any]]] = []
        configs.extend(("dataset", copy.deepcopy(config)) for config in self.task_config.get("datasets", []))
        configs.extend(("vqa", copy.deepcopy(config)) for config in self.task_config.get("vqa", []))
        descriptors: List[CacheDescriptor] = []
        for index, (kind, config) in enumerate(configs):
            name = _dataset_name(config, index, kind)
            config_hash = _stable_hash(config)
            cache_identity = {
                "config": config,
                "task_pipeline": self.task_pipeline,
                "effective_pipeline_args": self.effective_pipeline_args,
                "pipeline_signature": self.pipeline_signature,
                "cache_format": self.cache_format,
                "cache_shard_size_mb": self.shard_size_bytes // (1024 * 1024),
                "image_storage": self.image_storage,
            }
            # Preserve existing collator-level cache hashes. Processor-level
            # records have different semantics and must use a distinct identity.
            if self.cache_level != DEFAULT_CACHE_LEVEL:
                cache_identity["cache_level"] = self.cache_level
            dataset_hash = _stable_hash(cache_identity)
            directory = (
                self.policy_dir
                / "datasets"
                / f"{_slug(name, kind)}--{dataset_hash[:16]}"
            )
            descriptors.append(
                CacheDescriptor(
                    index=index,
                    kind=kind,
                    name=name,
                    config=config,
                    config_hash=config_hash,
                    dataset_hash=dataset_hash,
                    cache_dir=directory,
                    cache_file=directory / "cache.hdf5",
                    manifest_file=directory / "manifest.json",
                )
            )
        if not descriptors:
            raise ValueError("There is no dataset in task config")
        return descriptors

    def _common_manifest(self) -> Dict[str, Any]:
        return {
            "policy": {
                "module_path": self.module_path,
                "source_hash": self.policy_source_hash,
                "configured_name": self.configured_policy_name,
                "cache_key": self.policy_key,
            },
            "cache_level": self.cache_level,
            "pipeline_signature": self.pipeline_signature,
            "collation_spec": self.collation_spec,
            "task_pipeline": self.task_pipeline,
            "effective_pipeline_args": self.effective_pipeline_args,
            "cache_format": self.cache_format,
            "image_storage": self.image_storage,
            "build_dataloader": {
                "shuffle": False,
                "batch_size": 1,
                "num_workers": self.num_workers,
                "prefetch_factor": self.prefetch_factor if self.num_workers else None,
            },
        }

    def _task_manifest_valid(self) -> bool:
        if not self.task_manifest_file.is_file():
            return False
        try:
            with self.task_manifest_file.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            if (
                manifest.get("format_version") != CACHE_FORMAT_VERSION
                or not manifest.get("complete")
                or manifest.get("policy_key") != self.policy_key
                or manifest.get("task_hash") != self.task_hash
                or manifest.get("cache_level", DEFAULT_CACHE_LEVEL) != self.cache_level
                or [item.get("dataset_hash") for item in manifest.get("datasets", [])]
                != [descriptor.dataset_hash for descriptor in self.descriptors]
            ):
                return False
            artifacts = manifest.get("normalizer_artifacts", [])
            return all((self.artifacts_dir / relative).is_file() for relative in artifacts)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def _is_valid(self, descriptor: CacheDescriptor) -> bool:
        if not descriptor.manifest_file.is_file():
            return False
        try:
            with descriptor.manifest_file.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            if (
                manifest.get("format_version") != CACHE_FORMAT_VERSION
                or not manifest.get("complete")
                or manifest.get("dataset", {}).get("dataset_hash") != descriptor.dataset_hash
                or manifest.get("policy", {}).get("cache_key") != self.policy_key
                or manifest.get("cache_format", "hdf5") != self.cache_format
                or manifest.get("cache_level", DEFAULT_CACHE_LEVEL) != self.cache_level
            ):
                return False
            if self.cache_format == "npy":
                shards = manifest.get("shards", [])
                if not shards or sum(int(shard["sample_count"]) for shard in shards) != int(
                    manifest.get("sample_count", -1)
                ):
                    return False
                return all(
                    (descriptor.cache_dir / shard["payload"]).is_file()
                    and (descriptor.cache_dir / shard["offsets"]).is_file()
                    for shard in shards
                )
            if not descriptor.cache_file.is_file():
                return False
            with h5py.File(descriptor.cache_file, "r") as h5_file:
                return bool(h5_file.attrs.get("complete", False)) and int(
                    h5_file.attrs.get("sample_count", -1)
                ) == int(manifest.get("sample_count", -2))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def missing(self) -> List[CacheDescriptor]:
        return [descriptor for descriptor in self.descriptors if not self._is_valid(descriptor)]

    def ensure(self) -> List[CacheDataset]:
        dataset_class = (
            CacheBeforeCollatorDataset
            if self.cache_level == "processor"
            else CacheDataset
        )
        self.task_dir.mkdir(parents=True, exist_ok=True)
        with _FileLock(self.lock_file):
            missing = self.missing()
            task_manifest_invalid = not self._task_manifest_valid()
            if missing or task_manifest_invalid:
                if missing:
                    logger.info(
                        "Task cache miss: " + ", ".join(descriptor.name for descriptor in missing)
                    )
                else:
                    logger.info("Task cache normalization artifacts are missing; rebuilding metadata")
                self._build(missing)
            else:
                logger.info(f"Task cache hit: {self.task_dir}")
        return [
            dataset_class(
                descriptor.cache_file,
                descriptor.manifest_file,
                local_cache_dir=self.local_cache_dir,
                max_open_shards=self.max_open_shards,
            )
            for descriptor in self.descriptors
        ]

    def _build(self, missing: Sequence[CacheDescriptor]) -> None:
        # Lazy import avoids a data_utils.utils -> task_cache import cycle.
        from data_utils.utils import load_datasets

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        previous_output_dir = getattr(self.args, "output_dir", None)
        self.args.output_dir = str(self.artifacts_dir)
        try:
            source_datasets = load_datasets(self.args, self.task_config, save_norm=True)
        finally:
            if previous_output_dir is None:
                with contextlib.suppress(AttributeError):
                    delattr(self.args, "output_dir")
            else:
                self.args.output_dir = previous_output_dir

        if len(source_datasets) != len(self.descriptors):
            raise RuntimeError(
                "Task cache source/config count mismatch: "
                f"{len(source_datasets)} datasets for {len(self.descriptors)} configs"
            )
        missing_hashes = {descriptor.dataset_hash for descriptor in missing}
        common_manifest = self._common_manifest()
        for dataset, descriptor in zip(source_datasets, self.descriptors):
            if descriptor.dataset_hash not in missing_hashes:
                continue
            # Dataset locks are separate from task locks because different tasks
            # can legitimately resolve to the same policy+dataset cache path.
            with _FileLock(descriptor.cache_dir / ".build.lock"):
                if self._is_valid(descriptor):
                    continue
                _build_cache(
                    dataset=dataset,
                    processor=self.processor,
                    collator=self.collator,
                    descriptor=descriptor,
                    common_manifest=common_manifest,
                    cache_format=self.cache_format,
                    shard_size_bytes=self.shard_size_bytes,
                    image_storage=self.image_storage,
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    cache_level=self.cache_level,
                )

        task_manifest = {
            "format_version": CACHE_FORMAT_VERSION,
            "complete": True,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "policy_key": self.policy_key,
            "task_hash": self.task_hash,
            "cache_level": self.cache_level,
            "datasets": [
                {
                    "name": descriptor.name,
                    "dataset_hash": descriptor.dataset_hash,
                    "cache_file": str(descriptor.cache_file),
                }
                for descriptor in self.descriptors
            ],
            "normalizer_artifacts": sorted(
                source.relative_to(self.artifacts_dir).as_posix()
                for source in self.artifacts_dir.rglob("*")
                if source.is_file()
            ),
        }
        _write_json_atomic(self.task_manifest_file, task_manifest)

    def restore_normalizer_artifacts(self, output_dir: os.PathLike[str] | str) -> None:
        if not self.artifacts_dir.is_dir():
            return
        destination_root = Path(output_dir)
        destination_root.mkdir(parents=True, exist_ok=True)
        for source in self.artifacts_dir.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(self.artifacts_dir)
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

    def load_data(self, output_dir: Optional[os.PathLike[str] | str] = None) -> Dict[str, Any]:
        from data_utils.utils import prepare_data_splits

        datasets = self.ensure()
        if output_dir is not None:
            self.restore_normalizer_artifacts(output_dir)
        return prepare_data_splits(datasets, self.args, self.task_config)


def load_task_cache(
    args: Any,
    task_config: Mapping[str, Any],
    policy_config: Mapping[str, Any],
    processor: Any,
    collator: Any,
    output_dir: Optional[os.PathLike[str] | str] = None,
) -> tuple[Dict[str, Any], Any, TaskCacheManager]:
    """Build missing caches, then return cached splits and the correct collator."""
    manager = TaskCacheManager(args, task_config, policy_config, processor, collator)
    data = manager.load_data(output_dir=output_dir)
    if manager.cache_level == "processor":
        return data, collator, manager
    cache_datasets: List[CacheDataset] = []
    for split in (data.get("train"), data.get("eval")):
        if isinstance(split, CacheDataset):
            cache_datasets.append(split)
        elif isinstance(split, list):
            cache_datasets.extend(item for item in split if isinstance(item, CacheDataset))
    if not cache_datasets:
        # random_split returns Subset objects; the manager's canonical datasets still
        # carry the collation spec and are safe to use for this check.
        cache_datasets = manager.ensure()
    cache_collator = CacheCollator.from_datasets(cache_datasets)
    return data, cache_collator, manager


__all__ = [
    "CACHE_FORMAT_VERSION",
    "DEFAULT_CACHE_LEVEL",
    "DEFAULT_MAX_OPEN_SHARDS",
    "CacheBeforeCollatorDataset",
    "CacheCollator",
    "CacheDataset",
    "TaskCacheManager",
    "get_task_cache_config",
    "hash_policy_module",
    "is_task_cache_enabled",
    "load_task_cache",
    "resolve_cache_root",
]
