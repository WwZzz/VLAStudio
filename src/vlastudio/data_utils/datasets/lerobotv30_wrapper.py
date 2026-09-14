"""
LeRobot v3.0 Dataset Wrapper for ILStudio.

This is a standalone implementation that loads LeRobot v3.0 format datasets
WITHOUT depending on the lerobot library. It reads the dataset structure directly
from parquet files and video files.

LeRobot v3.0 Dataset Structure:
.
├── data
│   └── chunk-{chunk_index:03d}
│       └── file-{file_index:03d}.parquet   # MULTIPLE episodes per file
├── meta
│   ├── episodes
│   │   └── chunk-000
│   │       └── file-000.parquet            # per-episode rows w/ dataset_from_index,
│   │                                        # dataset_to_index, data/chunk_index,
│   │                                        # data/file_index, videos/.../chunk_index ...
│   ├── info.json
│   ├── stats.json                          # aggregated stats (top-level keys = features)
│   └── tasks.parquet
└── videos
    └── {video_key}
        └── chunk-{chunk_index:03d}
            └── file-{file_index:03d}.mp4   # MULTIPLE episodes per video file,
                                             # segmented by from_timestamp/to_timestamp

Key differences vs v2.x:
- data files are chunked by (chunk_index, file_index) and contain multiple episodes each
- video files are chunked similarly; per-episode offsets given by from/to_timestamp
- per-episode metadata is a parquet under meta/episodes/
- tasks are stored in meta/tasks.parquet
- aggregated stats live in meta/stats.json (top-level keys are feature names)
"""

import os
import json
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data as tud
import pyarrow.parquet as pq
from PIL import Image
from loguru import logger

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None

from vlastudio.benchmark.utils import resize_with_pad
from vlastudio.data_utils.utils import ensure_uint8_image


# =============================================================================
# Constants / paths
# =============================================================================

def _get_lerobot_home() -> Path:
    lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if lerobot_home:
        return Path(lerobot_home)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "lerobot"
    return Path.home() / ".cache" / "huggingface" / "lerobot"


DEFAULT_LEROBOT_HOME = _get_lerobot_home()
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
TASKS_PATH = "meta/tasks.parquet"
EPISODES_DIR = "meta/episodes"
CODEBASE_VERSION = "v3.0"


# =============================================================================
# Small utilities
# =============================================================================

def load_json(fpath: Path) -> Any:
    with open(fpath, 'r', encoding='utf-8') as f:
        return json.load(f)


def cast_stats_to_numpy(stats: dict) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for k, v in stats.items():
        if isinstance(v, dict):
            out[k] = {sk: np.asarray(sv) for sk, sv in v.items()}
    return out


# =============================================================================
# Video decoding (same strategy as v21 wrapper)
# =============================================================================

def get_video_backend() -> str:
    try:
        import importlib.util
        if importlib.util.find_spec("torchcodec"):
            return "torchcodec"
    except Exception:
        pass
    return "pyav"


def decode_video_frames_torchcodec(
    video_path: Path,
    timestamps: List[float],
    tolerance_s: float = 0.1,
    fps: Optional[float] = None,
) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(str(video_path), device="cpu", seek_mode="approximate")
    metadata = decoder.metadata
    video_fps = fps if fps else metadata.average_fps
    frame_indices = [round(ts * video_fps) for ts in timestamps]
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    loaded_frames, loaded_ts = [], []
    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
    if not loaded_frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    query_ts = torch.tensor(timestamps)
    loaded_ts_t = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_dist, argmin_idx = dist.min(1)
    if not (min_dist < tolerance_s).all():
        warnings.warn(f"Some frames violate tolerance: {min_dist.max().item()} > {tolerance_s}")
    frames = torch.stack([loaded_frames[int(idx)] for idx in argmin_idx])
    return frames.float() / 255.0


def decode_video_frames_torchvision(
    video_path: Path,
    timestamps: List[float],
    tolerance_s: float = 0.1,
    backend: str = "pyav",
) -> torch.Tensor:
    import torchvision
    torchvision.set_video_backend(backend)
    keyframes_only = backend == "pyav"
    reader = torchvision.io.VideoReader(str(video_path), "video")
    first_ts, last_ts = min(timestamps), max(timestamps)
    reader.seek(first_ts, keyframes_only=keyframes_only)
    loaded_frames, loaded_ts = [], []
    for frame in reader:
        current_ts = frame["pts"]
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break
    if backend == "pyav":
        reader.container.close()
    reader = None
    if not loaded_frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    query_ts = torch.tensor(timestamps)
    loaded_ts_t = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_dist, argmin_idx = dist.min(1)
    if not (min_dist < tolerance_s).all():
        warnings.warn(f"Some frames violate tolerance: {min_dist.max().item()} > {tolerance_s}")
    frames = torch.stack([loaded_frames[int(idx)] for idx in argmin_idx])
    return frames.float() / 255.0


# =============================================================================
# Metadata
# =============================================================================

class LeRobotV30Metadata:
    """Metadata handler for LeRobot v3.0 datasets."""

    def __init__(
        self,
        repo_id: str,
        root: Optional[str] = None,
        revision: str = CODEBASE_VERSION,
        force_download: bool = False,
        local_only: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision
        self.local_only = local_only
        self.root = Path(root) if root else DEFAULT_LEROBOT_HOME / repo_id
        try:
            if force_download and not local_only:
                raise FileNotFoundError
            self._load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if local_only:
                raise FileNotFoundError(
                    f"Local dataset not found at {self.root}. "
                    f"Cannot download because local_only=True."
                )
            self._download_metadata()
            self._load_metadata()

    def _download_metadata(self):
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is required to download datasets")
        logger.info(f"Downloading metadata for {self.repo_id}...")
        self.root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns="meta/*",
        )

    def _load_metadata(self):
        info_path = self.root / INFO_PATH
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found at {info_path}")
        self.info = load_json(info_path)
        for ft in self.info.get("features", {}).values():
            if "shape" in ft:
                ft["shape"] = tuple(ft["shape"])

        # --- tasks (tasks.parquet) ---
        # LeRobot v3.0 stores tasks as a parquet file. Formats observed:
        #   1) columns ['task_index', 'task']  : classic (task_index -> task text)
        #   2) column ['task_index'] only      : index (task text) -> task_index int
        tasks_path = self.root / TASKS_PATH
        self.tasks: Dict[int, str] = {}
        if tasks_path.exists():
            t = pq.read_table(tasks_path)
            df = t.to_pandas()
            if 'task' in df.columns and 'task_index' in df.columns:
                for _, row in df.iterrows():
                    self.tasks[int(row['task_index'])] = str(row['task'])
            elif 'task_index' in df.columns:
                # Index is the task text, column holds task_index integer
                for task_text, row in df.iterrows():
                    if isinstance(task_text, bytes):
                        task_text = task_text.decode('utf-8', errors='ignore')
                    self.tasks[int(row['task_index'])] = str(task_text)
            else:
                for idx, row in df.iterrows():
                    if isinstance(idx, bytes):
                        idx = idx.decode('utf-8', errors='ignore')
                    self.tasks[int(list(row.values)[0])] = str(idx)
        self.task_to_task_index = {v: k for k, v in self.tasks.items()}

        # --- episodes (meta/episodes/chunk-*/file-*.parquet) ---
        ep_dir = self.root / EPISODES_DIR
        episodes: Dict[int, Dict[str, Any]] = {}
        if ep_dir.exists():
            ep_paths = sorted(ep_dir.glob("*/file-*.parquet"))
            for p in ep_paths:
                t = pq.read_table(p)
                df = t.to_pandas()
                for _, row in df.iterrows():
                    ep_idx = int(row['episode_index'])
                    ep: Dict[str, Any] = {
                        'episode_index': ep_idx,
                        'length': int(row.get('length', 0)),
                        'tasks': list(row['tasks']) if row.get('tasks') is not None else [],
                        'data_chunk_index': int(row['data/chunk_index']),
                        'data_file_index': int(row['data/file_index']),
                        'dataset_from_index': int(row['dataset_from_index']),
                        'dataset_to_index': int(row['dataset_to_index']),
                    }
                    # Per-video-key chunk/file/from_ts/to_ts
                    videos_ep: Dict[str, Dict[str, Any]] = {}
                    for col in df.columns:
                        if col.startswith('videos/') and col.endswith('/chunk_index'):
                            vid_key = col[len('videos/'):-len('/chunk_index')]
                            try:
                                videos_ep[vid_key] = {
                                    'chunk_index': int(row[f'videos/{vid_key}/chunk_index']),
                                    'file_index': int(row[f'videos/{vid_key}/file_index']),
                                    'from_timestamp': float(row[f'videos/{vid_key}/from_timestamp']),
                                    'to_timestamp': float(row[f'videos/{vid_key}/to_timestamp']),
                                }
                            except Exception:
                                pass
                    ep['videos'] = videos_ep
                    episodes[ep_idx] = ep
        self.episodes = episodes

        # --- stats (aggregated top-level per feature) ---
        stats_path = self.root / STATS_PATH
        if stats_path.exists():
            raw = load_json(stats_path)
            self.stats = cast_stats_to_numpy(raw)
        else:
            self.stats = {}

    # ---- simple accessors ----
    @property
    def fps(self) -> int:
        return int(self.info.get("fps", 30))

    @property
    def features(self) -> Dict[str, dict]:
        return self.info.get("features", {})

    @property
    def camera_keys(self) -> List[str]:
        return [k for k, ft in self.features.items() if ft.get("dtype") in ("video", "image")]

    @property
    def video_keys(self) -> List[str]:
        return [k for k, ft in self.features.items() if ft.get("dtype") == "video"]

    @property
    def image_keys(self) -> List[str]:
        return [k for k, ft in self.features.items() if ft.get("dtype") == "image"]

    @property
    def total_episodes(self) -> int:
        return int(self.info.get("total_episodes", len(self.episodes)))

    @property
    def total_frames(self) -> int:
        return int(self.info.get("total_frames", 0))

    @property
    def data_path(self) -> str:
        return self.info.get(
            "data_path",
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        )

    @property
    def video_path(self) -> Optional[str]:
        return self.info.get("video_path", None)

    def get_data_file_path(self, chunk_index: int, file_index: int) -> Path:
        return Path(self.data_path.format(chunk_index=chunk_index, file_index=file_index))

    def get_video_file_path(self, video_key: str, chunk_index: int, file_index: int) -> Optional[Path]:
        if not self.video_path:
            return None
        return Path(
            self.video_path.format(
                video_key=video_key, chunk_index=chunk_index, file_index=file_index
            )
        )

    def get_task_index(self, task: str) -> Optional[int]:
        return self.task_to_task_index.get(task)


# =============================================================================
# Main dataset
# =============================================================================

class WrappedLerobotV30Dataset(tud.Dataset):
    """Standalone wrapper for LeRobot v3.0 datasets.

    Returns ILStudio standard format from ``__getitem__``:
        {
          'image':      torch.uint8 (K, C, H, W),
          'state':      torch.float32 (state_dim,),
          'action':     torch.float32 (chunk_size, action_dim),
          'is_pad':     torch.bool  (chunk_size,),
          'raw_lang':   str,
          'reasoning':  {},
          'timestamp':  int,
          'episode_id': int,
        }
    """

    def __init__(
        self,
        dataset_path_list: List[str],
        camera_names: List[str] = [],
        root: Optional[str] = None,
        chunk_size: int = 16,
        ctrl_space: str = 'joint',
        ctrl_type: str = 'abs',
        image_size: Optional[Tuple[int, int]] = None,
        tolerance_s: float = 0.1,
        state_key: Union[str, List[str]] = 'observation.state',
        action_key: Union[str, List[str]] = 'action',
        episode_filter: Optional[dict] = None,
        download_videos: bool = True,
        video_backend: Optional[str] = None,
        no_ensure_download: bool = False,
        local_only: bool = False,
        parquet_cache_size: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.root = Path(root) if root else DEFAULT_LEROBOT_HOME
        self.state_key = state_key
        self.action_key = action_key
        self.episode_filter = episode_filter
        self.tolerance_s = float(tolerance_s)
        self.download_videos = download_videos
        self.parquet_cache_size = int(parquet_cache_size) if parquet_cache_size is not None else 0
        self.camera_names = camera_names if isinstance(camera_names, list) else [camera_names]
        self.image_size = image_size  # (W, H) if provided
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.local_only = local_only
        self.video_backend = video_backend if video_backend else get_video_backend()

        self.dataset_path_list = dataset_path_list
        self.dataset_metas: List[LeRobotV30Metadata] = []
        self.dataset_dirs: List[str] = []
        self.per_dataset_episodes: List[List[int]] = []
        self.per_dataset_num_episodes: List[int] = []
        self.per_dataset_num_frames: List[int] = []

        for dataset_path in dataset_path_list:
            potential_local_path = Path(dataset_path)
            is_local_path = (
                potential_local_path.is_absolute() and potential_local_path.exists()
            ) or (
                (self.root / dataset_path).exists()
                and (self.root / dataset_path / INFO_PATH).exists()
            ) or (
                potential_local_path.exists()
                and (potential_local_path / INFO_PATH).exists()
            )

            if is_local_path:
                if potential_local_path.is_absolute() and potential_local_path.exists():
                    ds_root = potential_local_path
                elif (self.root / dataset_path).exists():
                    ds_root = self.root / dataset_path
                else:
                    ds_root = potential_local_path.resolve()
                repo_id = ds_root.name
                meta_local_only = True
                logger.info(f"Using local dataset at: {ds_root}")
            else:
                repo_id = dataset_path
                ds_root = self.root / repo_id
                meta_local_only = self.local_only

            meta = LeRobotV30Metadata(repo_id, root=str(ds_root), local_only=meta_local_only)

            # filter episodes
            episodes = self._filter_episodes(meta, episode_filter)
            if episodes is None:
                episodes = sorted(meta.episodes.keys())
            if len(episodes) == 0:
                warnings.warn(f"No episodes found for {repo_id} with filter {episode_filter}")
                continue

            # download data/videos if needed
            if not no_ensure_download and not meta_local_only:
                self._ensure_data_downloaded(meta, episodes)

            num_frames = sum(meta.episodes[ep].get('length', 0) for ep in episodes)

            self.dataset_metas.append(meta)
            self.dataset_dirs.append(str(ds_root))
            self.per_dataset_episodes.append(episodes)
            self.per_dataset_num_episodes.append(len(episodes))
            self.per_dataset_num_frames.append(num_frames)

        if not self.dataset_metas:
            raise ValueError("No valid datasets loaded!")

        self.cumulative_num_episodes = np.cumsum(self.per_dataset_num_episodes)
        self.cumulative_num_frames = np.cumsum(self.per_dataset_num_frames)
        self.per_dataset_episode_start = self.cumulative_num_episodes - np.array(self.per_dataset_num_episodes)
        self.per_dataset_frame_start = self.cumulative_num_frames - np.array(self.per_dataset_num_frames)
        self.total_frames = int(sum(self.per_dataset_num_frames))
        self.total_episodes = int(sum(self.per_dataset_num_episodes))
        self.episode_ids = np.arange(self.total_episodes)
        self.freq = self.dataset_metas[0].fps

        self._build_index_mapping()

        from collections import OrderedDict
        self._parquet_cache: "OrderedDict[Tuple[int, int, int], Any]" = OrderedDict()

        self.state_dim = self._compute_feature_dim(self.state_key)
        self.action_dim = self._compute_feature_dim(self.action_key)
        self._default_image_size = self._infer_image_size() if self.image_size is None else (self.image_size[1], self.image_size[0])

        logger.info(
            f"LeRobot v3.0 dataset loaded: {self.total_episodes} episodes, {self.total_frames} frames, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, image_size={self._default_image_size}"
        )

    # ---------------- filtering ----------------
    def _filter_episodes(self, meta: LeRobotV30Metadata, episode_filter: Optional[dict]) -> Optional[List[int]]:
        if not episode_filter:
            return None
        if not meta.episodes:
            warnings.warn("Dataset metadata does not contain episodes information")
            return None

        if "episode_index" in episode_filter:
            idxs = episode_filter["episode_index"]
            if isinstance(idxs, (list, tuple)):
                return [i for i in idxs if i in meta.episodes]
            return None

        selected = set(meta.episodes.keys())
        if "invalid_episode_index" in episode_filter:
            bad = set(episode_filter["invalid_episode_index"])
            selected -= bad

        for key, values in episode_filter.items():
            if key in ("invalid_episode_index",):
                continue
            vset = set(values) if isinstance(values, (list, tuple)) else {values}
            if key == "tasks":
                cur = {ep for ep, info in meta.episodes.items() if any(t in vset for t in info.get('tasks', []))}
                selected &= cur
            elif key == "task_index":
                cur = {ep for ep, info in meta.episodes.items()
                       if any(meta.get_task_index(t) in vset for t in info.get('tasks', []))}
                selected &= cur
            else:
                cur = {ep for ep, info in meta.episodes.items() if info.get(key) in vset}
                selected &= cur
        return sorted(selected)

    # ---------------- download ----------------
    def _ensure_data_downloaded(self, meta: LeRobotV30Metadata, episodes: List[int]):
        """Download only the chunks needed for the given episodes."""
        if snapshot_download is None:
            return
        data_paths = set()
        video_paths = set()
        for ep in episodes:
            info = meta.episodes[ep]
            data_paths.add((info['data_chunk_index'], info['data_file_index']))
            if self.download_videos:
                for vk, v in info.get('videos', {}).items():
                    video_paths.add((vk, v['chunk_index'], v['file_index']))

        missing_data = []
        for (ci, fi) in data_paths:
            p = meta.root / meta.get_data_file_path(ci, fi)
            if not p.exists():
                missing_data.append(f"data/chunk-{ci:03d}/file-{fi:03d}.parquet")
        missing_vids = []
        for (vk, ci, fi) in video_paths:
            vp = meta.get_video_file_path(vk, ci, fi)
            if vp is None:
                continue
            p = meta.root / vp
            if not p.exists():
                missing_vids.append(str(vp))

        patterns: List[str] = list(missing_data) + list(missing_vids)
        if not patterns:
            return
        logger.info(f"Downloading {len(patterns)} files for {meta.repo_id}...")
        snapshot_download(
            meta.repo_id,
            repo_type="dataset",
            revision=meta.revision,
            local_dir=meta.root,
            allow_patterns=patterns,
        )

    # ---------------- index mapping ----------------
    def _build_index_mapping(self):
        self.index_to_sample_map: List[Tuple[int, int, int]] = []
        for dataset_idx, (meta, episodes) in enumerate(zip(self.dataset_metas, self.per_dataset_episodes)):
            for ep_idx in episodes:
                ep_len = meta.episodes[ep_idx]['length']
                for frame_offset in range(ep_len):
                    self.index_to_sample_map.append((dataset_idx, ep_idx, frame_offset))

    # ---------------- dims ----------------
    def _compute_feature_dim(self, keys: Union[str, List[str]]) -> int:
        if isinstance(keys, str):
            keys = [keys]
        total = 0
        meta = self.dataset_metas[0]
        for k in keys:
            ft = meta.features.get(k)
            if ft is None:
                continue
            shape = ft.get('shape', ())
            if shape:
                total += shape[-1] if len(shape) > 0 else 1
            else:
                total += 1
        return total if total > 0 else 1

    def _infer_image_size(self) -> Tuple[int, int]:
        meta = self.dataset_metas[0]
        for k in meta.camera_keys:
            shape = meta.features[k].get('shape', ())
            if len(shape) >= 2:
                # v3 stores video shape as (H, W, C)
                if shape[-1] in (1, 3, 4):
                    return (shape[0], shape[1])
                else:
                    return (shape[1], shape[2])
        return (224, 224)

    # ---------------- IO ----------------
    def _load_parquet_file(self, dataset_idx: int, chunk_index: int, file_index: int) -> Dict[str, np.ndarray]:
        cache_key = (dataset_idx, chunk_index, file_index)
        if self.parquet_cache_size > 0:
            cached = self._parquet_cache.get(cache_key)
            if cached is not None:
                self._parquet_cache.move_to_end(cache_key)
                return cached
        meta = self.dataset_metas[dataset_idx]
        path = meta.root / meta.get_data_file_path(chunk_index, file_index)
        table = pq.read_table(path)
        data: Dict[str, np.ndarray] = {}
        for col in table.column_names:
            arr = table[col].to_numpy()
            if arr.dtype == object:
                try:
                    arr = np.stack(arr)
                except Exception:
                    pass
            data[col] = arr
        if self.parquet_cache_size > 0:
            self._parquet_cache[cache_key] = data
            self._parquet_cache.move_to_end(cache_key)
            while len(self._parquet_cache) > self.parquet_cache_size:
                self._parquet_cache.popitem(last=False)
        return data

    def _get_data_by_keys(
        self,
        parquet_data: Dict[str, np.ndarray],
        keys: Union[str, List[str]],
        frame_abs_idx: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if isinstance(keys, str):
            keys = [keys]
        parts = []
        for k in keys:
            arr = parquet_data.get(k)
            if arr is None:
                logger.warning(f"Key '{k}' not found in parquet columns")
                continue
            if frame_abs_idx is not None:
                arr = arr[frame_abs_idx]
            arr = np.asarray(arr)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            parts.append(arr)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=-1)

    def _load_video_frame(
        self,
        meta: LeRobotV30Metadata,
        ep_idx: int,
        vid_key: str,
        frame_offset: int,
    ) -> Optional[torch.Tensor]:
        ep = meta.episodes[ep_idx]
        vmeta = ep.get('videos', {}).get(vid_key)
        if vmeta is None:
            return None
        vpath_rel = meta.get_video_file_path(vid_key, vmeta['chunk_index'], vmeta['file_index'])
        if vpath_rel is None:
            return None
        video_path = meta.root / vpath_rel
        if not video_path.exists():
            return None
        # target timestamp in the combined video = from_timestamp + frame_offset / fps
        ts = float(vmeta['from_timestamp']) + frame_offset / meta.fps
        try:
            if self.video_backend == "torchcodec":
                frames = decode_video_frames_torchcodec(
                    video_path, [ts], self.tolerance_s, fps=meta.fps
                )
            else:
                frames = decode_video_frames_torchvision(
                    video_path, [ts], self.tolerance_s, backend="pyav"
                )
            return frames[0]
        except Exception as e:
            if not hasattr(self, '_video_warned'):
                logger.warning(f"Failed decoding video {video_path}: {e}")
                self._video_warned = True
            return None

    def _load_image_from_parquet(
        self,
        parquet_data: Dict[str, np.ndarray],
        cam_key: str,
        frame_abs_idx: int,
    ) -> Optional[torch.Tensor]:
        import cv2
        img_data = parquet_data[cam_key][frame_abs_idx]
        if img_data is None:
            return None
        if isinstance(img_data, dict):
            if img_data.get('bytes') is not None:
                arr = np.frombuffer(img_data['bytes'], np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    return None
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return None
        if isinstance(img_data, (bytes, bytearray)):
            arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if isinstance(img_data, np.ndarray):
            if img_data.ndim == 3:
                if img_data.shape[0] in (1, 3, 4):
                    t = torch.from_numpy(np.ascontiguousarray(img_data)).float()
                else:
                    t = torch.from_numpy(np.ascontiguousarray(img_data)).permute(2, 0, 1).float()
                if t.max() > 1.0:
                    t = t / 255.0
                return t
        return None

    # ---------------- public API ----------------
    def __len__(self):
        return len(self.index_to_sample_map)

    @property
    def num_episodes(self):
        return self.total_episodes

    def get_dataset_dir(self):
        return self.dataset_dirs[0]

    def get_freq(self):
        return self.freq

    def get_episode_len(self) -> List[int]:
        lengths = []
        for meta, eps in zip(self.dataset_metas, self.per_dataset_episodes):
            for ep in eps:
                lengths.append(meta.episodes[ep].get('length', 0))
        return lengths

    def __getitem__(self, index: int) -> Dict[str, Any]:
        dataset_idx, ep_idx, frame_offset = self.index_to_sample_map[index]
        meta = self.dataset_metas[dataset_idx]
        ep = meta.episodes[ep_idx]
        pdata = self._load_parquet_file(dataset_idx, ep['data_chunk_index'], ep['data_file_index'])
        frame_abs_idx = int(ep['dataset_from_index']) + frame_offset

        # state (current frame)
        state_arr = self._get_data_by_keys(pdata, self.state_key, frame_abs_idx=frame_abs_idx)
        if state_arr is None:
            state = torch.zeros(self.state_dim, dtype=torch.float32)
        else:
            state = torch.tensor(state_arr, dtype=torch.float32)

        # action chunk
        ep_len = int(ep['length'])
        action_full = self._get_data_by_keys(pdata, self.action_key, frame_abs_idx=None)
        if action_full is None:
            actions = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
            is_pad = np.ones((self.chunk_size,), dtype=bool)
        else:
            # Slice to this episode's rows
            ep_actions = np.asarray(action_full)[ep['dataset_from_index']:ep['dataset_to_index']]
            end = min(frame_offset + self.chunk_size, ep_len)
            valid_count = max(0, end - frame_offset)
            valid = ep_actions[frame_offset:end]
            if valid_count < self.chunk_size:
                if valid_count > 0:
                    last = valid[-1:]
                else:
                    last = ep_actions[-1:]
                pad_n = self.chunk_size - valid_count
                padding = np.repeat(last, pad_n, axis=0)
                actions = np.concatenate([valid, padding], axis=0) if valid_count > 0 else padding
                is_pad = np.array([False] * valid_count + [True] * pad_n)
            else:
                actions = valid
                is_pad = np.array([False] * self.chunk_size)
        action = torch.tensor(np.asarray(actions, dtype=np.float32), dtype=torch.float32)
        is_pad = torch.tensor(is_pad, dtype=torch.bool)

        # task / language
        task_idx = int(pdata['task_index'][frame_abs_idx]) if 'task_index' in pdata else 0
        raw_lang = meta.tasks.get(task_idx, "")
        if not raw_lang and ep.get('tasks'):
            raw_lang = ep['tasks'][0]

        # images
        cam_keys_all = meta.camera_keys
        cam_keys = cam_keys_all if not self.camera_names else [c for c in self.camera_names if c in cam_keys_all]
        images = []
        for cam_key in cam_keys:
            frame = None
            if cam_key in meta.video_keys:
                frame = self._load_video_frame(meta, ep_idx, cam_key, frame_offset)
            if frame is None and cam_key in pdata:
                try:
                    frame = self._load_image_from_parquet(pdata, cam_key, frame_abs_idx)
                except Exception as e:
                    logger.warning(f"Failed parquet image for {cam_key}: {e}")
            if frame is None:
                h, w = self._default_image_size
                frame = torch.zeros(3, h, w, dtype=torch.float32)
            images.append(frame)

        if images:
            target_h, target_w = self._default_image_size
            needs_resize = any(img.shape[1] != target_h or img.shape[2] != target_w for img in images)
            if needs_resize:
                images = torch.cat(
                    [resize_with_pad(img.unsqueeze(0), height=target_h, width=target_w) for img in images],
                    dim=0,
                )
            else:
                images = torch.stack(images)
            images = (images * 255).clamp(0, 255).to(torch.uint8)
            images = ensure_uint8_image(images)
        else:
            images = None

        episode_id = int(self.per_dataset_episode_start[dataset_idx]) + \
            self.per_dataset_episodes[dataset_idx].index(ep_idx)

        return {
            'image': images,
            'state': state,
            'action': action,
            'is_pad': is_pad,
            'raw_lang': raw_lang,
            'reasoning': {},
            'timestamp': frame_offset,
            'episode_id': episode_id,
        }

    # ---------------- stats ----------------
    def _stats_for_key(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        for meta in self.dataset_metas:
            if meta.stats and key in meta.stats:
                return {k: np.asarray(v) for k, v in meta.stats[key].items()}
        return None

    def _stats_for_keys(self, keys: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        if isinstance(keys, str):
            keys = [keys]
        stats_list = []
        missing = []
        for k in keys:
            s = self._stats_for_key(k)
            if s is None:
                missing.append(k)
            else:
                stats_list.append(s)
        if missing:
            raise KeyError(f"Stats not found for keys: {missing}")
        if len(stats_list) == 1:
            return stats_list[0]
        stat_names = ['mean', 'std', 'min', 'max', 'q01', 'q99']
        out: Dict[str, np.ndarray] = {}
        for name in stat_names:
            parts = [np.asarray(s[name]).reshape(-1) for s in stats_list if name in s]
            if parts:
                out[name] = np.concatenate(parts, axis=-1)
        return out

    def get_dataset_statistics(self) -> Dict[str, Any]:
        state_stats = self._stats_for_keys(self.state_key)
        action_stats = self._stats_for_keys(self.action_key)

        def _flatten(v):
            a = np.asarray(v).astype(np.float32)
            return a.reshape(-1)

        # stats.json stores values with shape matching feature shape; ensure 1D
        state_stats = {k: _flatten(v) for k, v in state_stats.items()}
        action_stats = {k: _flatten(v) for k, v in action_stats.items()}

        if 'q01' not in state_stats:
            state_stats['q01'] = state_stats.get('min', np.zeros(self.state_dim))
            state_stats['q99'] = state_stats.get('max', np.ones(self.state_dim))
        if 'q01' not in action_stats:
            action_stats['q01'] = action_stats.get('min', np.zeros(self.action_dim))
            action_stats['q99'] = action_stats.get('max', np.ones(self.action_dim))

        return {
            'state': state_stats,
            'action': action_stats,
            'num_episodes': self.total_episodes,
            'num_transitions': self.total_frames,
        }

    def extract_from_episode(self, episode_idx: int, keyname: List[str] = []) -> Dict[str, np.ndarray]:
        dataset_idx = int(np.searchsorted(self.cumulative_num_episodes, episode_idx + 1))
        local_i = episode_idx - int(self.per_dataset_episode_start[dataset_idx])
        ep_idx = self.per_dataset_episodes[dataset_idx][local_i]
        meta = self.dataset_metas[dataset_idx]
        ep = meta.episodes[ep_idx]
        pdata = self._load_parquet_file(dataset_idx, ep['data_chunk_index'], ep['data_file_index'])
        frm = int(ep['dataset_from_index'])
        to = int(ep['dataset_to_index'])
        out: Dict[str, np.ndarray] = {}
        if 'state' in keyname:
            a = self._get_data_by_keys(pdata, self.state_key, frame_abs_idx=None)
            if a is not None:
                out['state'] = np.asarray(a)[frm:to]
        if 'action' in keyname:
            a = self._get_data_by_keys(pdata, self.action_key, frame_abs_idx=None)
            if a is not None:
                out['action'] = np.asarray(a)[frm:to]
        return out


if __name__ == '__main__':
    ds = WrappedLerobotV30Dataset(
        dataset_path_list=[os.environ.get("DS_PATH", "jellyho/aloha_dish_drainer")],
        state_key='observation.state.joint_pos',
        action_key='action.joint_pos',
        chunk_size=16,
        tolerance_s=0.5,
        image_size=(320, 240),
        local_only=True,
    )
    print('ep:', ds.total_episodes, 'frames:', ds.total_frames)
    s = ds[0]
    print({k: (type(v).__name__, getattr(v, 'shape', None)) for k, v in s.items()})
    print('raw_lang:', s['raw_lang'])
    stats = ds.get_dataset_statistics()
    print('state stats:', {k: v.shape for k, v in stats['state'].items()})
    print('action stats:', {k: v.shape for k, v in stats['action'].items()})
