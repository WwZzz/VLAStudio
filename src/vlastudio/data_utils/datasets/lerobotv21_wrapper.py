"""
LeRobot v2.1 Dataset Wrapper for ILStudio.

This is a standalone implementation that loads LeRobot v2.1 format datasets
WITHOUT depending on the lerobot library. It reads the dataset structure directly
from parquet files and video files.

LeRobot v2.1 Dataset Structure:
.
├── data
│   ├── chunk-000
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   └── ...
├── meta
│   ├── episodes.jsonl
│   ├── info.json
│   ├── stats.json (or episodes_stats.jsonl for v2.1+)
│   └── tasks.jsonl
└── videos
    ├── chunk-000
    │   ├── observation.images.laptop
    │   │   ├── episode_000000.mp4
    │   │   └── ...
    │   └── ...
    └── ...
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
from huggingface_hub import snapshot_download
from loguru import logger

from vlastudio.benchmark.utils import resize_with_pad
from vlastudio.data_utils.utils import ensure_uint8_image


# =============================================================================
# Constants
# =============================================================================

def _get_lerobot_home() -> Path:
    """Get LeRobot home directory with priority:
    1. HF_LEROBOT_HOME environment variable
    2. HF_HOME/lerobot (if HF_HOME is set)
    3. Default: ~/.cache/huggingface/lerobot
    """
    # Priority 1: HF_LEROBOT_HOME
    lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if lerobot_home:
        return Path(lerobot_home)
    
    # Priority 2: HF_HOME/lerobot
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "lerobot"
    
    # Priority 3: Default
    return Path.home() / ".cache" / "huggingface" / "lerobot"

DEFAULT_LEROBOT_HOME = _get_lerobot_home()
INFO_PATH = "meta/info.json"
EPISODES_PATH = "meta/episodes.jsonl"
STATS_PATH = "meta/stats.json"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
TASKS_PATH = "meta/tasks.jsonl"
CODEBASE_VERSION = "v2.1"


# =============================================================================
# Utility Functions
# =============================================================================

def load_json(fpath: Path) -> Any:
    """Load a JSON file."""
    with open(fpath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonlines(fpath: Path) -> List[Any]:
    """Load a JSONL file (one JSON object per line)."""
    items = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    """Unflatten a dictionary with separator keys."""
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = outdict
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return outdict


def cast_stats_to_numpy(stats: dict) -> Dict[str, Dict[str, np.ndarray]]:
    """Convert stats values to numpy arrays."""
    flat_stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(flat_stats)


def aggregate_stats_for_keys(
    episodes_stats: Dict[int, dict],
    keys: List[str],
    episode_indices: Optional[List[int]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Aggregate statistics from multiple episodes for specific keys only.
    
    This is more efficient than aggregating all keys when only a subset is needed.
    
    Args:
        episodes_stats: Dictionary mapping episode_index to stats dict
        keys: List of keys to aggregate (e.g., ['observation.state', 'action'])
        episode_indices: Optional list of episode indices to include. If None, use all.
        
    Returns:
        Dictionary mapping key to aggregated stats
    """
    if not episodes_stats:
        return {}
    
    # Determine which episodes to use
    if episode_indices is not None:
        ep_indices = [idx for idx in episode_indices if idx in episodes_stats]
    else:
        ep_indices = list(episodes_stats.keys())
    
    if not ep_indices:
        return {}
    
    result = {}
    stat_types = ['mean', 'std', 'min', 'max', 'q01', 'q99']
    
    for key in keys:
        # Collect all stat values in a single pass over episodes
        stat_values = {st: [] for st in stat_types}
        
        for ep_idx in tqdm(ep_indices, desc=f"Aggregating stats for key: {key}", leave=False):
            ep_stats = episodes_stats.get(ep_idx, {})
            key_ep_stats = ep_stats.get(key, {})
            for stat_type in stat_types:
                if stat_type in key_ep_stats:
                    stat_values[stat_type].append(np.asarray(key_ep_stats[stat_type]))
        
        # Aggregate collected values
        key_stats = {}
        for stat_type, values in stat_values.items():
            if values:
                stacked = np.stack(values)
                # Aggregate based on stat type
                if stat_type == 'mean':
                    key_stats[stat_type] = np.mean(stacked, axis=0)
                elif stat_type == 'std':
                    # For std, use RMS of stds (approximate)
                    key_stats[stat_type] = np.sqrt(np.mean(stacked**2, axis=0))
                elif stat_type in ['min', 'q01']:
                    key_stats[stat_type] = np.min(stacked, axis=0)
                elif stat_type in ['max', 'q99']:
                    key_stats[stat_type] = np.max(stacked, axis=0)
        
        if key_stats:
            result[key] = key_stats
    
    return result


# =============================================================================
# Video Decoding - Multiple backends supported
# =============================================================================

def get_video_backend() -> str:
    """Get the best available video backend."""
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
    """
    Decode video frames using torchcodec (faster than pyav).
    
    Args:
        video_path: Path to video file
        timestamps: List of timestamps (in seconds) to extract
        tolerance_s: Tolerance for timestamp matching
        fps: Video FPS (optional, will be read from metadata if not provided)
        
    Returns:
        Tensor of shape (N, C, H, W) with frames in [0, 1] range
    """
    from torchcodec.decoders import VideoDecoder
    
    # Initialize decoder
    decoder = VideoDecoder(str(video_path), device="cpu", seek_mode="approximate")
    metadata = decoder.metadata
    video_fps = fps if fps else metadata.average_fps
    
    # Convert timestamps to frame indices
    frame_indices = [round(ts * video_fps) for ts in timestamps]
    
    # Batch retrieve frames (much faster than one-by-one)
    frames_batch = decoder.get_frames_at(indices=frame_indices)
    
    loaded_frames = []
    loaded_ts = []
    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
    
    if not loaded_frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    
    # Match timestamps
    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)
    
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_dist, argmin_idx = dist.min(1)
    
    is_within_tol = min_dist < tolerance_s
    if not is_within_tol.all():
        warnings.warn(f"Some frames violate tolerance: {min_dist[~is_within_tol]} > {tolerance_s}")
    
    frames = torch.stack([loaded_frames[idx] for idx in argmin_idx])
    frames = frames.float() / 255.0
    
    return frames


def decode_video_frames_torchvision(
    video_path: Path,
    timestamps: List[float],
    tolerance_s: float = 0.1,
    backend: str = "pyav",
) -> torch.Tensor:
    """
    Decode video frames at specified timestamps using torchvision VideoReader.
    
    This is the same method used by the original LeRobot implementation.
    
    Args:
        video_path: Path to video file
        timestamps: List of timestamps (in seconds) to extract
        tolerance_s: Tolerance for timestamp matching
        backend: Video backend ("pyav" or "video_reader")
        
    Returns:
        Tensor of shape (N, C, H, W) with frames in [0, 1] range
    """
    import torchvision
    
    video_path = str(video_path)
    
    # Set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesn't support accurate seek
    
    # Create video reader
    reader = torchvision.io.VideoReader(video_path, "video")
    
    # Get first and last timestamps
    first_ts = min(timestamps)
    last_ts = max(timestamps)
    
    # Seek to closest key frame before first timestamp
    reader.seek(first_ts, keyframes_only=keyframes_only)
    
    # Load frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break
    
    # Close reader
    if backend == "pyav":
        reader.container.close()
    reader = None
    
    if not loaded_frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    
    # Match requested timestamps to loaded frames
    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)
    
    # Compute distances between each query timestamp and loaded timestamps
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_dist, argmin_idx = dist.min(1)
    
    # Check tolerance
    is_within_tol = min_dist < tolerance_s
    if not is_within_tol.all():
        warnings.warn(
            f"Some frames violate tolerance: {min_dist[~is_within_tol]} > {tolerance_s}"
        )
    
    # Select frames
    frames = torch.stack([loaded_frames[idx] for idx in argmin_idx])
    
    # Convert to float [0, 1]
    frames = frames.float() / 255.0
    
    return frames


# =============================================================================
# Dataset Metadata
# =============================================================================

class LeRobotV21Metadata:
    """
    Metadata handler for LeRobot v2.1 datasets.
    
    Handles loading and caching of dataset metadata without depending on lerobot.
    """
    
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
        
        # Try to load metadata, download if needed (unless local_only)
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
        """Download metadata files from HuggingFace Hub."""
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
        """Load metadata from local files."""
        # Load info.json
        info_path = self.root / INFO_PATH
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found at {info_path}")
        
        self.info = load_json(info_path)
        
        # Convert shape tuples
        for ft in self.info.get("features", {}).values():
            if "shape" in ft:
                ft["shape"] = tuple(ft["shape"])
        
        # Load tasks
        tasks_path = self.root / TASKS_PATH
        if tasks_path.exists():
            tasks_list = load_jsonlines(tasks_path)
            self.tasks = {item["task_index"]: item["task"] for item in tasks_list}
            self.task_to_task_index = {task: idx for idx, task in self.tasks.items()}
        else:
            self.tasks = {}
            self.task_to_task_index = {}
        
        # Load episodes
        episodes_path = self.root / EPISODES_PATH
        if episodes_path.exists():
            episodes_list = load_jsonlines(episodes_path)
            self.episodes = {item["episode_index"]: item for item in episodes_list}
        else:
            self.episodes = {}
        
        # Load stats (v2.1 uses episodes_stats.jsonl, older versions use stats.json)
        # Note: Aggregated stats are computed lazily in the Dataset class
        # to only include keys that are actually used (state_key, action_key)
        episodes_stats_path = self.root / EPISODES_STATS_PATH
        stats_path = self.root / STATS_PATH
        
        if episodes_stats_path.exists():
            episodes_stats_list = load_jsonlines(episodes_stats_path)
            self.episodes_stats = {
                item["episode_index"]: cast_stats_to_numpy(item["stats"])
                for item in tqdm(episodes_stats_list)
            }
            # Don't aggregate here - will be done lazily with only needed keys
            self.stats = None
        elif stats_path.exists():
            self.stats = cast_stats_to_numpy(load_json(stats_path))
            self.episodes_stats = None  # Use stats directly
        else:
            self.stats = {}
            self.episodes_stats = None
    
    @property
    def fps(self) -> int:
        return self.info.get("fps", 30)
    
    @property
    def features(self) -> Dict[str, dict]:
        return self.info.get("features", {})
    
    @property
    def camera_keys(self) -> List[str]:
        """Get keys for visual modalities (video or image)."""
        return [
            key for key, ft in self.features.items()
            if ft.get("dtype") in ["video", "image"]
        ]
    
    @property
    def video_keys(self) -> List[str]:
        """Get keys for video modalities."""
        return [key for key, ft in self.features.items() if ft.get("dtype") == "video"]
    
    @property
    def total_episodes(self) -> int:
        return self.info.get("total_episodes", len(self.episodes))
    
    @property
    def total_frames(self) -> int:
        return self.info.get("total_frames", 0)
    
    @property
    def chunks_size(self) -> int:
        return self.info.get("chunks_size", 1000)
    
    @property
    def data_path(self) -> str:
        """Formattable string for parquet files."""
        return self.info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
    
    @property
    def video_path(self) -> str:
        """Formattable string for video files."""
        return self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
    
    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size
    
    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)
    
    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(
            episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index
        )
        return Path(fpath)
    
    def get_task_index(self, task: str) -> Optional[int]:
        return self.task_to_task_index.get(task, None)


# =============================================================================
# Main Dataset Class
# =============================================================================

class WrappedLerobotV21Dataset(tud.Dataset):
    """
    Standalone wrapper for LeRobot v2.1 datasets.
    
    This implementation does NOT depend on the lerobot library. It reads
    parquet files and video files directly.
    
    Args:
        dataset_path_list: List of dataset repo IDs (e.g., ["lerobot/aloha_sim_transfer_cube_scripted"])
        camera_names: List of camera keys to use (empty = use all cameras)
        root: Local directory for caching datasets
        chunk_size: Action chunk size for delta_timestamps
        ctrl_space: Control space type ('ee', 'joint', 'other')
        ctrl_type: Control type ('abs', 'rel', 'delta')
        image_size: Target image size (width, height) or None to keep original
        tolerance_s: Tolerance for timestamp synchronization
        state_key: Key for state observations in the dataset
        action_key: Key for actions in the dataset
        episode_filter: Filter episodes by metadata (e.g., {"episode_index": [0,1,2]})
        download_videos: Whether to download video files
        filter_invalid_videos: Whether to filter out episodes with missing/corrupted video files
        parquet_cache_size: Max number of episodes to keep in memory cache (0 disables caching)
    """
    
    def __init__(
        self,
        dataset_path_list: List[str],
        camera_names: List[str] = [],
        root: Optional[str] = None,
        chunk_size: int = 16,
        ctrl_space: str = 'ee',
        ctrl_type: str = 'delta',
        image_size: Optional[Tuple[int, int]] = None,
        tolerance_s: float = 0.1,
        state_key: Union[str, List[str]] = 'observation.state',
        action_key: Union[str, List[str]] = 'action',
        episode_filter: Optional[dict] = None,
        download_videos: bool = True,
        filter_invalid_videos: bool = False,
        video_backend: Optional[str] = None,
        no_ensure_download: bool = False,
        local_only: bool = False,
        parquet_cache_size: int = 16,
        *args,
        **kwargs,
    ):
        super().__init__()
        
        self.chunk_size = chunk_size
        self.root = Path(root) if root else DEFAULT_LEROBOT_HOME
        self.state_key = state_key
        self.action_key = action_key
        self.episode_filter = episode_filter
        self.tolerance_s = tolerance_s
        self.download_videos = download_videos
        self.filter_invalid_videos = filter_invalid_videos
        self.parquet_cache_size = int(parquet_cache_size) if parquet_cache_size is not None else 0
        self.camera_names = camera_names if isinstance(camera_names, list) else [camera_names]
        self.image_size = image_size
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.local_only = local_only
        # Auto-detect best video backend if not specified
        self.video_backend = video_backend if video_backend else get_video_backend()
        logger.info(f"Using video backend: {self.video_backend}")
        
        # Load datasets
        self.dataset_path_list = dataset_path_list
        self.dataset_metas: List[LeRobotV21Metadata] = []
        self.dataset_dirs: List[str] = []
        self.per_dataset_episodes: List[List[int]] = []
        self.per_dataset_num_episodes: List[int] = []
        self.per_dataset_num_frames: List[int] = []
        
        for dataset_path in dataset_path_list:
            # Check if it's a local path (absolute or relative)
            potential_local_path = Path(dataset_path)
            is_local_path = (
                potential_local_path.is_absolute() and potential_local_path.exists()
            ) or (
                (self.root / dataset_path).exists() and 
                (self.root / dataset_path / INFO_PATH).exists()
            ) or (
                potential_local_path.exists() and 
                (potential_local_path / INFO_PATH).exists()
            )
            
            if is_local_path:
                # Use local path directly
                if potential_local_path.is_absolute() and potential_local_path.exists():
                    ds_root = potential_local_path
                elif (self.root / dataset_path).exists():
                    ds_root = self.root / dataset_path
                else:
                    ds_root = potential_local_path.resolve()
                repo_id = ds_root.name  # Use directory name as repo_id
                meta_local_only = True
                logger.info(f"Using local dataset at: {ds_root}")
            else:
                # Treat as HuggingFace repo_id
                repo_id = dataset_path
                ds_root = self.root / repo_id
                meta_local_only = self.local_only
            
            # Load metadata
            meta = LeRobotV21Metadata(
                repo_id, 
                root=str(ds_root),
                local_only=meta_local_only
            )
            
            # Filter episodes
            episodes = self._filter_episodes(meta, episode_filter)
            if episodes is None:
                episodes = list(range(meta.total_episodes))
            
            if len(episodes) == 0:
                warnings.warn(f"No episodes found for {repo_id} with filter {episode_filter}")
                continue
            
            # Download data files if needed (skip if local_only or already local)
            if not no_ensure_download and not meta_local_only:
                self._ensure_data_downloaded(meta, episodes)
            
            # Filter out episodes with missing/corrupted videos (only if enabled)
            if self.filter_invalid_videos:
                episodes = self._filter_episodes_with_missing_videos(meta, episodes)
                
                if len(episodes) == 0:
                    warnings.warn(f"No episodes with complete video data found for {repo_id}")
                    continue
            
            # Calculate frame count
            num_frames = sum(meta.episodes[ep_idx].get('length', 0) for ep_idx in episodes)
            
            self.dataset_metas.append(meta)
            self.dataset_dirs.append(str(ds_root))
            self.per_dataset_episodes.append(episodes)
            self.per_dataset_num_episodes.append(len(episodes))
            self.per_dataset_num_frames.append(num_frames)
        
        if not self.dataset_metas:
            raise ValueError("No valid datasets loaded!")
        
        # Compute cumulative indices
        self.cumulative_num_episodes = np.cumsum(self.per_dataset_num_episodes)
        self.cumulative_num_frames = np.cumsum(self.per_dataset_num_frames)
        self.per_dataset_episode_start = self.cumulative_num_episodes - np.array(self.per_dataset_num_episodes)
        self.per_dataset_frame_start = self.cumulative_num_frames - np.array(self.per_dataset_num_frames)
        
        self.total_frames = sum(self.per_dataset_num_frames)
        self.total_episodes = sum(self.per_dataset_num_episodes)
        self.episode_ids = np.arange(self.total_episodes)
        self.freq = self.dataset_metas[0].fps
        
        # Build index mapping for fast lookup
        self._build_index_mapping()
        # Cache parquet data for recent episodes (LRU, bounded to avoid unbounded RAM growth)
        from collections import OrderedDict
        self._parquet_cache: "OrderedDict[Tuple[int, int], Any]" = OrderedDict()
        # Cache for aggregated stats (computed lazily)
        self._aggregated_stats: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Compute state and action dimensions from dataset features
        self.state_dim = self._compute_feature_dim(self.state_key)
        self.action_dim = self._compute_feature_dim(self.action_key)
        
        # Determine default image size from dataset features or use specified size
        # This ensures all images have consistent size across samples
        if self.image_size is None:
            self._default_image_size = self._infer_image_size_from_features()
        else:
            self._default_image_size = (self.image_size[1], self.image_size[0])  # (H, W)
        
        logger.info(
            f"LeRobot v2.1 Dataset loaded: {self.total_episodes} episodes, "
            f"{self.total_frames} frames from {len(self.dataset_metas)} dataset(s), "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"image_size={self._default_image_size}"
        )
    
    def _filter_episodes(
        self,
        meta: LeRobotV21Metadata,
        episode_filter: Optional[dict],
    ) -> Optional[List[int]]:
        """Filter episodes based on metadata fields."""
        if episode_filter is None or len(episode_filter) == 0:
            return None
        
        if not meta.episodes:
            warnings.warn("Dataset metadata does not contain episodes information")
            return None
        
        # Direct episode indices (include only these)
        if "episode_index" in episode_filter:
            indices = episode_filter["episode_index"]
            if isinstance(indices, (list, tuple)):
                valid = [idx for idx in indices if idx in meta.episodes]
                return valid
            return None
        
        # Start with all episodes
        selected = set(meta.episodes.keys())
        
        # Handle invalid_episode_index (exclude these episodes)
        if "invalid_episode_index" in episode_filter:
            invalid_indices = episode_filter["invalid_episode_index"]
            if isinstance(invalid_indices, (list, tuple)):
                invalid_set = set(invalid_indices)
                selected -= invalid_set
                logger.info(f"Excluded {len(invalid_set)} invalid episodes from {meta.repo_id}")
        
        for filter_key, filter_values in episode_filter.items():
            if filter_key in ("invalid_episode_index",):
                # Already handled above
                continue
            elif filter_key == "tasks":
                # Filter by task names
                if not meta.tasks:
                    warnings.warn("Dataset does not have tasks metadata")
                    return None
                
                task_set = set(filter_values) if isinstance(filter_values, (list, tuple)) else {filter_values}
                episodes_for_tasks = set()
                
                for ep_idx, ep_info in meta.episodes.items():
                    ep_tasks = ep_info.get('tasks', [])
                    if isinstance(ep_tasks, str):
                        ep_tasks = [ep_tasks]
                    if any(t in task_set for t in ep_tasks):
                        episodes_for_tasks.add(ep_idx)
                
                selected &= episodes_for_tasks
            
            elif filter_key == "task_index":
                # Filter by task indices
                if not meta.tasks:
                    warnings.warn("Dataset does not have tasks metadata")
                    return None
                
                task_idx_set = set(filter_values) if isinstance(filter_values, (list, tuple)) else {filter_values}
                episodes_for_task_idx = set()
                
                for ep_idx, ep_info in meta.episodes.items():
                    ep_tasks = ep_info.get('tasks', [])
                    if isinstance(ep_tasks, str):
                        ep_tasks = [ep_tasks]
                    
                    for task in ep_tasks:
                        if meta.get_task_index(task) in task_idx_set:
                            episodes_for_task_idx.add(ep_idx)
                            break
                
                selected &= episodes_for_task_idx
            
            else:
                # Generic filter
                filter_set = set(filter_values) if isinstance(filter_values, (list, tuple)) else {filter_values}
                episodes_for_field = set()
                
                for ep_idx, ep_info in meta.episodes.items():
                    ep_value = ep_info.get(filter_key)
                    if ep_value is not None:
                        if isinstance(ep_value, (list, tuple)):
                            if any(v in filter_set for v in ep_value):
                                episodes_for_field.add(ep_idx)
                        elif ep_value in filter_set:
                            episodes_for_field.add(ep_idx)
                
                selected &= episodes_for_field
        
        return sorted(list(selected))
    
    def _is_video_valid(self, video_path: Path) -> bool:
        """
        Check if a video file exists and can be opened/decoded.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video is valid, False otherwise
        """
        if not video_path.exists():
            return False
        
        try:
            # Check file size
            if video_path.stat().st_size < 1000:  # Less than 1KB is likely corrupted
                return False
            
            # Try to open and read first frame to verify video is valid
            import torchvision
            torchvision.set_video_backend("pyav")
            reader = torchvision.io.VideoReader(str(video_path), "video")
            
            # Try to get first frame
            for frame in reader:
                # Successfully read a frame, video is valid
                reader.container.close()
                return True
            
            # No frames in video
            reader.container.close()
            return False
            
        except Exception:
            return False
    
    def _filter_episodes_with_missing_videos(
        self,
        meta: LeRobotV21Metadata,
        episodes: List[int],
    ) -> List[int]:
        """
        Filter out episodes that have missing or corrupted video files for required cameras.
        
        Only checks cameras specified in self.camera_names (if not empty),
        otherwise checks all video keys in the dataset.
        
        Args:
            meta: Dataset metadata
            episodes: List of episode indices to check
            
        Returns:
            List of episode indices with complete video data
        """
        if not self.download_videos:
            # If not downloading videos, don't filter by video availability
            return episodes
        
        # Determine which cameras to check
        if len(self.camera_names) > 0:
            # Only check specified cameras that are video keys
            cameras_to_check = [cam for cam in self.camera_names if cam in meta.video_keys]
        else:
            # Check all video keys
            cameras_to_check = meta.video_keys
        
        if not cameras_to_check:
            # No video cameras to check
            return episodes
        
        valid_episodes = []
        invalid_episodes = []
        
        logger.info(f"Validating video files for {len(episodes)} episodes in {meta.repo_id}...")
        
        for ep_idx in tqdm(episodes, desc=f"Validating videos for {meta.repo_id}", leave=False):
            has_all_videos = True
            
            for cam_key in cameras_to_check:
                video_path = meta.root / meta.get_video_file_path(ep_idx, cam_key)
                if not self._is_video_valid(video_path):
                    has_all_videos = False
                    break
            
            if has_all_videos:
                valid_episodes.append(ep_idx)
            else:
                invalid_episodes.append(ep_idx)
        
        if invalid_episodes:
            logger.warning(
                f"Filtered out {len(invalid_episodes)} episodes from {meta.repo_id} due to missing/corrupted video files. "
                f"Remaining: {len(valid_episodes)} episodes. "
                f"Invalid episodes (first 10): {invalid_episodes[:10]}"
            )
        
        return valid_episodes
    
    def _is_download_complete(self, meta: LeRobotV21Metadata, episodes: List[int]) -> bool:
        """
        Quick check if all required files for the given episodes are already downloaded.
        
        This is a fast local-only check that doesn't contact the remote server.
        Uses chunk-level directory checking for efficiency with large datasets.
        
        Strategy:
        1. For datasets with many episodes, check only a sample of chunks
        2. Use directory existence checks (faster than file listing)
        3. Skip video checks if download_videos=False
        
        Returns True if all required chunk directories exist and appear valid.
        """
        # For large datasets, check at chunk level instead of per-episode
        # This is much faster for datasets with many episodes
        chunks_needed = set()
        for ep_idx in episodes:
            chunks_needed.add(meta.get_episode_chunk(ep_idx))
        
        # Optimization: For very large datasets with many chunks, sample check
        # If we need almost all chunks, just verify the directories exist
        chunks_to_check = list(chunks_needed)
        if len(chunks_to_check) > 10:
            # Sample first, middle, and last chunks for quick verification
            sample_indices = [0, len(chunks_to_check) // 2, -1]
            chunks_to_check = [chunks_to_check[i] for i in sample_indices]
        
        # Check if data chunks exist
        for chunk_idx in chunks_to_check:
            chunk_dir = meta.root / "data" / f"chunk-{chunk_idx:03d}"
            if not chunk_dir.exists():
                return False
            # Quick check: directory should have at least one parquet file
            # Use iterdir() with early exit instead of glob for speed
            try:
                has_parquet = any(f.suffix == '.parquet' for f in chunk_dir.iterdir())
                if not has_parquet:
                    return False
            except (OSError, PermissionError):
                return False
        
        # Check video chunks if needed (only for sampled chunks)
        if self.download_videos and meta.video_keys:
            # Only check the first video key to save time
            vid_key = meta.video_keys[0]
            for chunk_idx in chunks_to_check:
                video_chunk_dir = meta.root / "videos" / f"chunk-{chunk_idx:03d}" / vid_key.replace(".", "/")
                if not video_chunk_dir.exists():
                    return False
                # Quick check: directory should have video files
                try:
                    has_video = any(f.suffix == '.mp4' for f in video_chunk_dir.iterdir())
                    if not has_video:
                        return False
                except (OSError, PermissionError):
                    return False
        
        return True
    
    def _ensure_data_downloaded(self, meta: LeRobotV21Metadata, episodes: List[int]):
        """
        Ensure parquet and video files are downloaded.
        
        Follows a multi-stage approach similar to lerobot_dataset.py:
        1. Quick local check using metadata - no network access
        2. If metadata suggests data is complete, verify with a fast chunk-level check
        3. Only download if local checks indicate missing data
        
        This avoids unnecessary network access when data is already present.
        """
        # Stage 1: Check if we should skip download based on metadata
        # If meta.total_episodes > 0 and episodes exist, assume metadata was loaded successfully
        # which means at least the meta files are present
        if meta.episodes is not None and len(meta.episodes) > 0:
            # Stage 2: Quick chunk-level local check - no network access
            if self._is_download_complete(meta, episodes):
                logger.debug(f"All required data already present locally for {meta.repo_id}")
                return
        
        # Stage 3: Data is missing - need to download
        # Group episodes by chunk for efficient downloading
        chunks_needed = set()
        for ep_idx in episodes:
            chunks_needed.add(meta.get_episode_chunk(ep_idx))
        
        total_chunks = meta.info.get("total_chunks", 1)
        
        # Decide download strategy based on how many chunks we need
        if len(chunks_needed) > total_chunks * 0.5:
            # Need most chunks - download everything
            logger.info(
                f"Downloading dataset {meta.repo_id} "
                f"(need {len(chunks_needed)}/{total_chunks} chunks)..."
            )
            allow_patterns = ["data/**"]
            if self.download_videos:
                allow_patterns.append("videos/**")
            
            snapshot_download(
                meta.repo_id,
                repo_type="dataset",
                revision=meta.revision,
                local_dir=meta.root,
                allow_patterns=allow_patterns,
            )
        else:
            # Need only some chunks - download selectively
            allow_patterns = []
            for chunk_idx in sorted(chunks_needed):
                allow_patterns.append(f"data/chunk-{chunk_idx:03d}/*")
                if self.download_videos:
                    allow_patterns.append(f"videos/chunk-{chunk_idx:03d}/**")
            
            logger.info(
                f"Downloading {len(chunks_needed)} chunk(s) for {meta.repo_id}..."
            )
            snapshot_download(
                meta.repo_id,
                repo_type="dataset",
                revision=meta.revision,
                local_dir=meta.root,
                allow_patterns=allow_patterns,
            )
        
        # Step 3: Verify critical files after download
        # Just check a sample to make sure download worked
        sample_ep = episodes[0] if episodes else None
        if sample_ep is not None:
            parquet_path = meta.root / meta.get_data_file_path(sample_ep)
            if not parquet_path.exists():
                logger.warning(
                    f"Sample parquet file still missing after download: {parquet_path}. "
                    f"Some files may not exist in the remote repository."
                )
    
    def _build_index_mapping(self):
        """Build index mapping for fast sample lookup."""
        self.index_to_sample_map = []  # (dataset_idx, episode_idx, frame_offset)
        
        for dataset_idx, (meta, episodes) in enumerate(
            zip(self.dataset_metas, self.per_dataset_episodes)
        ):
            for ep_idx in tqdm(episodes, desc=f"Building index mapping for {meta.repo_id}"):
                ep_len = meta.episodes[ep_idx].get('length', 0)
                for frame_offset in range(ep_len):
                    self.index_to_sample_map.append((dataset_idx, ep_idx, frame_offset))
    
    def _compute_feature_dim(self, keys: Union[str, List[str]]) -> int:
        """Compute the total dimension for one or more feature keys.
        
        Args:
            keys: Single key string or list of keys to concatenate
            
        Returns:
            Total dimension (sum of all keys' dimensions)
        """
        if isinstance(keys, str):
            keys = [keys]
        
        total_dim = 0
        meta = self.dataset_metas[0]  # Use first dataset's features as reference
        
        for key in keys:
            if key in meta.features:
                shape = meta.features[key].get('shape', ())
                # For 1D features, shape is (dim,); for 2D, shape is (seq, dim), etc.
                # We take the last dimension as the feature dimension
                if shape:
                    total_dim += shape[-1] if len(shape) > 0 else 1
                else:
                    total_dim += 1
            else:
                # Key not in features, try to infer from first episode's data
                logger.warning(f"Key '{key}' not found in dataset features, will infer dimension from data")
                # Default to 0, will be updated when data is loaded
                pass
        
        return total_dim if total_dim > 0 else 7  # Fallback to 7 if cannot determine
    
    def _infer_image_size_from_features(self) -> Tuple[int, int]:
        """Infer default image size from dataset features.
        
        Returns:
            Tuple of (height, width) for the default image size.
        """
        meta = self.dataset_metas[0]  # Use first dataset as reference
        
        # Look for camera/video features to get image size
        for key in meta.camera_keys:
            if key in meta.features:
                shape = meta.features[key].get('shape', ())
                # Video/image shape is typically (C, H, W) or (H, W, C)
                if len(shape) >= 2:
                    # Assume shape is (C, H, W) for video features
                    if shape[0] in (1, 3, 4):  # Likely channels first
                        return (shape[1], shape[2])  # (H, W)
                    else:  # Likely channels last or (H, W)
                        return (shape[0], shape[1])  # (H, W)
        
        # Default fallback size
        logger.warning("Could not infer image size from features, using default 224x224")
        return (224, 224)
    
    def _get_keys_to_aggregate(self) -> List[str]:
        """Get the list of keys that need statistics aggregation."""
        keys = []
        
        # Add state keys
        if isinstance(self.state_key, str):
            keys.append(self.state_key)
        else:
            keys.extend(self.state_key)
        
        # Add action keys
        if isinstance(self.action_key, str):
            keys.append(self.action_key)
        else:
            keys.extend(self.action_key)
        
        return keys
    
    def _compute_aggregated_stats(self):
        """
        Compute aggregated statistics for state_key and action_key only.
        
        This is called once during __init__ and only computes stats for the
        keys that are actually used, which is more efficient than aggregating
        all keys in the dataset.
        """
        self._aggregated_stats = {}
        
        keys_to_aggregate = self._get_keys_to_aggregate()
        
        for dataset_idx, (meta, episodes) in enumerate(
            zip(self.dataset_metas, self.per_dataset_episodes)
        ):
            if meta.stats is not None:
                # Pre-computed stats available (older format or already aggregated)
                # Just extract the keys we need
                for key in keys_to_aggregate:
                    if key in meta.stats and key not in self._aggregated_stats:
                        self._aggregated_stats[key] = meta.stats[key]
            elif meta.episodes_stats is not None:
                # Need to aggregate from per-episode stats
                # Only aggregate for the episodes we're using
                key_stats = aggregate_stats_for_keys(
                    meta.episodes_stats,
                    keys_to_aggregate,
                    episode_indices=episodes
                )
                # Merge into aggregated stats
                for key, stats in key_stats.items():
                    if key not in self._aggregated_stats:
                        self._aggregated_stats[key] = stats
        
        logger.debug(f"Computed aggregated stats for keys: {list(self._aggregated_stats.keys())}")
    
    def _load_episode_parquet(self, dataset_idx: int, ep_idx: int) -> Dict[str, np.ndarray]:
        """Load parquet data for an episode."""
        cache_key = (dataset_idx, ep_idx)

        # Cache disabled
        if self.parquet_cache_size <= 0:
            meta = self.dataset_metas[dataset_idx]
            parquet_path = meta.root / meta.get_data_file_path(ep_idx)
            table = pq.read_table(parquet_path)
            data = {col: table[col].to_numpy() for col in table.column_names}
            for key in data:
                if data[key].dtype == object:
                    try:
                        data[key] = np.stack(data[key])
                    except Exception:
                        pass
            return data

        cached = self._parquet_cache.get(cache_key)
        if cached is not None:
            self._parquet_cache.move_to_end(cache_key)
            return cached

        meta = self.dataset_metas[dataset_idx]
        parquet_path = meta.root / meta.get_data_file_path(ep_idx)

        table = pq.read_table(parquet_path)
        data = {col: table[col].to_numpy() for col in table.column_names}

        for key in data:
            if data[key].dtype == object:
                try:
                    data[key] = np.stack(data[key])
                except Exception:
                    pass

        self._parquet_cache[cache_key] = data
        self._parquet_cache.move_to_end(cache_key)
        while len(self._parquet_cache) > self.parquet_cache_size:
            self._parquet_cache.popitem(last=False)

        return data
    
    def _load_video_frame(
        self,
        meta: LeRobotV21Metadata,
        ep_idx: int,
        vid_key: str,
        timestamp: float,
        parquet_data: Optional[Dict[str, np.ndarray]] = None,
        frame_index: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Load a single frame from video using timestamp (same as original LeRobot).
        
        Args:
            meta: Dataset metadata
            ep_idx: Episode index
            vid_key: Video key (camera name)
            timestamp: Timestamp for frame extraction
            parquet_data: Optional parquet data to fallback to if video missing
            frame_index: Optional frame index for parquet fallback
            
        Returns:
            Frame tensor (C, H, W) or None if not available
        """
        video_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
        
        if video_path.exists():
            try:
                # Use torchcodec if available (faster), otherwise fallback to pyav
                if self.video_backend == "torchcodec":
                    frames = decode_video_frames_torchcodec(
                        video_path, [timestamp], self.tolerance_s, fps=meta.fps
                    )
                else:
                    frames = decode_video_frames_torchvision(
                        video_path, [timestamp], self.tolerance_s, backend="pyav"
                    )
                return frames[0]  # (C, H, W)
            except Exception as e:
                logger.warning(f"Failed to decode video frame from {video_path}: {e}")
        
        # Fallback: try to load from parquet data if available
        if parquet_data is not None and frame_index is not None:
            if vid_key in parquet_data:
                try:
                    img_data = parquet_data[vid_key][frame_index]
                    if isinstance(img_data, (bytes, np.ndarray)):
                        from io import BytesIO
                        if isinstance(img_data, bytes):
                            img = Image.open(BytesIO(img_data)).convert('RGB')
                        else:
                            img = Image.fromarray(img_data).convert('RGB')
                        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                        return img_tensor
                except Exception as e:
                    logger.warning(f"Failed to load image from parquet for {vid_key}: {e}")
        
        # If video file doesn't exist and we don't have parquet fallback, return None
        if not hasattr(self, '_video_warned'):
            logger.warning(
                f"Video file not found: {video_path}. "
                f"This may happen if the video wasn't downloaded or doesn't exist in the dataset. "
                f"Will try to use placeholder or skip this camera."
            )
            self._video_warned = True
        
        return None
    
    def _load_image_from_parquet(
        self,
        parquet_data: Dict[str, np.ndarray],
        cam_key: str,
        frame_index: int,
        meta: LeRobotV21Metadata,
    ) -> Optional[torch.Tensor]:
        """
        Load image from parquet data. Handles various image storage formats.
        
        Args:
            parquet_data: Dictionary of parquet data
            cam_key: Camera key
            frame_index: Frame index in episode
            meta: Dataset metadata
            
        Returns:
            Image tensor (C, H, W) in [0, 1] range, or None if failed
        """
        # Use OpenCV for fast image decoding (significantly faster than PIL).
        # Always return CHW float32 in [0, 1].
        import cv2
        from io import BytesIO
        
        img_data = parquet_data[cam_key][frame_index]
        
        # Handle different image data formats
        if img_data is None:
            return None
        
        # Case 1: Dictionary with 'bytes' key (HuggingFace datasets Image format)
        if isinstance(img_data, dict):
            if 'bytes' in img_data and img_data['bytes'] is not None:
                nparr = np.frombuffer(img_data['bytes'], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return None
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            elif 'path' in img_data and img_data['path'] is not None:
                # Image stored as file path
                img_path = img_data['path']
                # Check if it's a relative path
                if not os.path.isabs(img_path):
                    img_path = meta.root / img_path
                if Path(img_path).exists():
                    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img is None:
                        return None
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return None
        
        # Case 2: Raw bytes
        if isinstance(img_data, bytes):
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Case 3: numpy array (raw pixel values)
        if isinstance(img_data, np.ndarray):
            # Check if it's already an image array
            if img_data.ndim == 3:
                # Could be (H, W, C) or (C, H, W)
                if img_data.shape[0] in (1, 3, 4):  # Likely (C, H, W)
                    img_tensor = torch.from_numpy(img_data.copy()).float()
                    if img_tensor.max() > 1.0:
                        img_tensor = img_tensor / 255.0
                    return img_tensor
                else:  # Likely (H, W, C)
                    img = img_data.astype(np.uint8, copy=False)
                    if img.shape[-1] == 3:
                        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    img = np.ascontiguousarray(img)
                    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            elif img_data.ndim == 2:
                # Grayscale image
                img = img_data.astype(np.uint8, copy=False)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            elif img_data.ndim == 1:
                # Might be compressed bytes stored as uint8 array
                try:
                    nparr = np.frombuffer(img_data.tobytes(), np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        return None
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                except Exception:
                    pass
        
        # Case 4: PIL Image (unlikely but handle it)
        if isinstance(img_data, Image.Image):
            img = np.array(img_data.convert('RGB'))
            return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Case 5: String path
        if isinstance(img_data, str):
            img_path = img_data
            if not os.path.isabs(img_path):
                img_path = meta.root / img_path
            if Path(img_path).exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    return None
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        logger.warning(f"Unknown image data type for {cam_key}: {type(img_data)}")
        return None
    
    def _get_data_by_keys(
        self,
        parquet_data: Dict[str, np.ndarray],
        keys: Union[str, List[str]],
        frame_idx: Optional[int] = None,
        fallback_key: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Get data from parquet by key(s). If keys is a list, concatenate the data.
        
        Args:
            parquet_data: Dictionary of parquet data
            keys: Single key string or list of keys to concatenate
            frame_idx: If provided, index into the data at this frame
            fallback_key: Fallback key if primary keys not found
            
        Returns:
            numpy array of concatenated data, or None if not found
        """
        if isinstance(keys, str):
            # Single key - no concatenation
            data = parquet_data.get(keys)
            if data is None and fallback_key:
                data = parquet_data.get(fallback_key)
            if data is None:
                return None
            if frame_idx is not None:
                return data[frame_idx]
            return data
        else:
            # List of keys - concatenate along last axis
            data_parts = []
            for key in keys:
                part = parquet_data.get(key)
                if part is None:
                    logger.warning(f"Key '{key}' not found in parquet data, skipping")
                    continue
                if frame_idx is not None:
                    part = part[frame_idx]
                part = np.asarray(part)
                # Ensure at least 1D for concatenation
                if part.ndim == 0:
                    part = part.reshape(1)
                data_parts.append(part)
            
            if not data_parts:
                # Try fallback
                if fallback_key:
                    data = parquet_data.get(fallback_key)
                    if data is not None:
                        if frame_idx is not None:
                            return data[frame_idx]
                        return data
                return None
            
            # Concatenate along last axis
            return np.concatenate(data_parts, axis=-1)
    
    def _get_stats_by_keys(
        self,
        stats: Dict[str, Any],
        keys: Union[str, List[str]],
        fallback_key: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get statistics for key(s). If keys is a list, concatenate the stats.
        
        Args:
            stats: Statistics dictionary
            keys: Single key string or list of keys
            fallback_key: Fallback key if primary keys not found
            
        Returns:
            Dictionary of concatenated statistics
        """
        if isinstance(keys, str):
            # Single key
            result = stats.get(keys)
            if result is None and fallback_key:
                result = stats.get(fallback_key)
            return result if result else {}
        else:
            # List of keys - concatenate stats
            stat_keys = ['mean', 'std', 'min', 'max', 'q01', 'q99']
            result = {}
            
            for stat_name in stat_keys:
                parts = []
                for key in keys:
                    key_stats = stats.get(key, {})
                    if stat_name in key_stats:
                        parts.append(np.asarray(key_stats[stat_name]))
                
                if parts:
                    result[stat_name] = np.concatenate(parts, axis=-1)
            
            # If no stats found, try fallback
            if not result and fallback_key:
                return stats.get(fallback_key, {})
            
            return result
    
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
        """Get lengths of all episodes."""
        lengths = []
        for meta, episodes in zip(self.dataset_metas, self.per_dataset_episodes):
            for ep_idx in episodes:
                lengths.append(meta.episodes[ep_idx].get('length', 0))
        return lengths
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        dataset_idx, ep_idx, frame_offset = self.index_to_sample_map[index]
        meta = self.dataset_metas[dataset_idx]
        
        # Load parquet data
        parquet_data = self._load_episode_parquet(dataset_idx, ep_idx)
        
        # Get current frame data
        frame_index = frame_offset
        timestamp = float(parquet_data.get('timestamp', [0])[frame_offset])
        
        # Get state (supports single key or list of keys to concatenate)
        state_data = self._get_data_by_keys(
            parquet_data, self.state_key, 
            frame_idx=frame_offset, 
            fallback_key=None  # No fallback, use configured state_key
        )
        if state_data is not None:
            state = torch.tensor(state_data, dtype=torch.float32)
        else:
            state = torch.zeros(self.state_dim, dtype=torch.float32)
        
        # Get action data (supports single key or list of keys to concatenate)
        action_full = self._get_data_by_keys(
            parquet_data, self.action_key,
            frame_idx=None,  # Get full episode
            fallback_key=None  # No fallback, use configured action_key
        )
        ep_len = len(action_full) if action_full is not None else 0
        
        # Build action chunk using slicing (more efficient than loop)
        if action_full is not None and len(action_full) > 0:
            # Calculate how many valid actions we can get
            end_idx = min(frame_offset + self.chunk_size, ep_len)
            valid_count = max(0, end_idx - frame_offset)
            
            if valid_count > 0:
                # Get valid actions in one slice
                valid_actions = action_full[frame_offset:end_idx]
                
                if valid_count < self.chunk_size:
                    # Need padding: repeat last valid action
                    pad_count = self.chunk_size - valid_count
                    last_action = valid_actions[-1:] if len(valid_actions) > 0 else action_full[-1:]
                    padding = np.repeat(last_action, pad_count, axis=0)
                    actions = np.concatenate([valid_actions, padding], axis=0)
                    is_pad = np.array([False] * valid_count + [True] * pad_count)
                else:
                    actions = valid_actions
                    is_pad = np.array([False] * self.chunk_size)
            else:
                # All padding (frame_offset >= ep_len)
                last_action = action_full[-1:]
                actions = np.repeat(last_action, self.chunk_size, axis=0)
                is_pad = np.array([True] * self.chunk_size)
        else:
            # No action data, use zeros with correct dimension
            actions = np.zeros((self.chunk_size, self.action_dim))
            is_pad = np.array([True] * self.chunk_size)
        
        action = torch.tensor(actions, dtype=torch.float32)
        is_pad = torch.tensor(is_pad, dtype=torch.bool)
        
        # Get task/language instruction
        task_idx = int(parquet_data.get('task_index', [0])[frame_offset])
        raw_lang = meta.tasks.get(task_idx, "")
        
        # Load images from video or parquet
        cam_keys = meta.camera_keys if len(self.camera_names) == 0 else self.camera_names
        images = []
        
        for cam_key in cam_keys:
            frame = None
            
            if cam_key in meta.video_keys:
                # Try to load from video first (using timestamp, same as original LeRobot)
                frame = self._load_video_frame(
                    meta, ep_idx, cam_key, timestamp,
                    parquet_data=parquet_data, frame_index=frame_offset
                )
            
            # Fallback to parquet if video failed or camera is in parquet
            if frame is None and cam_key in parquet_data:
                try:
                    frame = self._load_image_from_parquet(
                        parquet_data, cam_key, frame_offset, meta
                    )
                except Exception as e:
                    logger.warning(f"Failed to load image from parquet for {cam_key}: {e}")
            
            # If still no frame, create a placeholder (black image)
            if frame is None:
                # Try to infer image size from other cameras or use default
                # Use default image size for placeholder
                h, w = self._default_image_size
                frame = torch.zeros(3, h, w, dtype=torch.float32)
                if not hasattr(self, '_placeholder_warned'):
                    logger.warning(
                        f"Using placeholder (black) image for camera '{cam_key}' "
                        f"in episode {ep_idx}. Video file may be missing."
                    )
                    self._placeholder_warned = True
            
            images.append(frame)
        
        if images:
            # Stack images - ensure all have same size using _default_image_size
            # This guarantees consistent image sizes across all samples
            target_h, target_w = self._default_image_size
            
            # Check if all images have the same size as target
            needs_resize = any(img.shape[1] != target_h or img.shape[2] != target_w for img in images)
            
            if needs_resize:
                images = torch.cat([
                    resize_with_pad(img.unsqueeze(0), height=target_h, width=target_w)
                    for img in images
                ], dim=0)
            else:
                images = torch.stack(images)
            
            # Convert to uint8 [0, 255]
            images = (images * 255).clamp(0, 255).to(torch.uint8)
            images = ensure_uint8_image(images)
        else:
            images = None
        
        # Compute global episode ID
        episode_id = int(self.per_dataset_episode_start[dataset_idx]) + \
                     self.per_dataset_episodes[dataset_idx].index(ep_idx)
        
        return {
            'image': images,
            'state': state,
            'action': action,
            'is_pad': is_pad,
            'raw_lang': raw_lang,
            'reasoning': {},
            'timestamp': frame_index,
            'episode_id': episode_id,
        }
    
    def _compute_stats_for_key(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        """Compute statistics for a single key across all datasets.
        
        Only computes stats for the specified key, avoiding unnecessary computation.
        """
        # Check if already computed
        if key in self._aggregated_stats:
            return self._aggregated_stats[key]
        
        aggregated = None
        
        for dataset_idx, (meta, episodes) in enumerate(
            zip(self.dataset_metas, self.per_dataset_episodes)
        ):
            key_stats = None
            
            if meta.stats is not None and key in meta.stats:
                # Pre-computed stats available
                key_stats = meta.stats[key]
            elif meta.episodes_stats is not None:
                # Aggregate from per-episode stats for this key only
                stats_dict = aggregate_stats_for_keys(
                    meta.episodes_stats,
                    [key],
                    episode_indices=episodes
                )
                if key in stats_dict:
                    key_stats = stats_dict[key]
            
            if key_stats is not None:
                if aggregated is None:
                    aggregated = {k: np.array(v) for k, v in key_stats.items()}
                else:
                    # Merge stats from multiple datasets
                    for stat_name, stat_val in key_stats.items():
                        stat_val = np.array(stat_val)
                        if stat_name == 'min':
                            aggregated['min'] = np.minimum(aggregated.get('min', stat_val), stat_val)
                        elif stat_name == 'max':
                            aggregated['max'] = np.maximum(aggregated.get('max', stat_val), stat_val)
                        elif stat_name == 'q01':
                            aggregated['q01'] = np.minimum(aggregated.get('q01', stat_val), stat_val)
                        elif stat_name == 'q99':
                            aggregated['q99'] = np.maximum(aggregated.get('q99', stat_val), stat_val)
                        elif stat_name in ('mean', 'std'):
                            # For mean/std, we'd need weighted average; for simplicity, use first dataset's
                            if stat_name not in aggregated:
                                aggregated[stat_name] = stat_val
        
        # Cache the result
        if aggregated is not None:
            self._aggregated_stats[key] = aggregated
        
        return aggregated
    
    def _compute_stats_for_keys(self, keys: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """Compute and concatenate statistics for one or more keys.
        
        Args:
            keys: Single key string or list of keys to concatenate
            
        Returns:
            Dictionary with 'mean', 'std', 'min', 'max', 'q01', 'q99' arrays
            
        Raises:
            KeyError: If any key's statistics cannot be found
        """
        if isinstance(keys, str):
            keys = [keys]
        
        # Compute stats for each key
        all_stats = []
        missing_keys = []
        for key in keys:
            stats = self._compute_stats_for_key(key)
            if stats is not None:
                all_stats.append(stats)
            else:
                missing_keys.append(key)
        
        # Raise error if any key not found
        if missing_keys:
            raise KeyError(
                f"Statistics not found for key(s): {missing_keys}. "
                f"Available keys in dataset may not include these. "
                f"Please check your state_key and action_key configuration."
            )
        
        # If single key, return directly
        if len(all_stats) == 1:
            return all_stats[0]
        
        # Concatenate stats from multiple keys
        result = {}
        stat_names = ['mean', 'std', 'min', 'max', 'q01', 'q99']
        
        for stat_name in stat_names:
            values = [s.get(stat_name) for s in all_stats if stat_name in s]
            if values:
                result[stat_name] = np.concatenate(values, axis=-1)
        
        return result
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics. Supports concatenated keys.
        
        Statistics are computed lazily for only the required keys.
        
        Raises:
            KeyError: If state_key or action_key statistics cannot be found
        """
        # Compute stats only for the keys we need (state_key and action_key)
        state_stats = self._compute_stats_for_keys(self.state_key)
        action_stats = self._compute_stats_for_keys(self.action_key)
        
        # Add q01/q99 if not present (use min/max as fallback, with correct dimensions)
        if state_stats and 'q01' not in state_stats:
            state_stats['q01'] = state_stats.get('min', np.zeros(self.state_dim))
            state_stats['q99'] = state_stats.get('max', np.ones(self.state_dim))
        
        if action_stats and 'q01' not in action_stats:
            action_stats['q01'] = action_stats.get('min', np.zeros(self.action_dim))
            action_stats['q99'] = action_stats.get('max', np.ones(self.action_dim))
        
        return {
            'state': state_stats,
            'action': action_stats,
            'num_episodes': self.total_episodes,
            'num_transitions': self.total_frames,
        }
    
    def extract_from_episode(self, episode_idx: int, keyname: List[str] = []) -> Dict[str, np.ndarray]:
        """Extract specific features from an episode. Supports concatenated keys."""
        # Find which dataset and episode
        dataset_idx = np.searchsorted(self.cumulative_num_episodes, episode_idx + 1)
        local_ep_list_idx = episode_idx - int(self.per_dataset_episode_start[dataset_idx])
        ep_idx = self.per_dataset_episodes[dataset_idx][local_ep_list_idx]
        
        # Load parquet data
        parquet_data = self._load_episode_parquet(dataset_idx, ep_idx)
        
        result = {}
        
        if 'state' in keyname:
            state_data = self._get_data_by_keys(
                parquet_data, self.state_key,
                frame_idx=None,
                fallback_key='observation.state'
            )
            if state_data is not None:
                result['state'] = np.array(state_data)
        
        if 'action' in keyname:
            action_data = self._get_data_by_keys(
                parquet_data, self.action_key,
                frame_idx=None,
                fallback_key='action'
            )
            if action_data is not None:
                result['action'] = np.array(action_data)
        
        return result


if __name__ == '__main__':
    # Test the wrapper
    print("=" * 60)
    print("Testing LeRobot v2.1 Wrapper")
    print("=" * 60)
    
    # Test with a sample dataset
    dataset = WrappedLerobotV21Dataset(
        dataset_path_list=["lerobot/aloha_sim_transfer_cube_scripted"],
        tolerance_s=0.1,
    )
    
    print(f"Total episodes: {dataset.total_episodes}")
    print(f"Total frames: {dataset.total_frames}")
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape if sample['image'] is not None else None}")
    print(f"State shape: {sample['state'].shape}")
    print(f"Action shape: {sample['action'].shape}")
    print(f"Language: {sample['raw_lang']}")

