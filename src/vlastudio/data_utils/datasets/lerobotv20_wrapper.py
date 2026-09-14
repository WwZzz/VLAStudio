"""
LeRobot v2.0 Dataset Wrapper for ILStudio.

This is a standalone implementation that loads LeRobot v2.0 format datasets
WITHOUT depending on the lerobot library. It reads the dataset structure directly
from parquet files and video files.

LeRobot v2.0 Dataset Structure:
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
│   ├── stats.json
│   └── tasks.jsonl
└── videos
    ├── chunk-000
    │   ├── observation.images.laptop
    │   │   ├── episode_000000.mp4
    │   │   └── ...
    │   └── ...
    └── ...

Key differences from v2.1:
- v2.0 uses stats.json (aggregated), v2.1 uses episodes_stats.jsonl (per-episode)
- v2.0 may have slightly different metadata structure
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
TASKS_PATH = "meta/tasks.jsonl"

CODEBASE_VERSION = "v2.0"


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


# =============================================================================
# Video Decoding - Multiple backends supported
# =============================================================================

def get_video_backend() -> str:
    """Get the best available video backend."""
    # Allow explicit override (useful for debugging / reliability).
    forced = os.environ.get("ILSTUDIO_VIDEO_BACKEND")
    if forced:
        return forced
    # Default to torchvision/pyav for reliability. Torchcodec can be enabled explicitly
    # via `ILSTUDIO_VIDEO_BACKEND=torchcodec` (or by passing `video_backend="torchcodec"`).
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
    # NOTE: In some environments / torchcodec versions, constructing VideoDecoder from a path
    # string can fail with: "No valid stream found in input file...".
    # Using a regular file handle is more robust.
    from torchcodec.decoders import VideoDecoder

    video_path_str = str(video_path)

    # Keep the file handle open for the lifetime of the decoder and `get_frames_at` call.
    # (We intentionally avoid caching here to prevent "too many open files" issues.)
    with open(video_path_str, "rb") as file_handle:
        decoder = VideoDecoder(file_handle, seek_mode="approximate")
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
    
    Memory-efficient: uses streaming decode - only keeps the frames needed for
    timestamp matching instead of accumulating all frames from keyframe to last_ts.
    This avoids OOM when pyav seeks to a keyframe far before the target.
    
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
    stop_ts = last_ts + tolerance_s
    
    # Seek to closest key frame before first timestamp
    reader.seek(first_ts, keyframes_only=keyframes_only)
    
    # Memory-efficient streaming: only keep best-match frame(s) per timestamp
    best_for_ts: Dict[int, Tuple[torch.Tensor, float, float]] = {}
    
    for frame in reader:
        current_ts = frame["pts"]
        frame_data = frame["data"]
        
        for i, ts in enumerate(timestamps):
            dist = abs(current_ts - ts)
            if i not in best_for_ts or dist < best_for_ts[i][2]:
                best_for_ts[i] = (frame_data.clone(), current_ts, dist)
        
        if current_ts >= stop_ts:
            break
    
    # Close reader to release video buffers
    if backend == "pyav":
        reader.container.close()
    reader = None
    
    if not best_for_ts:
        raise RuntimeError(f"No frames decoded from {video_path}")
    
    frames_list = []
    for i in range(len(timestamps)):
        if i not in best_for_ts:
            raise RuntimeError(f"No frame found for timestamp {timestamps[i]} in {video_path}")
        frame_tensor, pts, dist = best_for_ts[i]
        if dist >= tolerance_s:
            warnings.warn(
                f"Frame at {pts:.3f}s violates tolerance for ts={timestamps[i]}: "
                f"dist={dist:.3f} > {tolerance_s}"
            )
        frames_list.append(frame_tensor)
    
    frames = torch.stack(frames_list)
    frames = frames.float() / 255.0
    return frames


# =============================================================================
# Dataset Metadata
# =============================================================================

class LeRobotV20Metadata:
    """
    Metadata handler for LeRobot v2.0 datasets.
    
    Handles loading and caching of dataset metadata without depending on lerobot.
    """
    
    def __init__(
        self,
        repo_id: str,
        root: Optional[str] = None,
        revision: str = CODEBASE_VERSION,
        force_download: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision
        self.root = Path(root) if root else DEFAULT_LEROBOT_HOME / repo_id
        
        # Try to load metadata, download if needed
        try:
            if force_download:
                raise FileNotFoundError
            self._load_metadata()
        except (FileNotFoundError, NotADirectoryError):
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
        
        # Load stats (v2.0 uses stats.json - aggregated stats)
        stats_path = self.root / STATS_PATH
        if stats_path.exists():
            self.stats = cast_stats_to_numpy(load_json(stats_path))
        else:
            self.stats = {}
    
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
    def image_keys(self) -> List[str]:
        """Get keys for image modalities (stored as images, not video)."""
        return [key for key, ft in self.features.items() if ft.get("dtype") == "image"]
    
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

class WrappedLerobotV20Dataset(tud.Dataset):
    """
    Standalone wrapper for LeRobot v2.0 datasets.
    
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
        video_backend: Video decoding backend ("pyav", "torchcodec", or None for auto)
        parquet_cache_size: Max number of episodes to keep in memory cache (0 disables caching).
            With num_workers>0, each worker has its own cache. Default 8 balances memory vs I/O.
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
        parquet_cache_size: int = 8,
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
        # Auto-detect best video backend if not specified
        self.video_backend = video_backend if video_backend else get_video_backend()
        logger.info(f"Using video backend: {self.video_backend}")
        
        # Load datasets
        self.dataset_path_list = dataset_path_list
        self.dataset_metas: List[LeRobotV20Metadata] = []
        self.dataset_dirs: List[str] = []
        self.per_dataset_episodes: List[List[int]] = []
        self.per_dataset_num_episodes: List[int] = []
        self.per_dataset_num_frames: List[int] = []
        
        for repo_id in dataset_path_list:
            # Load metadata
            ds_root = self.root / repo_id
            meta = LeRobotV20Metadata(repo_id, root=str(ds_root))
            
            # Filter episodes
            episodes = self._filter_episodes(meta, episode_filter)
            if episodes is None:
                episodes = list(range(meta.total_episodes))
            
            if len(episodes) == 0:
                warnings.warn(f"No episodes found for {repo_id} with filter {episode_filter}")
                continue
            
            # Download data files if needed
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
        self._parquet_available_columns = self._discover_parquet_columns()
        self._core_parquet_columns = [
            self._build_core_parquet_columns(i) for i in range(len(self.dataset_metas))
        ]
        
        # Compute state and action dimensions from dataset features
        self.state_dim = self._compute_feature_dim(self.state_key)
        self.action_dim = self._compute_feature_dim(self.action_key)
        
        # Determine default image size from dataset features or use specified size
        if self.image_size is None:
            self._default_image_size = self._infer_image_size_from_features()
        else:
            self._default_image_size = (self.image_size[1], self.image_size[0])  # (H, W)
        
        logger.info(
            f"LeRobot v2.0 Dataset loaded: {self.total_episodes} episodes, "
            f"{self.total_frames} frames from {len(self.dataset_metas)} dataset(s), "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"image_size={self._default_image_size}"
        )
    
    def _filter_episodes(
        self,
        meta: LeRobotV20Metadata,
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
        """Check if a video file exists and can be opened/decoded."""
        if not video_path.exists():
            return False
        
        try:
            if video_path.stat().st_size < 1000:
                return False
            
            import torchvision
            torchvision.set_video_backend("pyav")
            reader = torchvision.io.VideoReader(str(video_path), "video")
            
            for frame in reader:
                reader.container.close()
                return True
            
            reader.container.close()
            return False
            
        except Exception:
            return False
    
    def _filter_episodes_with_missing_videos(
        self,
        meta: LeRobotV20Metadata,
        episodes: List[int],
    ) -> List[int]:
        """Filter out episodes that have missing or corrupted video files."""
        if not self.download_videos:
            return episodes
        
        if len(self.camera_names) > 0:
            cameras_to_check = [cam for cam in self.camera_names if cam in meta.video_keys]
        else:
            cameras_to_check = meta.video_keys
        
        if not cameras_to_check:
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
                f"Remaining: {len(valid_episodes)} episodes."
            )
        
        return valid_episodes
    
    def _is_download_complete(self, meta: LeRobotV20Metadata, episodes: List[int]) -> bool:
        """
        Quick check if all required files for the given episodes are already downloaded.
        
        This is a fast local-only check that doesn't contact the remote server.
        Returns True if all parquet files (and video files if download_videos=True) exist locally.
        """
        for ep_idx in episodes:
            # Check parquet file
            parquet_path = meta.root / meta.get_data_file_path(ep_idx)
            if not parquet_path.exists():
                return False
            
            # Check video files (only if download_videos is True)
            if self.download_videos:
                for vid_key in meta.video_keys:
                    video_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False
        
        return True
    
    def _ensure_data_downloaded(self, meta: LeRobotV20Metadata, episodes: List[int]):
        """
        Ensure parquet and video files are downloaded.
        
        Uses a simple and efficient approach:
        1. Quick local check if all files exist
        2. If not complete, download everything needed in one batch
        3. No per-file remote checking - just download and let HF handle caching
        
        This is much faster than checking each file individually against the remote server.
        """
        # Step 1: Quick local check - if all files exist, we're done
        if self._is_download_complete(meta, episodes):
            logger.debug(f"All files already downloaded for {meta.repo_id}")
            return
        
        # Step 2: Files are missing - download them
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
        self.index_to_sample_map = []
        
        for dataset_idx, (meta, episodes) in enumerate(
            zip(self.dataset_metas, self.per_dataset_episodes)
        ):
            for ep_idx in tqdm(episodes, desc=f"Building index mapping for {meta.repo_id}"):
                ep_len = meta.episodes[ep_idx].get('length', 0)
                for frame_offset in range(ep_len):
                    self.index_to_sample_map.append((dataset_idx, ep_idx, frame_offset))
    
    def _compute_feature_dim(self, keys: Union[str, List[str]]) -> int:
        """Compute the total dimension for one or more feature keys."""
        if isinstance(keys, str):
            keys = [keys]
        
        total_dim = 0
        meta = self.dataset_metas[0]
        
        for key in keys:
            if key in meta.features:
                shape = meta.features[key].get('shape', ())
                if shape:
                    total_dim += shape[-1] if len(shape) > 0 else 1
                else:
                    total_dim += 1
            else:
                logger.warning(f"Key '{key}' not found in dataset features")
        
        return total_dim if total_dim > 0 else 7
    
    def _infer_image_size_from_features(self) -> Tuple[int, int]:
        """Infer default image size from dataset features."""
        meta = self.dataset_metas[0]
        
        for key in meta.camera_keys:
            if key in meta.features:
                shape = meta.features[key].get('shape', ())
                if len(shape) >= 2:
                    if shape[0] in (1, 3, 4):
                        return (shape[1], shape[2])
                    else:
                        return (shape[0], shape[1])
        
        logger.warning("Could not infer image size from features, using default 224x224")
        return (224, 224)
    
    def _discover_parquet_columns(self) -> List[set]:
        """Discover available parquet columns once per dataset."""
        available_columns = []
        for meta, episodes in zip(self.dataset_metas, self.per_dataset_episodes):
            if not episodes:
                available_columns.append(set())
                continue
            parquet_path = meta.root / meta.get_data_file_path(episodes[0])
            # Use the Arrow schema instead of the raw Parquet schema. For nested
            # list/struct columns, `ParquetSchema.names` exposes leaf names such
            # as `element`, which causes columns like
            # `observation.state.joint_position` to be treated as missing.
            schema = pq.ParquetFile(parquet_path).schema_arrow
            available_columns.append(set(schema.names))
        return available_columns

    def _normalize_parquet_arrays(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert object arrays to stacked numpy arrays when possible."""
        for key, value in list(data.items()):
            if getattr(value, "dtype", None) == object:
                try:
                    data[key] = np.stack(value)
                except Exception:
                    pass
        return data

    def _get_requested_camera_keys(self, meta: LeRobotV20Metadata) -> List[str]:
        """Get the camera keys actually requested for this dataset."""
        if len(self.camera_names) == 0:
            return list(meta.camera_keys)
        return [cam_key for cam_key in self.camera_names if cam_key in meta.camera_keys]

    def _build_core_parquet_columns(self, dataset_idx: int) -> List[str]:
        """Build the lightweight parquet column set cached for each episode."""
        meta = self.dataset_metas[dataset_idx]
        available = self._parquet_available_columns[dataset_idx]

        columns = {"timestamp", "task_index"}
        if isinstance(self.state_key, str):
            columns.add(self.state_key)
        else:
            columns.update(self.state_key)
        if isinstance(self.action_key, str):
            columns.add(self.action_key)
        else:
            columns.update(self.action_key)

        # Non-video cameras must come from parquet every sample, so include them in
        # the core cache. Video cameras stay lazy to avoid bloating memory.
        for cam_key in self._get_requested_camera_keys(meta):
            if cam_key not in meta.video_keys:
                columns.add(cam_key)

        return sorted(col for col in columns if col in available)

    def _read_episode_parquet_columns(
        self,
        dataset_idx: int,
        ep_idx: int,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Read a subset of parquet columns for one episode."""
        meta = self.dataset_metas[dataset_idx]
        parquet_path = meta.root / meta.get_data_file_path(ep_idx)
        available = self._parquet_available_columns[dataset_idx]

        read_columns = None
        if columns is not None:
            read_columns = [col for col in columns if col in available]
            if not read_columns:
                return {}

        table = pq.read_table(parquet_path, columns=read_columns)
        data = {col: table[col].to_numpy() for col in table.column_names}
        return self._normalize_parquet_arrays(data)

    def _load_episode_parquet(
        self,
        dataset_idx: int,
        ep_idx: int,
        columns: Optional[List[str]] = None,
        cache_result: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Load parquet data for an episode, caching only the lightweight core columns."""
        core_columns = self._core_parquet_columns[dataset_idx]
        requested_columns = core_columns if columns is None else list(dict.fromkeys(columns))
        use_core_cache = requested_columns == core_columns and cache_result and self.parquet_cache_size > 0

        if use_core_cache:
            cache_key = (dataset_idx, ep_idx)
            cached = self._parquet_cache.get(cache_key)
            if cached is not None:
                self._parquet_cache.move_to_end(cache_key)
                return cached

        data = self._read_episode_parquet_columns(dataset_idx, ep_idx, requested_columns)

        if use_core_cache:
            cache_key = (dataset_idx, ep_idx)
            self._parquet_cache[cache_key] = data
            self._parquet_cache.move_to_end(cache_key)
            while len(self._parquet_cache) > self.parquet_cache_size:
                self._parquet_cache.popitem(last=False)

        return data
    
    def _load_video_frame(
        self,
        meta: LeRobotV20Metadata,
        ep_idx: int,
        vid_key: str,
        timestamp: float,
        parquet_data: Optional[Dict[str, np.ndarray]] = None,
        frame_index: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Load a single frame from video using timestamp."""
        video_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
        
        if video_path.exists():
            try:
                if self.video_backend == "torchcodec":
                    try:
                        frames = decode_video_frames_torchcodec(
                            video_path, [timestamp], self.tolerance_s, fps=meta.fps
                        )
                    except Exception as e_torchcodec:
                        # Torchcodec can be sensitive to FFmpeg / shared-lib setup in some
                        # environments. Fall back to torchvision/pyav for robustness.
                        logger.warning(
                            f"torchcodec decode failed for {video_path}: {e_torchcodec}. Falling back to pyav."
                        )
                        frames = decode_video_frames_torchvision(
                            video_path, [timestamp], self.tolerance_s, backend="pyav"
                        )
                else:
                    frames = decode_video_frames_torchvision(
                        video_path, [timestamp], self.tolerance_s, backend="pyav"
                    )
                return frames[0]
            except Exception as e:
                logger.warning(f"Failed to decode video frame from {video_path}: {e}")
        
        # Fallback to parquet
        if parquet_data is not None and frame_index is not None:
            frame = self._load_image_from_parquet(parquet_data, vid_key, frame_index, meta)
            if frame is not None:
                return frame
        
        return None
    
    def _load_image_from_parquet(
        self,
        parquet_data: Dict[str, np.ndarray],
        cam_key: str,
        frame_index: int,
        meta: LeRobotV20Metadata,
    ) -> Optional[torch.Tensor]:
        """Load image from parquet data. Handles various image storage formats."""
        # Use OpenCV for fast image decoding (significantly faster than PIL).
        # Always return CHW float32 in [0, 1].
        import cv2
        from io import BytesIO
        
        if cam_key not in parquet_data:
            return None
        
        img_data = parquet_data[cam_key][frame_index]
        
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
                img_path = img_data['path']
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
        
        # Case 3: numpy array
        if isinstance(img_data, np.ndarray):
            if img_data.ndim == 3:
                if img_data.shape[0] in (1, 3, 4):
                    img_tensor = torch.from_numpy(img_data.copy()).float()
                    if img_tensor.max() > 1.0:
                        img_tensor = img_tensor / 255.0
                    return img_tensor
                else:
                    # Assume HWC uint8
                    img = img_data.astype(np.uint8, copy=False)
                    # If grayscale-like last dim, fall back to cv2 conversion via PIL-like path
                    if img.shape[-1] == 3:
                        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    # Unknown format
                    img = np.ascontiguousarray(img)
                    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            elif img_data.ndim == 2:
                img = img_data.astype(np.uint8, copy=False)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            elif img_data.ndim == 1:
                try:
                    nparr = np.frombuffer(img_data.tobytes(), np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        return None
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                except Exception:
                    pass
        
        # Case 4: PIL Image
        if isinstance(img_data, Image.Image):
            # Keep PIL support for rare cases, but convert via numpy only.
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
        """Get data from parquet by key(s). If keys is a list, concatenate the data."""
        if isinstance(keys, str):
            data = parquet_data.get(keys)
            if data is None and fallback_key:
                data = parquet_data.get(fallback_key)
            if data is None:
                return None
            if frame_idx is not None:
                return data[frame_idx]
            return data
        else:
            data_parts = []
            for key in keys:
                part = parquet_data.get(key)
                if part is None:
                    logger.warning(f"Key '{key}' not found in parquet data, skipping")
                    continue
                if frame_idx is not None:
                    part = part[frame_idx]
                part = np.asarray(part)
                if part.ndim == 0:
                    part = part.reshape(1)
                data_parts.append(part)
            
            if not data_parts:
                if fallback_key:
                    data = parquet_data.get(fallback_key)
                    if data is not None:
                        if frame_idx is not None:
                            return data[frame_idx]
                        return data
                return None
            
            return np.concatenate(data_parts, axis=-1)
    
    def _get_stats_by_keys(
        self,
        stats: Dict[str, Any],
        keys: Union[str, List[str]],
        fallback_key: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Get statistics for key(s). If keys is a list, concatenate the stats."""
        if isinstance(keys, str):
            result = stats.get(keys)
            if result is None and fallback_key:
                result = stats.get(fallback_key)
            return result if result else {}
        else:
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
        
        # Get state
        state_data = self._get_data_by_keys(
            parquet_data, self.state_key, 
            frame_idx=frame_offset, 
            fallback_key=None
        )
        if state_data is not None:
            state = torch.tensor(state_data, dtype=torch.float32)
        else:
            state = torch.zeros(self.state_dim, dtype=torch.float32)
        
        # Get action data
        action_full = self._get_data_by_keys(
            parquet_data, self.action_key,
            frame_idx=None,
            fallback_key=None
        )
        ep_len = len(action_full) if action_full is not None else 0
        
        # Build action chunk
        if action_full is not None and len(action_full) > 0:
            end_idx = min(frame_offset + self.chunk_size, ep_len)
            valid_count = max(0, end_idx - frame_offset)
            
            if valid_count > 0:
                valid_actions = action_full[frame_offset:end_idx]
                
                if valid_count < self.chunk_size:
                    pad_count = self.chunk_size - valid_count
                    last_action = valid_actions[-1:] if len(valid_actions) > 0 else action_full[-1:]
                    padding = np.repeat(last_action, pad_count, axis=0)
                    actions = np.concatenate([valid_actions, padding], axis=0)
                    is_pad = np.array([False] * valid_count + [True] * pad_count)
                else:
                    actions = valid_actions
                    is_pad = np.array([False] * self.chunk_size)
            else:
                last_action = action_full[-1:]
                actions = np.repeat(last_action, self.chunk_size, axis=0)
                is_pad = np.array([True] * self.chunk_size)
        else:
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
                frame = self._load_video_frame(
                    meta, ep_idx, cam_key, timestamp,
                    parquet_data=parquet_data, frame_index=frame_offset
                )
            
            # Fallback to parquet if video failed or camera is stored directly in parquet.
            if frame is None and cam_key not in parquet_data and cam_key in self._parquet_available_columns[dataset_idx]:
                parquet_data = {
                    **parquet_data,
                    **self._load_episode_parquet(
                        dataset_idx,
                        ep_idx,
                        columns=[cam_key],
                        cache_result=False,
                    ),
                }

            if frame is None and cam_key in parquet_data:
                try:
                    frame = self._load_image_from_parquet(
                        parquet_data, cam_key, frame_offset, meta
                    )
                except Exception as e:
                    logger.warning(f"Failed to load image from parquet for {cam_key}: {e}")
            
            # If still no frame, create a placeholder
            if frame is None:
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
            target_h, target_w = self._default_image_size
            needs_resize = any(img.shape[1] != target_h or img.shape[2] != target_w for img in images)
            
            if needs_resize:
                images = torch.cat([
                    resize_with_pad(img.unsqueeze(0), height=target_h, width=target_w)
                    for img in images
                ], dim=0)
            else:
                images = torch.stack(images)
            
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
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        # v2.0 uses pre-computed stats in stats.json
        state_stats = self._get_stats_by_keys(
            self.dataset_metas[0].stats, self.state_key
        )
        action_stats = self._get_stats_by_keys(
            self.dataset_metas[0].stats, self.action_key
        )
        
        # Add q01/q99 if not present
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
        """Extract specific features from an episode."""
        dataset_idx = np.searchsorted(self.cumulative_num_episodes, episode_idx + 1)
        local_ep_list_idx = episode_idx - int(self.per_dataset_episode_start[dataset_idx])
        ep_idx = self.per_dataset_episodes[dataset_idx][local_ep_list_idx]
        
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
    print("Testing LeRobot v2.0 Wrapper")
    print("=" * 60)
    
    # Test with a sample dataset
    dataset = WrappedLerobotV20Dataset(
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

