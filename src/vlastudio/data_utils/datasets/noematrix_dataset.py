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
from typing import Optional, List, Dict, Any, Tuple, Union, Sequence
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data as tud
import pyarrow.parquet as pq
from PIL import Image
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download, HfApi
from loguru import logger
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation

from vlastudio.benchmark.utils import resize_with_pad
from vlastudio.data_utils.utils import ensure_uint8_image
import vlastudio.data_utils.pose_utils as pose_utils


# =============================================================================
# Constants
# =============================================================================

DEFAULT_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"
INFO_PATH = "meta/info.json"
EPISODES_PATH = "meta/episodes.jsonl"
STATS_PATH = "meta/stats.json"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
TASKS_PATH = "meta/tasks.jsonl"

CODEBASE_VERSION = "v2.1"


# =============================================================================
# Running Statistics (from openpi/shared/normalize.py)
# =============================================================================

class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            batch (np.ndarray): An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> Dict[str, np.ndarray]:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return {
            'mean': self._mean,
            'std': stddev,
            'q01': q01,
            'q99': q99,
            'min': self._min,
            'max': self._max,
        }

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


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
# Video Decoding (using PyAV)
# =============================================================================

def decode_video_frames_pyav(
    video_path: Path,
    timestamps: List[float],
    tolerance_s: float = 0.1,
) -> torch.Tensor:
    """
    Decode video frames at specified timestamps using PyAV.
    
    Args:
        video_path: Path to video file
        timestamps: List of timestamps (in seconds) to extract
        tolerance_s: Tolerance for timestamp matching
        
    Returns:
        Tensor of shape (N, C, H, W) with frames in [0, 1] range
    """
    import av
    
    video_path = str(video_path)
    container = av.open(video_path)
    stream = container.streams.video[0]
    
    # Get video properties
    fps = float(stream.average_rate)
    time_base = float(stream.time_base)
    
    # Sort timestamps
    sorted_ts = sorted(set(timestamps))
    first_ts = min(sorted_ts)
    last_ts = max(sorted_ts)
    
    # Seek to near the first timestamp
    # Convert timestamp to pts
    seek_ts = max(0, first_ts - 0.5)  # Seek a bit before
    container.seek(int(seek_ts / time_base), stream=stream)
    
    # Decode frames
    loaded_frames = []
    loaded_ts = []
    
    for frame in container.decode(stream):
        current_ts = float(frame.pts * time_base)
        
        # Skip frames before first requested timestamp
        if current_ts < first_ts - tolerance_s:
            continue
            
        # Convert frame to numpy
        img = frame.to_ndarray(format='rgb24')
        loaded_frames.append(img)
        loaded_ts.append(current_ts)
        
        # Stop if we've passed the last timestamp
        if current_ts >= last_ts + tolerance_s:
            break
    
    container.close()
    
    if not loaded_frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    
    # Match requested timestamps to loaded frames
    loaded_ts = np.array(loaded_ts)
    result_frames = []
    
    for ts in timestamps:
        # Find closest frame
        distances = np.abs(loaded_ts - ts)
        closest_idx = np.argmin(distances)
        
        if distances[closest_idx] > tolerance_s:
            warnings.warn(
                f"Frame at ts={ts:.4f} not found within tolerance. "
                f"Closest frame at ts={loaded_ts[closest_idx]:.4f}"
            )
        
        result_frames.append(loaded_frames[closest_idx])
    
    # Convert to tensor (N, H, W, C) -> (N, C, H, W)
    frames = np.stack(result_frames)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    
    return frames


# =============================================================================
# Dataset Metadata
# =============================================================================

class NoematrixMetadata:
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
    def image_keys(self) -> List[str]:
        """Get keys for image modalities (stored as images in parquet)."""
        return [key for key, ft in self.features.items() if ft.get("dtype") == "image"]
    
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

class NoematrixDataset(tud.Dataset):
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
        state_type: State output type - 'zero' returns all-zeros state (default, for UMI transform),
                   'raw' returns original state values without zeroing
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
        state_type: str = 'zero',
        *args,
        **kwargs,
    ):
        super().__init__()
        
        # Validate state_type
        if state_type not in ('zero', 'raw'):
            raise ValueError(f"state_type must be 'zero' or 'raw', got '{state_type}'")
        
        self.chunk_size = chunk_size
        self.root = Path(root) if root else DEFAULT_LEROBOT_HOME
        self.state_key = state_key
        self.action_key = action_key
        self.episode_filter = episode_filter
        self.tolerance_s = tolerance_s
        self.download_videos = download_videos
        self.camera_names = camera_names if isinstance(camera_names, list) else [camera_names]
        self.image_size = image_size
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.state_type = state_type
        
        # Load datasets
        self.dataset_path_list = dataset_path_list
        self.dataset_metas: List[NoematrixMetadata] = []
        self.dataset_dirs: List[str] = []
        self.per_dataset_episodes: List[List[int]] = []
        self.per_dataset_num_episodes: List[int] = []
        self.per_dataset_num_frames: List[int] = []
        
        for repo_id in dataset_path_list:
            # Load metadata
            ds_root = self.root / repo_id
            meta = NoematrixMetadata(repo_id, root=str(ds_root))
            
            # Filter episodes
            episodes = self._filter_episodes(meta, episode_filter)
            if episodes is None:
                episodes = list(range(meta.total_episodes))
            
            if len(episodes) == 0:
                warnings.warn(f"No episodes found for {repo_id} with filter {episode_filter}")
                continue
            
            # Download data files if needed
            self._ensure_data_downloaded(meta, episodes)
            
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
        
        # Initialize parquet cache
        self._parquet_cache: Dict[Tuple[int, int], Any] = {}
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
        
        # Cache for computed statistics (after transform)
        self._computed_stats: Optional[Dict[str, Any]] = None
        
        logger.info(
            f"LeRobot v2.1 Dataset loaded: {self.total_episodes} episodes, "
            f"{self.total_frames} frames from {len(self.dataset_metas)} dataset(s), "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"image_size={self._default_image_size}"
        )
    
    # =========================================================================
    # UMI Zero State and Real Relative Actions Transform
    # =========================================================================
    
    def _array_to_pose_and_gripper(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert array to pose matrices and gripper values.
        
        Args:
            x: Array of shape (T, 14) or (14,) containing:
               [left_pos(3), left_axis_angle(3), left_grip(1), 
                right_pos(3), right_axis_angle(3), right_grip(1)]
        
        Returns:
            Tuple of (left_pose_mat, left_grip, right_pose_mat, right_grip)
        """
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        
        left_pos = x[:, :3]
        left_axis_angle = x[:, 3:6]
        left_grip = x[:, 6:7]
        right_pos = x[:, 7:10]
        right_axis_angle = x[:, 10:13]
        right_grip = x[:, 13:14]
        
        left_pose = np.concatenate([left_pos, left_axis_angle], axis=1)
        right_pose = np.concatenate([right_pos, right_axis_angle], axis=1)
        
        left_pose_mat = pose_utils.pose_to_mat(left_pose)
        right_pose_mat = pose_utils.pose_to_mat(right_pose)
        
        return left_pose_mat, left_grip, right_pose_mat, right_grip
    
    def _transform_to_zero_state_relative_actions(
        self, 
        state: np.ndarray, 
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform absolute state and actions into zero state and relative actions space.
        
        This is the UmiZeroStateAndRealRelativeActions transform from openpi/transforms.py.
        
        IMPORTANT: This follows the exact same logic as openpi:
        1. Parse state (14,) and actions (T, 14) into pose matrices and grippers
        2. Convert action poses to relative poses based on state pose
        3. Convert rotation matrices to euler angles (xyz order)
        4. Merge: [left_pos(3), left_rpy(3), left_grip(1), right_pos(3), right_rpy(3), right_grip(1)]
        5. Set state to zeros AFTER action transformation
        
        Args:
            state: State array of shape (14,) - absolute pose
                   Format: [left_pos(3), left_axis_angle(3), left_grip(1),
                           right_pos(3), right_axis_angle(3), right_grip(1)]
            actions: Actions array of shape (T, 14) - absolute poses, same format as state
            
        Returns:
            Tuple of (transformed_state, transformed_actions)
            - transformed_state: zeros array of same shape as state (14,)
            - transformed_actions: relative actions (T, 14) in format 
              [left_rel_pos(3), left_rel_rpy(3), left_grip(1),
               right_rel_pos(3), right_rel_rpy(3), right_grip(1)]
        """
        # Parse state and actions into pose matrices and grippers
        # state: (14,) -> state_*_pose_mat: (1, 4, 4), state_*_grip: (1, 1)
        # actions: (T, 14) -> action_*_pose_mat: (T, 4, 4), action_*_grip: (T, 1)
        state_left_pose_mat, state_left_grip, state_right_pose_mat, state_right_grip = \
            self._array_to_pose_and_gripper(state)
        action_left_pose_mat, action_left_grip, action_right_pose_mat, action_right_grip = \
            self._array_to_pose_and_gripper(actions)
        
        # Convert to relative poses based on the state (state_*_pose_mat[0] is the base pose)
        # This computes: relative_pose = inv(base_pose) @ action_pose
        action_left_relative_pose_mat = pose_utils.convert_pose_mat_rep(
            action_left_pose_mat, state_left_pose_mat[0], "relative"
        )
        action_right_relative_pose_mat = pose_utils.convert_pose_mat_rep(
            action_right_pose_mat, state_right_pose_mat[0], "relative"
        )
        
        # Convert rotation matrices to euler angles (xyz order, i.e., roll-pitch-yaw)
        action_left_relative_rpy = Rotation.from_matrix(
            action_left_relative_pose_mat[:, :3, :3]
        ).as_euler('xyz', degrees=False)
        action_right_relative_rpy = Rotation.from_matrix(
            action_right_relative_pose_mat[:, :3, :3]
        ).as_euler('xyz', degrees=False)
        
        # Extract relative positions from transformation matrices
        action_left_relative_pos = action_left_relative_pose_mat[:, :3, 3]
        action_right_relative_pos = action_right_relative_pose_mat[:, :3, 3]
        
        # Merge actions: [left_pos(3), left_rpy(3), left_grip(1), right_pos(3), right_rpy(3), right_grip(1)]
        action_left = np.concatenate([
            action_left_relative_pos, action_left_relative_rpy, action_left_grip
        ], axis=1)
        action_right = np.concatenate([
            action_right_relative_pos, action_right_relative_rpy, action_right_grip
        ], axis=1)
        
        transformed_actions = np.concatenate([action_left, action_right], axis=1)
        
        # Set state to zeros AFTER action transformation (same as openpi line 285)
        transformed_state = np.zeros_like(state)
        
        return transformed_state, transformed_actions
    
    def _filter_episodes(
        self,
        meta: NoematrixMetadata,
        episode_filter: Optional[dict],
    ) -> Optional[List[int]]:
        """Filter episodes based on metadata fields."""
        if episode_filter is None or len(episode_filter) == 0:
            return None
        
        if not meta.episodes:
            warnings.warn("Dataset metadata does not contain episodes information")
            return None
        
        # Direct episode indices
        if "episode_index" in episode_filter:
            indices = episode_filter["episode_index"]
            if isinstance(indices, (list, tuple)):
                valid = [idx for idx in indices if idx in meta.episodes]
                return valid
            return None
        
        # Start with all episodes
        selected = set(meta.episodes.keys())
        
        for filter_key, filter_values in episode_filter.items():
            if filter_key == "tasks":
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
    
    def _ensure_data_downloaded(self, meta: NoematrixMetadata, episodes: List[int]):
        """Ensure parquet and video files are downloaded."""
        # Check which files need to be downloaded
        missing_parquet = []
        missing_videos = []
        
        for ep_idx in episodes:
            # Check parquet file
            parquet_path = meta.root / meta.get_data_file_path(ep_idx)
            if not parquet_path.exists():
                missing_parquet.append(ep_idx)
            
            # Check video files
            if self.download_videos:
                for vid_key in meta.video_keys:
                    video_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        missing_videos.append((ep_idx, vid_key))
        
        if not missing_parquet and not missing_videos:
            return  # All files already exist
        
        # Build chunk-based patterns for more efficient downloading
        # Group by chunk to reduce number of patterns
        chunks_needed = set()
        for ep_idx in missing_parquet:
            chunks_needed.add(meta.get_episode_chunk(ep_idx))
        for ep_idx, _ in missing_videos:
            chunks_needed.add(meta.get_episode_chunk(ep_idx))
        
        # If we need most chunks, just download everything
        # Otherwise, use chunk-based patterns
        total_chunks = meta.info.get("total_chunks", 1)
        
        if len(chunks_needed) > total_chunks * 0.5:
            # Download all data and videos
            logger.info(f"Downloading dataset {meta.repo_id} (need {len(chunks_needed)}/{total_chunks} chunks)...")
            allow_patterns = ["data/**", "videos/**"] if self.download_videos else ["data/**"]
            snapshot_download(
                meta.repo_id,
                repo_type="dataset",
                revision=meta.revision,
                local_dir=meta.root,
                allow_patterns=allow_patterns,
            )
        else:
            # Download specific chunks
            allow_patterns = []
            for chunk_idx in sorted(chunks_needed):
                allow_patterns.append(f"data/chunk-{chunk_idx:03d}/*")
                if self.download_videos:
                    # Include all subdirectories for videos in this chunk
                    allow_patterns.append(f"videos/chunk-{chunk_idx:03d}/**")
            
            logger.info(f"Downloading {len(chunks_needed)} chunk(s) for {meta.repo_id}...")
            snapshot_download(
                meta.repo_id,
                repo_type="dataset",
                revision=meta.revision,
                local_dir=meta.root,
                allow_patterns=allow_patterns,
            )
        
        # Verify critical files were downloaded (check a sample)
        if missing_parquet and len(missing_parquet) > 0:
            sample_ep = missing_parquet[0]
            parquet_path = meta.root / meta.get_data_file_path(sample_ep)
            if not parquet_path.exists():
                raise FileNotFoundError(
                    f"Critical parquet file still missing after download: {parquet_path}. "
                    f"This may indicate the dataset structure doesn't match expectations."
                )
        
        if missing_videos and len(missing_videos) > 0 and self.download_videos:
            # Check a sample video file
            sample_ep, sample_vid_key = missing_videos[0]
            video_path = meta.root / meta.get_video_file_path(sample_ep, sample_vid_key)
            if not video_path.exists():
                logger.warning(
                    f"Sample video file still missing after download: {video_path}. "
                    f"Some videos may be missing from the dataset. "
                    f"The loader will use placeholders for missing videos."
                )
    
    def _build_index_mapping(self):
        """Build index mapping for fast sample lookup."""
        self.index_to_sample_map = []  # (dataset_idx, episode_idx, frame_offset)
        
        for dataset_idx, (meta, episodes) in enumerate(
            zip(self.dataset_metas, self.per_dataset_episodes)
        ):
            for ep_idx in episodes:
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
    
    @staticmethod
    def _hf_transform_to_torch(items_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform function that converts items from HuggingFace dataset (pyarrow)
        to torch tensors. Similar to lerobot's hf_transform_to_torch.
        
        Images are converted from PIL (h w c, uint8) to torch (c h w, float32 in [0,1]).
        """
        to_tensor = transforms.ToTensor()
        
        for key in items_dict:
            values = items_dict[key]
            if not values:
                continue
                
            first_item = values[0]
            
            if isinstance(first_item, Image.Image):
                # Convert PIL Images to tensors
                items_dict[key] = [to_tensor(img) for img in values]
            elif isinstance(first_item, dict) and "bytes" in first_item:
                # Handle embedded image bytes (some datasets store images as dicts with bytes)
                converted = []
                for item in values:
                    if isinstance(item, dict) and "bytes" in item:
                        try:
                            from io import BytesIO
                            img = Image.open(BytesIO(item["bytes"])).convert('RGB')
                            converted.append(to_tensor(img))
                        except Exception:
                            converted.append(item)
                    else:
                        converted.append(item)
                items_dict[key] = converted
            elif first_item is None:
                # Keep None values as-is
                pass
            elif isinstance(first_item, str):
                # Keep strings as-is
                pass
            else:
                # Convert other types to torch tensors
                items_dict[key] = [
                    x if isinstance(x, str) else torch.tensor(x) 
                    for x in values
                ]
        
        return items_dict
    
    def _load_episode_parquet(self, dataset_idx: int, ep_idx: int) -> Dataset:
        """Load parquet data for an episode using datasets library for proper image handling."""
        cache_key = (dataset_idx, ep_idx)
        
        if cache_key not in self._parquet_cache:
            meta = self.dataset_metas[dataset_idx]
            parquet_path = meta.root / meta.get_data_file_path(ep_idx)
            
            # Set cache directory explicitly to avoid permission issues
            # Priority: HF_DATASETS_CACHE > HF_HOME parent/datasets > default ~/.cache/huggingface/datasets
            if "HF_DATASETS_CACHE" in os.environ:
                cache_dir = os.environ["HF_DATASETS_CACHE"]
            elif "HF_HOME" in os.environ:
                # Use HF_HOME's parent directory for datasets cache (HF_HOME is usually .../hub)
                hf_home = Path(os.environ["HF_HOME"])
                cache_dir = str(hf_home.parent / "datasets")
            else:
                # Use default huggingface datasets cache location
                cache_dir = str(Path.home() / ".cache" / "huggingface" / "datasets")
            
            # Ensure cache directory exists (will raise error if not writable, which is expected)
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Use datasets library to load parquet, which automatically handles image types
            # Explicitly set cache_dir to avoid issues with path-based cache inference
            # This prevents datasets library from trying to create cache in /inspire or other paths
            # Disable progress bars and logging to avoid "Generating train split" messages
            import datasets
            datasets.disable_progress_bar()
            
            hf_dataset = load_dataset(
                "parquet",
                data_files=str(parquet_path),
                split="train",
                cache_dir=cache_dir,
            )
            
            # Re-enable progress bar for other operations
            datasets.enable_progress_bar()
            
            # Use set_transform (like lerobot) instead of map
            # set_transform applies transformation dynamically when accessing items
            # This properly handles PIL Images from datasets library
            hf_dataset.set_transform(self._hf_transform_to_torch)
            
            self._parquet_cache[cache_key] = hf_dataset
        
        return self._parquet_cache[cache_key]
    
    def _load_video_frame(
        self,
        meta: NoematrixMetadata,
        ep_idx: int,
        vid_key: str,
        timestamp: float,
    ) -> Optional[torch.Tensor]:
        """
        Load a single frame from video file.
        
        Args:
            meta: Dataset metadata
            ep_idx: Episode index
            vid_key: Video key (camera name)
            timestamp: Timestamp for frame extraction
            
        Returns:
            Frame tensor (C, H, W) in range [0, 1] or None if not available
        """
        video_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
        
        if video_path.exists():
            try:
                frames = decode_video_frames_pyav(video_path, [timestamp], self.tolerance_s)
                return frames[0]  # (C, H, W) in range [0, 1]
            except Exception as e:
                logger.warning(f"Failed to decode video frame from {video_path}: {e}")
                return None
        
        # Video file doesn't exist
        if not hasattr(self, '_video_warned'):
            logger.warning(
                f"Video file not found: {video_path}. "
                f"This may happen if the video wasn't downloaded or doesn't exist in the dataset. "
                f"Will try to use placeholder or skip this camera."
            )
            self._video_warned = True
        
        return None
    
    def _extract_tensor_from_item(
        self,
        item: Dict[str, Any],
        keys: Union[str, List[str]],
        fallback_key: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """
        Extract tensor data from a single item (frame) from hf_dataset.
        Supports single key or list of keys to concatenate.
        """
        if isinstance(keys, str):
            # Single key
            data = item.get(keys)
            if data is None and fallback_key:
                data = item.get(fallback_key)
            if data is None:
                return None
            if isinstance(data, torch.Tensor):
                return data.float()
            return torch.tensor(data, dtype=torch.float32)
        else:
            # List of keys - concatenate along last axis
            data_parts = []
            for key in keys:
                part = item.get(key)
                if part is None:
                    continue
                if isinstance(part, torch.Tensor):
                    data_parts.append(part.float())
                else:
                    data_parts.append(torch.tensor(part, dtype=torch.float32))
            
            if not data_parts:
                # Try fallback
                if fallback_key:
                    data = item.get(fallback_key)
                    if data is not None:
                        if isinstance(data, torch.Tensor):
                            return data.float()
                        return torch.tensor(data, dtype=torch.float32)
                return None
            
            # Concatenate along last axis
            return torch.cat(data_parts, dim=-1)
    
    def _extract_tensor_from_episode(
        self,
        hf_dataset: Dataset,
        keys: Union[str, List[str]],
        fallback_key: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """
        Extract tensor data from full episode hf_dataset.
        Supports single key or list of keys to concatenate.
        """
        # Convert Dataset to dict format for easier access
        dataset_dict = hf_dataset.to_dict()
        
        if isinstance(keys, str):
            # Single key
            if keys in dataset_dict:
                data_list = dataset_dict[keys]
                # Convert list of values to tensor
                tensors = []
                for x in data_list:
                    if isinstance(x, torch.Tensor):
                        tensors.append(x.float())
                    else:
                        tensors.append(torch.tensor(x, dtype=torch.float32))
                if tensors:
                    try:
                        return torch.stack(tensors)
                    except:
                        # If shapes don't match, return as list of tensors (will handle in caller)
                        return torch.tensor(data_list, dtype=torch.float32)
            elif fallback_key and fallback_key in dataset_dict:
                data_list = dataset_dict[fallback_key]
                tensors = []
                for x in data_list:
                    if isinstance(x, torch.Tensor):
                        tensors.append(x.float())
                    else:
                        tensors.append(torch.tensor(x, dtype=torch.float32))
                if tensors:
                    try:
                        return torch.stack(tensors)
                    except:
                        return torch.tensor(data_list, dtype=torch.float32)
            return None
        else:
            # List of keys - concatenate along last axis
            data_parts = []
            for key in keys:
                if key in dataset_dict:
                    data_list = dataset_dict[key]
                    tensors = []
                    for x in data_list:
                        if isinstance(x, torch.Tensor):
                            tensors.append(x.float())
                        else:
                            tensors.append(torch.tensor(x, dtype=torch.float32))
                    if tensors:
                        try:
                            part = torch.stack(tensors)
                        except:
                            part = torch.tensor(data_list, dtype=torch.float32)
                        data_parts.append(part)
            
            if not data_parts:
                # Try fallback
                if fallback_key and fallback_key in dataset_dict:
                    data_list = dataset_dict[fallback_key]
                    tensors = []
                    for x in data_list:
                        if isinstance(x, torch.Tensor):
                            tensors.append(x.float())
                        else:
                            tensors.append(torch.tensor(x, dtype=torch.float32))
                    if tensors:
                        try:
                            return torch.stack(tensors)
                        except:
                            return torch.tensor(data_list, dtype=torch.float32)
                return None
            
            # Concatenate along last axis (assuming shape is [T, D] for each part)
            # Each part should have shape [T, D_i], concatenate to [T, sum(D_i)]
            return torch.cat(data_parts, dim=-1)
    
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
                data_parts.append(np.asarray(part))
            
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
            if frame_idx is not None:
                # Single frame: concatenate 1D arrays
                return np.concatenate(data_parts, axis=-1)
            else:
                # Full episode: concatenate along last axis (T, D1), (T, D2) -> (T, D1+D2)
                return np.concatenate(data_parts, axis=-1)
    
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
        """Get a sample from the dataset. Reference lerobot implementation."""
        dataset_idx, ep_idx, frame_offset = self.index_to_sample_map[index]
        meta = self.dataset_metas[dataset_idx]
        
        # Load hf_dataset for this episode (contains all frames including images)
        hf_dataset = self._load_episode_parquet(dataset_idx, ep_idx)
        
        # Get current frame from hf_dataset (similar to lerobot's item = self.hf_dataset[idx])
        item = hf_dataset[frame_offset]
        frame_index = frame_offset
        timestamp = float(item.get('timestamp', 0))
        if isinstance(timestamp, torch.Tensor):
            timestamp = timestamp.item()
        
        # Get state (supports single key or list of keys to concatenate)
        state = self._extract_tensor_from_item(item, self.state_key, fallback_key='observation.state')
        if state is None:
            state = torch.zeros(7, dtype=torch.float32)
        
        # Get action from current frame
        # This dataset has action already in chunk format: (max_chunk_size, action_dim), e.g., (100, 14)
        # We just need to truncate to (chunk_size, action_dim)
        action_chunk = self._extract_tensor_from_item(item, self.action_key, fallback_key='action')
        
        if action_chunk is not None:
            # action_chunk shape: (max_chunk_size, action_dim), e.g., (100, 14)
            max_chunk_size = action_chunk.shape[0]
            
            # Validate chunk_size doesn't exceed max_chunk_size
            if self.chunk_size > max_chunk_size:
                logger.warning(
                    f"chunk_size ({self.chunk_size}) exceeds max_chunk_size ({max_chunk_size}) in dataset. "
                    f"Truncating to {max_chunk_size}."
                )
                effective_chunk_size = max_chunk_size
            else:
                effective_chunk_size = self.chunk_size
            
            # Truncate to chunk_size
            action = action_chunk[:effective_chunk_size]  # (chunk_size, action_dim)
            
            # Create is_pad mask (all False since we have valid actions)
            is_pad = torch.zeros(effective_chunk_size, dtype=torch.bool)
            
            # Pad if chunk_size > effective_chunk_size (shouldn't happen normally)
            if self.chunk_size > effective_chunk_size:
                pad_size = self.chunk_size - effective_chunk_size
                action_dim = action.shape[-1]
                # Pad with last action
                last_action = action[-1:].expand(pad_size, -1)
                action = torch.cat([action, last_action], dim=0)
                is_pad = torch.cat([is_pad, torch.ones(pad_size, dtype=torch.bool)], dim=0)
        else:
            # Fallback: create zero action
            action_dim = state.shape[-1] if state is not None else 14
            action = torch.zeros(self.chunk_size, action_dim, dtype=torch.float32)
            is_pad = torch.ones(self.chunk_size, dtype=torch.bool)
        
        # Ensure action is float32
        action = action.float()
        
        # Apply UMI relative actions transform
        # This converts absolute poses to relative actions
        # Following openpi/transforms.py UmiZeroStateAndRealRelativeActions
        state_np = state.numpy() if isinstance(state, torch.Tensor) else np.array(state)
        action_np = action.numpy() if isinstance(action, torch.Tensor) else np.array(action)
        
        # Store original state for 'raw' mode
        original_state = state.clone()
        
        try:
            transformed_state, transformed_action = self._transform_to_zero_state_relative_actions(
                state_np, action_np
            )
            # Action is always transformed to relative
            action = torch.from_numpy(transformed_action).float()
            # State depends on state_type setting
            if self.state_type == 'zero':
                state = torch.from_numpy(transformed_state).float()  # zeros
            else:  # state_type == 'raw'
                state = original_state  # keep original
        except Exception as e:
            # If transform fails (e.g., wrong dimension), log warning
            if not hasattr(self, '_transform_warning_logged'):
                logger.warning(
                    f"Failed to apply UMI transform: {e}. "
                    f"State shape: {state_np.shape}, Action shape: {action_np.shape}. "
                    f"Keeping original action."
                )
                self._transform_warning_logged = True
            # For state: zero out if state_type is 'zero', otherwise keep original
            if self.state_type == 'zero':
                state = torch.zeros_like(state)
        
        # Get task/language instruction
        task_idx = int(item.get('task_index', 0))
        if isinstance(task_idx, torch.Tensor):
            task_idx = task_idx.item()
        raw_lang = meta.tasks.get(task_idx, "")
        
        # Load images following lerobot's approach:
        # 1. Image keys (dtype == "image") are already loaded from hf_dataset (embedded in parquet)
        # 2. Video keys (dtype == "video") need to be loaded from video files
        # If camera_names is specified, use that; otherwise use all camera_keys from metadata
        cam_keys = meta.camera_keys if len(self.camera_names) == 0 else self.camera_names
        images = []
        
        for cam_key in cam_keys:
            frame = None
            
            # First, try to get from hf_dataset item (works for image keys)
            # The transform already converted PIL Images to tensors
            if cam_key in item:
                frame = item[cam_key]
                if isinstance(frame, torch.Tensor):
                    # Already a tensor in [0, 1] range with shape (C, H, W)
                    pass
                elif isinstance(frame, Image.Image):
                    # Convert PIL Image to tensor (shouldn't happen if transform worked)
                    to_tensor = transforms.ToTensor()
                    frame = to_tensor(frame)
                elif isinstance(frame, list) and len(frame) > 0:
                    # If it's a list (batched), take first element
                    first = frame[0]
                    if isinstance(first, torch.Tensor):
                        frame = first
                    elif isinstance(first, Image.Image):
                        to_tensor = transforms.ToTensor()
                        frame = to_tensor(first)
                    else:
                        frame = None
                else:
                    frame = None
            
            # For video keys, load from video file (similar to lerobot's _query_videos)
            # This overrides any parquet data for video keys
            if cam_key in meta.video_keys:
                video_frame = self._load_video_frame(
                    meta, ep_idx, cam_key, timestamp
                )
                if video_frame is not None:
                    frame = video_frame
            
            # If still no frame, create a placeholder (black image)
            if frame is None:
                # Use default image size for placeholder
                h, w = self._default_image_size
                frame = torch.zeros(3, h, w, dtype=torch.float32)
                if not hasattr(self, '_placeholder_warned'):
                    logger.warning(
                        f"Using placeholder (black) image for camera '{cam_key}' "
                        f"in episode {ep_idx}. Video file may be missing. "
                        f"Available keys in item: {list(item.keys())}"
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
    
    def _create_stats_only_dataset(self) -> tud.Dataset:
        """
        Create a lightweight dataset that only loads state and action data (no images).
        This is used for fast statistics computation.
        """
        class StatsOnlyDataset(tud.Dataset):
            """Lightweight dataset for statistics computation - no image loading."""
            
            def __init__(inner_self, parent: 'NoematrixDataset'):
                inner_self.parent = parent
                inner_self.index_to_sample_map = parent.index_to_sample_map
                inner_self.dataset_metas = parent.dataset_metas
                inner_self.state_key = parent.state_key
                inner_self.action_key = parent.action_key
                inner_self.chunk_size = parent.chunk_size
                inner_self.state_type = parent.state_type
                # Create a separate parquet cache for stats computation
                inner_self._stats_parquet_cache: Dict[Tuple[int, int], Any] = {}
            
            def __len__(inner_self):
                return len(inner_self.index_to_sample_map)
            
            def _load_parquet_for_stats(inner_self, dataset_idx: int, ep_idx: int) -> Dict[str, np.ndarray]:
                """Load parquet data without image columns for faster loading."""
                cache_key = (dataset_idx, ep_idx)
                
                if cache_key not in inner_self._stats_parquet_cache:
                    meta = inner_self.dataset_metas[dataset_idx]
                    parquet_path = meta.root / meta.get_data_file_path(ep_idx)
                    
                    # Read parquet with pyarrow directly (faster, no image decoding)
                    table = pq.read_table(parquet_path)
                    
                    # Only keep columns we need (state and action keys)
                    needed_cols = set()
                    state_keys = [inner_self.state_key] if isinstance(inner_self.state_key, str) else inner_self.state_key
                    action_keys = [inner_self.action_key] if isinstance(inner_self.action_key, str) else inner_self.action_key
                    needed_cols.update(state_keys)
                    needed_cols.update(action_keys)
                    
                    # Filter columns that exist in the table
                    available_cols = [col for col in needed_cols if col in table.column_names]
                    
                    data = {}
                    for col in available_cols:
                        arr = table[col].to_numpy()
                        # Handle nested arrays
                        if arr.dtype == object:
                            try:
                                arr = np.stack(arr)
                            except:
                                pass
                        data[col] = arr
                    
                    inner_self._stats_parquet_cache[cache_key] = data
                
                return inner_self._stats_parquet_cache[cache_key]
            
            def _to_numpy_array(inner_self, data) -> np.ndarray:
                """Convert data to a proper numpy array, handling nested structures."""
                if isinstance(data, np.ndarray):
                    if data.dtype == object:
                        # Handle object arrays (nested lists/arrays)
                        try:
                            return np.array(data.tolist(), dtype=np.float32)
                        except:
                            return np.stack([np.asarray(x, dtype=np.float32) for x in data])
                    return data.astype(np.float32)
                elif isinstance(data, (list, tuple)):
                    return np.array(data, dtype=np.float32)
                else:
                    return np.asarray(data, dtype=np.float32)
            
            def __getitem__(inner_self, index: int) -> Dict[str, np.ndarray]:
                dataset_idx, ep_idx, frame_offset = inner_self.index_to_sample_map[index]
                
                # Load parquet data (no images)
                parquet_data = inner_self._load_parquet_for_stats(dataset_idx, ep_idx)
                
                # Extract state
                state_keys = [inner_self.state_key] if isinstance(inner_self.state_key, str) else inner_self.state_key
                state_parts = []
                for key in state_keys:
                    if key in parquet_data:
                        raw_data = parquet_data[key][frame_offset]
                        state_parts.append(inner_self._to_numpy_array(raw_data).flatten())
                
                if state_parts:
                    state = np.concatenate(state_parts, axis=-1).astype(np.float32)
                else:
                    state = np.zeros(14, dtype=np.float32)
                
                # Extract action chunk
                action_keys = [inner_self.action_key] if isinstance(inner_self.action_key, str) else inner_self.action_key
                action_parts = []
                for key in action_keys:
                    if key in parquet_data:
                        raw_data = parquet_data[key][frame_offset]
                        arr = inner_self._to_numpy_array(raw_data)
                        # Ensure 2D: (chunk_size, action_dim)
                        if arr.ndim == 1:
                            arr = arr.reshape(1, -1)
                        action_parts.append(arr)
                
                if action_parts:
                    # Concatenate along action dimension
                    action_chunk = np.concatenate(action_parts, axis=-1)
                    # Truncate to chunk_size
                    max_chunk = action_chunk.shape[0]
                    effective_chunk = min(inner_self.chunk_size, max_chunk)
                    action = action_chunk[:effective_chunk]
                    
                    # Pad if needed
                    if inner_self.chunk_size > effective_chunk:
                        pad_size = inner_self.chunk_size - effective_chunk
                        last_action = action[-1:]
                        padding = np.repeat(last_action, pad_size, axis=0)
                        action = np.concatenate([action, padding], axis=0)
                    
                    action = action.astype(np.float32)
                else:
                    action = np.zeros((inner_self.chunk_size, 14), dtype=np.float32)
                
                # Apply UMI transform (following openpi/transforms.py)
                # Store original state for 'raw' mode
                original_state = state.copy()
                
                try:
                    transformed_state, transformed_action = inner_self.parent._transform_to_zero_state_relative_actions(
                        state, action
                    )
                    # Action is always transformed to relative
                    action = transformed_action
                    # State depends on state_type setting
                    if inner_self.state_type == 'zero':
                        state = transformed_state  # zeros
                    else:  # state_type == 'raw'
                        state = original_state  # keep original
                except Exception:
                    # If transform fails, handle state based on state_type
                    if inner_self.state_type == 'zero':
                        state = np.zeros_like(state)
                    # else keep original state
                
                return {'state': state.astype(np.float32), 'action': action.astype(np.float32)}
        
        return StatsOnlyDataset(self)
    
    def get_dataset_statistics(self, batch_size: int = 512, num_workers: int = 4) -> Dict[str, Any]:
        """
        Get dataset statistics by iterating through the entire dataset.
        
        Since this dataset applies UMI transform (zero state + relative actions),
        we cannot use pre-computed statistics from the dataset. Instead, we compute
        statistics by iterating through all samples and applying the transform.
        
        This optimized version:
        - Uses a lightweight dataset that skips image loading
        - Uses PyTorch DataLoader with multiple workers for parallel loading
        - Processes data in batches for efficiency
        
        Args:
            batch_size: Batch size for computing statistics
            num_workers: Number of workers for data loading
            
        Returns:
            Dictionary with 'state' and 'action' statistics, plus metadata
        """
        # Return cached stats if already computed
        if self._computed_stats is not None:
            return self._computed_stats
        
        logger.info(
            f"Computing dataset statistics by iterating through {len(self)} samples "
            f"(batch_size={batch_size}, num_workers={num_workers})..."
        )
        
        # Initialize running statistics
        state_stats = RunningStats()
        action_stats = RunningStats()
        
        # Create lightweight dataset and dataloader
        stats_dataset = self._create_stats_only_dataset()
        
        # Custom collate function
        def collate_fn(batch):
            states = np.stack([item['state'] for item in batch], axis=0)
            actions = np.stack([item['action'] for item in batch], axis=0)
            return {'state': states, 'action': actions}
        
        dataloader = tud.DataLoader(
            stats_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            persistent_workers=False if num_workers == 0 else True,
        )
        
        try:
            for batch in tqdm(dataloader, desc="Computing statistics"):
                batch_states = batch['state']  # (B, state_dim)
                batch_actions = batch['action']  # (B, chunk_size, action_dim)
                
                # Flatten actions: (B, chunk_size, action_dim) -> (B * chunk_size, action_dim)
                batch_actions_flat = batch_actions.reshape(-1, batch_actions.shape[-1])
                
                state_stats.update(batch_states)
                action_stats.update(batch_actions_flat)
        finally:
            # Clean up dataloader and dataset
            del dataloader
            del stats_dataset
        
        # Get final statistics
        try:
            state_result = state_stats.get_statistics()
            action_result = action_stats.get_statistics()
        except ValueError as e:
            logger.error(f"Failed to compute statistics: {e}")
            return {
                'state': {},
                'action': {},
                'num_episodes': self.total_episodes,
                'num_transitions': self.total_frames,
            }
        
        # Cache the results
        self._computed_stats = {
            'state': state_result,
            'action': action_result,
            'num_episodes': self.total_episodes,
            'num_transitions': self.total_frames,
        }
        
        logger.info("Dataset statistics computation completed.")
        
        return self._computed_stats
    
    def extract_from_episode(self, episode_idx: int, keyname: List[str] = []) -> Dict[str, np.ndarray]:
        """Extract specific features from an episode. Supports concatenated keys."""
        # Find which dataset and episode
        dataset_idx = np.searchsorted(self.cumulative_num_episodes, episode_idx + 1)
        local_ep_list_idx = episode_idx - int(self.per_dataset_episode_start[dataset_idx])
        ep_idx = self.per_dataset_episodes[dataset_idx][local_ep_list_idx]
        
        # Load hf_dataset for this episode
        hf_dataset = self._load_episode_parquet(dataset_idx, ep_idx)
        
        result = {}
        
        if 'state' in keyname:
            state_data = self._extract_tensor_from_episode(
                hf_dataset, self.state_key,
                fallback_key='observation.state'
            )
            if state_data is not None:
                result['state'] = state_data.cpu().numpy() if isinstance(state_data, torch.Tensor) else np.array(state_data)
        
        if 'action' in keyname:
            action_data = self._extract_tensor_from_episode(
                hf_dataset, self.action_key,
                fallback_key='action'
            )
            if action_data is not None:
                result['action'] = action_data.cpu().numpy() if isinstance(action_data, torch.Tensor) else np.array(action_data)
        
        return result


if __name__ == '__main__':
    # Test the wrapper
    print("=" * 60)
    print("Testing LeRobot v2.1 Wrapper")
    print("=" * 60)
    
    # Test with a sample dataset
    dataset = NoematrixDataset(
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

