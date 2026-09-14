
# --- Mean-Variance Normalization (also known as Standardization or Z-Score Normalization) ---
#
# Description:
#   This method rescales the data to have a mean of 0 and a standard deviation of 1.
#   It is the most common normalization technique, especially for data that is
#   approximately Gaussian (normally distributed).
#
# Formula:
#   x_normalized = (x - mean) / (std_dev + epsilon)
#
# Where:
#   - `mean` is the mean of the training dataset.
#   - `std_dev` is the standard deviation of the training dataset.
#   - `epsilon` is a small constant (e.g., 1e-8) to prevent division by zero.
#
# Usage:
#   Typically, the mean and std_dev are pre-computed from a large, representative
#   offline dataset and then stored. These same values are used for normalization
#   during both training and inference.


# --- Percentile Normalization (Robust Normalization) ---
#
# Description:
#   This method is robust to outliers and non-Gaussian distributions. Instead of
#   using mean/std_dev, it normalizes based on percentile ranks within the data distribution.
#   It works by clipping the data to a specified percentile range and then scaling it
#   to a target interval, typically [-1, 1].
#
# A common implementation involves these steps:
#   1. Determine percentile boundaries from a representative dataset (e.g., 5th and 95th).
#   2. Clip incoming data to these boundaries.
#   3. Linearly scale the clipped value to the target range (e.g., [-1, 1]).
#
# Formula (for scaling to [-1, 1] using p5 and p95):
#   x_clipped = clip(x, p5, p95)
#   x_normalized = 2 * (x_clipped - p5) / (p95 - p5) - 1
#
# Where:
#   - `p5` is the 5th percentile of the dataset.
#   - `p95` is the 95th percentile of the dataset.
#
# Usage:
#   Excellent for actions or states with skewed distributions or extreme outliers,
#   such as force sensor readings or tool velocities.

# --- Min-Max Normalization ---
#
# Description:
#   This method linearly rescales the data to a fixed range, most commonly [0, 1] or [-1, 1].
#   It is useful when the upper and lower bounds of the data are known or fixed.
#
# Formula (for scaling to [0, 1]):
#   x_normalized = (x - min_val) / (max_val - min_val)
#
# Formula (for scaling to [-1, 1]):
#   x_normalized = 2 * (x - min_val) / (max_val - min_val) - 1
#
# Where:
#   - `min_val` is the minimum value of the dataset.
#   - `max_val` is the maximum value of the dataset.
#
# Usage:
#   Suitable for data with well-defined boundaries, such as robot joint angles (which have
#   physical limits) or image pixel intensities.
#   Note: This method is highly sensitive to outliers.

# ----------------------- h5 structure
# # ACTION
# joint_action: (T, qpos_dim)
# ee_action: (T, ee_dim)
#
# # OBSERVATIONS
# observations:
# 	joint_pos: (T, qpos_dim)
# 	ee_pos: (T, ee_dim)
#   ...

import pickle
import numpy as np
import os
from collections import defaultdict
from benchmark.base import MetaAction, MetaObs  
from typing import List
import fnmatch
import hashlib
import warnings
import torch.distributed as dist
import torch
import json
from loguru import logger

def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


MIN_SCALE_THRESHOLD = 1e-2


class RunningStats:
    """Compute statistics in a streaming/running manner using Welford's algorithm.
    
    This class enables memory-efficient statistics computation for large datasets
    by processing data incrementally without storing all samples.
    
    Also tracks E[X²] (square_mean) for cross-dataset aggregation:
        Combined mean = (n1*mean1 + n2*mean2) / (n1 + n2)
        Combined E[X²] = (n1*square_mean1 + n2*square_mean2) / (n1 + n2)
        Combined std = sqrt(Combined E[X²] - Combined mean²)
    """
    
    def __init__(self, shape: tuple):
        """Initialize running statistics tracker.
        
        Args:
            shape: Shape of each data sample (excluding batch dimension)
        """
        self.shape = shape
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)  # Sum of squared differences from mean
        self.sum_squares = np.zeros(shape, dtype=np.float64)  # Sum of x^2 for square_mean
        self.min_val = np.full(shape, np.inf, dtype=np.float64)
        self.max_val = np.full(shape, -np.inf, dtype=np.float64)
        
        # Reservoir sampling for percentiles
        self.reservoir_size = 10000
        self.reservoir = []
        self.seen = 0
    
    def update(self, x: np.ndarray):
        """Update statistics with a new sample or batch of samples.
        
        Args:
            x: Data array. Can be single sample of shape `self.shape` or
               batch of shape (batch_size, *self.shape)
        """
        x = np.asarray(x, dtype=np.float64)
        
        # Handle batch dimension
        if x.shape == self.shape:
            x = x.reshape(1, *self.shape)
        
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            sample = x[i]
            self.n += 1
            
            # Welford's algorithm for mean and variance
            delta = sample - self.mean
            self.mean += delta / self.n
            delta2 = sample - self.mean
            self.M2 += delta * delta2
            
            # Track sum of squares for square_mean
            self.sum_squares += sample ** 2
            
            # Min/max
            self.min_val = np.minimum(self.min_val, sample)
            self.max_val = np.maximum(self.max_val, sample)
            
            # Reservoir sampling for percentiles
            self.seen += 1
            if len(self.reservoir) < self.reservoir_size:
                self.reservoir.append(sample.copy())
            else:
                # Replace with probability reservoir_size/seen
                j = np.random.randint(0, self.seen)
                if j < self.reservoir_size:
                    self.reservoir[j] = sample.copy()
    
    def get_stats(self) -> dict:
        """Get computed statistics.
        
        Returns:
            Dictionary with keys: mean, square_mean, std, min, max, q01, q99, count
        """
        if self.n == 0:
            raise ValueError("No data has been added")
        
        mean = self.mean.astype(np.float32)
        square_mean = (self.sum_squares / self.n).astype(np.float32)
        
        # Variance from M2 (Welford's algorithm)
        variance = self.M2 / self.n
        std = np.sqrt(np.maximum(variance, 0)).astype(np.float32)
        
        # Percentiles from reservoir
        if self.reservoir:
            reservoir_array = np.array(self.reservoir)
            q01 = np.percentile(reservoir_array, 1, axis=0).astype(np.float32)
            q99 = np.percentile(reservoir_array, 99, axis=0).astype(np.float32)
        else:
            q01 = mean.copy()
            q99 = mean.copy()
        
        return {
            "mean": mean,
            "square_mean": square_mean,
            "std": std,
            "min": self.min_val.astype(np.float32),
            "max": self.max_val.astype(np.float32),
            "q01": q01,
            "q99": q99,
            "count": np.array(self.n, dtype=np.int64),
        }

def str2hash(s: str):
    return str(hashlib.md5(s.encode()).hexdigest())


def merge_stats(stats_list: list, keys: list = None) -> dict:
    """Merge statistics from multiple datasets using count-weighted averaging.
    
    This function combines statistics from multiple datasets into a single set
    of statistics, using the sample counts for weighted averaging.
    
    For merging to work correctly, each stats dict must contain:
    - 'count' or be available in parent dict as 'num_transitions'
    - 'mean', 'square_mean', 'std', 'min', 'max', 'q01', 'q99'
    
    The merge formulas are:
    - Combined count: n = n1 + n2 + ...
    - Combined mean: E[X] = (n1*mean1 + n2*mean2 + ...) / n
    - Combined E[X²]: E[X²] = (n1*square_mean1 + n2*square_mean2 + ...) / n
    - Combined std: std = sqrt(E[X²] - E[X]²)
    - Combined min: min of all mins
    - Combined max: max of all maxs
    - Combined q01/q99: weighted average (approximation)
    
    Args:
        stats_list: List of stats dictionaries. Each dict should have format:
            {
                'num_transitions': int,  # or 'count' inside each key's stats
                'state': {'mean': ..., 'square_mean': ..., 'std': ..., ...},
                'action': {'mean': ..., 'square_mean': ..., 'std': ..., ...},
            }
        keys: Optional list of keys to merge (e.g., ['state', 'action']).
              If None, auto-detect from first stats dict.
    
    Returns:
        Merged stats dictionary with the same structure.
    """
    if not stats_list:
        raise ValueError("stats_list cannot be empty")
    
    if len(stats_list) == 1:
        return stats_list[0]
    
    # Auto-detect keys if not provided
    if keys is None:
        keys = [k for k in stats_list[0].keys() 
                if isinstance(stats_list[0][k], dict) and 'mean' in stats_list[0][k]]
    
    merged = {}
    total_count = 0
    
    # First pass: collect counts
    counts = []
    for stats in stats_list:
        if 'num_transitions' in stats:
            count = stats['num_transitions']
        elif keys and keys[0] in stats and 'count' in stats[keys[0]]:
            count = int(stats[keys[0]]['count'])
        else:
            raise ValueError("Stats must contain 'num_transitions' or 'count' for merging")
        counts.append(count)
        total_count += count
    
    merged['num_transitions'] = total_count
    
    # Merge num_trajectories if present
    if all('num_trajectories' in s for s in stats_list):
        merged['num_trajectories'] = sum(s['num_trajectories'] for s in stats_list)
    
    # Second pass: merge statistics for each key
    for key in keys:
        key_stats_list = []
        for stats, count in zip(stats_list, counts):
            if key not in stats:
                raise ValueError(f"Key '{key}' not found in stats")
            key_stats_list.append((stats[key], count))
        
        merged[key] = _merge_key_stats(key_stats_list, total_count)
    
    return merged


def _merge_key_stats(key_stats_list: list, total_count: int) -> dict:
    """Merge statistics for a single key (e.g., 'state' or 'action').
    
    Args:
        key_stats_list: List of (stats_dict, count) tuples
        total_count: Total sample count
    
    Returns:
        Merged stats dict for this key
    """
    if not key_stats_list:
        raise ValueError("key_stats_list cannot be empty")
    
    # Get shape from first stats
    first_stats = key_stats_list[0][0]
    shape = first_stats['mean'].shape
    
    # Initialize accumulators
    weighted_mean = np.zeros(shape, dtype=np.float64)
    weighted_square_mean = np.zeros(shape, dtype=np.float64)
    min_val = np.full(shape, np.inf, dtype=np.float64)
    max_val = np.full(shape, -np.inf, dtype=np.float64)
    weighted_q01 = np.zeros(shape, dtype=np.float64)
    weighted_q99 = np.zeros(shape, dtype=np.float64)
    
    for stats, count in key_stats_list:
        weight = count / total_count
        
        # Weighted mean
        weighted_mean += weight * stats['mean'].astype(np.float64)
        
        # Weighted square_mean (E[X²])
        if 'square_mean' in stats:
            weighted_square_mean += weight * stats['square_mean'].astype(np.float64)
        else:
            # Fallback: compute from mean and std
            # E[X²] = Var(X) + E[X]² = std² + mean²
            mean = stats['mean'].astype(np.float64)
            std = stats['std'].astype(np.float64)
            square_mean = std ** 2 + mean ** 2
            weighted_square_mean += weight * square_mean
        
        # Min/max: take extremes
        min_val = np.minimum(min_val, stats['min'].astype(np.float64))
        max_val = np.maximum(max_val, stats['max'].astype(np.float64))
        
        # Percentiles: weighted average (approximation)
        if 'q01' in stats:
            weighted_q01 += weight * stats['q01'].astype(np.float64)
        if 'q99' in stats:
            weighted_q99 += weight * stats['q99'].astype(np.float64)
    
    # Compute std from E[X²] and E[X]
    # Var(X) = E[X²] - E[X]²
    variance = weighted_square_mean - weighted_mean ** 2
    std = np.sqrt(np.maximum(variance, 0))
    
    return {
        'mean': weighted_mean.astype(np.float32),
        'square_mean': weighted_square_mean.astype(np.float32),
        'std': std.astype(np.float32),
        'min': min_val.astype(np.float32),
        'max': max_val.astype(np.float32),
        'q01': weighted_q01.astype(np.float32),
        'q99': weighted_q99.astype(np.float32),
    }

def find_all_hdf5(dataset_dir):
    """
    Find all HDF5 files in the dataset directory.
    Note: This function is deprecated. Use dataset class internal method instead.
    """
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        if 'pointcloud' in root: continue
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            hdf5_files.append(os.path.join(root, filename))
    return hdf5_files

class BaseNormalizer:
    def __init__(self, dataset, dataset_name:str=None, ctrl_type='delta', ctrl_space='ee', mask=None, *args, **kwargs):
        """Initialize BaseNormalizer
        
        Args:
            dataset: Dataset object or string path (for loading from cache)
            dataset_name: Explicit name for the dataset (required for new format)
            ctrl_type: Control type ('abs', 'delta', etc.)
            ctrl_space: Control space ('ee', 'joint', etc.)
            mask: Optional mask for selective normalization. Can be:
                  - None: normalize all dimensions (default behavior)
                  - Boolean array: True for dimensions to normalize, False to skip
                  - List of indices: indices of dimensions NOT to normalize (e.g., [-1] for last dim)
        """
        # Get centralized cache directory
        self.cache_dir = os.path.join(os.environ.get('ILSTD_CACHE', os.path.expanduser('~/.cache/ilstd')), 'normalize')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Store mask specification (will be converted to bool array later when data shape is known)
        self.mask_spec = mask
        self.mask = None  # Will be initialized on first normalize/denormalize call
        
        if isinstance(dataset, str):
            # Loading mode: dataset is a path (cache_dir or dataset_dir for backward compat)
            self.dataset = None
            self.dataset_dir = dataset  # Keep for backward compatibility
            self.ctrl_space = ctrl_space
            self.ctrl_type = ctrl_type
            # When loading, dataset_name must be provided
            if dataset_name is None:
                raise ValueError("dataset_name must be provided when loading normalizer from path")
            self.dataset_name = dataset_name
        else:
            # Training mode: dataset is an actual dataset object
            self.dataset = dataset
            self.dataset_dir = dataset.get_dataset_dir() if hasattr(dataset, 'get_dataset_dir') else None
            self.ctrl_type = getattr(dataset, 'ctrl_type', ctrl_type)
            self.ctrl_space = getattr(dataset, 'ctrl_space', ctrl_space)
            
            # Use dataset.dataset_id if available, otherwise fall back to hash-based name
            if hasattr(dataset, 'dataset_id'):
                self.dataset_name = dataset.dataset_id
            elif dataset_name is not None:
                self.dataset_name = dataset_name
            elif self.dataset_dir:
                self.dataset_name = 'd' + str2hash(self.dataset_dir)
            else:
                raise ValueError("Cannot determine dataset name: dataset has no 'dataset_id' attribute and dataset_name not provided")
        
        self.stats_filename = f"{self.dataset_name}_stats_{self.ctrl_space}_{self.ctrl_type}.pkl"
        
        rank = dist.get_rank() if is_distributed() else 0
        if rank == 0:
            if self.is_stats_exist():
                self.all_stats = self.load_stats()
            else:
                assert self.dataset is not None, "dataset cannot be None when stats file does not exist"
                _ = self.compute_and_save_stats()
                self.all_stats = self.load_stats()
            
                
        if is_distributed(): dist.barrier()
        if rank != 0: self.all_stats = self.load_stats()

    def _safe_scale(self, scale, min_scale=None):
        if isinstance(scale, torch.Tensor):
            safe_scale = scale
            if min_scale is not None:
                safe_scale = torch.clamp(safe_scale, min=min_scale)
            return torch.where(
                scale.abs() <= MIN_SCALE_THRESHOLD,
                torch.ones_like(scale),
                safe_scale,
            )
        safe_scale = scale
        if min_scale is not None:
            safe_scale = np.clip(safe_scale, min_scale, np.inf)
        return np.where(np.abs(scale) <= MIN_SCALE_THRESHOLD, 1.0, safe_scale)

    @classmethod
    def meta2name(cls, dataset_dir:str, ctrl_space:str='ee', ctrl_type:str='delta'):
        return f"{'d' + str2hash(dataset_dir)}"

    def is_stats_exist(self):
        """Check if stats file exists in cache directory or dataset directory (backward compat)"""
        # Check in centralized cache first (new format)
        cache_path = os.path.join(self.cache_dir, self.stats_filename)
        if os.path.exists(cache_path):
            return True
        else:
            cache_path=os.path.join(self.dataset_dir, self.stats_filename)
            if os.path.exists(cache_path):
                self.cache_dir = self.dataset_dir
                return True
            return False
        
        # # Backward compatibility: check in dataset_dir if it exists
        # if self.dataset_dir:
        #     old_path = os.path.join(self.dataset_dir, self.stats_filename)
        #     old_path_alt = os.path.join(self.dataset_dir, f'dataset_stats_{self.ctrl_space}_{self.ctrl_type}.pkl')
        #     if os.path.exists(old_path) or os.path.exists(old_path_alt):
        #         return True
        
        # return False

    def compute_stats_for_array(self, data_k):
        """Compute statistics for an array.
        
        Statistics include:
        - mean: E[X] - first moment
        - square_mean: E[X²] - second raw moment (for aggregating std across datasets)
        - std: standard deviation
        - max, min: range statistics
        - q01, q99: percentile statistics
        
        Note: square_mean enables aggregation of statistics across multiple datasets:
            Combined variance = E[X²]_combined - E[X]_combined²
            where E[X]_combined = (n1*E[X]_1 + n2*E[X]_2) / (n1 + n2)
            and E[X²]_combined = (n1*E[X²]_1 + n2*E[X²]_2) / (n1 + n2)
        """
        return {
            "mean": data_k.mean(0),
            "square_mean": (data_k ** 2).mean(0),  # E[X²] for aggregating std
            "std": data_k.std(0),
            "max": data_k.max(0),
            "min": data_k.min(0),
            "q01": np.quantile(data_k, 0.01, axis=0),
            "q99": np.quantile(data_k, 0.99, axis=0),
        }
    
    def compute_and_save_stats(self):
        """Compute and save normalization statistics using streaming computation.
        
        Uses RunningStats for memory-efficient computation - processes data
        incrementally without storing all samples in memory.
        
        Supports multiple dataset types with flexible data extraction:
        1. Episodic datasets with num_episodes and extract_from_episode
        2. Datasets with extract_all() method
        3. Map-style datasets with __len__ and __getitem__
        4. Iterable datasets
        """
        num_trajectories = None
        running_stats = {}  # key -> RunningStats
        
        # Method 0: Check if dataset has get_dataset_statistics()
        if hasattr(self.dataset, 'get_dataset_statistics') and callable(getattr(self.dataset, 'get_dataset_statistics')):
            logger.info(f"Using get_dataset_statistics() method")
            all_stats = self.dataset.get_dataset_statistics()
            self.save_stats(all_stats)
            return {k:{kk:np.array(vv) for kk,vv in v.items()} if isinstance(v, dict) else v for k,v in all_stats.items()}
        
        def _to_numpy(value):
            """Convert value to numpy array."""
            if hasattr(value, 'cpu'):
                return value.cpu().numpy()
            elif not isinstance(value, np.ndarray):
                return np.array(value)
            return value
        
        def _update_running_stats(key: str, value: np.ndarray):
            """Update running stats for a key, initializing if needed."""
            value = _to_numpy(value)
            # Flatten batch dimension if present for shape detection
            if len(value.shape) >= 2:
                sample_shape = value.shape[1:]  # (batch, *shape) -> (*shape,)
            else:
                sample_shape = value.shape
            
            if key not in running_stats:
                running_stats[key] = RunningStats(sample_shape)
            
            # Update with batch
            running_stats[key].update(value)
        
        # Method 1: Check if dataset has num_episodes and extract_from_episode
        use_running = False
        if hasattr(self.dataset, 'num_episodes') and hasattr(self.dataset, 'extract_from_episode'):
            logger.info(f"Using episodic extraction (streaming): {self.dataset.num_episodes} episodes")
            num_trajectories = self.dataset.num_episodes
            use_running = True
            for idx in range(self.dataset.num_episodes):
                res_each = self.dataset.extract_from_episode(idx, ['state', 'action'])
                for k in res_each:
                    _update_running_stats(k, res_each[k])
                if (idx + 1) % 100 == 0:
                    logger.info(f"  Processed {idx + 1}/{self.dataset.num_episodes} episodes...")
        
        # Method 2: Check if dataset has extract_all() method
        elif hasattr(self.dataset, 'extract_all') and callable(getattr(self.dataset, 'extract_all')):
            logger.info("Using extract_all() method (streaming)")
            try:
                extracted = self.dataset.extract_all(['state', 'action'])
                use_running = True
                for k, v in extracted.items():
                    v = _to_numpy(v)
                    # Process in chunks to use running stats
                    chunk_size = 10000
                    sample_shape = v.shape[1:] if len(v.shape) >= 2 else v.shape
                    if k not in running_stats:
                        running_stats[k] = RunningStats(sample_shape)
                    for i in range(0, len(v), chunk_size):
                        running_stats[k].update(v[i:i+chunk_size])
            except Exception as e:
                logger.warning(f"extract_all() failed: {e}, falling back to iteration")
                use_running = False
        
        # Method 3 & 4: Use DataLoader for map-style or iterable datasets (streaming)
        if not use_running or len(running_stats) == 0:
            from torch.utils.data import DataLoader
            
            # Determine if it's a map-style dataset
            is_map_style = hasattr(self.dataset, '__len__') and hasattr(self.dataset, '__getitem__')
            
            # Try to detect if the dataset returns batched data
            returns_batches = self._detect_if_returns_batches()
            
            if is_map_style:
                logger.info(f"Using DataLoader (streaming) for map-style dataset: {len(self.dataset)} samples")
                batch_size = 1 if returns_batches else 32
                if returns_batches:
                    logger.info("  Detected: Dataset returns batches, using batch_size=1")
                else:
                    logger.info(f"  Detected: Dataset returns samples, using batch_size={batch_size}")
                
                dataloader = DataLoader(
                    self.dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=None if returns_batches else self._collate_for_stats
                )
            else:
                logger.info("Using DataLoader (streaming) for iterable dataset")
                batch_size = 1 if returns_batches else 32
                if returns_batches:
                    logger.info("  Detected: Dataset returns batches, using batch_size=1")
                else:
                    logger.info(f"  Detected: Dataset returns samples, using batch_size={batch_size}")
                
                dataloader = DataLoader(
                    self.dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=None if returns_batches else self._collate_for_stats
                )
            
            # Stream data through running stats
            batch_count = 0
            for batch_or_sample in dataloader:
                if batch_or_sample is None:
                    continue
                
                # Handle both dict and tuple/list formats
                if isinstance(batch_or_sample, dict):
                    # Dict format
                    for key in ['state', 'action']:
                        if key in batch_or_sample:
                            _update_running_stats(key, batch_or_sample[key])
                
                elif isinstance(batch_or_sample, (tuple, list)) and len(batch_or_sample) >= 2:
                    # Tuple/list format: (state, action, ...)
                    _update_running_stats('state', batch_or_sample[0])
                    _update_running_stats('action', batch_or_sample[1])
                
                batch_count += 1
                if batch_count % 100 == 0:
                    logger.info(f"Processed {batch_count} batches...")
            
            logger.info(f"Total batches processed: {batch_count}")
        
        # Extract statistics from RunningStats
        if len(running_stats) == 0:
            raise ValueError("No data collected for statistics computation")
        
        all_stats = {}
        if num_trajectories is not None:
            all_stats['num_trajectories'] = num_trajectories
        
        for k, rs in running_stats.items():
            stats_k = rs.get_stats()
            if 'num_transitions' not in all_stats:
                all_stats['num_transitions'] = int(stats_k['count'])
            # Store stats (excluding count which is top-level)
            key_name = k.split('/')[-1]
            all_stats[key_name] = {kk: vv for kk, vv in stats_k.items() if kk != 'count'}
        
        logger.info(f"Statistics computed (streaming): {all_stats.get('num_transitions', 0)} transitions")
        self.save_stats(all_stats)
        return {k:{kk:np.array(vv) for kk,vv in v.items()} if isinstance(v, dict) else v for k,v in all_stats.items()}
    
    def _detect_if_returns_batches(self):
        """Detect if the dataset returns batches or individual samples
        
        This is important for correctly constructing the DataLoader:
        - If returns batches: use batch_size=1, no collate_fn
        - If returns samples: use batch_size=32, with collate_fn
        
        Detection rules:
        1. Map-style datasets (__len__ + __getitem__) always return samples
        2. Iterable datasets: check 'raw_lang' field
           - If raw_lang is a list → returns batches
           - If raw_lang is a string → returns samples
        
        Returns:
            bool: True if dataset returns batches, False if returns samples
        """
        # Rule 1: Map-style datasets always return samples
        if hasattr(self.dataset, '__len__') and hasattr(self.dataset, '__getitem__'):
            logger.debug(f"    Map-style dataset always returns samples")
            return False
        
        # Rule 2: For iterable datasets, check raw_lang field
        # Strategy 1: Check for explicit attributes first
        if hasattr(self.dataset, 'returns_batches'):
            result = bool(self.dataset.returns_batches)
            logger.debug(f"    Explicit attribute returns_batches={result}")
            return result
        
        # Strategy 2: Peek at the first item and check raw_lang
        try:
            # For iterable datasets
            if hasattr(self.dataset, '__iter__'):
                iterator = iter(self.dataset)
                sample = next(iterator)
            else:
                # Can't detect, assume returns samples
                logger.debug(f"    Cannot iterate dataset, assuming samples")
                return False
            
            # Check raw_lang field
            if isinstance(sample, dict) and 'raw_lang' in sample:
                raw_lang = sample['raw_lang']
                if isinstance(raw_lang, list):
                    logger.debug(f"    Detected batch: raw_lang is a list (length={len(raw_lang)})")
                    return True
                elif isinstance(raw_lang, str):
                    logger.debug(f"    Detected sample: raw_lang is a string")
                    return False
            
            # If no raw_lang field, assume returns samples
            logger.debug(f"    No raw_lang field found, assuming samples")
            return False
            
        except Exception as e:
            # If peek fails, assume returns samples (safer default)
            logger.debug(f"    Could not detect batch/sample format ({e}), assuming samples")
            return False
    
    def _collate_for_stats(self, batch):
        """Collate function for DataLoader to extract state and action"""
        if not batch:
            return None
        
        # Handle different batch formats
        collated = {}
        
        # Check if batch items are dictionaries
        if isinstance(batch[0], dict):
            # Batch of dictionaries
            for key in ['state', 'action']:
                if key in batch[0]:
                    values = [item[key] for item in batch]
                    # Stack or concatenate
                    try:
                        if hasattr(values[0], 'cpu'):
                            values = [v.cpu() if hasattr(v, 'cpu') else v for v in values]
                        collated[key] = torch.stack([torch.as_tensor(v) for v in values])
                    except:
                        # If stack fails, try to handle differently
                        collated[key] = values
        
        # Handle tuple/list format (obs, action, ...)
        elif isinstance(batch[0], (tuple, list)) and len(batch[0]) >= 2:
            # Assume format: (state/obs, action, ...)
            states = [item[0] for item in batch]
            actions = [item[1] for item in batch]
            
            try:
                collated['state'] = torch.stack([torch.as_tensor(s) for s in states])
                collated['action'] = torch.stack([torch.as_tensor(a) for a in actions])
            except:
                collated['state'] = states
                collated['action'] = actions
        
        return collated if collated else None
    
    def save_stats(self, all_stats, save_path=None):
        """Save stats to centralized cache directory"""
        if save_path is None: save_path = os.path.join(self.cache_dir, self.stats_filename)
        with open(save_path, 'wb') as file:
            pickle.dump(all_stats, file)

    def save_stats_to_(self, target_dir:str):
        """Save the dataset's stats to `target_dir` (typically for training checkpoints)
        
        This saves stats to both the target_dir (for checkpoint) and cache_dir (for future use).
        """
        assert hasattr(self, 'all_stats') and self.all_stats is not None, "No stats found."
        save_path = os.path.join(target_dir, self.stats_filename)
        # Save to target_dir (training checkpoint)
        if not os.path.exists(save_path):
            self.save_stats(self.all_stats, save_path=save_path)
        else:
            warnings.warn(f"Stats file {save_path} already exists in training dir.")


    def load_stats(self):
        """Load stats from cache directory or dataset directory (backward compat)"""
        # Try cache directory first (new format)
        stats_path = os.path.join(self.cache_dir, self.stats_filename)
        
        # if not os.path.exists(stats_path) and self.dataset_dir:
        #     # Backward compatibility: try dataset_dir
        #     stats_path = os.path.join(self.dataset_dir, self.stats_filename)
        #     if not os.path.exists(stats_path):
        #         stats_path = os.path.join(self.dataset_dir, f'dataset_stats_{self.ctrl_space}_{self.ctrl_type}.pkl')
        
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Stats file not found. Searched in:\n"
                f"  - {os.path.join(self.cache_dir, self.stats_filename)}\n"
                f"  - {os.path.join(self.dataset_dir, self.stats_filename) if self.dataset_dir else 'N/A'}"
            )
        
        with open(stats_path, 'rb') as file:
            all_stats = pickle.load(file)
        all_stats = {k:{kk:np.array(vv) for kk,vv in v.items()} if isinstance(v, dict) else v for k,v in all_stats.items()}
        return all_stats
    
    def get_stat_by_key(self, key='action'):
        if key not in self.all_stats: raise KeyError(f"Cannot find {key} in stats.")
        return self.all_stats[key]
    
    def _build_mask(self, data_shape):
        """Build boolean mask from mask_spec based on data shape
        
        Args:
            data_shape: Shape of the data to be normalized (last dimension is feature dim)
        
        Returns:
            Boolean numpy array of shape (feature_dim,) or None if no masking
        """
        if self.mask_spec is None:
            return None
        
        # Get feature dimension (last dimension of data)
        if isinstance(data_shape, (tuple, list)):
            feature_dim = data_shape[-1]
        else:
            feature_dim = data_shape
        
        # Case 1: mask_spec is already a boolean array
        if isinstance(self.mask_spec, (np.ndarray, list, tuple)):
            mask_array = np.array(self.mask_spec)
            
            # If it's boolean, use directly
            if mask_array.dtype == bool:
                if len(mask_array) != feature_dim:
                    raise ValueError(f"Mask length {len(mask_array)} doesn't match feature dimension {feature_dim}")
                return mask_array
            
            # Otherwise, treat as indices of dimensions NOT to normalize
            else:
                # Create a mask with all True (normalize all by default)
                mask = np.ones(feature_dim, dtype=bool)
                # Set specified indices to False (don't normalize)
                indices = np.array(self.mask_spec, dtype=int)
                # Handle negative indices
                indices = np.where(indices < 0, feature_dim + indices, indices)
                mask[indices] = False
                return mask
        
        else:
            raise ValueError(f"Invalid mask_spec type: {type(self.mask_spec)}. Expected None, boolean array, or list of indices.")
    
    def _apply_mask(self, data, normalized_data, mask):
        """Apply mask to selectively use normalized or original data
        
        Args:
            data: Original data
            normalized_data: Normalized data
            mask: Boolean mask (True = use normalized, False = use original)
        
        Returns:
            Data with selective normalization applied
        """
        if mask is None:
            return normalized_data
        
        # Expand mask to match data shape (broadcast over batch dimensions)
        # data shape: (batch_size, ..., feature_dim)
        # mask shape: (feature_dim,)
        if isinstance(data, torch.Tensor):
            mask_tensor = torch.from_numpy(mask).to(data.device)
            result = torch.where(mask_tensor, normalized_data, data)
        else:
            result = np.where(mask, normalized_data, data)
        
        return result
    
    def normalize_metaobs(self, mobs: MetaObs, ctrl_space='ee'):
        assert ctrl_space==self.ctrl_space, f"the space of observation {ctrl_space} does not match the normalizer's {self.ctrl_space}"
        mobs.state = self.normalize(mobs.state, datatype='state')
        return mobs
    
    def denormalize_metaact(self, mact: MetaAction):
        assert mact.ctrl_type==self.ctrl_type, f"the contrlling type of action {mact.ctrl_type} does not match the normalizer's {self.ctrl_type}"
        assert mact.ctrl_space==self.ctrl_space, f"the space of action {mact.ctrl_type} does not match the normalizer's {self.ctrl_space}"
        is_chunked = len(mact.action.shape) >= 3
        if is_chunked: mact.action = mact.action.swapaxes(0, 1)
        mact.action = self.denormalize(mact.action, datatype='action')
        if is_chunked: mact.action = mact.action.swapaxes(0, 1)
        return mact
    
    def normalize(self, *args, **kwargs):
        raise NotImplementedError
    
    def denormalize(self, *args, **kwargs):
        raise NotImplementedError
    
class MinMaxNormalizer(BaseNormalizer):
    def __init__(
        self,
        dataset,
        dataset_name=None,
        low: float = -1,
        high: float = 1,
        ctrl_type='delta',
        ctrl_space='ee',
        mask=None,
    ):
        super().__init__(dataset, dataset_name, ctrl_type=ctrl_type, ctrl_space=ctrl_space, mask=mask)
        assert low!=high, "low is equal to high"
        self.low = low
        self.high = high
        self.delta = self.high-self.low
        
    def __str__(self):
        return "minmax"
    
    def normalize(self, data, datatype='action'):
        dtype = data.dtype
        stats = self.get_stat_by_key(datatype)
        
        # Initialize mask on first call
        if self.mask is None and self.mask_spec is not None:
            self.mask = self._build_mask(data.shape)
        
        # Perform normalization
        safe_range = self._safe_scale(stats['max'] - stats['min'])
        normalized = (data - stats['min']) / safe_range * self.delta + self.low
        
        # Apply mask to selectively normalize
        result = self._apply_mask(data, normalized, self.mask)
        
        return result.to(dtype) if isinstance(result, torch.Tensor) else result.astype(dtype)
    
    def denormalize(self, data, datatype='action'):
        dtype = data.dtype
        stats = self.get_stat_by_key(datatype)
        
        # Initialize mask on first call
        if self.mask is None and self.mask_spec is not None:
            self.mask = self._build_mask(data.shape)
        
        # Perform denormalization
        safe_range = self._safe_scale(stats['max'] - stats['min'])
        denormalized = ((data - self.low) / self.delta) * safe_range + stats['min']
        
        # Apply mask to selectively denormalize
        result = self._apply_mask(data, denormalized, self.mask)
        
        return result.to(dtype) if isinstance(result, torch.Tensor) else result.astype(dtype)

class PercentileNormalizer(BaseNormalizer):
    def __init__(
        self,
        dataset_dir,
        dataset_name=None,
        low: float = -1,
        high: float = 1,
        ctrl_type='delta',
        ctrl_space='ee',
        mask=None,
    ):
        super().__init__(dataset_dir, dataset_name, ctrl_type=ctrl_type, ctrl_space=ctrl_space, mask=mask)
        assert low!=high, "low is equal to high"
        self.low = low
        self.high = high
        self.delta = self.high-self.low
    
    def __str__(self):
        return "percentile"
    
    def normalize(self, data, datatype='action'):
        dtype = data.dtype
        stats = self.get_stat_by_key(datatype)
        
        # Initialize mask on first call
        if self.mask is None and self.mask_spec is not None:
            self.mask = self._build_mask(data.shape)
        
        # Perform normalization
        safe_range = self._safe_scale(stats['q99'] - stats['q01'])
        normalized = (data - stats['q01']) / safe_range * self.delta + self.low
        if isinstance(normalized, torch.Tensor):
            normalized = torch.clip(normalized, self.low, self.high)
        else:
            normalized = np.clip(normalized, self.low, self.high)
        
        # Apply mask to selectively normalize
        result = self._apply_mask(data, normalized, self.mask)
        
        return result.to(dtype) if isinstance(result, torch.Tensor) else result.astype(dtype)
    
    def denormalize(self, data, datatype='action'):
        dtype = data.dtype
        stats = self.get_stat_by_key(datatype)
        
        # Initialize mask on first call
        if self.mask is None and self.mask_spec is not None:
            self.mask = self._build_mask(data.shape)
        
        # Perform denormalization
        safe_range = self._safe_scale(stats['q99'] - stats['q01'])
        denormalized = ((data - self.low) / self.delta) * safe_range + stats['q01']
        if isinstance(denormalized, torch.Tensor):
            denormalized = torch.clip(denormalized, stats['q01'], stats['q99'])
        else:
            denormalized = np.clip(denormalized, stats['q01'], stats['q99'])
        
        # Apply mask to selectively denormalize
        result = self._apply_mask(data, denormalized, self.mask)
        
        return result.to(dtype) if isinstance(result, torch.Tensor) else result.astype(dtype)

class ZScoreNormalizer(BaseNormalizer):
    def __init__(
        self,
        dataset,
        dataset_name=None,
        ctrl_type='delta',
        ctrl_space='ee',
        min_std=1e-2,
        mask=None,
        *args,
        **kwargs,
    ):
        super().__init__(dataset, dataset_name, ctrl_type=ctrl_type, ctrl_space=ctrl_space, mask=mask)
        self.min_std = min_std
        
    def __str__(self):
        return "zscore"
    
    def normalize(self, data, datatype='action'):
        dtype = data.dtype
        stats = self.get_stat_by_key(datatype)
        
        # Initialize mask on first call
        if self.mask is None and self.mask_spec is not None:
            self.mask = self._build_mask(data.shape)
        
        # Perform normalization
        std = self._safe_scale(stats['std'], self.min_std)
        normalized = (data - stats['mean']) / std
        
        # Apply mask to selectively normalize
        result = self._apply_mask(data, normalized, self.mask)
        
        return result.to(dtype) if isinstance(result, torch.Tensor) else result.astype(dtype) 
    
    def denormalize(self, data, datatype='action'):
        dtype = data.dtype
        stats = self.get_stat_by_key(datatype)
        
        # Initialize mask on first call
        if self.mask is None and self.mask_spec is not None:
            self.mask = self._build_mask(data.shape)
        
        # Perform denormalization
        std = self._safe_scale(stats['std'], self.min_std)
        denormalized = data * std + stats['mean']
        
        # Apply mask to selectively denormalize
        result = self._apply_mask(data, denormalized, self.mask)
        
        return result.to(dtype) if isinstance(result, torch.Tensor) else result.astype(dtype)
    
class Identity(BaseNormalizer):
    def __init__(self, *args, ctrl_type: str = 'delta', ctrl_space: str = 'ee', **kwargs):
        # Accept same (load_dir, dataset_name=..., ctrl_*) call shape as other normalizers.
        # Previously ctrl_type was positional #1, so Identity(load_dir, ctrl_type=...) crashed
        # and load_normalizers fell back to ee/delta — breaking abs-EE replay tagging.
        logger.info("Perform no normalization on actions and state")
        self.ctrl_type = ctrl_type
        self.ctrl_space = ctrl_space

    def __str__(self):
        return 'identity'
    
    def normalize(self, data, *args, **kwargs):
        return data
    
    def denormalize(self, data, *args, **kwargs):
        return data
    
# Normalize Class
NORMTYPE2CLASS = {
    'minmax': MinMaxNormalizer,
    'percentile': PercentileNormalizer, 
    'zscore': ZScoreNormalizer,
    'identity': Identity,
}


def save_norm_meta_to_json(file_path: str, data: dict):
    """
    Save normalization meta information to json file
    
    Saves complete metadata including datasets (with per-dataset ctrl_space/ctrl_type), state, and action.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_normalizer_from_meta(norm_meta, src_dir='', dataset_id=None):
    """Load normalizers from metadata
    
    Uses dataset_id as keys and includes per-dataset ctrl_space/ctrl_type.
    
    Args:
        norm_meta: Metadata dictionary
        src_dir: Source directory where normalize.json and stats files are located.
                 If empty or stats not found, will fallback to cache directory.
        dataset_id: Specific dataset_id to load. If None, loads the first dataset from metadata.
                    Note: This is read from metadata, not from args.dataset_id.
    
    Returns:
        Dictionary with 'state' and 'action' normalizers
    """
    datasets_info = norm_meta.get('datasets', [])
    
    if not datasets_info:
        raise ValueError("No datasets found in metadata")
    
    # Determine which dataset to load
    if dataset_id is None:
        # Use first dataset as default (typical for single-dataset training)
        dataset_meta = datasets_info[0]
        dataset_id = dataset_meta['dataset_id']
        if len(datasets_info) > 1:
            logger.info(f"Multiple datasets found in metadata. Using first: {dataset_id}")
    else:
        # Find the matching dataset by dataset_id
        dataset_meta = None
        for ds_meta in datasets_info:
            if ds_meta['dataset_id'] == dataset_id:
                dataset_meta = ds_meta
                break
        
        if dataset_meta is None:
            raise ValueError(f"Dataset '{dataset_id}' not found in metadata. Available: {[d['dataset_id'] for d in datasets_info]}")
    
    # Get ctrl info from metadata
    ctrl_space = dataset_meta.get('ctrl_space', 'ee')
    ctrl_type = dataset_meta.get('ctrl_type', 'delta')
    
    # Get mask info from metadata
    action_norm_mask = dataset_meta.get('action_norm_mask', None)
    state_norm_mask = dataset_meta.get('state_norm_mask', None)
    
    # Log mask information for transparency
    if action_norm_mask is not None or state_norm_mask is not None:
        logger.info(f"Loading normalizers with mask configuration for dataset '{dataset_id}':")
        if action_norm_mask is not None:
            logger.info(f"  - action_norm_mask: {action_norm_mask}")
        if state_norm_mask is not None:
            logger.info(f"  - state_norm_mask: {state_norm_mask}")
    
    # Get normalizer types from metadata
    state_norm_type = norm_meta['state'].get(dataset_id, 'zscore')
    action_norm_type = norm_meta['action'].get(dataset_id, 'zscore')
    
    # Create normalizers with dataset_id and ctrl info
    kwargs = {'ctrl_space': ctrl_space, 'ctrl_type': ctrl_type}
    state_kwargs = kwargs.copy()
    action_kwargs = kwargs.copy()
    
    # Add mask to kwargs if present
    if state_norm_mask is not None:
        state_kwargs['mask'] = state_norm_mask
    if action_norm_mask is not None:
        action_kwargs['mask'] = action_norm_mask
    
    # Determine load directory: prefer src_dir, fallback to cache
    cache_dir = os.path.join(os.environ.get('ILSTD_CACHE', os.path.expanduser('~/.cache/ilstd')), 'normalize')
    
    # Check if stats exist in src_dir
    if src_dir and os.path.exists(src_dir):
        stats_filename = f"{dataset_id}_stats_{ctrl_space}_{ctrl_type}.pkl"
        src_stats_path = os.path.join(src_dir, stats_filename)
        
        if os.path.exists(src_stats_path):
            # Load from src_dir (checkpoint directory)
            load_dir = src_dir
        else:
            # Fallback to cache
            load_dir = cache_dir
            warnings.warn(f"Stats not found in {src_dir}, using cache directory: {cache_dir}")
    else:
        # Use cache directory
        load_dir = cache_dir
    
    # Create normalizers
    state_normalizer = NORMTYPE2CLASS[state_norm_type](
        load_dir, dataset_name=dataset_id, **state_kwargs
    )
    action_normalizer = NORMTYPE2CLASS[action_norm_type](
        load_dir, dataset_name=dataset_id, **action_kwargs
    )
    
    return {'state': state_normalizer, 'action': action_normalizer}

def load_normalizers(args):
    """Load normalizers from saved metadata
    
    Loads normalizers using dataset_id as key with per-dataset ctrl_space/ctrl_type.
    
    Args:
        args: Arguments object with model_name_or_path and optional dataset_id
    
    Returns:
        tuple: (normalizers_dict, ctrl_space, ctrl_type) or (normalizers_dict, datasets_info)
               For new format, returns list of dataset info dicts
    """
    try:
        # load normalizers
        if isinstance(args.model_name_or_path, str) and args.model_name_or_path.endswith('/'): args.model_name_or_path = args.model_name_or_path[:-1]
        policy_normalize_file = os.path.join(os.path.dirname(args.model_name_or_path), 'normalize.json')
        if not os.path.exists(policy_normalize_file):
            policy_normalize_file = os.path.join(args.model_name_or_path, 'normalize.json')
            if not os.path.exists(policy_normalize_file):
                raise FileNotFoundError("No normalize.json found")
        with open(policy_normalize_file, 'r') as f:
            norm_meta = json.load(f)
        
        # Get dataset_id from args if specified, otherwise use first dataset
        dataset_id = getattr(args, 'dataset_id', None)
        if dataset_id == '':  # Empty string means not specified
            dataset_id = None
        
        # Load normalizer from metadata
        normalizers = load_normalizer_from_meta(
            norm_meta, 
            src_dir=os.path.dirname(policy_normalize_file),
            dataset_id=dataset_id  # Will use first dataset if None
        )
        
        # Get ctrl info from the specified dataset or first dataset
        datasets_info = norm_meta.get('datasets', [])
        if datasets_info:
            # Find the dataset that was actually loaded
            target_dataset = None
            if dataset_id:
                # Look for the specified dataset
                for dataset in datasets_info:
                    if dataset.get('dataset_id') == dataset_id:
                        target_dataset = dataset
                        break
            
            # If not found or no dataset_id specified, use first dataset
            if target_dataset is None:
                target_dataset = datasets_info[0]
            
            ctrl_space = target_dataset.get('ctrl_space', 'ee')
            ctrl_type = target_dataset.get('ctrl_type', 'delta')
            
            logger.info(f"   ✓ Using ctrl_space='{ctrl_space}', ctrl_type='{ctrl_type}' from dataset '{target_dataset.get('dataset_id', 'unknown')}'")
        else:
            ctrl_space, ctrl_type = 'ee', 'delta'
            logger.warning(f"   ⚠ No dataset info found, using default ctrl_space='{ctrl_space}', ctrl_type='{ctrl_type}'")
        
        return normalizers, ctrl_space, ctrl_type
            
    except Exception as e:
        warnings.warn(f"Failed to load normalizers from {args.model_name_or_path} because {e}")
        identity_normalizer = {'state':Identity(), 'action':Identity()}
        return identity_normalizer, 'ee', 'delta'

