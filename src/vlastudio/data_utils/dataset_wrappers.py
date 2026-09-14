"""
Dataset wrappers for applying normalization without modifying the original dataset classes.

This module provides wrapper classes that apply normalization to dataset outputs,
supporting both map-style and iterable datasets.
"""
import torch
from torch.utils.data import Dataset, IterableDataset, ConcatDataset
from typing import Dict, Optional, List
from loguru import logger
from .normalize import MIN_SCALE_THRESHOLD
# TensorFlow imports for RLDS dataset handling
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class MapVQAWrapper(Dataset):
    def __init__(self, dataset: Dataset, action_dim: int, state_dim: int, chunk_size: int):
        self.dataset = dataset
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.is_vqa = True
        self.dummy_state = torch.zeros(state_dim)
        self.dummy_action = torch.zeros(chunk_size, action_dim)
        self.dummy_is_pad = torch.ones(chunk_size, dtype=torch.bool)
        self.dummy_timestamp = 0
        self.dummy_episode_id = 0
    
    def __getitem__(self, index):
        item = self.dataset[index]
        return {
            "image": item['image'],
            "raw_lang": item['raw_lang'],
            "reasoning": item['reasoning'],
            "state": self.dummy_state,
            "action": self.dummy_action,
            "is_pad": self.dummy_is_pad,
            "timestamp": self.dummy_timestamp,
            "episode_id": self.dummy_episode_id,
            "__index__": index,
        }
    
    def __len__(self):
        return len(self.dataset)

class IterVQAWrapper(IterableDataset):
    def __init__(self, dataset: IterableDataset, action_dim: int, state_dim: int, chunk_size: int):
        self.dataset = dataset
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.is_vqa = True
        self.dummy_state = torch.zeros(state_dim)
        self.dummy_action = torch.zeros(chunk_size, action_dim)
        self.dummy_is_pad = torch.ones(chunk_size, dtype=torch.bool)
        self.dummy_timestamp = 0
        self.dummy_episode_id = 0
    
    def __iter__(self):
        for item in self.dataset:
            yield {
                "image": item['image'],
                "raw_lang": item['raw_lang'],
                "reasoning": item['reasoning'],
                "state": self.dummy_state,
                "action": self.dummy_action,
                "is_pad": self.dummy_is_pad,
                "timestamp": self.dummy_timestamp,
                "episode_id": self.dummy_episode_id,
                "__index__": -1,
            }

class NormalizedMapDataset(Dataset):
    """
    Wrapper for map-style datasets that applies normalization to actions and states.
    
    This wrapper preserves the map-style nature of the dataset while transparently
    applying normalization to the output samples.
    
    Args:
        dataset: The underlying map-style dataset (torch.utils.data.Dataset)
        action_normalizers: Dictionary mapping dataset names/paths to action normalizers
        state_normalizers: Dictionary mapping dataset names/paths to state normalizers
        dataset_name: Name or path of the dataset (used to lookup the correct normalizer)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        action_normalizers: Optional[Dict] = None,
        state_normalizers: Optional[Dict] = None,
        dataset_name: Optional[str] = None
    ):
        self.dataset = dataset
        self.action_normalizers = action_normalizers or {}
        self.state_normalizers = state_normalizers or {}
        
        # Determine dataset name for normalizer lookup
        if dataset_name is None:
            # Try to get dataset_dir from the dataset
            if hasattr(dataset, 'dataset_dir'):
                dataset_name = dataset.dataset_dir
            elif hasattr(dataset, 'dataset_path_list') and len(dataset.dataset_path_list) > 0:
                dataset_name = dataset.dataset_path_list[0]
            elif hasattr(dataset, 'get_dataset_dir'):
                dataset_name = dataset.get_dataset_dir()
            else:
                dataset_name = 'default'
        
        self.dataset_name = dataset_name
        
        # Get the specific normalizers for this dataset
        self.action_normalizer = self.action_normalizers.get(dataset_name, None)
        self.state_normalizer = self.state_normalizers.get(dataset_name, None)
    
    def __len__(self):
        """Return the length of the underlying dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset and apply normalization.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Sample dictionary with normalized actions and states
        """
        # Get the original sample from the underlying dataset
        sample = self.dataset[idx]
        
        # Apply action normalization if available
        if self.action_normalizer is not None and 'action' in sample:
            sample['action'] = self.action_normalizer.normalize(sample['action'], datatype='action')
        
        # Apply state normalization if available
        if self.state_normalizer is not None and 'state' in sample:
            sample['state'] = self.state_normalizer.normalize(sample['state'], datatype='state')
        
        return sample
    
    def __getattr__(self, name):
        """
        Forward attribute access to the underlying dataset.
        
        This allows the wrapper to be transparent and expose all methods
        and attributes of the wrapped dataset.
        """
        # Avoid infinite recursion for special attributes
        if name in ['dataset', 'action_normalizers', 'state_normalizers', 
                    'dataset_name', 'action_normalizer', 'state_normalizer']:
            return object.__getattribute__(self, name)
        
        # Forward to the underlying dataset
        return getattr(self.dataset, name)


class NormalizedIterableDataset(IterableDataset):
    """
    Wrapper for iterable datasets that applies normalization to actions and states.
    
    This wrapper preserves the iterable nature of the dataset while transparently
    applying normalization to the output samples.
    
    Args:
        dataset: The underlying iterable dataset (torch.utils.data.IterableDataset)
        action_normalizers: Dictionary mapping dataset names/paths to action normalizers
        state_normalizers: Dictionary mapping dataset names/paths to state normalizers
        dataset_name: Name or path of the dataset (used to lookup the correct normalizer)
    """
    
    def __init__(
        self,
        dataset: IterableDataset,
        action_normalizers: Optional[Dict] = None,
        state_normalizers: Optional[Dict] = None,
        dataset_name: Optional[str] = None
    ):
        super().__init__()
        self.dataset = dataset
        self.action_normalizers = action_normalizers or {}
        self.state_normalizers = state_normalizers or {}
        
        # Determine dataset name for normalizer lookup
        if dataset_name is None:
            # Try to get dataset name from the dataset
            if hasattr(dataset, 'dataset_dir'):
                dataset_name = dataset.dataset_dir
            elif hasattr(dataset, 'dataset_path_list') and len(dataset.dataset_path_list) > 0:
                dataset_name = dataset.dataset_path_list[0]
            elif hasattr(dataset, 'dataset_path'):
                dataset_name = dataset.dataset_path
            else:
                dataset_name = 'default'
        
        self.dataset_name = dataset_name
        
        # Get the specific normalizers for this dataset
        self.action_normalizer = self.action_normalizers.get(dataset_name, None)
        self.state_normalizer = self.state_normalizers.get(dataset_name, None)
    
    def __iter__(self):
        """
        Iterate over the dataset and apply normalization to each sample.
        
        Yields:
            Sample dictionary with normalized actions and states
        """
        for sample in self.dataset:
            # Apply action normalization if available
            if self.action_normalizer is not None and 'action' in sample:
                sample['action'] = self.action_normalizer.normalize(sample['action'])
            
            # Apply state normalization if available
            if self.state_normalizer is not None and 'state' in sample:
                sample['state'] = self.state_normalizer.normalize(sample['state'], datatype='state')
            
            yield sample
    
    def __getattr__(self, name):
        """
        Forward attribute access to the underlying dataset.
        
        This allows the wrapper to be transparent and expose all methods
        and attributes of the wrapped dataset.
        """
        # Avoid infinite recursion for special attributes
        if name in ['dataset', 'action_normalizers', 'state_normalizers', 
                    'dataset_name', 'action_normalizer', 'state_normalizer']:
            return object.__getattribute__(self, name)
        
        # Forward to the underlying dataset
        return getattr(self.dataset, name)


def wrap_dataset_with_normalizers(
    dataset,
    action_normalizers: Optional[Dict] = None,
    state_normalizers: Optional[Dict] = None,
    dataset_name: Optional[str] = None
):
    """
    Automatically wrap a dataset with the appropriate normalizer wrapper.
    
    This function uses duck typing to detect whether the dataset is map-style 
    or iterable by checking for the presence of __getitem__ and __iter__ methods.
    
    Args:
        dataset: The dataset to wrap (either map-style or iterable)
        action_normalizers: Dictionary mapping dataset names/paths to action normalizers
        state_normalizers: Dictionary mapping dataset names/paths to state normalizers
        dataset_name: Name or path of the dataset (used to lookup the correct normalizer)
    
    Returns:
        Wrapped dataset with normalization applied
    """
    # Check if normalization is needed
    if not action_normalizers and not state_normalizers:
        # No normalizers provided, return original dataset
        return dataset
    
    # Use duck typing to detect dataset type
    # Check for __getitem__ (map-style) vs __iter__ (iterable)
    has_getitem = hasattr(dataset, '__getitem__') and callable(getattr(dataset, '__getitem__'))
    has_iter = hasattr(dataset, '__iter__') and callable(getattr(dataset, '__iter__'))
    has_len = hasattr(dataset, '__len__') and callable(getattr(dataset, '__len__'))
    
    # Map-style datasets have both __getitem__ and __len__
    # Iterable datasets have __iter__ but may not have __getitem__ or __len__
    if has_getitem and has_len:
        # Map-style dataset
        return NormalizedMapDataset(
            dataset=dataset,
            action_normalizers=action_normalizers,
            state_normalizers=state_normalizers,
            dataset_name=dataset_name
        )
    elif has_iter:
        import dlimp as dl
        # Check if this is an RLDS dataset (tf.data with dlimp Dataset)
        if hasattr(dataset, 'dataset') and isinstance(dataset.dataset, dl.DLataset):
            # This is an RLDS dataset, use TensorFlow pipeline for normalization
            return _wrap_rlds_dataset_with_tf_normalizers(
                dataset=dataset,
                action_normalizers=action_normalizers,
                state_normalizers=state_normalizers,
                dataset_name=dataset_name
            )
        else:
            # Regular iterable dataset
            return NormalizedIterableDataset(
                dataset=dataset,
                action_normalizers=action_normalizers,
                state_normalizers=state_normalizers,
                dataset_name=dataset_name
            )
    else:
        # Unknown dataset type, return as-is and let it fail later if needed
        import warnings
        warnings.warn(
            f"Dataset type {type(dataset)} does not implement standard dataset interface "
            f"(__getitem__+__len__ for map-style or __iter__ for iterable). "
            f"Returning unwrapped dataset."
        )
        return dataset


def _wrap_rlds_dataset_with_tf_normalizers(
    dataset,
    action_normalizers: Optional[Dict] = None,
    state_normalizers: Optional[Dict] = None,
    dataset_name: Optional[str] = None
):
    """
    Apply normalization to RLDS datasets using pure TensorFlow pipeline operations.
    This function extracts normalization parameters outside the TF graph and creates a
    pure TF function to be mapped onto the dataset, ensuring graph compatibility.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for RLDS dataset normalization but is not available.")

    action_normalizer = action_normalizers.get(dataset_name) if action_normalizers else None
    state_normalizer = state_normalizers.get(dataset_name) if state_normalizers else None

    if not action_normalizer and not state_normalizer:
        return dataset

    # --- Step 1: Extract and convert all parameters to TF constants outside the pipeline ---
    # Assume float32 as the rlds_wrapper already casts actions to float32.
    sample_dtype = tf.float32 
    action_tf_params = _get_tf_norm_params(action_normalizer, 'action', sample_dtype) if action_normalizer else None
    state_tf_params = _get_tf_norm_params(state_normalizer, 'state', sample_dtype) if state_normalizer else None

    # --- Step 2: Define the pure TensorFlow function for the pipeline ---
    def normalize_in_pipeline(sample):
        """Pure TensorFlow function to normalize a single sample."""
        if action_tf_params and 'action' in sample:
            sample['action'] = _apply_tf_norm_from_tf_params(sample['action'], action_tf_params)
        
        if state_tf_params and 'state' in sample:
            sample['state'] = _apply_tf_norm_from_tf_params(sample['state'], state_tf_params)
        return sample

    # --- Step 3: Apply the pure function to the dataset ---
    dataset.dataset = dataset.dataset.map(
        normalize_in_pipeline,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset


def _get_tf_norm_params(normalizer, data_type: str, sample_dtype=None) -> Optional[Dict]:
    """
    Extracts all necessary statistics from a normalizer object and converts them to
    TensorFlow constants, ready to be used in a tf.data pipeline.
    """
    if sample_dtype is None: sample_dtype = tf.float32
    if not hasattr(normalizer, 'get_stat_by_key'):
        return None

    stats = normalizer.get_stat_by_key(data_type)
    if not stats:
        return None

    # Map normalizer type string to a graph-compatible integer index
    norm_type_str = str(normalizer)
    type_map = {'zscore': 0, 'minmax': 1, 'percentile': 2}
    type_index = type_map.get(norm_type_str, -1)  # -1 for identity/unsupported

    params = {'type_index': tf.constant(type_index, dtype=tf.int32)}

    # Convert stats to TF constants
    for key in ['mean', 'std', 'min', 'max', 'q01', 'q99']:
        if key in stats and stats[key] is not None:
            params[key] = tf.constant(stats[key], dtype=sample_dtype)

    # Extract mask_spec, NOT the mask itself
    params['mask_spec'] = getattr(normalizer, 'mask_spec', None)
    
    # Convert other normalizer-specific parameters to TF constants
    params['low'] = tf.constant(getattr(normalizer, 'low', 0.0), dtype=sample_dtype)
    params['high'] = tf.constant(getattr(normalizer, 'high', 1.0), dtype=sample_dtype)
    params['min_std'] = tf.constant(getattr(normalizer, 'min_std', 1e-8), dtype=sample_dtype)

    return params


def _apply_tf_norm_from_tf_params(data, tf_params: Dict):
    """
    Applies normalization using pre-extracted parameters (as TF constants) and pure
    TensorFlow operations, including TF control flow.
    """
    # --- Dynamically build the mask inside the TF graph ---
    mask = None
    mask_spec = tf_params.get('mask_spec')
    if mask_spec is not None:
        feature_dim = tf.shape(data)[-1]
        mask_spec_tensor = tf.constant(mask_spec)

        if mask_spec_tensor.dtype == tf.bool:
            tf.Assert(tf.equal(tf.shape(mask_spec_tensor)[0], feature_dim), ["Mask length mismatch"])
            mask = mask_spec_tensor
        else:
            indices = tf.cast(mask_spec_tensor, dtype=tf.int32)
            indices = tf.where(indices < 0, feature_dim + indices, indices)
            mask = tf.tensor_scatter_nd_update(
                tensor=tf.ones(shape=[feature_dim], dtype=tf.bool),
                indices=tf.expand_dims(indices, axis=1),
                updates=tf.zeros(tf.shape(indices), dtype=tf.bool)
            )

    # --- Define pure TF functions for each normalization type ---
    def zscore_fn():
        clipped_std = tf.clip_by_value(tf_params['std'], tf_params['min_std'], tf.float32.max)
        safe_std = tf.where(tf.abs(tf_params['std']) < MIN_SCALE_THRESHOLD, 1.0, clipped_std)
        normalized = (data - tf_params['mean']) / safe_std
        return tf.where(tf.abs(tf_params['std']) < MIN_SCALE_THRESHOLD, tf.zeros_like(normalized), normalized)

    def minmax_fn():
        delta = tf_params['high'] - tf_params['low']
        raw_range = tf_params['max'] - tf_params['min']
        safe_range = tf.where(tf.abs(raw_range) < MIN_SCALE_THRESHOLD, 1.0, raw_range)
        normalized = (data - tf_params['min']) / safe_range * delta + tf_params['low']
        return tf.where(tf.abs(raw_range) < MIN_SCALE_THRESHOLD, tf.zeros_like(normalized), normalized)

    def percentile_fn():
        delta = tf_params['high'] - tf_params['low']
        raw_range = tf_params['q99'] - tf_params['q01']
        safe_range = tf.where(tf.abs(raw_range) < MIN_SCALE_THRESHOLD, 1.0, raw_range)
        normalized = (data - tf_params['q01']) / safe_range * delta + tf_params['low']
        normalized = tf.where(tf.abs(raw_range) < MIN_SCALE_THRESHOLD, tf.zeros_like(normalized), normalized)
        return tf.clip_by_value(normalized, tf_params['low'], tf_params['high'])

    # --- Use TF control flow to select the normalization branch ---
    normalized = tf.switch_case(
        branch_index=tf_params['type_index'],
        branch_fns={
            0: zscore_fn,
            1: minmax_fn,
            2: percentile_fn,
        },
        default=lambda: data  # Identity or unsupported
    )

    if mask is not None:
        normalized = tf.where(mask, normalized, data)
    
    return normalized

class WrappedDataset(Dataset):
    """Wrapper for map-style datasets"""
    def __init__(self, dataset, processor=None):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        return sample if self.processor is None else self.processor(sample)


class WrappedIterableDataset(IterableDataset):
    """Wrapper for iterable datasets with processor support"""
    def __init__(self, dataset, processor=None):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
    
    def __iter__(self):
        for sample in self.dataset:
            if self.processor is not None:
                sample = self.processor(sample)
            yield sample



class MapToIterableDataset(IterableDataset):
    """Convert a map-style dataset to an iterable dataset with optional shuffling"""
    def __init__(self, dataset,*args, **kwargs):
        super().__init__()
        self.dataset = dataset
    
    def __iter__(self):
        for idx in range(len(self.dataset)):
            yield self.dataset[idx]


class MergedDataset(Dataset):
    """
    A dataset that merges multiple datasets into a single dataset.
    
    This is different from ConcatDataset in that it:
    1. Has a single dataset_id for identification
    2. Supports merged normalization statistics
    3. Maintains a consistent interface for the merged group
    
    The merged dataset concatenates all underlying datasets and provides
    a unified view with a single normalizer computed from merged statistics.
    
    Args:
        datasets: List of datasets to merge
        merged_id: Unique identifier for this merged dataset
    
    Example config:
        ```yaml
        datasets:
          - merged: transfer_cube_all  # merged_id
              - type: AlohaSimDataset
                name: sim_transfer_cube_scripted
                args: {...}
              - type: AlohaSimDataset
                name: sim_transfer_cube_human
                args: {...}
        ```
    """
    
    def __init__(self, datasets: List[Dataset], merged_id: str):
        if not datasets:
            raise ValueError("datasets list cannot be empty")
        
        self.datasets = datasets
        self.merged_id = merged_id
        
        # Set dataset_id for normalizer identification
        self.dataset_id = merged_id
        self.name = merged_id
        
        # Compute cumulative lengths for indexing
        self._lengths = [len(ds) for ds in datasets]
        self._cumulative_lengths = []
        cumsum = 0
        for length in self._lengths:
            self._cumulative_lengths.append(cumsum)
            cumsum += length
        self._total_length = cumsum
        
        # Collect source dataset info for debugging
        self._source_datasets = []
        for ds in datasets:
            ds_id = getattr(ds, 'dataset_id', getattr(ds, 'name', str(type(ds).__name__)))
            self._source_datasets.append(ds_id)
        
        logger.info(f"Created MergedDataset '{merged_id}' with {len(datasets)} datasets, "
                    f"total {self._total_length} samples")
        logger.info(f"  Source datasets: {self._source_datasets}")
        logger.info(f"  Sample counts: {self._lengths}")
    
    def __len__(self) -> int:
        return self._total_length
    
    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx = self._total_length + idx
        if idx < 0 or idx >= self._total_length:
            raise IndexError(f"Index {idx} out of range for MergedDataset of size {self._total_length}")
        
        # Find which dataset this index belongs to
        for i, (ds, cumlen) in enumerate(zip(self.datasets, self._cumulative_lengths)):
            if idx < cumlen + self._lengths[i]:
                local_idx = idx - cumlen
                return ds[local_idx]
        
        raise IndexError(f"Index {idx} could not be mapped to any dataset")
    
    def get_dataset_dir(self) -> str:
        """Return the merged_id as the dataset directory for compatibility."""
        return self.merged_id
    
    @property
    def num_episodes(self) -> int:
        """Return total number of episodes if all datasets have this attribute."""
        if all(hasattr(ds, 'num_episodes') for ds in self.datasets):
            return sum(ds.num_episodes for ds in self.datasets)
        return 0
    
    def extract_from_episode(self, episode_idx: int, keys: list):
        """
        Extract data from a specific episode across all merged datasets.
        
        This method maps the global episode index to the correct dataset.
        """
        # Map episode_idx to the correct dataset
        cumulative_episodes = 0
        for ds in self.datasets:
            if not hasattr(ds, 'num_episodes'):
                raise AttributeError(f"Dataset {getattr(ds, 'dataset_id', 'unknown')} "
                                     "does not have 'num_episodes' attribute")
            
            ds_num_episodes = ds.num_episodes
            if episode_idx < cumulative_episodes + ds_num_episodes:
                local_episode_idx = episode_idx - cumulative_episodes
                return ds.extract_from_episode(local_episode_idx, keys)
            cumulative_episodes += ds_num_episodes
        
        raise IndexError(f"Episode index {episode_idx} out of range")
    
    def get_dataset_statistics(self) -> dict:
        """Get merged statistics from all sub-datasets.
        
        This method collects statistics from each sub-dataset and merges them
        using the merge_stats function. This ensures accurate statistics for
        merged datasets, especially when sub-datasets have pre-computed stats
        (e.g., LeRobot datasets with stats.json).
        """
        from .normalize import merge_stats
        
        # Collect statistics from each sub-dataset
        sub_stats_list = []
        for ds in self.datasets:
            if hasattr(ds, 'get_dataset_statistics') and callable(getattr(ds, 'get_dataset_statistics')):
                stats = ds.get_dataset_statistics()
            else:
                # Fallback: compute stats using normalizer
                from .normalize import BaseNormalizer
                normalizer = BaseNormalizer(ds, dataset_name=getattr(ds, 'dataset_id', 'unknown'))
                stats = normalizer.all_stats
            sub_stats_list.append(stats)
        
        # Merge statistics
        merged_stats = merge_stats(sub_stats_list, keys=['state', 'action'])
        return merged_stats
    
    def __getattr__(self, name):
        """Forward unknown attributes to the first dataset for compatibility."""
        if name in ['datasets', 'merged_id', 'dataset_id', 'name', 
                    '_lengths', '_cumulative_lengths', '_total_length', '_source_datasets',
                    'get_dataset_statistics']:
            return object.__getattribute__(self, name)
        
        # Try to get from first dataset
        if self.datasets:
            return getattr(self.datasets[0], name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
