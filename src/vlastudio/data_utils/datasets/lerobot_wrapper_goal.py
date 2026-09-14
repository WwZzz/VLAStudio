from pprint import pprint
import torch.utils.data as tud
import torch
from huggingface_hub import HfApi
from typing import List, Union, Dict, Any, Optional
try:
    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
except ImportError:
    import lerobot
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
import warnings
from vlastudio.benchmark.utils import resize_with_pad
from vlastudio.data_utils.utils import ensure_uint8_image
from loguru import logger
from tqdm import tqdm

class WrappedLerobotDataset(tud.Dataset):
    """
    LeRobot dataset wrapper with goal image support.
    
    This wrapper extends the base WrappedLerobotDataset to include a future "goal" image
    in the reasoning dict. The goal image corresponds to a randomly selected future frame
    using the first camera's perspective.
    
    Goal image selection:
        - Randomly selects an action index k from valid indices
        - Valid indices: k > 0, k < chunk_size, and action[k] is not padded
        - Returns the image at frame (current_frame + k)
    
    Returns in reasoning dict:
        - goal_image: (C, H, W) tensor of the goal frame from first camera
        - goal_action_index: The randomly selected action index k
    """
    def __init__(self, 
            dataset_path_list: list, 
            camera_names: list=[], 
            root: str = None,
            chunk_size: int = 16,  
            ctrl_space: str = 'ee', 
            ctrl_type: str = 'delta',
            image_size: tuple = None,
            tolerance_s: float = 1e-3,
            state_key: Union[str, List[str]] = 'observation.state',
            action_key: Union[str, List[str]] = 'action',
            episode_filter: dict = None,  # Filter episodes by metadata fields, e.g., {"tasks": ["task1"], "episode_index": [0, 1, 2]}
            video_backend: str = None,
            *args, 
            **kwargs,
            ):
        super().__init__()
        self.chunk_size = chunk_size
        self.root = root
        self.state_key = state_key  # Can be str or List[str]
        self.action_key = action_key  # Can be str or List[str]
        self.episode_filter = episode_filter
        self.video_backend = video_backend
        
        # Get primary action key for delta_timestamps (use first key if list)
        self._primary_action_key = action_key[0] if isinstance(action_key, list) else action_key
        datasets = []
        data_metas = []
        dataset_dirs = []
        num_episodes = []
        num_frames = []
        all_camera_keys = dict()
        for data_path in dataset_path_list:
            ds_meta = LeRobotDatasetMetadata(data_path, root=self.root)
            
            # Filter episodes by provided filter conditions
            episodes_to_load = None
            if episode_filter is not None and len(episode_filter) > 0:
                episodes_to_load = self._filter_episodes(ds_meta, episode_filter)
                if episodes_to_load is not None and len(episodes_to_load) == 0:
                    warnings.warn(f"No episodes found matching filter {episode_filter} in dataset {data_path}")
                    continue
            
            delta_timestamps = {self._primary_action_key: [t / ds_meta.fps for t in range(chunk_size)]}
            dataset = LeRobotDataset(
                data_path, 
                root=self.root, 
                delta_timestamps=delta_timestamps, 
                tolerance_s=tolerance_s,
                episodes=episodes_to_load,
                video_backend = self.video_backend,
            )
            
            # Optimize: Remove unused columns from hf_dataset to reduce I/O
            # This is critical for performance - only load columns we actually need
            dataset = self._optimize_dataset_columns(dataset, ds_meta, camera_names)
            
            # Log filtering info and calculate actual frames count
            if episodes_to_load is not None:
                logger.info(f"Selected {len(episodes_to_load)} episodes from {data_path}")
                logger.info(f"Episode indices: {episodes_to_load[:10]}{'...' if len(episodes_to_load) > 10 else ''}")
                
                # Calculate actual frame count for filtered episodes
                # LeRobot doesn't actually reduce hf_dataset size, so we need to calculate manually
                actual_frames = sum(ds_meta.episodes[ep_idx]['length'] for ep_idx in episodes_to_load)
                # logger.info(f"Actual frames for selected episodes: {actual_frames}")
            else:
                actual_frames = dataset.num_frames
            
            data_metas.append(ds_meta)
            datasets.append(dataset)
            dataset_dirs.append(str(dataset.root))
            # Use actual dataset size after filtering
            num_episodes.append(dataset.num_episodes)
            num_frames.append(actual_frames)  # Use calculated frames, not dataset.num_frames
            all_camera_keys[data_path] = ds_meta.camera_keys
        self.dataset_path_list = dataset_path_list
        self.datasets = datasets
        self.dataset_metas= data_metas
        self.dataset_dirs = dataset_dirs
        self.per_dataset_num_episodes = num_episodes
        self.per_dataset_num_frames = num_frames
        self.cumulative_num_episodes = np.cumsum(self.per_dataset_num_episodes)
        self.cumulative_num_frames = np.cumsum(self.per_dataset_num_frames)
        self.per_dataset_episode_start = self.cumulative_num_episodes - np.array(self.per_dataset_num_episodes)
        self.per_dataset_frame_start = self.cumulative_num_frames - np.array(self.per_dataset_num_frames)
        self.total_frames = sum(self.per_dataset_num_frames)
        self.total_episodes = sum(num_episodes)
        self.camera_names = camera_names if isinstance(camera_names, list) else [camera_names]
        self.episode_ids = np.arange(sum(self.per_dataset_num_episodes))
        self.image_size = image_size
        self.ctrl_space = ctrl_space  # ['ee', 'joint', 'other']
        self.ctrl_type = ctrl_type  # ['abs', 'rel', 'delta']
        self.freq = self.dataset_metas[0].fps
        self.max_workers = 8
        self.initialize()
        
    def _optimize_dataset_columns(
        self, 
        dataset: LeRobotDataset, 
        ds_meta: LeRobotDatasetMetadata,
        camera_names: list
    ) -> LeRobotDataset:
        """
        Optimize dataset by removing unused columns and disabling unused video cameras.
        
        This is critical for performance - loading unused image/video data
        causes significant I/O waste even if we don't use them in __getitem__.
        
        LeRobot datasets have two types of camera data storage:
        1. Image type (ds_meta.image_keys): Images stored directly in hf_dataset columns
           - Optimized by removing unused columns from hf_dataset
        2. Video type (ds_meta.video_keys): Images stored in .mp4 files, decoded on-the-fly
           - Optimized by modifying meta.info['features'] to hide unused video cameras
           - This prevents LeRobotDataset._query_videos from decoding unused videos
        
        Args:
            dataset: LeRobotDataset instance
            ds_meta: Dataset metadata
            camera_names: List of camera names to keep (empty/None = keep all cameras)
            
        Returns:
            The same dataset with optimized hf_dataset and meta info
        """
        all_camera_keys = set(ds_meta.camera_keys)
        image_keys = set(ds_meta.image_keys)
        video_keys = set(ds_meta.video_keys)
        
        # Determine which cameras to keep
        if camera_names is not None and len(camera_names) > 0:
            cameras_to_keep = set(camera_names) & all_camera_keys
            if not cameras_to_keep:
                logger.warning(
                    f"None of the specified cameras {camera_names} found in dataset. "
                    f"Available cameras: {all_camera_keys}. Keeping all cameras."
                )
                cameras_to_keep = all_camera_keys
        else:
            # Keep all cameras
            cameras_to_keep = all_camera_keys
        
        # === Optimize Video Cameras ===
        # Modify meta.info['features'] to hide unused video cameras
        # This prevents LeRobotDataset._query_videos from decoding them
        video_cameras_to_disable = video_keys - cameras_to_keep
        if video_cameras_to_disable:
            logger.info(f"Disabling unused video cameras: {video_cameras_to_disable}")
            for vid_key in video_cameras_to_disable:
                if vid_key in dataset.meta.info['features']:
                    # Change dtype from 'video' to 'disabled_video' to exclude from video_keys
                    dataset.meta.info['features'][vid_key]['dtype'] = 'disabled_video'
            
            # Log remaining video cameras
            remaining_video_keys = [k for k, v in dataset.meta.info['features'].items() 
                                   if v.get('dtype') == 'video']
            logger.debug(f"Remaining active video cameras: {remaining_video_keys}")
        
        # === Optimize Image Columns in hf_dataset ===
        if dataset.hf_dataset is None:
            return dataset
        
        current_columns = set(dataset.hf_dataset.column_names)
        
        # Essential columns that are always needed
        essential_columns = {
            'index', 'episode_index', 'frame_index', 'timestamp', 'task_index',
        }
        
        # Add state keys (supports list)
        if isinstance(self.state_key, list):
            essential_columns.update(self.state_key)
        else:
            essential_columns.add(self.state_key)
        
        # Add action keys (supports list)
        if isinstance(self.action_key, list):
            essential_columns.update(self.action_key)
            # Add padding mask for primary action key
            essential_columns.add(f'{self._primary_action_key}_is_pad')
        else:
            essential_columns.add(self.action_key)
            essential_columns.add(f'{self.action_key}_is_pad')
        
        # Image columns to keep (intersection of cameras_to_keep and actual image columns in hf_dataset)
        image_cameras_to_keep = cameras_to_keep & image_keys & current_columns
        
        # Build the set of columns to keep
        columns_to_keep = essential_columns | image_cameras_to_keep
        
        # Find columns to remove
        columns_to_remove = [col for col in current_columns - columns_to_keep]
        
        if columns_to_remove:
            logger.info(
                f"Removing {len(columns_to_remove)} unused hf_dataset columns: "
                f"{columns_to_remove[:5]}{'...' if len(columns_to_remove) > 5 else ''}"
            )
            dataset.hf_dataset = dataset.hf_dataset.remove_columns(columns_to_remove)
            logger.debug(f"Remaining columns: {dataset.hf_dataset.column_names}")
        
        # Log final camera configuration
        final_image_keys = [k for k in dataset.hf_dataset.column_names if k in image_keys]
        final_video_keys = [k for k, v in dataset.meta.info['features'].items() 
                          if v.get('dtype') == 'video']
        logger.info(f"Active cameras - Image: {final_image_keys}, Video: {final_video_keys}")
        
        return dataset
    
    def _filter_episodes(self, ds_meta: LeRobotDatasetMetadata, episode_filter: dict) -> list:
        """
        Filter episodes based on metadata fields. This is a universal filtering method
        that works across all LeRobot datasets.
        
        Args:
            ds_meta: LeRobotDatasetMetadata object
            episode_filter: Dictionary containing filter conditions. Supported keys:
                - "episode_index": list of episode indices to include (directly)
                - "tasks": list of task names (if dataset has tasks)
                - "task_index": list of task indices (if dataset has tasks)
                - Any other field in episodes metadata can be used
                
        Returns:
            List of episode indices that match the filter conditions.
            Returns None if filtering is not possible.
            
        Examples:
            # Direct episode indices
            {"episode_index": [0, 1, 2, 5, 10]}
            
            # Filter by tasks (if dataset has tasks)
            {"tasks": ["reach-v2", "push-v2"]}
            
            # Filter by task index (if dataset has tasks)
            {"task_index": [0, 1, 2]}
            
            # Combine multiple filters (AND logic)
            {"tasks": ["reach-v2"], "episode_index": [0, 1, 2]}
        """
        if ds_meta.episodes is None:
            warnings.warn(f"Dataset metadata does not contain episodes information")
            return None
        
        # If episode_index is directly provided, use it
        if "episode_index" in episode_filter:
            episode_indices = episode_filter["episode_index"]
            if isinstance(episode_indices, (list, tuple)):
                # Validate indices
                valid_indices = [idx for idx in episode_indices if 0 <= idx < len(ds_meta.episodes)]
                if len(valid_indices) != len(episode_indices):
                    warnings.warn(f"Some episode indices are out of range. Using {len(valid_indices)}/{len(episode_indices)} valid indices.")
                return valid_indices
            else:
                warnings.warn(f"episode_index should be a list, got {type(episode_indices)}")
                return None
        
        # Start with all episodes
        selected_episodes = set(range(len(ds_meta.episodes)))
        
        # Apply each filter condition
        for filter_key, filter_values in episode_filter.items():
            if filter_key == "tasks":
                # Filter by task names
                if ds_meta.tasks is None:
                    warnings.warn(f"Dataset does not have tasks metadata, cannot filter by 'tasks'")
                    return None
                
                task_set = set(filter_values) if isinstance(filter_values, (list, tuple)) else {filter_values}
                episodes_for_tasks = set()
                
                for episode_idx in range(len(ds_meta.episodes)):
                    episode_tasks = ds_meta.episodes[episode_idx].get('tasks', None)
                    if episode_tasks is None:
                        continue
                    
                    # episode_tasks could be a list or single value
                    if isinstance(episode_tasks, (list, tuple)):
                        if any(task in task_set for task in episode_tasks):
                            episodes_for_tasks.add(episode_idx)
                    else:
                        if episode_tasks in task_set:
                            episodes_for_tasks.add(episode_idx)
                
                selected_episodes &= episodes_for_tasks
                
            elif filter_key == "task_index":
                # Filter by task indices
                if ds_meta.tasks is None:
                    warnings.warn(f"Dataset does not have tasks metadata, cannot filter by 'task_index'")
                    return None
                
                task_idx_set = set(filter_values) if isinstance(filter_values, (list, tuple)) else {filter_values}
                episodes_for_task_idx = set()
                
                for episode_idx in range(len(ds_meta.episodes)):
                    # Get task names for this episode
                    episode_tasks = ds_meta.episodes[episode_idx].get('tasks', None)
                    if episode_tasks is None:
                        continue
                    
                    # Convert task names to indices
                    if isinstance(episode_tasks, (list, tuple)):
                        episode_task_indices = [ds_meta.get_task_index(task) for task in episode_tasks]
                    else:
                        episode_task_indices = [ds_meta.get_task_index(episode_tasks)]
                    
                    # Check if any task index matches
                    if any(idx in task_idx_set for idx in episode_task_indices if idx is not None):
                        episodes_for_task_idx.add(episode_idx)
                
                selected_episodes &= episodes_for_task_idx
                
            else:
                # Generic filter by any episode metadata field
                filter_value_set = set(filter_values) if isinstance(filter_values, (list, tuple)) else {filter_values}
                episodes_for_field = set()
                
                for episode_idx in range(len(ds_meta.episodes)):
                    episode_value = ds_meta.episodes[episode_idx].get(filter_key, None)
                    if episode_value is None:
                        continue
                    
                    # Handle list values
                    if isinstance(episode_value, (list, tuple)):
                        if any(val in filter_value_set for val in episode_value):
                            episodes_for_field.add(episode_idx)
                    else:
                        if episode_value in filter_value_set:
                            episodes_for_field.add(episode_idx)
                
                if len(episodes_for_field) == 0:
                    warnings.warn(f"No episodes found with {filter_key} in {filter_values}")
                
                selected_episodes &= episodes_for_field
        
        return sorted(list(selected_episodes))
        
    def initialize(self):
        self.episode_len = self.get_episode_len() 
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(self.episode_len)
        
        # Build index mapping table for fast lookup (critical for performance!)
        self._build_index_mapping()
        
        # Log dataset info
        logger.info(f"Dataset initialized: {self.total_episodes} episodes, {self.total_frames} frames")
        # logger.info(f"Episode lengths: min={min(self.episode_len)}, max={max(self.episode_len)}, mean={np.mean(self.episode_len):.1f}")
        return
    
    def _build_index_mapping(self):
        """
        Pre-compute index mapping for filtered datasets to avoid runtime loops.
        This is CRITICAL for performance with multi-worker dataloaders!
        """
        self.index_to_episode_map = []  # [(dataset_idx, episode_idx, frame_offset_in_episode, actual_hf_index), ...]
        
        for dataset_idx, dataset in enumerate(self.datasets):
            if hasattr(dataset, 'episodes') and dataset.episodes is not None:
                # Filtered dataset - need to build mapping
                ds_meta = self.dataset_metas[dataset_idx]
                for ep_idx in dataset.episodes:
                    ep_len = ds_meta.episodes[ep_idx]['length']
                    ep_start_in_hf = ds_meta.episodes[ep_idx]['dataset_from_index']
                    
                    for frame_offset in range(ep_len):
                        actual_hf_idx = ep_start_in_hf + frame_offset
                        self.index_to_episode_map.append((dataset_idx, ep_idx, frame_offset, actual_hf_idx))
            else:
                # No filtering - direct 1:1 mapping
                total_frames = self.per_dataset_num_frames[dataset_idx]
                for frame_idx in range(total_frames):
                    self.index_to_episode_map.append((dataset_idx, -1, -1, frame_idx))
        
        # logger.info(f"Built index mapping table with {len(self.index_to_episode_map)} entries")
        
    def _load_file_into_memory(self, *args, **kwargs):
        warnings.warn("Cannot load LerobotDataset into memory")
        return
    
    def _load_all_episodes_into_memory(self):
        warnings.warn("Cannot load LerobotDataset into memory")
        return
    
    def get_episode_len(self):
        """
        Get the length of each episode in the filtered dataset.
        Always use metadata for speed - never iterate through hf_dataset.
        """
        episode_lens = []
        for dataset in self.datasets:
            ds_meta = dataset.meta
            
            # Determine which episodes to use
            if hasattr(dataset, 'episodes') and dataset.episodes is not None:
                # Filtered dataset - use specified episodes
                ep_indices = dataset.episodes
            else:
                # No filtering - use all episodes from metadata
                ep_indices = range(len(ds_meta.episodes))
            
            # Get length from metadata (fast, no I/O)
            for ep_idx in tqdm(ep_indices):
                episode_lens.append(ds_meta.episodes[ep_idx]['length'])
        
        return episode_lens
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        # Use index mapping table length for accuracy
        if hasattr(self, 'index_to_episode_map'):
            return len(self.index_to_episode_map)
        return self.total_frames
        
    @property
    def num_episodes(self):
        return self.total_episodes

    def get_dataset_dir(self):
        """Get the dataset directory path."""
        return self.dataset_dirs[0]
       
    def get_freq(self):
        """Get the dataset frequency."""
        return self.freq
    
    def _locate_dataset_for_transition(self, index):
        """
        Locate which dataset and frame index for a given wrapped index.
        Uses pre-computed mapping table for O(1) lookup - FAST!
        """
        assert index < len(self.index_to_episode_map), f"Index {index} out of range {len(self.index_to_episode_map)}"
        
        # O(1) lookup from pre-computed table!
        dataset_idx, ep_idx, frame_offset, actual_hf_idx = self.index_to_episode_map[index]
        
        return int(dataset_idx), int(actual_hf_idx)
    
    def _locate_transition(self, index):
        """
        Convert sample index to episode index and internal timestep.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (episode_id, start_ts)
        """
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)  # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts
    
    def _get_data_from_sample(
        self,
        sample: Dict[str, Any],
        keys: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        Get data from sample by key(s). If keys is a list, concatenate the data.
        
        Args:
            sample: Sample dictionary from LeRobotDataset
            keys: Single key string or list of keys to concatenate
            
        Returns:
            Tensor of data (concatenated along last axis if multiple keys)
        """
        if isinstance(keys, str):
            # Single key - return directly
            return sample[keys]
        else:
            # List of keys - concatenate along last axis
            data_parts = []
            for key in keys:
                if key in sample:
                    part = sample[key]
                    if not isinstance(part, torch.Tensor):
                        part = torch.tensor(part)
                    data_parts.append(part)
                else:
                    logger.warning(f"Key '{key}' not found in sample, skipping")
            
            if not data_parts:
                raise KeyError(f"None of the keys {keys} found in sample")
            
            # Concatenate along last axis
            return torch.cat(data_parts, dim=-1)
    
    def _get_stats_by_keys(
        self,
        stats: Dict[str, Any],
        keys: Union[str, List[str]],
    ) -> Dict[str, np.ndarray]:
        """
        Get statistics for key(s). If keys is a list, concatenate the stats.
        
        Args:
            stats: Statistics dictionary from dataset metadata
            keys: Single key string or list of keys
            
        Returns:
            Dictionary of concatenated statistics
        """
        if isinstance(keys, str):
            # Single key
            return stats.get(keys, {})
        else:
            # List of keys - concatenate stats
            stat_names = ['mean', 'std', 'min', 'max', 'q01', 'q99']
            result = {}
            
            for stat_name in stat_names:
                parts = []
                for key in keys:
                    key_stats = stats.get(key, {})
                    if stat_name in key_stats:
                        parts.append(np.asarray(key_stats[stat_name]))
                
                if parts:
                    result[stat_name] = np.concatenate(parts, axis=-1)
            
            return result
    
    def extract_from_episode(self, episode_idx, keyname=[]):
        """Extract specific features from an episode. Supports concatenated keys."""
        dataset_idx = np.argmax(self.cumulative_num_episodes > episode_idx)
        inner_episode_idx = episode_idx - self.per_dataset_episode_start[dataset_idx]
        ds_meta = self.dataset_metas[dataset_idx]
        all_features = ds_meta.features
        preserved_keys = []
        ori_k = {}
        
        # Handle state keys (supports list)
        if 'state' in keyname:
            if isinstance(self.state_key, list):
                preserved_keys.extend(self.state_key)
                for k in self.state_key:
                    ori_k[k] = 'state'
            else:
                preserved_keys.append(self.state_key)
                ori_k[self.state_key] = 'state'
        
        # Handle action keys (supports list)
        if 'action' in keyname:
            if isinstance(self.action_key, list):
                preserved_keys.extend(self.action_key)
                for k in self.action_key:
                    ori_k[k] = 'action'
            else:
                preserved_keys.append(self.action_key)
                ori_k[self.action_key] = 'action'
        
        if 'image' in keyname or 'images' in keyname:
            preserved_keys.extend(ds_meta.camera_keys)
            for i,k in enumerate(ds_meta.camera_keys):
                ori_k[k] = f'images_{i}'
            ignore_image = False
        else:
            ignore_image = all([ckey not in keyname for ckey in ds_meta.camera_keys])
        ignore_keys = [feat for feat in all_features if feat not in preserved_keys and feat not in keyname]
        subdata = LeRobotDataset(
            self.dataset_path_list[dataset_idx], 
            episodes=[inner_episode_idx],
        )
        if ignore_image:
            for k,v in subdata.meta.features.items():
                if v['dtype']=='video': subdata.meta.info['features'][k]['dtype'] = 'hidden'
        extracted_feats = [{k:s[k].numpy() for k in preserved_keys if k in s} for s in subdata]
        if ignore_image:
            for k,v in subdata.meta.features.items():
                if v['dtype']=='hidden': subdata.meta.info['features'][k]['dtype'] = 'video'
        
        # Build result dict, concatenating keys that map to same output
        res_dict = {}
        for output_name in set(ori_k.values()):
            keys_for_output = [k for k, v in ori_k.items() if v == output_name]
            if len(keys_for_output) == 1:
                k = keys_for_output[0]
                if k in extracted_feats[0]:
                    res_dict[output_name] = np.stack([efeat[k] for efeat in extracted_feats])
            else:
                # Concatenate multiple keys
                parts = []
                for k in keys_for_output:
                    if k in extracted_feats[0]:
                        parts.append(np.stack([efeat[k] for efeat in extracted_feats]))
                if parts:
                    res_dict[output_name] = np.concatenate(parts, axis=-1)
        
        return res_dict
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing the sample data
        """
        # find dataset_id by index: start_index for the target dataset by the num_frames of each dataset
        # find sample_id in dataset: index-start_index
        
        dataset_idx, start_ts = self._locate_dataset_for_transition(index)
        sample = self.datasets[dataset_idx][start_ts]
        data_dict = {}
        episode_id = self.per_dataset_episode_start[dataset_idx] + sample['episode_index'].item()
        raw_lang = sample['task']
        
        # Get action (supports single key or list of keys to concatenate)
        action = self._get_data_from_sample(sample, self.action_key)
        
        # Get state (supports single key or list of keys to concatenate)
        state = self._get_data_from_sample(sample, self.state_key)
        
        # Get timestamp/frame_index with fallback for different dataset versions
        if 'frame_index' in sample:
            timestamp = sample['frame_index'].item()
        elif 'index' in sample:
            timestamp = sample['index'].item()
        else:
            # Fallback to using the index from locate
            timestamp = start_ts
        
        # Get padding mask
        pad_key = f'{self._primary_action_key}_is_pad'
        if pad_key in sample:
            is_pad = sample[pad_key]
        elif 'action_is_pad' in sample:
            is_pad = sample['action_is_pad']
        else:
            is_pad = torch.zeros(action.shape[0], dtype=torch.bool)
        
        # Process images
        # LeRobot datasets have two types of camera storage:
        # 1. Image type: stored in hf_dataset columns (ds_meta.image_keys)
        # 2. Video type: stored in .mp4 files, decoded by LeRobotDataset._query_videos (ds_meta.video_keys)
        # Both types are returned in the sample dict by LeRobotDataset.__getitem__
        ds_meta = self.datasets[dataset_idx].meta
        all_camera_keys = ds_meta.camera_keys  # Includes both image and video keys
        
        # Determine which cameras to use
        if len(self.camera_names) > 0:
            # Use specified cameras (filter to only those available in this dataset)
            cam_keys = [k for k in self.camera_names if k in all_camera_keys]
        else:
            # Use all cameras from this dataset
            cam_keys = all_camera_keys
        
        # Collect images from sample (works for both image and video type cameras)
        images_list = []
        for cam_key in cam_keys:
            if cam_key in sample:
                cam_img = sample[cam_key]
                if self.image_size is not None:
                    cam_img = resize_with_pad(cam_img.unsqueeze(0), height=self.image_size[1], width=self.image_size[0])
                else:
                    cam_img = cam_img.unsqueeze(0)
                images_list.append(cam_img)
            else:
                logger.warning(f"Camera key '{cam_key}' not found in sample, skipping")
        
        if images_list:
            images = torch.cat(images_list, dim=0)
        else:
            # Fallback: create empty tensor if no images found
            logger.warning(f"No camera images found for index {index}")
            images = torch.zeros(1, 3, 224, 224, dtype=torch.uint8)
        
        # Safety check: ensure images are uint8 with values in [0, 255]
        images = ensure_uint8_image(images)
        
        # === Get goal image (future frame corresponding to a random valid action index k) ===
        reasoning = {}
        goal_image = None
        goal_action_idx = None
        
        # Determine valid action indices: > 0, < chunk_size, and not padded
        # is_pad is a boolean tensor of shape (chunk_size,)
        valid_indices = []
        for k in range(1, self.chunk_size):  # Start from 1 to ensure not current frame
            if k < len(is_pad) and not is_pad[k].item():
                valid_indices.append(k)
        
        # Randomly select one valid index
        if valid_indices:
            goal_action_idx = valid_indices[np.random.randint(len(valid_indices))]
        else:
            # Fallback: if all actions are padded, use index 1 (or chunk_size-1 if only 1 action)
            goal_action_idx = min(1, self.chunk_size - 1)
        
        # Get the first camera key (for goal image)
        first_cam_key = cam_keys[0] if cam_keys else None
        
        if first_cam_key is not None and goal_action_idx is not None:
            # Calculate the goal frame index
            # The goal image corresponds to the frame at current_frame + goal_action_index
            ds_meta = self.datasets[dataset_idx].meta
            dataset = self.datasets[dataset_idx]
            
            # Get current episode info
            current_episode_idx = sample['episode_index'].item()
            current_frame_in_episode = sample['frame_index'].item() if 'frame_index' in sample else timestamp
            
            # Get episode length from metadata
            if hasattr(dataset, 'episodes') and dataset.episodes is not None:
                # Filtered dataset - find the actual episode
                ep_info = ds_meta.episodes[current_episode_idx]
            else:
                ep_info = ds_meta.episodes[current_episode_idx]
            
            episode_length = ep_info['length']
            episode_start_idx = ep_info['dataset_from_index']
            
            # Calculate goal frame index within episode
            goal_frame_in_episode = current_frame_in_episode + goal_action_idx
            
            # Clamp to episode boundary (don't go beyond episode end)
            if goal_frame_in_episode >= episode_length:
                goal_frame_in_episode = episode_length - 1
                # Update the actual goal action index used
                goal_action_idx = max(1, goal_frame_in_episode - current_frame_in_episode)
            
            # Calculate absolute index in dataset
            goal_abs_idx = episode_start_idx + goal_frame_in_episode
            
            try:
                # Get the goal frame sample
                goal_sample = dataset[goal_abs_idx]
                
                # Extract goal image from first camera
                if first_cam_key in goal_sample:
                    goal_img = goal_sample[first_cam_key]
                    
                    # Apply same resize as main images
                    if self.image_size is not None:
                        goal_img = resize_with_pad(goal_img.unsqueeze(0), 
                                                   height=self.image_size[1], 
                                                   width=self.image_size[0])
                        goal_img = goal_img.squeeze(0)  # Remove batch dim: (C, H, W)
                    
                    # Ensure uint8
                    goal_img = ensure_uint8_image(goal_img)
                    goal_image = goal_img
                    
            except Exception as e:
                logger.warning(f"Failed to get goal image at index {goal_abs_idx}: {e}")
                goal_image = None
        
        # Build reasoning dict with goal image info
        reasoning = {
            'goal_image': goal_image,  # (C, H, W) or None
            'goal_action_index': goal_action_idx,  # The randomly selected action index k (> 0, not padded)
        }
        
        data_dict = {
            'image': images,
            'state': state,
            'action': action,
            'is_pad': is_pad,
            'raw_lang': raw_lang,
            'reasoning': reasoning,
            'timestamp': timestamp,  
            'episode_id': episode_id,
        }  
        return data_dict

    def get_dataset_statistics(self):
        """Get dataset statistics. Supports concatenated keys."""
        meta_stats = self.dataset_metas[0].stats
        
        # Get state stats (supports list of keys)
        state_stats = self._get_stats_by_keys(meta_stats, self.state_key)
        
        # Get action stats (supports list of keys)
        action_stats = self._get_stats_by_keys(meta_stats, self.action_key)
        
        # Add q01/q99 if not present (use min/max as fallback)
        if state_stats and 'q01' not in state_stats:
            state_stats['q01'] = state_stats.get('min', np.zeros(1))
            state_stats['q99'] = state_stats.get('max', np.ones(1))
        if action_stats and 'q01' not in action_stats:
            action_stats['q01'] = action_stats.get('min', np.zeros(1))
            action_stats['q99'] = action_stats.get('max', np.ones(1))
        
        stats = {
            'state': state_stats,
            'action': action_stats,
            'num_episodes': self.total_episodes,
            'num_transitions': self.total_frames,
        }
        return stats
    
        
if __name__=='__main__':
    ##################################### Liberos #################################
    dataset = WrappedLerobotDataset(
            ["HuggingFaceVLA/libero"], 
            tolerance_s=10.0,
        )
    print(f"Total episodes: {dataset.total_episodes}")
    print(f"Total frames: {dataset.total_frames}")
    dataset = WrappedLerobotDataset(
            ["HuggingFaceVLA/libero"], 
            tolerance_s=10.0,
            episode_filter={"task_index": [24]}
        )
    print(f"Task0 Total episodes: {dataset.total_episodes}")
    print(f"Task0 Total frames: {dataset.total_frames}")
    ##################################### MetaWorld #################################
    # # Example 1: Load dataset without filtering
    # print("=" * 60)
    # print("Example 1: Load entire dataset (no filtering)")
    # print("=" * 60)
    # dataset = WrappedLerobotDataset(["lerobot/metaworld_mt50", ], tolerance_s=10.0)
    # print(f"Total episodes: {dataset.total_episodes}")
    # print(f"Total frames: {dataset.total_frames}")
    
    # # Print available tasks if exists
    # print("\nDataset info:")
    # if dataset.dataset_metas[0].tasks is not None:
    #     print(f"Has tasks: Yes ({len(dataset.dataset_metas[0].tasks)} tasks)")
    #     print("Available tasks:", list(dataset.dataset_metas[0].tasks.index[:5]))
    # else:
    #     print("Has tasks: No")
    
    # # Example 2: Filter by episode_index directly (works for all datasets)
    # print("\n" + "=" * 60)
    # print("Example 2: Filter by episode_index (universal method)")
    # print("=" * 60)
    # print("Loading episodes [0, 1, 2, 5, 10]")
    # dataset_by_index = WrappedLerobotDataset(
    #     ["lerobot/metaworld_mt50"], 
    #     tolerance_s=10.0,
    #     episode_filter={"episode_index": [0, 1, 2, 5, 10]}
    # )
    # print(f"Filtered episodes: {dataset_by_index.total_episodes}")
    # print(f"Filtered frames: {dataset_by_index.total_frames}")
    # print(f"Dataset length (__len__): {len(dataset_by_index)}")
    # print(f"Episode lengths: {dataset_by_index.episode_len[:10]}")  # Show first 10
    
    # # Example 3: Filter by tasks (only if dataset has tasks)
    # print("\n" + "=" * 60)
    # print("Example 3: Filter by tasks (if available)")
    # print("=" * 60)
    
    # if dataset.dataset_metas[0].tasks is not None and len(dataset.dataset_metas[0].tasks) > 0:
    #     example_tasks = list(dataset.dataset_metas[0].tasks.index[:2])
    #     print(f"Filtering by tasks: {example_tasks}")
        
    #     dataset_by_task = WrappedLerobotDataset(
    #         ["lerobot/metaworld_mt50"], 
    #         tolerance_s=10.0,
    #         episode_filter={"tasks": example_tasks}
    #     )
    #     print(f"Filtered episodes: {dataset_by_task.total_episodes}")
    #     print(f"Filtered frames: {dataset_by_task.total_frames}")
    #     print(f"Dataset length (__len__): {len(dataset_by_task)}")
    #     print(f"First 5 episode lengths: {dataset_by_task.episode_len[:5]}")
    # else:
    #     print("No tasks found in dataset metadata")
    

    # print(f"Filtered frames: {dataset_by_index.total_frames}")
    # print(f"Dataset length (__len__): {len(dataset_by_index)}")
    # print(f"Episode lengths: {dataset_by_index.episode_len[:10]}")  # Show first 10
    
    # # Example 3: Filter by tasks (only if dataset has tasks)
    # print("\n" + "=" * 60)
    # print("Example 3: Filter by tasks (if available)")
    # print("=" * 60)
    
    # if dataset.dataset_metas[0].tasks is not None and len(dataset.dataset_metas[0].tasks) > 0:
    #     example_tasks = list(dataset.dataset_metas[0].tasks.index[:2])
    #     print(f"Filtering by tasks: {example_tasks}")
        
    #     dataset_by_task = WrappedLerobotDataset(
    #         ["lerobot/metaworld_mt50"], 
    #         tolerance_s=10.0,
    #         episode_filter={"tasks": example_tasks}
    #     )
    #     print(f"Filtered episodes: {dataset_by_task.total_episodes}")
    #     print(f"Filtered frames: {dataset_by_task.total_frames}")
    #     print(f"Dataset length (__len__): {len(dataset_by_task)}")
    #     print(f"First 5 episode lengths: {dataset_by_task.episode_len[:5]}")
    # else:
    #     print("No tasks found in dataset metadata")
    