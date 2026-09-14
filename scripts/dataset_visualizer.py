#!/usr/bin/env python3
"""
Dataset Visualizer - Fast visualization for IL-Studio datasets.

Usage:
    python scripts/dataset_visualizer.py --task <task_config>

Example:
    python scripts/dataset_visualizer.py --task sim_transfer_cube_scripted
"""

import os
import sys
import argparse
import threading
import tempfile
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import torch
import cv2
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from vlastudio.configs.loader import ConfigLoader
from vlastudio.data_utils.utils import _create_dataset_from_config, is_map_data
from torch.utils.data import DataLoader, Subset


# ============================================================================
# Configuration
# ============================================================================

@dataclass 
class Config:
    max_cached_episodes: int = 100  # Max episodes in cache (increased for smooth preview)
    image_size: Tuple[int, int] = (320, 240)
    preload_workers: int = 2  # Background thread workers
    dataloader_workers: int = 4  # PyTorch DataLoader workers for parallel loading
    default_fps: int = 10
    cache_window: int = 5  # Keep K episodes before and after current cached
    cache_expire_seconds: int = 600  # Expire cache after 10 minutes of no access
    cleanup_interval_seconds: int = 120  # Run cleanup every 2 minutes
    max_dropdown_episodes: int = 200  # Max episodes to show in dropdown
    max_episode_samples: int = 500  # Max samples to load per episode (subsample if longer)
    long_episode_threshold: int = 300  # Episodes longer than this will be subsampled


CFG = Config()


# ============================================================================
# Episode Manager with Raw/Normalized Data Support
# ============================================================================

class EpisodeManager:
    """Episode loading with normalization toggle support and lazy indexing."""
    
    def __init__(self):
        self.task_config = None
        self.current_task_name = ""  # Track current loaded task
        self.datasets = []
        self.current_dataset = None
        self.current_name = ""
        
        # Episode data - lazy loading with access timestamps
        self.episode_map: Dict[int, List[int]] = {}  # episode_id -> [sample_indices]
        self.episode_cache: Dict[int, List[Dict]] = {}
        self.cache_access_time: Dict[int, float] = {}  # episode_id -> last access timestamp
        self.cache_order: List[int] = []
        
        # Episode range index for fast lookup: episode_id -> (start_idx, end_idx)
        # This enables O(1) lookup for known episodes and O(log n) estimation for unknown ones
        self.episode_range: Dict[int, Tuple[int, int]] = {}
        
        # Video cache - store generated video paths with timestamps
        self.video_cache: Dict[int, str] = {}  # episode_id -> video_path
        self.video_access_time: Dict[int, float] = {}  # episode_id -> last access timestamp
        self.video_cache_order: List[int] = []
        self.max_video_cache: int = 20  # Limit video cache size
        
        # Priority cache for instant startup
        self.priority_visuals: Dict[int, Dict] = {} # episode_id -> {traj, curves, etc}
        
        # Lazy indexing state
        self._index_complete = False
        self._indexed_up_to = 0  # How many samples have been indexed
        self._discovered_episodes: List[int] = []  # Ordered list of discovered episodes
        self._index_lock = threading.Lock()
        self._bg_indexing = False  # Background indexing in progress
        
        # Normalization stats (for denormalization)
        self.action_stats = None
        self.state_stats = None
        self.norm_type = 'zscore'
        
        # Threading - with safety check
        self._executor = None
        self.loading: set = set()
        self.video_generating: set = set()  # Episodes currently generating videos
        self.lock = threading.Lock()
        
        # Callback for UI updates
        self._episode_update_callback = None
        
        # Global stats cache
        self._stats_cache: Dict[str, Dict] = {}
        
        # Background cleanup
        self._cleanup_running = False
        self._start_cleanup_thread()
        
        # Current episode tracking for smart cache management
        self._current_episode: Optional[int] = None

    @property
    def executor(self):
        """Lazy and safe executor initialization."""
        with self.lock:
            if self._executor is None or self._executor._shutdown:
                self._executor = ThreadPoolExecutor(max_workers=CFG.preload_workers)
            return self._executor
    
    def _start_cleanup_thread(self):
        """Start background thread for periodic cache cleanup."""
        if self._cleanup_running:
            return
        
        self._cleanup_running = True
        
        def _cleanup_loop():
            while self._cleanup_running:
                time.sleep(CFG.cleanup_interval_seconds)
                self._cleanup_expired_cache()
        
        cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired_cache(self):
        """Remove episodes that haven't been accessed for a while.
        
        IMPORTANT: Never remove episodes within cache_window of current episode.
        """
        current_time = time.time()
        expire_threshold = CFG.cache_expire_seconds
        K = CFG.cache_window
        current_ep = self._current_episode
        
        # Define protected range around current episode
        protected_range = set()
        if current_ep is not None:
            for d in range(-K, K + 1):
                protected_range.add(current_ep + d)
        
        # Cleanup episode cache
        expired_episodes = []
        with self.lock:
            for eid, access_time in list(self.cache_access_time.items()):
                # Skip protected episodes
                if eid in protected_range:
                    continue
                if current_time - access_time > expire_threshold:
                    expired_episodes.append(eid)
            
            for eid in expired_episodes:
                if eid in self.episode_cache:
                    del self.episode_cache[eid]
                if eid in self.cache_access_time:
                    del self.cache_access_time[eid]
                if eid in self.cache_order:
                    self.cache_order.remove(eid)
        
        # Cleanup video cache
        expired_videos = []
        with self.lock:
            for eid, access_time in list(self.video_access_time.items()):
                # Skip protected episodes
                if eid in protected_range:
                    continue
                if current_time - access_time > expire_threshold:
                    expired_videos.append(eid)
            
            for eid in expired_videos:
                video_path = self.video_cache.pop(eid, None)
                if video_path and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except:
                        pass
                if eid in self.video_access_time:
                    del self.video_access_time[eid]
                if eid in self.video_cache_order:
                    self.video_cache_order.remove(eid)
        
        if expired_episodes or expired_videos:
            print(f"[Cleanup] Removed {len(expired_episodes)} episodes, {len(expired_videos)} videos from cache")
    
    def get_cache_stats(self) -> Dict:
        """Get current cache statistics."""
        with self.lock:
            return {
                'episodes_cached': len(self.episode_cache),
                'videos_cached': len(self.video_cache),
                'max_episodes': CFG.max_cached_episodes,
                'max_videos': self.max_video_cache,
                'expire_seconds': CFG.cache_expire_seconds,
            }
    
    def clear_all_caches(self):
        """Clear all caches when switching to a different task."""
        with self.lock:
            # Clear episode cache
            self.episode_cache.clear()
            self.cache_access_time.clear()
            self.cache_order.clear()
            self.episode_map.clear()
            self.episode_range.clear()
            
            # Clear video cache and delete temp files
            for video_path in self.video_cache.values():
                if video_path and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except:
                        pass
            self.video_cache.clear()
            self.video_access_time.clear()
            self.video_cache_order.clear()
            
            # Clear priority visuals
            self.priority_visuals.clear()
            
            # Clear stats cache
            self._stats_cache.clear()
            
            # Reset indexing state
            self._index_complete = False
            self._indexed_up_to = 0
            self._discovered_episodes = []
            self._estimated_total_episodes = None
            self._bg_indexing = False
            
            # Reset current tracking
            self._current_episode = None
            self.loading.clear()
            self.video_generating.clear()
        
        print("🧹 All caches cleared")
    
    def load_task(self, task_name: str, force_reload: bool = False) -> Tuple[bool, str, Dict]:
        """Load task config and datasets. Skip if already loaded.
        
        OPTIMIZED: Skip normalization during load - use cached stats or load lazily.
        This makes initial load MUCH faster.
        
        Args:
            task_name: Task configuration name
            force_reload: If True, force reload even if already loaded
        """
        # CRITICAL: Check if already loaded to avoid re-initialization
        if not force_reload and task_name == self.current_task_name and self.task_config and self.datasets:
            # Already loaded, return success WITHOUT re-selecting dataset
            print(f"✅ Task '{task_name}' already loaded, skipping re-initialization")
            return True, f"Using already loaded task: {task_name}", self.task_config.get('meta', {})
        
        # Different task - clear all caches before loading new one
        if task_name != self.current_task_name and self.current_task_name:
            print(f"🔄 Switching from '{self.current_task_name}' to '{task_name}' - clearing caches...")
            self.clear_all_caches()
            
        try:
            class Args:
                def __init__(self):
                    self.task = task_name
                    self.policy = 'act'
                    self.training_config = 'default'
                    self.output_dir = '/tmp/viz'
                    self.unknown_args = []
            
            cfg_loader = ConfigLoader(args=Args(), unknown_args=[])
            self.task_config, path = cfg_loader.load_task(task_name)
            self.current_task_name = task_name  # Update current task
            
            # Load datasets DIRECTLY without normalization (FAST!)
            # Normalization stats will be loaded from cache in _load_norm_stats()
            print("⚡ Loading datasets (skipping normalization for speed)...")
            self.datasets = []
            datasets_config = self.task_config.get('datasets', [])
            for dc in datasets_config:
                try:
                    ds = _create_dataset_from_config(dc, None)
                    name = dc.get('name', 'unknown')
                    self.datasets.append({'dataset': ds, 'name': name})
                    print(f"  ✅ Loaded {name}")
                except Exception as e:
                    print(f"  ⚠️ Skip {dc.get('name')}: {e}")
            
            # Get normalization type
            meta = self.task_config.get('meta', {})
            self.norm_type = meta.get('action_normalize', 'zscore')
            
            if self.datasets:
                self._select_dataset(0)
            
            return True, path, meta
            
        except Exception as e:
            import traceback
            return False, f"{e}\n{traceback.format_exc()}", {}
    
    def _select_dataset(self, idx: int):
        """Select dataset - FAST initialization, defer heavy work to background.
        
        OPTIMIZED: Only does minimal setup, episode 0 scan happens on first access.
        """
        self.current_dataset = self.datasets[idx]['dataset']
        self.current_name = self.datasets[idx]['name']
        self.episode_map.clear()
        self.episode_cache.clear()
        self.cache_order.clear()
        self.episode_range.clear()  # Clear episode range index
        
        # Reset lazy indexing state
        self._index_complete = False
        self._indexed_up_to = 0
        self._discovered_episodes = []
        self._estimated_total_episodes = None
        
        # Try to load normalization stats from cache (fast)
        self._load_norm_stats()
        
        if not is_map_data(self.current_dataset):
            return
        
        # Get dataset length (usually fast - just metadata)
        ds_len = len(self.current_dataset)
        print(f"⚡ Dataset has {ds_len} samples")
        
        # DEFER episode estimation to background - accessing last sample can be slow
        def _estimate_episodes_bg():
            if ds_len > 0:
                try:
                    last_sample = self.current_dataset[ds_len - 1]
                    last_eid = last_sample.get('episode_id', 0)
                    if isinstance(last_eid, (torch.Tensor, np.ndarray)):
                        last_eid = int(last_eid.item() if hasattr(last_eid, 'item') else last_eid)
                    self._estimated_total_episodes = last_eid + 1
                    print(f"⚡ Estimated {self._estimated_total_episodes} episodes")
                except Exception as e:
                    print(f"Could not estimate episode count: {e}")
        
        # Run estimation in background
        threading.Thread(target=_estimate_episodes_bg, daemon=True).start()
        
        # FAST: Use binary search to find episode 0 boundary
        print("⚡ Finding episode 0 boundary using binary search...")
        ep0_end = self._find_episode_boundary_fast(0, ds_len)
        
        if ep0_end > 0:
            # Record episode 0 range
            self.episode_range[0] = (0, ep0_end - 1)
            self.episode_map[0] = list(range(ep0_end))
            self._discovered_episodes.append(0)
            print(f"✅ Episode 0: {ep0_end} samples (indices 0-{ep0_end - 1})")
            
            # Also record episode 1 start if we found it
            if ep0_end < ds_len:
                try:
                    s = self.current_dataset[ep0_end]
                    s_eid = s.get('episode_id', 0)
                    if isinstance(s_eid, (torch.Tensor, np.ndarray)):
                        s_eid = int(s_eid.item() if hasattr(s_eid, 'item') else s_eid)
                    if s_eid == 1:
                        self._discovered_episodes.append(1)
                        print(f"   Episode 1 starts at index {ep0_end}")
                except:
                    pass
        else:
            print(f"⚠️ Could not find episode 0 boundary")
    
    def _get_episode_id_at(self, idx: int) -> Optional[int]:
        """Get episode ID at a given index. Returns None on error."""
        try:
            s = self.current_dataset[idx]
            eid = s.get('episode_id', 0)
            if isinstance(eid, (torch.Tensor, np.ndarray)):
                eid = int(eid.item() if hasattr(eid, 'item') else eid)
            return eid
        except:
            return None
    
    def _find_episode_boundary_fast(self, target_eid: int, ds_len: int) -> int:
        """Find the end index (exclusive) of target episode using binary search.
        
        Strategy:
        1. Start with initial guess (200 for episode 0)
        2. Double if still in target episode, halve if past it
        3. Once we have bounds, use binary search to find exact boundary
        
        Returns:
            End index (exclusive) of the target episode, or 0 if not found
        """
        if ds_len == 0:
            return 0
        
        # Verify episode 0 exists at index 0
        first_eid = self._get_episode_id_at(0)
        if first_eid is None or first_eid != target_eid:
            return 0
        
        # Phase 1: Find upper bound using exponential search
        # Start with 200, double if still in episode, halve if past
        initial_guess = 200
        probe = min(initial_guess, ds_len - 1)
        
        # First, find a position that's past the target episode
        upper_bound = None
        lower_bound = 0
        
        while probe < ds_len:
            probe_eid = self._get_episode_id_at(probe)
            
            if probe_eid is None:
                # Error reading, try smaller
                probe = probe // 2
                if probe <= lower_bound:
                    break
                continue
            
            if probe_eid == target_eid:
                # Still in target episode, this is our new lower bound
                lower_bound = probe
                # Double the probe distance
                probe = min(probe * 2, ds_len - 1)
                if probe == lower_bound:
                    # We're at the end of dataset
                    break
            else:
                # Past target episode, this is our upper bound
                upper_bound = probe
                break
        
        # If we never found an upper bound, the episode extends to the end
        if upper_bound is None:
            # Check the last sample
            last_eid = self._get_episode_id_at(ds_len - 1)
            if last_eid == target_eid:
                return ds_len
            else:
                upper_bound = ds_len - 1
        
        # Phase 2: Binary search between lower_bound and upper_bound
        # Find the first index where episode_id != target_eid
        left = lower_bound
        right = upper_bound
        
        while left < right:
            mid = (left + right) // 2
            mid_eid = self._get_episode_id_at(mid)
            
            if mid_eid is None:
                # Error, try to recover
                right = mid
                continue
            
            if mid_eid == target_eid:
                # Still in target episode, boundary is to the right
                left = mid + 1
            else:
                # Past target episode, boundary is at or before mid
                right = mid
        
        return left
    
    def _lazy_index_batch(self, min_episodes: int = 5, max_samples: int = 500):
        """Lazily index a batch of samples to discover more episodes."""
        if self._index_complete or not is_map_data(self.current_dataset):
            return
        
        with self._index_lock:
            ds_len = len(self.current_dataset)
            start_idx = self._indexed_up_to
            initial_episode_count = len(self._discovered_episodes)
            
            # Index until we find enough new episodes or hit max samples
            samples_scanned = 0
            while self._indexed_up_to < ds_len:
                try:
                    sample = self.current_dataset[self._indexed_up_to]
                    eid = sample.get('episode_id', 0)
                    if isinstance(eid, (torch.Tensor, np.ndarray)):
                        eid = int(eid.item() if hasattr(eid, 'item') else eid)
                    
                    if eid not in self.episode_map:
                        self.episode_map[eid] = []
                        self._discovered_episodes.append(eid)
                    self.episode_map[eid].append(self._indexed_up_to)
                    
                    self._indexed_up_to += 1
                    samples_scanned += 1
                    
                    # Stop conditions
                    new_episodes = len(self._discovered_episodes) - initial_episode_count
                    if new_episodes >= min_episodes and samples_scanned >= 50:
                        break
                    if samples_scanned >= max_samples:
                        break
                        
                except Exception as e:
                    print(f"Index error at {self._indexed_up_to}: {e}")
                    self._indexed_up_to += 1
                    samples_scanned += 1
            
            if self._indexed_up_to >= ds_len:
                self._index_complete = True
                print(f"Indexing complete: {len(self._discovered_episodes)} episodes")
            else:
                print(f"Lazy indexed {samples_scanned} samples, found {len(self._discovered_episodes)} episodes so far")
    
    def _ensure_episode_indexed(self, eid: int) -> bool:
        """Quick check if episode exists, do minimal indexing if needed."""
        if eid in self.episode_map:
            return True
        
        if self._index_complete:
            return False
        
        # Quick batch to find this episode (don't block too long)
        for _ in range(3):  # Max 3 quick batches
            self._lazy_index_batch(min_episodes=10, max_samples=500)
            if eid in self.episode_map or self._index_complete:
                break
        
        return eid in self.episode_map
    
    def _index_more_episodes(self, count: int = 10):
        """Index more episodes for the dropdown."""
        if self._index_complete:
            return
        self._lazy_index_batch(min_episodes=count, max_samples=1000)
    
    def start_background_indexing(self):
        """Start background indexing of all episodes. Optimized for minimal lock contention."""
        if self._index_complete or self._bg_indexing:
            return
        
        self._bg_indexing = True
        
        def _bg_index():
            try:
                ds_len = len(self.current_dataset)
                print(f"[BG] Scanning episodes from {self._indexed_up_to}/{ds_len}...")
                
                # SMALLER batch size to reduce lock contention
                batch_size = 50
                while self._indexed_up_to < ds_len:
                    # Collect batch data WITHOUT holding lock
                    batch_data = []
                    start_idx = self._indexed_up_to
                    end_idx = min(start_idx + batch_size, ds_len)
                    
                    for i in range(start_idx, end_idx):
                        try:
                            sample = self.current_dataset[i]
                            eid = sample.get('episode_id', 0)
                            if isinstance(eid, (torch.Tensor, np.ndarray)):
                                eid = int(eid.item() if hasattr(eid, 'item') else eid)
                            batch_data.append((i, eid))
                        except:
                            batch_data.append((i, None))
                    
                    # Quick lock to update maps
                    with self._index_lock:
                        for i, eid in batch_data:
                            if eid is not None:
                                if eid not in self.episode_map:
                                    self.episode_map[eid] = []
                                    self._discovered_episodes.append(eid)
                                self.episode_map[eid].append(i)
                        self._indexed_up_to = end_idx
                    
                    if self._indexed_up_to % 5000 == 0:
                        print(f"[BG] Indexed {len(self._discovered_episodes)} episodes, {self._indexed_up_to}/{ds_len} samples")
                    
                    time.sleep(0.05)  # Longer yield to give priority to UI
                            
                self._index_complete = True
            except Exception as e:
                print(f"[BG] Indexing failed: {e}")
            finally:
                self._bg_indexing = False
                print(f"[BG] Indexing complete: {len(self._discovered_episodes)} episodes total")
        
        thread = threading.Thread(target=_bg_index, daemon=True)
        thread.start()
    
    def get_current_episode_count(self) -> Tuple[int, bool]:
        """Get current episode count and whether indexing is complete."""
        return len(self._discovered_episodes), self._index_complete
    
    def get_estimated_total_episodes(self) -> Optional[int]:
        """Get estimated total episode count (from last sample's episode_id)."""
        return getattr(self, '_estimated_total_episodes', None)
    
    def get_dataset_size_info(self) -> Dict[str, any]:
        """Get dataset size information (frames and episodes).
        
        Returns:
            Dict with 'total_frames', 'total_episodes', 'indexed_episodes', 'indexing_complete'
        """
        ds_len = len(self.current_dataset) if self.current_dataset else 0
        estimated_eps = getattr(self, '_estimated_total_episodes', None)
        indexed_eps = len(self._discovered_episodes) if hasattr(self, '_discovered_episodes') else 0
        index_complete = getattr(self, '_index_complete', False)
        
        return {
            'total_frames': ds_len,
            'total_episodes': estimated_eps,
            'indexed_episodes': indexed_eps,
            'indexing_complete': index_complete,
        }
    
    def get_stats_info(self) -> str:
        """Get formatted statistics information for display."""
        lines = []
        
        if self.action_stats:
            lines.append("📊 **Action Statistics:**")
            for key, val in self.action_stats.items():
                if isinstance(val, (list, np.ndarray)):
                    val_str = f"[{len(val)} dims]"
                else:
                    val_str = str(val)[:50]
                lines.append(f"  - {key}: {val_str}")
        else:
            lines.append("⚠️ No action statistics available")
        
        if self.state_stats:
            lines.append("\n📊 **State Statistics:**")
            for key, val in self.state_stats.items():
                if isinstance(val, (list, np.ndarray)):
                    val_str = f"[{len(val)} dims]"
                else:
                    val_str = str(val)[:50]
                lines.append(f"  - {key}: {val_str}")
        
        return "\n".join(lines) if lines else "No statistics loaded"
    
    def jump_to_sample(self, sample_idx: int) -> Tuple[bool, int, str]:
        """Jump to a specific sample index and return its episode ID.
        
        Searches backward and forward to find all samples in the same episode.
        Returns: (success, episode_id, message)
        """
        if not is_map_data(self.current_dataset):
            return False, -1, "Dataset is not map-style"
        
        ds_len = len(self.current_dataset)
        if sample_idx < 0 or sample_idx >= ds_len:
            return False, -1, f"Sample index {sample_idx} out of range [0, {ds_len-1}]"
        
        try:
            sample = self.current_dataset[sample_idx]
            eid = sample.get('episode_id', 0)
            if isinstance(eid, (torch.Tensor, np.ndarray)):
                eid = int(eid.item() if hasattr(eid, 'item') else eid)
            
            # Search backward and forward to find all samples in this episode
            episode_indices = [sample_idx]
            
            # Search backward
            for i in range(sample_idx - 1, max(0, sample_idx - 1000) - 1, -1):
                try:
                    s = self.current_dataset[i]
                    s_eid = s.get('episode_id', 0)
                    if isinstance(s_eid, (torch.Tensor, np.ndarray)):
                        s_eid = int(s_eid.item() if hasattr(s_eid, 'item') else s_eid)
                    if s_eid == eid:
                        episode_indices.insert(0, i)
                    else:
                        break  # Different episode, stop searching
                except:
                    break
            
            # Search forward
            for i in range(sample_idx + 1, min(ds_len, sample_idx + 1000)):
                try:
                    s = self.current_dataset[i]
                    s_eid = s.get('episode_id', 0)
                    if isinstance(s_eid, (torch.Tensor, np.ndarray)):
                        s_eid = int(s_eid.item() if hasattr(s_eid, 'item') else s_eid)
                    if s_eid == eid:
                        episode_indices.append(i)
                    else:
                        break  # Different episode, stop searching
                except:
                    break
            
            # Update episode map with full episode
            with self._index_lock:
                if eid not in self.episode_map:
                    self._discovered_episodes.append(eid)
                self.episode_map[eid] = sorted(set(episode_indices))
            
            return True, eid, f"Sample {sample_idx} → Episode {eid} ({len(episode_indices)} frames)"
        except Exception as e:
            return False, -1, f"Error accessing sample {sample_idx}: {e}"
    
    def get_available_norm_types(self) -> List[str]:
        """Get available normalization types based on loaded stats."""
        types = []
        if self.action_stats:
            if 'mean' in self.action_stats and 'std' in self.action_stats:
                types.append('zscore')
            if 'min' in self.action_stats and 'max' in self.action_stats:
                types.append('minmax')
            if 'q01' in self.action_stats and 'q99' in self.action_stats:
                types.append('percentile')
        return types if types else ['zscore']  # Default
    
    def _load_norm_stats(self):
        """Load normalization statistics from dataset's normalizers."""
        self.action_stats = None
        self.state_stats = None
        
        # Try to get stats from dataset's normalizers (preferred method)
        if self.current_dataset is not None:
            # Check for action_normalizer
            if hasattr(self.current_dataset, 'action_normalizer') and self.current_dataset.action_normalizer is not None:
                an = self.current_dataset.action_normalizer
                if hasattr(an, 'all_stats') and 'action' in an.all_stats:
                    self.action_stats = an.all_stats['action']
                    print(f"📊 Loaded action stats from normalizer: {list(self.action_stats.keys())}")
            
            # Check for state_normalizer
            if hasattr(self.current_dataset, 'state_normalizer') and self.current_dataset.state_normalizer is not None:
                sn = self.current_dataset.state_normalizer
                if hasattr(sn, 'all_stats') and 'state' in sn.all_stats:
                    self.state_stats = sn.all_stats['state']
                    print(f"📊 Loaded state stats from normalizer: {list(self.state_stats.keys())}")
        
        # Fallback: Try to find stats from cache file
        if self.action_stats is None or self.state_stats is None:
            cache_dir = os.path.join(os.environ.get('ILSTD_CACHE', os.path.expanduser('~/.cache/ilstd')), 'normalize')
            ctrl_space = getattr(self.current_dataset, 'ctrl_space', 'ee')
            ctrl_type = getattr(self.current_dataset, 'ctrl_type', 'delta')
            
            stats_file = os.path.join(cache_dir, f"{self.current_name}_stats_{ctrl_space}_{ctrl_type}.pkl")
            
            # Check memory cache first
            if stats_file in self._stats_cache:
                all_stats = self._stats_cache[stats_file]
                if self.action_stats is None:
                    self.action_stats = all_stats.get('action', {})
                if self.state_stats is None:
                    self.state_stats = all_stats.get('state', {})
                return
            
            if os.path.exists(stats_file):
                try:
                    import pickle
                    with open(stats_file, 'rb') as f:
                        all_stats = pickle.load(f)
                    self._stats_cache[stats_file] = all_stats  # Cache in memory
                    if self.action_stats is None:
                        self.action_stats = all_stats.get('action', {})
                    if self.state_stats is None:
                        self.state_stats = all_stats.get('state', {})
                    print(f"Loaded and cached normalization stats from {stats_file}")
                except Exception as e:
                    print(f"Failed to load stats from file: {e}")
    
    def denormalize(self, data: np.ndarray, stats: Dict, norm_type: str) -> np.ndarray:
        """Denormalize data using stats."""
        if stats is None or not stats:
            return data
        
        try:
            if norm_type == 'zscore':
                mean = np.array(stats.get('mean', 0))
                std = np.array(stats.get('std', 1))
                std = np.clip(std, 1e-6, None)
                return data * std + mean
            elif norm_type == 'minmax':
                min_val = np.array(stats.get('min', 0))
                max_val = np.array(stats.get('max', 1))
                # Assume normalized to [-1, 1]
                return (data + 1) / 2 * (max_val - min_val) + min_val
            elif norm_type == 'percentile':
                q01 = np.array(stats.get('q01', 0))
                q99 = np.array(stats.get('q99', 1))
                return (data + 1) / 2 * (q99 - q01) + q01
        except:
            pass
        return data
    
    def get_episode_ids(self, load_more: bool = False) -> List[int]:
        """Get discovered episode IDs (sorted). Optionally load more."""
        if load_more and not self._index_complete:
            self._index_more_episodes(20)
        # Return sorted list for consistent UI ordering
        return sorted(self._discovered_episodes)
    
    def get_all_episode_ids(self) -> List[int]:
        """Get discovered episode IDs for dropdown display.
        
        OPTIMIZED: Only returns discovered episodes to prevent dropdown freeze.
        Limited to max_dropdown_episodes to prevent UI freeze.
        """
        # Only return discovered episodes - prevents dropdown from freezing
        # with thousands of items
        sorted_eps = sorted(self._discovered_episodes)
        # Limit to prevent dropdown freeze
        if len(sorted_eps) > CFG.max_dropdown_episodes:
            return sorted_eps[:CFG.max_dropdown_episodes]
        return sorted_eps
    
    def has_more_episodes(self) -> bool:
        """Check if there are potentially more episodes to discover."""
        return not self._index_complete
    
    def get_episode_length(self, eid: int) -> int:
        """Get episode length. If not in map, check cache or scan."""
        # Check episode_map first
        if eid in self.episode_map:
            return len(self.episode_map[eid])
        # Check cache
        with self.lock:
            if eid in self.episode_cache:
                return len(self.episode_cache[eid])
        # Not found, return 0 (will be updated after loading)
        return 0
    
    def get_episode(self, eid: int, denormalize: bool = False) -> List[Dict]:
        """Get episode with optional denormalization.
        
        FAST PATH: Uses DataLoader with multiple workers for parallel loading.
        Thread-safe: prevents duplicate loading of the same episode.
        """
        # Check cache first and determine if we should load or wait
        should_load = False
        should_wait = False
        
        with self.lock:
            if eid in self.episode_cache:
                self.cache_access_time[eid] = time.time()
                cached = self.episode_cache[eid]
                if denormalize and self.action_stats:
                    return self._denormalize_samples(cached)
                return cached
            
            # Check if another thread is already loading this episode
            if eid in self.loading:
                # Another thread is loading, we should wait
                should_wait = True
            else:
                # We are the first to load this episode
                self.loading.add(eid)
                should_load = True
        
        # If another thread is loading, wait for it
        if should_wait:
            max_wait = 30  # seconds
            wait_start = time.time()
            while True:
                time.sleep(0.05)
                with self.lock:
                    if eid in self.episode_cache:
                        self.cache_access_time[eid] = time.time()
                        cached = self.episode_cache[eid]
                        if denormalize and self.action_stats:
                            return self._denormalize_samples(cached)
                        return cached
                    if eid not in self.loading:
                        # Loading finished but not in cache - episode doesn't exist
                        return []
                if time.time() - wait_start > max_wait:
                    return []
        
        # We are the loading thread - proceed with loading
        assert should_load, "Logic error: should_load must be True here"
        try:
            # Try to get indices from episode_map first
            indices = self.episode_map.get(eid, [])
            
            # If not in map, try direct scanning (FAST for sequential datasets)
            if not indices:
                # Use faster scan for early episodes (0-10)
                if eid <= 10:
                    indices = self._scan_episode_from_start(eid)
                else:
                    indices = self._scan_episode_directly(eid)
                
                if indices:
                    print(f"✅ Found {len(indices)} samples for episode {eid}")
                else:
                    print(f"⚠️ No samples found for episode {eid}")
            
            if not indices:
                return []
            
            # Use DataLoader for parallel loading when episode is large enough
            samples = self._load_samples_parallel(indices)
            
            if not samples:
                return []
            
            # Cache normalized data with access time
            with self.lock:
                # Only evict if really necessary, and only evict unprotected episodes
                K = CFG.cache_window
                current_ep = self._current_episode
                protected = set()
                if current_ep is not None:
                    for d in range(-K, K + 1):
                        protected.add(current_ep + d)
                
                while len(self.episode_cache) >= CFG.max_cached_episodes and self.cache_order:
                    # Find oldest unprotected episode to evict
                    evicted = False
                    for old in list(self.cache_order):
                        if old not in protected:
                            self.cache_order.remove(old)
                            self.episode_cache.pop(old, None)
                            self.cache_access_time.pop(old, None)
                            evicted = True
                            break
                    if not evicted:
                        # All are protected, just allow cache to grow
                        break
                
                self.episode_cache[eid] = samples
                self.cache_access_time[eid] = time.time()
                if eid in self.cache_order:
                    self.cache_order.remove(eid)
                self.cache_order.append(eid)
            
            if denormalize and self.action_stats:
                return self._denormalize_samples(samples)
            return samples
        finally:
            # Always remove from loading set when done
            with self.lock:
                self.loading.discard(eid)
    
    def _subsample_indices(self, indices: List[int], max_samples: int) -> List[int]:
        """Subsample indices to max_samples while keeping first, last, and evenly spaced samples.
        
        This ensures we capture the full trajectory shape even with subsampling.
        """
        n = len(indices)
        if n <= max_samples:
            return indices
        
        # Always include first and last
        # Evenly space the rest
        step = (n - 1) / (max_samples - 1)
        sampled_indices = []
        for i in range(max_samples):
            idx = int(round(i * step))
            idx = min(idx, n - 1)  # Clamp to valid range
            sampled_indices.append(indices[idx])
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for idx in sampled_indices:
            if idx not in seen:
                seen.add(idx)
                result.append(idx)
        
        return result
    
    def _load_samples_parallel(self, indices: List[int], max_samples: Optional[int] = None) -> List[Dict]:
        """Load samples using DataLoader for parallel I/O.
        
        Uses PyTorch DataLoader with multiple workers for faster loading.
        For long episodes, subsamples to max_samples for performance.
        
        Args:
            indices: List of sample indices to load
            max_samples: Max samples to load (None = use CFG.max_episode_samples)
        """
        if not indices or not is_map_data(self.current_dataset):
            return []
        
        # Subsample long episodes for performance
        original_len = len(indices)
        max_samples = max_samples or CFG.max_episode_samples
        
        if original_len > CFG.long_episode_threshold:
            indices = self._subsample_indices(indices, max_samples)
            print(f"⚡ Subsampled episode: {original_len} → {len(indices)} samples")
        
        # For small episodes, use sequential loading (overhead not worth it)
        if len(indices) < 20:
            samples = []
            for idx in indices:
                try:
                    s = self.current_dataset[idx]
                    samples.append(self._to_numpy(s))
                except Exception as e:
                    print(f"Error loading sample {idx}: {e}")
            return samples
        
        # Use DataLoader for parallel loading
        try:
            # Create a Subset of the dataset with only the indices we need
            subset = Subset(self.current_dataset, indices)
            
            # Create DataLoader with multiple workers
            loader = DataLoader(
                subset,
                batch_size=1,  # Load one sample at a time to preserve order
                shuffle=False,
                num_workers=CFG.dataloader_workers,
                prefetch_factor=2,
                persistent_workers=False,  # Don't persist for one-off loads
                pin_memory=False,
            )
            
            samples = []
            for batch in loader:
                # batch is a dict with batched tensors, unbatch them
                sample = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        # Remove batch dimension
                        sample[k] = v[0].numpy() if v.dim() > 0 else v.numpy()
                    elif isinstance(v, (list, tuple)) and len(v) == 1:
                        sample[k] = v[0]
                    else:
                        sample[k] = v
                samples.append(sample)
            
            return samples
        except Exception as e:
            print(f"DataLoader failed, falling back to sequential: {e}")
            # Fallback to sequential loading
            samples = []
            for idx in indices:
                try:
                    s = self.current_dataset[idx]
                    samples.append(self._to_numpy(s))
                except Exception as e2:
                    print(f"Error loading sample {idx}: {e2}")
            return samples
    
    def _get_average_episode_length(self) -> int:
        """Get average episode length from known episode ranges."""
        if not self.episode_range:
            # Fallback: estimate from dataset metadata
            ds_len = len(self.current_dataset) if self.current_dataset else 0
            num_episodes = getattr(self.current_dataset, 'total_episodes', 
                                  getattr(self.current_dataset, 'num_episodes', 10))
            if num_episodes > 0 and ds_len > 0:
                return ds_len // num_episodes
            return 500  # Default
        
        # Calculate from known ranges
        total_len = 0
        for start, end in self.episode_range.values():
            total_len += (end - start + 1)
        return total_len // len(self.episode_range)
    
    def _find_episode_start_binary(self, target_eid: int, ds_len: int) -> int:
        """Find the start index of target episode using binary search.
        
        Returns:
            Start index of the episode, or -1 if not found
        """
        # First, find any sample with target_eid
        est_start, est_end = self._estimate_episode_range(target_eid)
        mid_guess = (est_start + est_end) // 2
        
        found_idx = self._binary_search_episode(target_eid, mid_guess)
        if found_idx < 0:
            return -1
        
        # Now binary search backward to find the start
        left = 0
        right = found_idx
        
        while left < right:
            mid = (left + right) // 2
            mid_eid = self._get_episode_id_at(mid)
            
            if mid_eid is None:
                left = mid + 1
                continue
            
            if mid_eid < target_eid:
                # Target episode starts after mid
                left = mid + 1
            else:
                # mid_eid >= target_eid, start is at or before mid
                right = mid
        
        # Verify we found the right episode
        verify_eid = self._get_episode_id_at(left)
        if verify_eid == target_eid:
            return left
        return -1
    
    def _find_episode_end_binary(self, target_eid: int, start_idx: int, ds_len: int) -> int:
        """Find the end index (inclusive) of target episode using binary search.
        
        Args:
            target_eid: Target episode ID
            start_idx: Known start index of the episode
            ds_len: Dataset length
            
        Returns:
            End index (inclusive) of the episode
        """
        # Estimate upper bound based on average episode length
        avg_len = self._get_average_episode_length()
        # Use 2x average as initial upper bound estimate
        initial_upper = min(start_idx + avg_len * 2, ds_len - 1)
        
        # Exponential search to find upper bound
        probe = initial_upper
        lower_bound = start_idx
        upper_bound = None
        
        while probe < ds_len:
            probe_eid = self._get_episode_id_at(probe)
            
            if probe_eid is None:
                probe = (probe + lower_bound) // 2
                if probe <= lower_bound:
                    break
                continue
            
            if probe_eid == target_eid:
                # Still in target episode
                lower_bound = probe
                # Expand by average episode length (not double, more conservative)
                probe = min(probe + avg_len, ds_len - 1)
                if probe == lower_bound:
                    break
            else:
                # Past target episode
                upper_bound = probe
                break
        
        if upper_bound is None:
            # Check if episode extends to the end
            last_eid = self._get_episode_id_at(ds_len - 1)
            if last_eid == target_eid:
                return ds_len - 1
            upper_bound = ds_len - 1
        
        # Binary search between lower_bound and upper_bound
        left = lower_bound
        right = upper_bound
        
        while left < right:
            mid = (left + right + 1) // 2  # Round up to find last valid index
            mid_eid = self._get_episode_id_at(mid)
            
            if mid_eid is None:
                right = mid - 1
                continue
            
            if mid_eid == target_eid:
                # Still in target episode
                left = mid
            else:
                # Past target episode
                right = mid - 1
        
        return left
    
    def _scan_episode_from_start(self, eid: int) -> List[int]:
        """Find episode using binary search. Works for any episode.
        
        OPTIMIZED: Uses binary search instead of sequential scan.
        """
        if not is_map_data(self.current_dataset):
            return []
        
        ds_len = len(self.current_dataset)
        if ds_len == 0:
            return []
        
        # For episode 0, we know it starts at index 0
        if eid == 0:
            first_eid = self._get_episode_id_at(0)
            if first_eid != 0:
                return []
            
            # Find end using binary search
            end_idx = self._find_episode_end_binary(0, 0, ds_len)
            indices = list(range(0, end_idx + 1))
        else:
            # For other episodes, find start and end using binary search
            start_idx = self._find_episode_start_binary(eid, ds_len)
            if start_idx < 0:
                return []
            
            end_idx = self._find_episode_end_binary(eid, start_idx, ds_len)
            indices = list(range(start_idx, end_idx + 1))
        
        # Update episode_map and episode_range for future use
        if indices:
            with self._index_lock:
                self.episode_map[eid] = indices
                self.episode_range[eid] = (indices[0], indices[-1])
                if eid not in self._discovered_episodes:
                    self._discovered_episodes.append(eid)
        
        return indices
    
    def _estimate_episode_range(self, eid: int) -> Tuple[int, int]:
        """Estimate the sample index range for an unknown episode.
        
        Uses known episode ranges to interpolate/extrapolate.
        Returns (estimated_start, estimated_end).
        """
        if not self.episode_range:
            # No known ranges, use simple estimation
            ds_len = len(self.current_dataset) if self.current_dataset else 0
            estimated = self._estimated_total_episodes or 1
            avg_len = ds_len // estimated
            start = eid * avg_len
            end = start + avg_len - 1
            return (max(0, start), min(ds_len - 1, end))
        
        # Find nearest known episodes
        known_eids = sorted(self.episode_range.keys())
        
        # Check if we have episodes before and after
        lower_eid = None
        upper_eid = None
        for k in known_eids:
            if k < eid:
                lower_eid = k
            elif k > eid:
                upper_eid = k
                break
        
        ds_len = len(self.current_dataset) if self.current_dataset else 0
        
        if lower_eid is not None and upper_eid is not None:
            # Interpolate between two known episodes
            lower_start, lower_end = self.episode_range[lower_eid]
            upper_start, upper_end = self.episode_range[upper_eid]
            
            # Calculate average episode length in this range
            total_samples = upper_end - lower_start + 1
            num_episodes = upper_eid - lower_eid + 1
            avg_len = total_samples // num_episodes
            
            # Estimate position
            offset = eid - lower_eid
            est_start = lower_start + offset * avg_len
            est_end = est_start + avg_len - 1
            return (max(0, est_start), min(ds_len - 1, est_end))
        
        elif lower_eid is not None:
            # Extrapolate forward from last known episode
            lower_start, lower_end = self.episode_range[lower_eid]
            avg_len = lower_end - lower_start + 1
            offset = eid - lower_eid
            est_start = lower_end + 1 + (offset - 1) * avg_len
            est_end = est_start + avg_len - 1
            return (max(0, est_start), min(ds_len - 1, est_end))
        
        elif upper_eid is not None:
            # Extrapolate backward from first known episode
            upper_start, upper_end = self.episode_range[upper_eid]
            avg_len = upper_end - upper_start + 1
            offset = upper_eid - eid
            est_end = upper_start - 1 - (offset - 1) * avg_len
            est_start = est_end - avg_len + 1
            return (max(0, est_start), min(ds_len - 1, est_end))
        
        # Fallback
        estimated = self._estimated_total_episodes or 1
        avg_len = ds_len // estimated
        start = eid * avg_len
        return (max(0, start), min(ds_len - 1, start + avg_len - 1))
    
    def _scan_episode_directly(self, eid: int) -> List[int]:
        """Find episode using binary search.
        
        OPTIMIZED: Uses binary search for both finding start and end of episode.
        Falls back to _scan_episode_from_start which now also uses binary search.
        """
        if not is_map_data(self.current_dataset):
            return []
        
        ds_len = len(self.current_dataset)
        if ds_len == 0:
            return []
        
        # FAST PATH: Check if we already have the range indexed
        if eid in self.episode_range:
            start, end = self.episode_range[eid]
            indices = list(range(start, end + 1))
            # Update episode_map
            with self._index_lock:
                self.episode_map[eid] = indices
                if eid not in self._discovered_episodes:
                    self._discovered_episodes.append(eid)
            return indices
        
        # Use binary search to find the episode
        # First, find the start of the episode
        start_idx = self._find_episode_start_binary(eid, ds_len)
        if start_idx < 0:
            return []
        
        # Then find the end using binary search with average length estimation
        end_idx = self._find_episode_end_binary(eid, start_idx, ds_len)
        
        indices = list(range(start_idx, end_idx + 1))
        
        # Update episode_map and episode_range for future use
        if indices:
            with self._index_lock:
                self.episode_map[eid] = indices
                self.episode_range[eid] = (indices[0], indices[-1])
                if eid not in self._discovered_episodes:
                    self._discovered_episodes.append(eid)
        
        return indices
    
    def _binary_search_episode(self, target_eid: int, start_guess: int) -> int:
        """Binary search to find a sample with the target episode_id."""
        ds_len = len(self.current_dataset)
        
        # First check the guess position
        try:
            s = self.current_dataset[start_guess]
            s_eid = s.get('episode_id', -1)
            if isinstance(s_eid, (torch.Tensor, np.ndarray)):
                s_eid = int(s_eid.item() if hasattr(s_eid, 'item') else s_eid)
            if s_eid == target_eid:
                return start_guess
        except:
            pass
        
        # Binary search
        left, right = 0, ds_len - 1
        iterations = 0
        while left <= right:
            mid = (left + right) // 2
            iterations += 1
            try:
                s = self.current_dataset[mid]
                s_eid = s.get('episode_id', -1)
                if isinstance(s_eid, (torch.Tensor, np.ndarray)):
                    s_eid = int(s_eid.item() if hasattr(s_eid, 'item') else s_eid)
                
                if s_eid == target_eid:
                    return mid
                elif s_eid < target_eid:
                    left = mid + 1
                else:
                    right = mid - 1
            except Exception as e:
                print(f"⚠️ Binary search error at mid={mid}: {e}")
                left = mid + 1
        
        print(f"⚠️ Binary search failed for episode {target_eid} after {iterations} iterations (ds_len={ds_len})")
        return -1

    def _ensure_episode_fully_indexed(self, eid: int) -> bool:
        """Helper to complete indexing for a specific episode.
        
        OPTIMIZED: Uses smaller batches and releases lock frequently.
        """
        if self._index_complete:
            return True
        
        if eid not in self.episode_map:
            return False
        
        ds_len = len(self.current_dataset)
        max_iterations = 200  # Limit to prevent blocking too long
        iterations = 0
        
        while self._indexed_up_to < ds_len and iterations < max_iterations:
            # Process in small batches to avoid blocking
            batch_data = []
            start_idx = self._indexed_up_to
            end_idx = min(start_idx + 20, ds_len)  # Small batch
            
            for i in range(start_idx, end_idx):
                try:
                    sample = self.current_dataset[i]
                    sample_eid = sample.get('episode_id', 0)
                    if isinstance(sample_eid, (torch.Tensor, np.ndarray)):
                        sample_eid = int(sample_eid.item() if hasattr(sample_eid, 'item') else sample_eid)
                    batch_data.append((i, sample_eid))
                except:
                    batch_data.append((i, None))
            
            # Quick lock to update
            with self._index_lock:
                for i, sample_eid in batch_data:
                    if sample_eid is not None:
                        if sample_eid not in self.episode_map:
                            self.episode_map[sample_eid] = []
                            self._discovered_episodes.append(sample_eid)
                        self.episode_map[sample_eid].append(i)
                        
                        # If we found a different episode, our target is complete
                        if sample_eid != eid and eid in self.episode_map:
                            self._indexed_up_to = end_idx
                            return True
                
                self._indexed_up_to = end_idx
            
            iterations += 1
        
        if self._indexed_up_to >= ds_len:
            self._index_complete = True
        
        return eid in self.episode_map
    
    def _denormalize_samples(self, samples: List[Dict]) -> List[Dict]:
        """Denormalize action and state in samples."""
        result = []
        for s in samples:
            new_s = s.copy()
            if 'action' in s and s['action'] is not None:
                new_s['action'] = self.denormalize(s['action'], self.action_stats, self.norm_type)
            if 'state' in s and s['state'] is not None:
                new_s['state'] = self.denormalize(s['state'], self.state_stats, self.norm_type)
            result.append(new_s)
        return result
    
    def get_frame(self, eid: int, frame_idx: int) -> Optional[Dict]:
        """Get single frame."""
        # Ensure episode is indexed
        self._ensure_episode_indexed(eid)
        
        indices = self.episode_map.get(eid, [])
        if not indices or frame_idx >= len(indices):
            return None
        
        with self.lock:
            if eid in self.episode_cache:
                samples = self.episode_cache[eid]
                if frame_idx < len(samples):
                    return samples[frame_idx]
        
        try:
            s = self.current_dataset[indices[frame_idx]]
            return self._to_numpy(s)
        except:
            return None
    
    def preload_episode(self, eid: int, priority: bool = False):
        """Background preload a single episode.
        
        Args:
            eid: Episode ID to preload
            priority: If True, preload with higher priority (for nearby episodes)
        """
        # Only check cache - don't add to loading here since get_episode handles it
        with self.lock:
            if eid in self.episode_cache or eid in self.loading:
                return
        
        def _load():
            # get_episode handles all the loading logic including loading set management
            self.get_episode(eid)
        
        self.executor.submit(_load)
    
    def preload_nearby(self, eid: int):
        """Preload episodes around the current one for smooth navigation.
        
        Caches K episodes before and after current episode (K = CFG.cache_window).
        Loads one by one in background without blocking main thread.
        Priority: next episode first, then alternating forward/backward.
        """
        estimated = self._estimated_total_episodes
        max_eid = (estimated - 1) if estimated else 999999
        K = CFG.cache_window
        
        # Build priority queue: next first, then alternate forward/backward
        to_preload = []
        
        # Next episode has highest priority
        if eid + 1 <= max_eid:
            to_preload.append(eid + 1)
        
        # Then alternate: +2, -1, +3, -2, +4, -3, etc.
        for d in range(2, K + 1):
            if eid + d <= max_eid:
                to_preload.append(eid + d)
            if eid - (d - 1) >= 0:
                to_preload.append(eid - (d - 1))
        
        # Add remaining behind episodes
        for d in range(K, K + 1):
            if eid - d >= 0 and (eid - d) not in to_preload:
                to_preload.append(eid - d)
        
        # Preload each one sequentially in background (one at a time to not block)
        for target_eid in to_preload:
            self.preload_episode(target_eid, priority=False)
    
    def preload_initial_episodes(self, count: int = 10):
        """Preload first N episodes on dataset load for instant access."""
        # Use continuous episode IDs (0, 1, 2, ..., count-1)
        estimated = getattr(self, '_estimated_total_episodes', None)
        max_eid = estimated if estimated else count
        for eid in range(min(count, max_eid)):
            self.preload_episode(eid, priority=True)
    
    def get_cached_video(self, eid: int) -> Optional[str]:
        """Get cached video path if available."""
        with self.lock:
            path = self.video_cache.get(eid)
            if path and os.path.exists(path):
                # Update access time
                self.video_access_time[eid] = time.time()
                return path
            # Remove invalid cache entry
            if eid in self.video_cache:
                del self.video_cache[eid]
                self.video_access_time.pop(eid, None)
                if eid in self.video_cache_order:
                    self.video_cache_order.remove(eid)
        return None
    
    def cache_video(self, eid: int, video_path: str):
        """Cache a video path."""
        if not video_path or not os.path.exists(video_path):
            return
        
        with self.lock:
            # Evict old videos if cache is full
            while len(self.video_cache) >= self.max_video_cache and self.video_cache_order:
                old_eid = self.video_cache_order.pop(0)
                old_path = self.video_cache.pop(old_eid, None)
                self.video_access_time.pop(old_eid, None)
                if old_path and os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except:
                        pass
            
            self.video_cache[eid] = video_path
            self.video_access_time[eid] = time.time()
            if eid in self.video_cache_order:
                self.video_cache_order.remove(eid)
            self.video_cache_order.append(eid)
    
    def pregenerate_video(self, eid: int):
        """Pre-generate video for an episode in background."""
        with self.lock:
            if eid in self.video_cache or eid in self.video_generating:
                return
            self.video_generating.add(eid)
        
        def _generate():
            try:
                samples = self.get_episode(eid)
                if samples:
                    video_path = create_episode_video(samples)
                    if video_path:
                        self.cache_video(eid, video_path)
            finally:
                with self.lock:
                    self.video_generating.discard(eid)
        
        self.executor.submit(_generate)
    
    def pregenerate_nearby_videos(self, eid: int):
        """Pre-generate video for next episode only (prioritize speed)."""
        estimated = self._estimated_total_episodes
        max_eid = (estimated - 1) if estimated else 999999
        
        # Only pre-generate next episode's video
        next_eid = eid + 1
        if next_eid <= max_eid:
            self.pregenerate_video(next_eid)
    
    def _to_numpy(self, sample: Dict) -> Dict:
        result = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.cpu().numpy()
            else:
                result[k] = v
        return result
    
    def get_ctrl_space(self) -> str:
        if self.current_dataset:
            return getattr(self.current_dataset, 'ctrl_space', 'unknown')
        return 'unknown'


# Global manager
MGR = EpisodeManager()


# ============================================================================
# Video Generation - Real MP4 Video
# ============================================================================

def create_episode_video(samples: List[Dict], fps: int = 30, max_frames: int = None) -> Optional[str]:
    """Create MP4 video efficiently with maximum speed optimization.
    
    Args:
        samples: List of sample dictionaries with 'image' key
        fps: Target video FPS
        max_frames: Max frames to include (None = use CFG.max_episode_samples)
    """
    if not samples:
        return None
    
    max_frames = max_frames or CFG.max_episode_samples
    
    # Subsample frames if too many
    if len(samples) > max_frames:
        step = len(samples) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        indices = list(set(indices))  # Remove duplicates
        indices.sort()
        samples = [samples[i] for i in indices if i < len(samples)]
        print(f"⚡ Video: subsampled to {len(samples)} frames")
    
    # Fast path: process images quickly
    frames = []
    for sample in samples:
        img = create_multiview_image(sample)
        if img is not None:
            frames.append(img)
    
    if not frames:
        return None
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    # Adjust FPS based on frame count for reasonable video duration
    # Target: 5-15 seconds video
    video_fps = max(10, min(30, len(frames) // 10))
    
    try:
        import imageio
        # Use maximum speed settings for ffmpeg
        writer = imageio.get_writer(
            temp_path,
            fps=video_fps,
            codec='libx264',
            quality=5,  # Lower quality for faster encoding
            pixelformat='yuv420p',
            macro_block_size=16,
            ffmpeg_params=['-preset', 'ultrafast', '-tune', 'fastdecode', '-crf', '28']
        )
        
        for frame in frames:
            writer.append_data(frame)
        
        writer.close()
        return temp_path
            
    except Exception as e:
        print(f"Video creation failed: {e}")
        return None
    
    return None


def create_multiview_image(sample: Dict) -> Optional[np.ndarray]:
    """Create horizontal concat of all views efficiently."""
    img_data = sample.get('image')
    if img_data is None:
        return None
    
    # Check if we have pre-processed image in some form
    if isinstance(img_data, np.ndarray):
        if len(img_data.shape) == 4:
            # Multi-view: (N, C, H, W)
            views = []
            for i in range(img_data.shape[0]):
                view = process_image(img_data, i)
                if view is not None:
                    views.append(resize_image(view))
            return np.hstack(views) if views else None
        else:
            view = process_image(img_data, 0)
            return resize_image(view) if view is not None else None
            
    return None


def process_image(img_data: np.ndarray, view_idx: int = 0) -> Optional[np.ndarray]:
    """Convert to RGB numpy array."""
    if img_data is None:
        return None
    
    if len(img_data.shape) == 4:
        if view_idx >= img_data.shape[0]:
            view_idx = 0
        img = img_data[view_idx]
    elif len(img_data.shape) == 3:
        img = img_data
    else:
        return None
    
    # (c, h, w) -> (h, w, c)
    if img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    
    # Normalize
    if img.dtype in [np.float32, np.float64]:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    # Grayscale to RGB
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    
    return img


def resize_image(img: np.ndarray) -> np.ndarray:
    """Resize image to display size using OpenCV (much faster)."""
    target_w, target_h = CFG.image_size
    h, w = img.shape[:2]
    if h != target_h or w != target_w:
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return img


# ============================================================================
# Visualization
# ============================================================================

def make_stats_plot(stats: Dict, title: str, dim_names: Optional[List[str]] = None) -> go.Figure:
    """Create a wide plot showing mean, std, and range for each dimension."""
    try:
        fig = go.Figure()
        
        if not stats:
            fig.add_annotation(text="No statistics available", x=0.5, y=0.5, 
                              xref="paper", yref="paper", showarrow=False, font=dict(size=16))
            fig.update_layout(height=250, title=dict(text=title, x=0.5), template='plotly_white')
            return fig
        
        # Extract statistics - handle both list and numpy array
        def to_array(v):
            if v is None:
                return np.array([])
            if isinstance(v, (list, tuple)):
                return np.array(v, dtype=float)
            if isinstance(v, np.ndarray):
                return v.astype(float)
            return np.array([float(v)])
        
        mean = to_array(stats.get('mean'))
        std = to_array(stats.get('std'))
        min_val = to_array(stats.get('min'))
        max_val = to_array(stats.get('max'))
        q01 = to_array(stats.get('q01'))
        q99 = to_array(stats.get('q99'))
        
        n_dims = max(len(mean), len(min_val), len(max_val)) if any([len(mean), len(min_val), len(max_val)]) else 0
        if n_dims == 0:
            fig.add_annotation(text="No dimension data in stats", x=0.5, y=0.5, 
                              xref="paper", yref="paper", showarrow=False)
            fig.update_layout(height=250, title=dict(text=title, x=0.5), template='plotly_white')
            return fig
        
        x = list(range(n_dims))
        x_labels = dim_names if dim_names and len(dim_names) == n_dims else [f"D{i}" for i in range(n_dims)]
        
        # Add range bars (min-max)
        if len(min_val) == n_dims and len(max_val) == n_dims:
            for i in range(n_dims):
                fig.add_trace(go.Scatter(
                    x=[i, i], y=[float(min_val[i]), float(max_val[i])],
                    mode='lines', line=dict(color='rgba(100,149,237,0.5)', width=10),
                    name='Min-Max' if i == 0 else None, showlegend=(i == 0),
                    hovertemplate=f"D{i}: [{float(min_val[i]):.3f}, {float(max_val[i]):.3f}]<extra>Range</extra>"
                ))
        
        # Add Q01-Q99 bars
        if len(q01) == n_dims and len(q99) == n_dims:
            for i in range(n_dims):
                fig.add_trace(go.Scatter(
                    x=[i, i], y=[float(q01[i]), float(q99[i])],
                    mode='lines', line=dict(color='rgba(50,205,50,0.7)', width=5),
                    name='Q01-Q99' if i == 0 else None, showlegend=(i == 0),
                    hovertemplate=f"D{i}: [{float(q01[i]):.3f}, {float(q99[i]):.3f}]<extra>Q01-Q99</extra>"
                ))
        
        # Add mean points with error bars (std)
        if len(mean) == n_dims:
            error_y_config = None
            if len(std) == n_dims:
                error_y_config = dict(type='data', array=[float(s) for s in std], visible=True, color='red', thickness=2)
            
            fig.add_trace(go.Scatter(
                x=x, y=[float(m) for m in mean], mode='markers',
                marker=dict(color='red', size=10, symbol='diamond'),
                error_y=error_y_config,
                name='Mean ± Std',
                hovertemplate="D%{x}: μ=%{y:.4f}<extra>Mean</extra>"
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=14)),
            xaxis=dict(title="Dimension", tickmode='array', tickvals=x, ticktext=x_labels, tickangle=45 if n_dims > 10 else 0),
            yaxis=dict(title="Value"),
            height=280,
            margin=dict(l=50, r=20, t=50, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    except Exception as e:
        print(f"Error in make_stats_plot: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)[:40]}", x=0.5, y=0.5, 
                          xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=250, title=dict(text=title, x=0.5), template='plotly_white')
        return fig


def make_stats_plot_mpl(stats: Dict, title: str, dim_names: Optional[List[str]] = None):
    """Matplotlib fallback for stats plot (robust rendering in Gradio).
    
    Uses transparent background so Gradio theme colors show through.
    """
    # Use transparent background - Gradio theme will handle colors
    fig, ax = plt.subplots(figsize=(10, 3), facecolor='none')
    ax.set_facecolor('none')
    ax.set_title(title)
    
    if not stats:
        ax.text(0.5, 0.5, "No statistics available", ha="center", va="center", 
               transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    def to_array(v):
        if v is None:
            return np.array([])
        if isinstance(v, (list, tuple)):
            return np.array(v, dtype=float)
        if isinstance(v, np.ndarray):
            return v.astype(float)
        return np.array([float(v)])

    mean = to_array(stats.get("mean"))
    std = to_array(stats.get("std"))
    min_val = to_array(stats.get("min"))
    max_val = to_array(stats.get("max"))
    q01 = to_array(stats.get("q01"))
    q99 = to_array(stats.get("q99"))

    n_dims = max(len(mean), len(min_val), len(max_val)) if any([len(mean), len(min_val), len(max_val)]) else 0
    if n_dims == 0:
        ax.text(0.5, 0.5, "No dimension data in stats", ha="center", va="center", 
               transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    x = np.arange(n_dims)
    labels = dim_names if dim_names and len(dim_names) == n_dims else [f"D{i}" for i in range(n_dims)]

    # Range (min-max)
    if len(min_val) == n_dims and len(max_val) == n_dims:
        ax.vlines(x, min_val, max_val, colors="#6495ED", linewidth=6, alpha=0.5, label="Min-Max")

    # Q01-Q99
    if len(q01) == n_dims and len(q99) == n_dims:
        ax.vlines(x, q01, q99, colors="#32CD32", linewidth=3, alpha=0.7, label="Q01-Q99")

    # Mean ± std
    if len(mean) == n_dims:
        if len(std) == n_dims:
            ax.errorbar(x, mean, yerr=std, fmt="D", color="#FF6B6B", ecolor="#FF6B6B", 
                       elinewidth=1.5, capsize=2, label="Mean ± Std")
        else:
            ax.plot(x, mean, "D", color="#FF6B6B", label="Mean")

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    if n_dims <= 20:
        ax.set_xticklabels(labels, rotation=45 if n_dims > 10 else 0, ha="right" if n_dims > 10 else "center")
    else:
        ax.set_xticklabels([str(i) for i in x])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def make_trajectory_plot(samples: List[Dict], action_dim: int, ctrl_space: str) -> go.Figure:
    """3D trajectory plot for end-effector control space.
    
    Uses transparent/auto background - Gradio theme handles colors.
    """
    try:
        fig = go.Figure()
        
        if ctrl_space != 'ee':
            fig.add_annotation(text="3D trajectory only for 'ee' control space", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False, font=dict(size=16))
            fig.update_layout(height=400, title=dict(text="3D Trajectory", x=0.5), 
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig
        
        actions = []
        for s in samples:
            a = s.get('action')
            if a is not None:
                if isinstance(a, torch.Tensor):
                    a = a.numpy()
                if len(a.shape) == 2:
                    a = a[0]
                actions.append(a.copy() if isinstance(a, np.ndarray) else a)
        
        if not actions:
            fig.add_annotation(text="No action data", x=0.5, y=0.5, 
                              xref="paper", yref="paper", showarrow=False, font=dict(size=16))
            fig.update_layout(height=400, title=dict(text="3D Trajectory", x=0.5), 
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig
        
        actions = np.array(actions, dtype=float)
        dual_arm = action_dim >= 12
        
        if dual_arm:
            half = action_dim // 2
            fig.add_trace(go.Scatter3d(
                x=actions[:, 0].tolist(), y=actions[:, 1].tolist(), z=actions[:, 2].tolist(),
                mode='lines+markers', name='Left Arm',
                line=dict(color='#FF6B6B', width=4), marker=dict(size=2)
            ))
            if actions.shape[1] > half + 2:
                fig.add_trace(go.Scatter3d(
                    x=actions[:, half].tolist(), y=actions[:, half+1].tolist(), z=actions[:, half+2].tolist(),
                    mode='lines+markers', name='Right Arm',
                    line=dict(color='#4ECDC4', width=4), marker=dict(size=2)
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=actions[:, 0].tolist(), y=actions[:, 1].tolist(), z=actions[:, 2].tolist(),
                mode='lines+markers', name='Trajectory',
                line=dict(color='#FF6B6B', width=4),
                marker=dict(size=3, color=list(range(len(actions))), colorscale='Viridis', showscale=True)
            ))
        
        fig.update_layout(
            title=dict(text='🗺️ 3D End-Effector Trajectory', x=0.5),
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data',
            ),
            height=400, margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig
    except Exception as e:
        print(f"Error in make_trajectory_plot: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)[:40]}", x=0.5, y=0.5, 
                          xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=400, title=dict(text="3D Trajectory", x=0.5), 
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig


def make_curves_plot(samples: List[Dict], key: str, title: str) -> go.Figure:
    """Time-series curves for action/state data."""
    try:
        fig = go.Figure()
        
        data = []
        for s in samples:
            d = s.get(key)
            if d is not None:
                if isinstance(d, torch.Tensor):
                    d = d.numpy()
                if len(d.shape) == 2:
                    d = d[0]  # Take first step if multi-step
                data.append(d.copy() if isinstance(d, np.ndarray) else d)
        
        if not data:
            fig.add_annotation(text=f"No {key} data", x=0.5, y=0.5, 
                              xref="paper", yref="paper", showarrow=False, font=dict(size=16))
            fig.update_layout(height=400, title=dict(text=title, x=0.5), template='plotly_white')
            return fig
        
        data = np.array(data, dtype=float)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_frames = data.shape[0]
        n_dims = data.shape[1] if len(data.shape) > 1 else 1
        
        # Create single plot with all dimensions
        colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel
        
        for i in range(min(n_dims, 20)):  # Limit to 20 dimensions for performance
            y_data = data[:, i].tolist() if len(data.shape) > 1 else data.tolist()
            fig.add_trace(go.Scatter(
                x=list(range(n_frames)),
                y=y_data,
                mode='lines',
                name=f'D{i}',
                line=dict(color=colors[i % len(colors)], width=1.5),
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=14)),
            xaxis_title="Frame",
            yaxis_title="Value",
            height=400,
            margin=dict(l=50, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified',
            template='plotly_white'
        )
        return fig
    except Exception as e:
        print(f"Error in make_curves_plot({key}): {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)[:40]}", x=0.5, y=0.5, 
                          xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=400, title=dict(text=title, x=0.5), template='plotly_white')
        return fig


def make_curves_plot_mpl(samples: List[Dict], key: str, title: str, max_dims: int = 20):
    """Matplotlib fallback for time-series curves (robust rendering in Gradio).
    
    Uses transparent background so Gradio theme colors show through.
    """
    # Collect data first
    data = []
    for s in samples:
        d = s.get(key)
        if d is None:
            continue
        if isinstance(d, torch.Tensor):
            d = d.numpy()
        if isinstance(d, np.ndarray) and d.ndim == 2:
            d = d[0]  # first step for chunked action/state
        data.append(d.copy() if isinstance(d, np.ndarray) else d)

    # Use transparent background
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='none')
    ax.set_facecolor('none')
    ax.set_title(title)
    
    if not data:
        ax.text(0.5, 0.5, f"No {key} data", ha="center", va="center", 
               transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    arr = np.array(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_frames, n_dims = arr.shape[0], arr.shape[1]
    dims_to_plot = min(n_dims, max_dims)

    x = np.arange(n_frames)
    colors = plt.cm.tab10.colors + plt.cm.Set2.colors
    for i in range(dims_to_plot):
        ax.plot(x, arr[:, i], linewidth=1.2, label=f"D{i}", color=colors[i % len(colors)])

    ax.set_xlabel("Frame")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    if dims_to_plot <= 10:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def make_data_table(samples: List[Dict], key: str) -> pd.DataFrame:
    """Create data table for action/state."""
    rows = []
    for fi, s in enumerate(samples):
        d = s.get(key)
        if d is None:
            continue
        if isinstance(d, torch.Tensor):
            d = d.numpy()
        if len(d.shape) == 2:
            for si, sd in enumerate(d):
                row = {'Frame': fi, 'Step': si}
                for di, v in enumerate(sd):
                    row[f'D{di}'] = round(float(v), 4)
                rows.append(row)
        else:
            row = {'Frame': fi, 'Step': 0}
            for di, v in enumerate(d):
                row[f'D{di}'] = round(float(v), 4)
            rows.append(row)
    
    return pd.DataFrame(rows) if rows else pd.DataFrame({'Info': ['No data']})


# ============================================================================
# Gradio Callbacks
# ============================================================================

# Track when load_dataset_cb was called to prevent duplicate processing
import time as _time
_LOAD_TIMESTAMP = 0
_LOAD_EPISODE = None
_LOAD_CALL_COUNT = 0  # Track number of calls for debugging

def _format_dataset_info(size_info: Dict) -> str:
    """Format dataset size info for display."""
    frames = size_info.get('total_frames', 0)
    total_eps = size_info.get('total_episodes')
    indexed_eps = size_info.get('indexed_episodes', 0)
    complete = size_info.get('indexing_complete', False)
    
    # Format frames with K/M suffix
    if frames >= 1_000_000:
        frames_str = f"{frames / 1_000_000:.2f}M"
    elif frames >= 1_000:
        frames_str = f"{frames / 1_000:.1f}K"
    else:
        frames_str = str(frames)
    
    # Format episodes
    if total_eps:
        if complete:
            eps_str = f"{total_eps}"
        else:
            eps_str = f"{indexed_eps}/{total_eps}"
    else:
        eps_str = f"{indexed_eps}+" if not complete else str(indexed_eps)
    
    status = "✅" if complete else "⏳"
    return f"📊 Frames: {frames_str} | Episodes: {eps_str} {status}"


def load_dataset_cb(task_name: str):
    """Load dataset with lazy indexing and return pre-warmed visuals if available.
    
    OPTIMIZED: If data is already cached (e.g., after theme switch refresh),
    returns cached data immediately without reloading.
    """
    global _LOAD_TIMESTAMP, _LOAD_EPISODE, _LOAD_CALL_COUNT
    
    _LOAD_CALL_COUNT += 1
    call_id = _LOAD_CALL_COUNT
    
    if not task_name or not task_name.strip():
        return ("❌ Please enter a task name", "", gr.update(), gr.update(), 0, 0, None, None, "", None, None, None, None, None, gr.update(), gr.update(), None, None)
    
    task_name = task_name.strip()
    
    # Debounce: Skip if called too recently (within 3 seconds) for same task
    time_since_last = _time.time() - _LOAD_TIMESTAMP
    if task_name == MGR.current_task_name and time_since_last < 3.0 and _LOAD_EPISODE is not None:
        print(f"⏭️ [Call #{call_id}] Skipping duplicate load_dataset_cb (last call {time_since_last:.2f}s ago)")
        # Return current state without reloading
        ep_ids = MGR.get_episode_ids()
        first_ep = _LOAD_EPISODE
        size_info = MGR.get_dataset_size_info()
        ds_names = [d['name'] for d in MGR.datasets]
        norm_types = MGR.get_available_norm_types()
        
        return (
            f"✅ Loaded: {task_name}",
            _format_dataset_info(size_info),
            gr.update(choices=ds_names, value=ds_names[0] if ds_names else None),
            gr.update(choices=[str(e) for e in ep_ids], value=str(first_ep) if ep_ids else None),
            gr.update(), gr.update(),  # Keep slider values
            gr.update(), gr.update(), gr.update(),  # Keep video, image, lang
            gr.update(), gr.update(), gr.update(),  # Keep plots
            gr.update(), gr.update(),  # Keep tables
            gr.update(), gr.update(),  # Keep checkboxes
            gr.update(), gr.update()   # Keep stats plots
        )
    
    print(f"🔄 [Call #{call_id}] load_dataset_cb: task={task_name}")
    
    # Check if data is already loaded and cached (e.g., after theme switch)
    # This prevents reloading on page refresh caused by theme changes
    is_already_loaded = (
        task_name == MGR.current_task_name and 
        MGR.task_config is not None and 
        len(MGR.episode_cache) > 0
    )
    
    # Check if pre-warmed visuals are available
    is_prewarmed = is_already_loaded and len(MGR.priority_visuals) > 0
    
    success, msg, meta = MGR.load_task(task_name)
    if not success:
        return (f"❌ {msg}", "", gr.update(), gr.update(), 0, 0, None, None, "", None, None, None, None, None, gr.update(), gr.update(), None, None)
    
    ds_names = [d['name'] for d in MGR.datasets]
    ep_ids = MGR.get_episode_ids()  # Use discovered episodes only (prevents dropdown freeze)
    first_ep = ep_ids[0] if ep_ids else 0
    
    # Mark this episode as just loaded with timestamp to prevent duplicate processing
    _LOAD_TIMESTAMP = _time.time()
    _LOAD_EPISODE = first_ep
    
    # Get dataset size info
    size_info = MGR.get_dataset_size_info()
    size_str = _format_dataset_info(size_info)
    norm_types = MGR.get_available_norm_types()
    
    # Create stats plots (matplotlib for robust rendering)
    action_stats_plot = make_stats_plot_mpl(MGR.action_stats, "📊 Action Statistics (per dimension)")
    state_stats_plot = make_stats_plot_mpl(MGR.state_stats, "📊 State Statistics (per dimension)")
    
    # If pre-warmed visuals available, return them immediately
    if is_prewarmed and first_ep in MGR.priority_visuals:
        v = MGR.priority_visuals[first_ep]
        ep_len = MGR.get_episode_length(first_ep)
        
        # Get language from cached first sample if available
        samples = MGR.episode_cache.get(first_ep, [])
        lang = samples[0].get('raw_lang', '') if samples else ''
        
        print(f"✅ Returning pre-warmed data for episode {first_ep} (instant)")
        
        # Use ALL cached visuals - no regeneration needed
        return (
            f"✅ Loaded: {task_name} (cached)",
            size_str,
            gr.update(choices=ds_names, value=ds_names[0] if ds_names else None),
            gr.update(choices=[str(e) for e in ep_ids], value=str(first_ep) if ep_ids else None),
            max(0, ep_len - 1), 0, 
            v.get('video_path'), v.get('first_img'), f"📝 {lang}",
            v.get('traj'), v.get('act_curve'), v.get('state_curve'),
            v.get('act_table'), v.get('state_table'),
            gr.update(value=True),  # Default to show raw data
            gr.update(choices=norm_types, value=MGR.norm_type),
            action_stats_plot,
            state_stats_plot
        )
    
    # If data is already loaded but no pre-warmed visuals, use cached episode data
    if is_already_loaded and first_ep in MGR.episode_cache:
        print(f"✅ Using cached data for episode {first_ep} (after theme switch)")
        samples = MGR.episode_cache.get(first_ep, [])
        if samples:
            # Denormalize if needed
            if MGR.action_stats:
                samples = MGR._denormalize_samples(samples)
            
            ep_len = len(samples)
            video_path = MGR.get_cached_video(first_ep)
            first_img = create_multiview_image(samples[0]) if samples else None
            lang = samples[0].get('raw_lang', '') if samples else ''
            
            action_dim = meta.get('action_dim', 7)
            ctrl_space = MGR.get_ctrl_space()
            
            return (
                f"✅ Loaded: {task_name} (cached)",
                size_str,
                gr.update(choices=ds_names, value=ds_names[0] if ds_names else None),
                gr.update(choices=[str(e) for e in ep_ids], value=str(first_ep) if ep_ids else None),
                max(0, ep_len - 1), 0, 
                video_path, first_img, f"📝 {lang}",
                make_trajectory_plot(samples, action_dim, ctrl_space),
                make_curves_plot_mpl(samples, 'action', '📈 Action Curves (Raw Data)'),
                make_curves_plot_mpl(samples, 'state', '📈 State Curves (Raw Data)'),
                make_data_table(samples, 'action'),
                make_data_table(samples, 'state'),
                gr.update(value=True),
                gr.update(choices=norm_types, value=MGR.norm_type),
                action_stats_plot,
                state_stats_plot
            )

    # PRIORITY: Load first episode IMMEDIATELY before any background tasks
    # This ensures the user sees content as fast as possible
    print(f"⚡ Loading episode {first_ep} with highest priority...")
    MGR._current_episode = first_ep
    
    samples = MGR.get_episode(first_ep, denormalize=True) if ep_ids else None
    if samples:
        # Generate video for first episode
        video_path = MGR.get_cached_video(first_ep)
        if not video_path:
            video_path = create_episode_video(samples)
            if video_path:
                MGR.cache_video(first_ep, video_path)
        
        action_dim = meta.get('action_dim', 7)
        ctrl_space = MGR.get_ctrl_space()
        first_img = create_multiview_image(samples[0])
        lang = samples[0].get('raw_lang', '')
        
        # NOW start background tasks AFTER first episode is ready
        def _start_background_tasks():
            MGR.start_background_indexing()
            MGR.preload_nearby(first_ep)
            MGR.pregenerate_nearby_videos(first_ep)
        threading.Thread(target=_start_background_tasks, daemon=True).start()
        
        return (
            f"✅ {msg}",
            size_str,
            gr.update(choices=ds_names, value=ds_names[0] if ds_names else None),
            gr.update(choices=[str(e) for e in ep_ids], value=str(first_ep) if ep_ids else None),
            max(0, len(samples) - 1), 0, video_path, first_img, f"📝 {lang}",
            make_trajectory_plot(samples, action_dim, ctrl_space),
            make_curves_plot_mpl(samples, 'action', '📈 Action Curves (Raw Data)'),
            make_curves_plot_mpl(samples, 'state', '📈 State Curves (Raw Data)'),
            make_data_table(samples, 'action'),
            make_data_table(samples, 'state'),
            gr.update(value=True),  # Default to show raw data
            gr.update(choices=norm_types, value=MGR.norm_type),
            action_stats_plot,
            state_stats_plot
        )
    
    return (
        f"✅ {msg}",
        size_str,
        gr.update(choices=ds_names, value=ds_names[0] if ds_names else None),
        gr.update(choices=[str(e) for e in ep_ids], value=str(first_ep) if ep_ids else None),
        0, 0, None, None, "", None, None, None, None, None,
        gr.update(value=True),  # Default to show raw data
        gr.update(choices=norm_types, value=MGR.norm_type),
        action_stats_plot,
        state_stats_plot
    )


def refresh_episode_list_cb():
    """Refresh episode dropdown with newly indexed episodes and dataset info."""
    ep_ids = MGR.get_episode_ids()  # Use discovered episodes only
    size_info = MGR.get_dataset_size_info()
    size_str = _format_dataset_info(size_info)
    
    return gr.update(choices=[str(e) for e in ep_ids]), size_str


def prev_episode_cb(ep_str: str):
    """Navigate to previous episode (simple decrement)."""
    if not ep_str:
        return gr.update()
    
    try:
        current_eid = int(ep_str)
        if current_eid > 0:
            new_eid = current_eid - 1
            # Get discovered episodes for dropdown
            ep_ids = MGR.get_episode_ids()
            # Add new_eid if not in list (will be discovered on load)
            if new_eid not in ep_ids:
                ep_ids = sorted(set(ep_ids) | {new_eid})
            return gr.update(choices=[str(e) for e in ep_ids], value=str(new_eid))
        return gr.update()
    except (ValueError, IndexError):
        return gr.update()


def next_episode_cb(ep_str: str):
    """Navigate to next episode (simple increment)."""
    if not ep_str:
        return gr.update()
    
    try:
        current_eid = int(ep_str)
        estimated = MGR.get_estimated_total_episodes()
        max_eid = (estimated - 1) if estimated else 999999
        
        if current_eid < max_eid:
            new_eid = current_eid + 1
            # Get discovered episodes for dropdown
            ep_ids = MGR.get_episode_ids()
            # Add new_eid if not in list (will be discovered on load)
            if new_eid not in ep_ids:
                ep_ids = sorted(set(ep_ids) | {new_eid})
            return gr.update(choices=[str(e) for e in ep_ids], value=str(new_eid))
        return gr.update()
    except (ValueError, IndexError):
        return gr.update()


def jump_to_sample_cb(sample_idx_str: str, show_raw: bool):
    """Jump to a specific sample index and display its episode."""
    global _LOAD_TIMESTAMP, _LOAD_EPISODE
    
    if not sample_idx_str or not sample_idx_str.strip():
        return (gr.update(), "❌ Please enter a sample index",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    
    try:
        sample_idx = int(sample_idx_str.strip())
    except ValueError:
        return (gr.update(), f"❌ Invalid sample index: {sample_idx_str}",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    
    success, eid, msg = MGR.jump_to_sample(sample_idx)
    
    if not success:
        return (gr.update(), f"❌ {msg}",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    
    # Prevent duplicate processing
    _LOAD_TIMESTAMP = _time.time()
    _LOAD_EPISODE = eid
    
    # Get updated episode list (discovered only)
    ep_ids = MGR.get_episode_ids()
    
    # Load episode data
    samples = MGR.get_episode(eid, denormalize=show_raw)
    if not samples:
        return (gr.update(choices=[str(e) for e in ep_ids], value=str(eid)),
                f"✅ {msg} (but no data loaded)",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    
    # Generate video
    video_path = MGR.get_cached_video(eid)
    if not video_path:
        norm_samples = MGR.get_episode(eid, denormalize=False) if show_raw else samples
        video_path = create_episode_video(norm_samples)
        if video_path:
            MGR.cache_video(eid, video_path)
    
    # Generate visuals
    meta = MGR.task_config.get('meta', {}) if MGR.task_config else {}
    action_dim = meta.get('action_dim', 7)
    ctrl_space = MGR.get_ctrl_space()
    data_type = "Raw Data" if show_raw else "Normalized Data"
    
    first_img = create_multiview_image(samples[0])
    lang = samples[0].get('raw_lang', '')
    
    return (
        gr.update(choices=[str(e) for e in ep_ids], value=str(eid)),
        f"✅ {msg}",
        max(0, len(samples) - 1), 0, video_path, first_img, f"📝 {lang}",
        make_trajectory_plot(samples, action_dim, ctrl_space),
        make_curves_plot_mpl(samples, 'action', f'📈 Action Curves ({data_type})'),
        make_curves_plot_mpl(samples, 'state', f'📈 State Curves ({data_type})'),
        make_data_table(samples, 'action'),
        make_data_table(samples, 'state')
    )


def select_dataset_cb(ds_name: str):
    """Dataset change with lazy indexing."""
    global _LOAD_TIMESTAMP, _LOAD_EPISODE
    
    if not ds_name:
        return gr.update(), 0, 0
    
    for i, d in enumerate(MGR.datasets):
        if d['name'] == ds_name:
            MGR._select_dataset(i)
            break
    
    # Start background indexing for new dataset
    MGR.start_background_indexing()
    
    # Preload first several episodes
    MGR.preload_initial_episodes(count=10)
    
    ep_ids = MGR.get_episode_ids()  # Use discovered episodes only
    first_ep = ep_ids[0] if ep_ids else 0
    first_len = MGR.get_episode_length(first_ep) if ep_ids else 0
    
    # Mark to prevent duplicate processing
    _LOAD_TIMESTAMP = _time.time()
    _LOAD_EPISODE = first_ep
    
    return (
        gr.update(choices=[str(e) for e in ep_ids], value=str(first_ep) if ep_ids else None),
        max(0, first_len - 1),
        0
    )


def select_episode_cb(ep_str: str, show_raw: bool):
    """Episode change - generate video and plots.
    
    Returns 10 values (NOT including show_raw to prevent event cascade).
    """
    global _LOAD_TIMESTAMP, _LOAD_EPISODE
    
    print(f"🔄 select_episode_cb called: ep={ep_str}, show_raw={show_raw}")
    
    if not ep_str:
        return 0, 0, None, None, "", None, None, None, None, None
    
    eid = int(ep_str)
    
    # Update current episode for cache management
    MGR._current_episode = eid
    
    # Skip if this episode was just loaded by load_dataset_cb within last 2 seconds
    # This prevents the duplicate processing caused by Gradio's change event
    time_since_load = _time.time() - _LOAD_TIMESTAMP
    if _LOAD_EPISODE == eid and time_since_load < 2.0:
        print(f"⏭️ Skipping select_episode_cb for episode {eid} (loaded {time_since_load:.2f}s ago)")
        # Don't clear _LOAD_EPISODE here - let toggle_raw_cb also skip
        # Return gr.update() for all outputs to keep current values (10 outputs)
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    
    samples = MGR.get_episode(eid, denormalize=show_raw)
    
    if not samples:
        return 0, 0, None, None, "No data", None, None, None, None, None
    
    # Try to use cached video first, otherwise generate in background
    video_path = MGR.get_cached_video(eid)
    if not video_path:
        # Generate video synchronously for current episode (user is waiting)
        # Use samples directly - they contain original images regardless of denormalize flag
        video_path = create_episode_video(samples)
        if video_path:
            MGR.cache_video(eid, video_path)
    
    # Preload next episode AFTER returning data (in background thread)
    def _bg_preload():
        MGR.preload_nearby(eid)
        MGR.pregenerate_nearby_videos(eid)
    threading.Thread(target=_bg_preload, daemon=True).start()
    
    # First frame for preview
    first_img = create_multiview_image(samples[0])
    
    # Language
    lang = samples[0].get('raw_lang', '')
    lang_text = f"📝 {lang}" if lang else ""
    
    # Plots and tables - use denormalized data if show_raw
    meta = MGR.task_config.get('meta', {}) if MGR.task_config else {}
    action_dim = meta.get('action_dim', 7)
    ctrl_space = MGR.get_ctrl_space()
    
    data_type = "Raw Data" if show_raw else "Normalized Data"
    
    traj = make_trajectory_plot(samples, action_dim, ctrl_space)
    act_curve = make_curves_plot_mpl(samples, 'action', f'📈 Action Curves ({data_type})')
    state_curve = make_curves_plot_mpl(samples, 'state', f'📈 State Curves ({data_type})')
    act_table = make_data_table(samples, 'action')
    state_table = make_data_table(samples, 'state')
    
    ep_len = len(samples)
    
    # Return 10 values (no show_raw to prevent event cascade)
    return (
        max(0, ep_len - 1),
        0,
        video_path,
        first_img,
        lang_text,
        traj,
        act_curve,
        state_curve,
        act_table,
        state_table
    )


def toggle_raw_cb(ep_str: str, speed: float, show_raw: bool, norm_type: str):
    """Toggle between normalized and raw data - update ONLY plots and tables.
    
    FAST PATH: Uses cached episode data, only applies denormalization transform.
    Does NOT update any other components to prevent event cascades.
    """
    global _LOAD_TIMESTAMP, _LOAD_EPISODE
    
    print(f"🔄 toggle_raw_cb called: ep={ep_str}, show_raw={show_raw}, norm_type={norm_type}")
    
    if not ep_str:
        return None, None, None, None, None
    
    eid = int(ep_str)
    
    # Skip if this was just loaded by load_dataset_cb within last 2 seconds
    # This prevents overwriting the plots that were just set by load_dataset_cb
    time_since_load = _time.time() - _LOAD_TIMESTAMP
    if _LOAD_EPISODE == eid and time_since_load < 2.0:
        print(f"⏭️ Skipping toggle_raw_cb for episode {eid} (loaded {time_since_load:.2f}s ago)")
        # Return gr.update() to keep current values
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    
    # Update normalization type if changed
    if norm_type and norm_type != MGR.norm_type:
        MGR.norm_type = norm_type
    
    # FAST PATH: Get from cache directly
    samples = None
    with MGR.lock:
        if eid in MGR.episode_cache:
            # Cache stores normalized data
            cached_samples = MGR.episode_cache[eid]
            if show_raw and MGR.action_stats:
                # Apply denormalization with current norm_type
                samples = MGR._denormalize_samples(cached_samples)
            else:
                # Return normalized data as-is
                samples = cached_samples
    
    # Fallback if not in cache
    if samples is None:
        samples = MGR.get_episode(eid, denormalize=show_raw)
    
    if not samples:
        return None, None, None, None, None
    
    meta = MGR.task_config.get('meta', {}) if MGR.task_config else {}
    action_dim = meta.get('action_dim', 7)
    ctrl_space = MGR.get_ctrl_space()
    
    data_type = f"Raw Data ({norm_type})" if show_raw else "Normalized Data"
    
    traj = make_trajectory_plot(samples, action_dim, ctrl_space)
    act_curve = make_curves_plot_mpl(samples, 'action', f'📈 Action Curves ({data_type})')
    state_curve = make_curves_plot_mpl(samples, 'state', f'📈 State Curves ({data_type})')
    act_table = make_data_table(samples, 'action')
    state_table = make_data_table(samples, 'state')
    
    print(f"✅ toggle_raw_cb completed for ep={eid}, show_raw={show_raw}")
    return traj, act_curve, state_curve, act_table, state_table


def select_frame_cb(frame_idx: int, ep_str: str, slider_max: int):
    """Frame slider change."""
    if not ep_str:
        return None, ""
    
    frame_idx = int(frame_idx)
    if frame_idx < 0:
        frame_idx = 0
    if slider_max > 0 and frame_idx > slider_max:
        frame_idx = slider_max
    
    eid = int(ep_str)
    sample = MGR.get_frame(eid, frame_idx)
    
    if sample is None:
        return None, ""
    
    img = create_multiview_image(sample)
    lang = sample.get('raw_lang', '')
    ep_len = MGR.get_episode_length(eid)
    
    return img, f"📝 帧 {frame_idx + 1}/{ep_len}: {lang}" if lang else f"📝 帧 {frame_idx + 1}/{ep_len}"


def refresh_data_cb(ep_str: str, show_raw: bool):
    """Refresh all data with current settings."""
    return select_episode_cb(ep_str, show_raw)




# ============================================================================
# Build Interface
# ============================================================================

def create_app(default_task: str = 'sim_transfer_cube_scripted', 
               default_lang: str = 'en', 
               default_theme: str = 'system'):
    """Create Gradio app.
    
    Args:
        default_task: Default task configuration name
        default_lang: Default language (en, zh, etc.)
        default_theme: Default theme (system, light, dark)
    """
    
    css = """
    /* ===== MINIMAL CLEAN STYLE ===== */
    /* Let Gradio theme handle most styling, only add essential customizations */
    
    /* Title */
    .title { 
        text-align: center; 
        font-size: 2em; 
        font-weight: bold; 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 15px;
    }
    
    /* Section headers */
    .section { 
        padding: 8px 15px; 
        border-radius: 5px; 
        margin: 8px 0; 
        font-weight: bold; 
        font-size: 1.1em;
    }
    
    /* Refresh button styling */
    .refresh-btn { 
        font-weight: bold !important; 
        border-radius: 8px !important; 
    }
    
    /* Make dropdowns have clear borders */
    .wrap, .wrap-inner, .secondary-wrap {
        border: 1px solid var(--border-color-primary) !important;
        border-radius: 8px !important;
    }
    
    /* Dropdown arrow icon - larger */
    .wrap svg, .wrap-inner svg {
        width: 20px !important;
        height: 20px !important;
        transform: scale(1.2) !important;
    }
    
    /* Dropdown list */
    .wrap ul, .wrap-inner ul {
        border: 1px solid var(--border-color-primary) !important;
        border-radius: 8px !important;
        margin-top: 2px !important;
        max-height: 400px !important;
    }
    
    /* Dropdown items */
    .wrap ul li, .wrap-inner ul li {
        padding: 10px 14px !important;
        font-size: 1em !important;
    }
    
    /* Checkbox styling */
    input[type="checkbox"] {
        width: 20px !important;
        height: 20px !important;
        cursor: pointer !important;
    }
    
    /* Compact navigation buttons */
    .refresh-btn {
        min-width: 36px !important;
        max-width: 50px !important;
        padding: 6px 8px !important;
        font-size: 0.9em !important;
    }
    
    /* Align row items */
    .gr-row {
        align-items: flex-end !important;
    }
    """
    
    # JavaScript to control video playback speed
    js_speed_control = """
    function(speed) {
        const videos = document.querySelectorAll('video');
        videos.forEach(v => {
            v.playbackRate = speed;
        });
        return speed;
    }
    """
    
    # JavaScript to apply speed after video loads (called after episode change)
    js_apply_speed_after_load = """
    function(speed) {
        // Apply speed immediately
        const videos = document.querySelectorAll('video');
        videos.forEach(v => {
            v.playbackRate = speed;
            // Also set up listener for when video loads
            v.onloadeddata = function() {
                v.playbackRate = speed;
            };
        });
        // Use MutationObserver to catch newly added videos
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.tagName === 'VIDEO') {
                        node.playbackRate = speed;
                        node.onloadeddata = function() {
                            node.playbackRate = speed;
                        };
                    }
                    if (node.querySelectorAll) {
                        node.querySelectorAll('video').forEach(v => {
                            v.playbackRate = speed;
                            v.onloadeddata = function() {
                                v.playbackRate = speed;
                            };
                        });
                    }
                });
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
        // Disconnect after 5 seconds to avoid memory leak
        setTimeout(() => observer.disconnect(), 5000);
        return speed;
    }
    """
    
    # HTML/JS to set default theme on first load (configurable via CLI args)
    # Theme: Gradio uses URL param __theme (light/dark/system)
    # Language: Gradio uses svelte-i18n, detected from browser's accept-language header
    # 
    # NOTE: Gradio 6.x language is determined by browser settings, not configurable via API.
    # Users can change language in Settings (?view=settings).
    # The --lang parameter is informational only for Gradio 6.x.
    head_html = f"""
    <script>
    (function() {{
        // Check if this is a fresh visit (no theme param and no saved preference)
        const url = new URL(window.location.href);
        const hasThemeParam = url.searchParams.has('__theme');
        const savedTheme = localStorage.getItem('gradio-user-theme');
        
        // If user hasn't explicitly set a theme, apply the CLI default
        if (!hasThemeParam && !savedTheme) {{
            const defaultTheme = '{default_theme}';
            if (defaultTheme !== 'system') {{
                // Redirect to URL with theme param
                url.searchParams.set('__theme', defaultTheme);
                window.location.replace(url.toString());
            }}
        }}
        
        // Save user's theme choice when they change it (for persistence)
        // This is done by intercepting the Settings page theme buttons
        window.addEventListener('click', function(e) {{
            const btn = e.target.closest('.theme-button');
            if (btn) {{
                const text = btn.textContent || '';
                if (text.includes('Light')) {{
                    localStorage.setItem('gradio-user-theme', 'light');
                }} else if (text.includes('Dark')) {{
                    localStorage.setItem('gradio-user-theme', 'dark');
                }} else if (text.includes('System')) {{
                    localStorage.setItem('gradio-user-theme', 'system');
                }}
            }}
        }});
    }})();
    </script>
    """
    
    # Use default Soft theme - supports Gradio's built-in light/dark toggle
    with gr.Blocks(title="Dataset Visualizer", theme=gr.themes.Soft(primary_hue="red"), css=css, head=head_html) as app:
        
        # State
        slider_max_state = gr.State(0)
        
        gr.HTML("<div class='title'>🤖 IL-Studio Dataset Visualizer</div>")
        
        # ===== Controls =====
        with gr.Row():
            # Left column: Task loading
            with gr.Column(scale=2):
                gr.HTML("<div class='section'>⚙️ Task Configuration</div>")
                with gr.Row():
                    task_input = gr.Textbox(label="Task Config", value=default_task, scale=4)
                    load_btn = gr.Button("🔄 Load", variant="primary", scale=1)
                status = gr.Textbox(label="Status", lines=1, interactive=False)
                dataset_info = gr.Textbox(label="Dataset Info", lines=1, interactive=False)
                dataset_dd = gr.Dropdown(label="📊 Dataset", choices=[])
            
            # Right column: Episode navigation
            with gr.Column(scale=2):
                gr.HTML("<div class='section'>📁 Episode Navigation</div>")
                with gr.Row():
                    episode_dd = gr.Dropdown(label="Episode", choices=[], scale=5)
                    prev_ep_btn = gr.Button("◀", scale=1, min_width=40, elem_classes=["refresh-btn"])
                    next_ep_btn = gr.Button("▶", scale=1, min_width=40, elem_classes=["refresh-btn"])
                    refresh_ep_btn = gr.Button("🔄", scale=1, min_width=40, elem_classes=["refresh-btn"])
                with gr.Row():
                    sample_idx_input = gr.Textbox(label="Jump to Sample", placeholder="e.g. 12345", scale=5)
                    jump_btn = gr.Button("Go", scale=1, min_width=40, elem_classes=["refresh-btn"])
        
        gr.HTML("<hr style='border: 1px solid #cccccc; margin: 10px 0;'>")
        
        # ===== Video =====
        gr.HTML("<div class='section'>🎬 Video Player</div>")
        
        with gr.Row():
            with gr.Column(scale=3):
                video_player = gr.Video(label="Episode Video (Multi-view)", autoplay=True, loop=True, height=280)
                with gr.Row():
                    speed_slider = gr.Slider(label="Speed", minimum=0.25, maximum=4.0, value=1.0, step=0.25, scale=3)
                    speed_label = gr.HTML("<span style='color:#666;'>1.0x=30fps</span>", scale=1)
            with gr.Column(scale=2):
                frame_img = gr.Image(label="Frame Preview", type="numpy", height=180)
                frame_slider = gr.Slider(label="Frame", minimum=0, maximum=0, step=1, value=0)
                lang_box = gr.Textbox(label="Language", lines=1, interactive=False)
        
        gr.HTML("<hr style='border: 1px solid #cccccc; margin: 10px 0;'>")
        
        # ===== Analysis =====
        gr.HTML("<div class='section'>📊 Data Analysis</div>")
        
        with gr.Row():
            show_raw = gr.Checkbox(label="✅ Show Raw Data (Denormalized)", value=True, scale=2)
            norm_type_dd = gr.Dropdown(label="Norm Type", choices=['zscore', 'minmax', 'percentile'], value='zscore', scale=2)
        
        # Statistics plots - side by side
        gr.Markdown("### 📊 Normalization Statistics")
        with gr.Row():
            action_stats_plot = gr.Plot(label="Action Stats")
            state_stats_plot = gr.Plot(label="State Stats")
        
        # Data visualization - simple layout without Accordion
        gr.Markdown("### 🗺️ 3D Trajectory")
        traj_plot = gr.Plot(label="3D Trajectory")
        
        gr.Markdown("### 📈 Action & State Curves")
        with gr.Row():
            action_plot = gr.Plot(label="Action Curves")
            state_plot = gr.Plot(label="State Curves")
        
        gr.Markdown("### 📋 Data Tables")
        with gr.Row():
            action_table = gr.Dataframe(label="Action Table", max_height=300)
            state_table = gr.Dataframe(label="State Table", max_height=300)
        
        # ===== Event Bindings =====
        
        load_btn.click(
            fn=load_dataset_cb,
            inputs=[task_input],
            outputs=[status, dataset_info, dataset_dd, episode_dd, slider_max_state, frame_slider,
                    video_player, frame_img, lang_box, traj_plot, action_plot, state_plot, action_table, state_table,
                    show_raw, norm_type_dd, action_stats_plot, state_stats_plot]
        ).then(
            fn=lambda m: gr.update(maximum=m),
            inputs=[slider_max_state],
            outputs=[frame_slider]
        ).then(
            fn=None,
            inputs=[speed_slider],
            outputs=[],
            js=js_apply_speed_after_load
        )
        
        dataset_dd.change(
            fn=select_dataset_cb,
            inputs=[dataset_dd],
            outputs=[episode_dd, slider_max_state, frame_slider]
        ).then(
            fn=lambda m: gr.update(maximum=m),
            inputs=[slider_max_state],
            outputs=[frame_slider]
        )
        
        # Refresh episode list and dataset info (for background-indexed episodes)
        refresh_ep_btn.click(
            fn=refresh_episode_list_cb,
            inputs=[],
            outputs=[episode_dd, dataset_info]
        )
        
        # Prev/Next episode navigation
        prev_ep_btn.click(
            fn=prev_episode_cb,
            inputs=[episode_dd],
            outputs=[episode_dd]
        )
        
        next_ep_btn.click(
            fn=next_episode_cb,
            inputs=[episode_dd],
            outputs=[episode_dd]
        )
        
        # Jump to sample
        jump_btn.click(
            fn=jump_to_sample_cb,
            inputs=[sample_idx_input, show_raw],
            outputs=[episode_dd, status, slider_max_state, frame_slider, video_player, frame_img, lang_box,
                    traj_plot, action_plot, state_plot, action_table, state_table]
        ).then(
            fn=lambda m: gr.update(maximum=m),
            inputs=[slider_max_state],
            outputs=[frame_slider]
        ).then(
            fn=None,
            inputs=[speed_slider],
            outputs=[],
            js=js_apply_speed_after_load
        )
        
        # NOTE: Do NOT include show_raw in outputs to prevent event cascade
        episode_dd.change(
            fn=select_episode_cb,
            inputs=[episode_dd, show_raw],
            outputs=[slider_max_state, frame_slider, video_player, frame_img, lang_box,
                    traj_plot, action_plot, state_plot, action_table, state_table]
        ).then(
            fn=lambda m: gr.update(maximum=m),
            inputs=[slider_max_state],
            outputs=[frame_slider]
        ).then(
            fn=None,
            inputs=[speed_slider],
            outputs=[],
            js=js_apply_speed_after_load
        )
        
        frame_slider.change(
            fn=select_frame_cb,
            inputs=[frame_slider, episode_dd, slider_max_state],
            outputs=[frame_img, lang_box]
        )
        
        # Speed slider controls video playback rate via JavaScript
        speed_slider.change(
            fn=None,
            inputs=[speed_slider],
            outputs=[],
            js=js_speed_control
        )
        
        # Auto-refresh plots/tables when normalization toggle or type changes
        # NOTE: Do NOT include show_raw in outputs to prevent event cascade
        show_raw.change(
            fn=toggle_raw_cb,
            inputs=[episode_dd, speed_slider, show_raw, norm_type_dd],
            outputs=[traj_plot, action_plot, state_plot, action_table, state_table]
        )
        
        norm_type_dd.change(
            fn=toggle_raw_cb,
            inputs=[episode_dd, speed_slider, show_raw, norm_type_dd],
            outputs=[traj_plot, action_plot, state_plot, action_table, state_table]
        )
        
        # ===== Auto-load on page open =====
        app.load(
            fn=load_dataset_cb,
            inputs=[task_input],
            outputs=[status, dataset_info, dataset_dd, episode_dd, slider_max_state, frame_slider,
                    video_player, frame_img, lang_box, traj_plot, action_plot, state_plot, action_table, state_table,
                    show_raw, norm_type_dd, action_stats_plot, state_stats_plot]
        ).then(
            fn=lambda m: gr.update(maximum=m),
            inputs=[slider_max_state],
            outputs=[frame_slider]
        ).then(
            fn=None,
            inputs=[speed_slider],
            outputs=[],
            js=js_apply_speed_after_load
        )
    
    return app


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dataset Visualizer')
    parser.add_argument('-t', '--task', type=str, default='')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('-r', '--root_path', type=str, default='')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'zh', 'ja', 'ko', 'es', 'fr', 'de'],
                       help='Default language (en, zh, ja, ko, es, fr, de)')
    parser.add_argument('--theme', type=str, default='system', choices=['system', 'light', 'dark'],
                       help='Default theme (system, light, dark)')
    args = parser.parse_args()
    
    task_name = args.task if args.task else 'sim_transfer_cube_scripted'
    
    print("\n" + "="*50)
    print("🤖 IL-Studio Dataset Visualizer")
    print(f"   Task: {task_name}")
    print(f"   Language: {args.lang}, Theme: {args.theme}")
    print("="*50)
    
    # FAST START: Only load config, defer everything else to UI load
    # This gets the server up ASAP, first episode loads when UI opens
    print(f"\n⚡ Quick-loading task config '{task_name}'...")
    
    # Just load task config - actual data loading happens in load_dataset_cb
    success, msg, meta = MGR.load_task(task_name)
    
    if success:
        print(f"✅ Task config loaded: {msg}")
        estimated = MGR.get_estimated_total_episodes()
        if estimated:
            print(f"📊 Estimated {estimated} total episodes")
        print(f"   Action dim: {meta.get('action_dim', 'N/A')}, State dim: {meta.get('state_dim', 'N/A')}")
    else:
        print(f"⚠️ Task config failed: {msg}")
    
    print(f"\n🌐 Starting server at http://localhost:{args.port}")
    print("   First episode will load when you open the UI")
    print("="*50 + "\n")
    
    # Pass task and settings to create_app
    app = create_app(default_task=task_name, default_lang=args.lang, default_theme=args.theme)
    
    app.launch(server_port=args.port, share=args.share, show_error=True, root_path=args.root_path)


if __name__ == "__main__":
    main()
