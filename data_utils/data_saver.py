"""
data_saver.py

Data saving utilities for teleoperation data collection.
Supports both HDF5 (legacy) and LeRobotDataset v3.0 formats.

Key features:
- For non-teleoperator data: save raw data with `observation.` prefix
- For teleoperator data: save as `observation.teleop.xxx` (teleop signals are rarely used for training)
- Support for appending to existing datasets (resume recording)
"""

import json
import os
import queue
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from loguru import logger

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# =============================================================================
# Utility Functions
# =============================================================================

def as_uint8_rgb_image(img: Any) -> Optional[np.ndarray]:
    """
    Convert an input image-like object into uint8 RGB HWC numpy array.
    Returns None if conversion fails.
    """
    if img is None:
        return None
    try:
        arr = np.asarray(img)
    except Exception:
        return None
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        return None
    # Ensure last dim is channels
    if arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):
        # Likely CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        # If float in [0,1], scale up
        if np.issubdtype(arr.dtype, np.floating) and arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
            arr = (arr * 255.0).round()
        arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    return arr


def is_image_like(value: Any) -> bool:
    """Check if a value looks like an image (numpy array with 2-4 dims and image-like shape)."""
    if not isinstance(value, np.ndarray):
        try:
            arr = np.asarray(value)
            if arr.dtype == object:
                return False
        except Exception:
            return False
    else:
        arr = value
    
    if arr.dtype == object:
        return False
    
    # Check if it's a 2D or 3D array that looks like an image
    if arr.ndim == 2:
        # Grayscale image
        return arr.shape[0] > 1 and arr.shape[1] > 1
    elif arr.ndim == 3:
        # HWC or CHW format
        h, w, c = arr.shape
        # Check if it looks like HWC (channels last)
        if c in (1, 3, 4) and h > 1 and w > 1:
            return True
        # Check if it looks like CHW (channels first)
        if h in (1, 3, 4) and w > 1 and c > 1:
            return True
    return False


def is_scalar_or_1d(value: Any) -> bool:
    """Check if a value is a numeric scalar or 1D array (suitable for state/action)."""
    try:
        arr = np.asarray(value)
        if arr.dtype.kind in ("U", "S", "O"):
            return False
        return arr.ndim <= 1
    except Exception:
        return False


# =============================================================================
# Data Conversion
# =============================================================================

def convert_frame_data(
    frame: Dict[str, dict],
    device_names: List[str],
    teleop_device: Optional[str],
    enrich_saved_frame: Optional[Callable[[Dict[str, dict]], Dict[str, dict]]] = None,
) -> Dict[str, Any]:
    """
    Convert a raw frame dict to a standardized format.
    
    For non-teleoperator devices:
        - Save raw data with `observation.` prefix
        - Images are saved as `observation.images.{device_name}` or `observation.images.{key}`
    
    For teleoperator device:
        - Save each key as `observation.teleop.xxx` (teleop signals are rarely used for training)
        - Images are saved as `observation.teleop.images.{key}`
    
    Args:
        frame: Raw frame data {device_name: {key: value, ...}, ...}
        device_names: List of all device names
        teleop_device: Name of teleoperator device (or None)
        enrich_saved_frame: Optional hook to enrich raw device dicts before conversion
            (e.g. AliciaD FK → state_ee at save time).
    
    Returns:
        Converted frame dict with standardized keys
    """
    if enrich_saved_frame is not None:
        frame = enrich_saved_frame(frame)

    result: Dict[str, Any] = {}
    
    # Copy special keys
    if "_sync_timestamp" in frame:
        result["_sync_timestamp"] = frame["_sync_timestamp"]
    
    for dev_name in device_names:
        if dev_name not in frame:
            continue
        
        dev_data = frame.get(dev_name)
        if not isinstance(dev_data, dict):
            continue
        
        if dev_name == teleop_device:
            # Teleoperator data -> observation.teleop.xxx (teleop signals are rarely used for training)
            for key, value in dev_data.items():
                if key in ("__timestamp__", "timestamp") or str(key).startswith("_"):
                    continue
                try:
                    if isinstance(value, np.ndarray):
                        if is_image_like(value):
                            img = as_uint8_rgb_image(value)
                            if img is not None:
                                result[f"observation.teleop.images.{key}"] = img
                        else:
                            result[f"observation.teleop.{key}"] = value
                    elif is_scalar_or_1d(value):
                        result[f"observation.teleop.{key}"] = np.asarray(value, dtype=np.float32).reshape(-1)
                    elif is_image_like(value):
                        img = as_uint8_rgb_image(value)
                        if img is not None:
                            result[f"observation.teleop.images.{key}"] = img
                    else:
                        logger.debug("Skipped teleop key '{}' (type={})", key, type(value).__name__)
                except Exception as e:
                    logger.warning("Failed to convert teleop key '{}': {}", key, e)
        else:
            # Non-teleoperator data -> observation.xxx (save raw data)
            for key, value in dev_data.items():
                if key in ("__timestamp__", "timestamp") or str(key).startswith("_"):
                    continue
                try:
                    if isinstance(value, np.ndarray):
                        if is_image_like(value):
                            img = as_uint8_rgb_image(value)
                            if img is not None:
                                # Use key name if it looks like a camera name, otherwise use device name
                                if key in ("image",):
                                    result[f"observation.images.{dev_name}"] = img
                                else:
                                    result[f"observation.images.{key}"] = img
                        else:
                            # Non-image array data
                            result[f"observation.{key}"] = value
                    elif is_scalar_or_1d(value):
                        arr = np.asarray(value, dtype=np.float32).reshape(-1)
                        result[f"observation.{key}"] = arr
                    elif is_image_like(value):
                        img = as_uint8_rgb_image(value)
                        if img is not None:
                            if key in ("image",):
                                result[f"observation.images.{dev_name}"] = img
                            else:
                                result[f"observation.images.{key}"] = img
                    else:
                        # Log skipped values for debugging
                        logger.debug("Skipped non-serializable key '{}' (type={})", key, type(value).__name__)
                except Exception as e:
                    logger.warning("Failed to convert key '{}': {}", key, e)
    
    return result


# =============================================================================
# Base Data Saver
# =============================================================================

class BaseDataSaver(ABC):
    """
    Abstract base class for data savers with streaming support.
    
    Provides a unified interface for recording episodes:
    - start_episode(episode_idx=None) -> int: Start recording, returns actual episode index
    - add_frame(frame): Add a frame (non-blocking, streamed to disk)
    - finish_episode(save=True) -> int: End recording, returns frame count
    
    All subclasses use streaming by default for real-time non-blocking recording.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        fps: int,
        task: str = "",
        enrich_saved_frame: Optional[Callable[[Dict[str, dict]], Dict[str, dict]]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.fps = int(max(1, fps))
        self.task = task
        self._enrich_saved_frame = enrich_saved_frame
        self._episode_count = 0
        self._feature_schema: Dict[str, dict] = {}  # Track feature schema
        
        # Recording state
        self._recording = False
        self._current_episode_idx: Optional[int] = None
        self._current_device_names: List[str] = []
        self._current_teleop_device: Optional[str] = None
        self._first_frame_received = False
        
        # Streaming state
        self._stream_queue: Optional[queue.Queue] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_stop_event: Optional[threading.Event] = None
        self._stream_error: Optional[Exception] = None
        self._stream_frame_count = 0

    def _convert_frame(
        self,
        frame: Dict[str, dict],
        device_names: List[str],
        teleop_device: Optional[str],
    ) -> Dict[str, Any]:
        """Convert + optional per-device save-time enrichment."""
        return convert_frame_data(
            frame,
            device_names,
            teleop_device,
            enrich_saved_frame=self._enrich_saved_frame,
        )
    
    def start_episode(
        self,
        episode_idx: Optional[int] = None,
        device_names: Optional[List[str]] = None,
        teleop_device: Optional[str] = None,
    ) -> int:
        """
        Start recording an episode with streaming.
        
        Args:
            episode_idx: Episode index (None = append after last episode)
            device_names: List of device names (required for first episode)
            teleop_device: Name of teleop device (if any)
        
        Returns:
            The actual episode index being recorded
        """
        if self._recording:
            logger.warning("Already recording episode {}, finishing first", self._current_episode_idx)
            self.finish_episode(save=False)
        
        # Determine episode index
        if episode_idx is None:
            episode_idx = self._episode_count
        
        self._recording = True
        self._current_episode_idx = episode_idx
        self._current_device_names = device_names or []
        self._current_teleop_device = teleop_device
        self._first_frame_received = False
        
        return episode_idx
    
    def add_frame(self, frame: Dict[str, dict]) -> bool:
        """
        Add a frame to the current episode (non-blocking, streamed to disk).
        
        Args:
            frame: Frame data dict {device_name: device_data_dict}
        
        Returns:
            True if frame was added successfully
        """
        if not self._recording:
            logger.warning("Not recording, call start_episode() first")
            return False
        
        # Start streaming on first frame (to infer schema)
        if not self._first_frame_received:
            self._first_frame_received = True
            success = self._start_streaming(
                episode_idx=self._current_episode_idx,
                device_names=self._current_device_names,
                teleop_device=self._current_teleop_device,
                first_frame=frame,
            )
            if not success:
                return False
        
        # Add frame to streaming queue
        return self._add_frame_async(frame)
    
    def finish_episode(self, save: bool = True) -> int:
        """
        Finish recording and optionally save the episode.
        
        Args:
            save: If True, save the episode. If False, discard.
        
        Returns:
            Number of frames recorded (0 if discarded or error)
        """
        if not self._recording:
            return 0
        
        # If streaming was started, finish it
        if self._stream_thread is not None:
            frame_count = self._finish_streaming(save=save)
        else:
            frame_count = 0
        
        # Update episode count
        if save and frame_count > 0:
            self._episode_count = max(self._episode_count, self._current_episode_idx + 1)
        
        # Reset recording state
        self._recording = False
        self._first_frame_received = False
        
        return frame_count
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording an episode."""
        return self._recording
    
    @property
    def is_streaming(self) -> bool:
        """Check if streaming is currently active."""
        return self._stream_thread is not None and self._stream_thread.is_alive()
    
    @property
    def current_frame_count(self) -> int:
        """Get number of frames in current recording."""
        return self._stream_frame_count if self._recording else 0
    
    # =========================================================================
    # Streaming Methods (to be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    def _start_streaming(
        self,
        episode_idx: int,
        device_names: List[str],
        teleop_device: Optional[str],
        first_frame: Dict[str, dict],
    ) -> bool:
        """
        Start streaming recording. Called on first frame.
        
        Args:
            episode_idx: Episode index
            device_names: List of device names
            teleop_device: Name of teleop device (if any)
            first_frame: First frame data (for schema inference)
        
        Returns:
            True if streaming started successfully
        """
        pass
    
    def _add_frame_async(self, frame: Dict[str, dict]) -> bool:
        """
        Add a frame to the streaming queue (non-blocking).
        
        Args:
            frame: Frame data dict
        
        Returns:
            True if frame was queued, False if queue is full or streaming not active
        """
        if self._stream_queue is None:
            return False
        
        try:
            self._stream_queue.put_nowait(frame)
            return True
        except queue.Full:
            logger.warning("Streaming queue full, frame dropped")
            return False
    
    @abstractmethod
    def _finish_streaming(self, save: bool = True) -> int:
        """
        Finish streaming and optionally save the episode.
        
        Args:
            save: If True, save the episode. If False, discard.
        
        Returns:
            Number of frames recorded, or -1 if error
        """
        pass
    
    @abstractmethod
    def finalize(self) -> None:
        """Finalize the dataset (close writers, flush buffers)."""
        pass
    
    @property
    @abstractmethod
    def dataset_path(self) -> str:
        """Return the path to the dataset."""
        pass
    
    @property
    def episode_count(self) -> int:
        """Return the number of episodes in the dataset."""
        return self._episode_count
    
    def get_feature_schema(self) -> Dict[str, dict]:
        """
        Return the feature schema of the dataset.
        
        Returns:
            Dict mapping feature keys to their properties (dtype, shape, etc.)
        """
        return self._feature_schema.copy()


# =============================================================================
# HDF5 Data Saver (with streaming support)
# =============================================================================

class HDF5DataSaver(BaseDataSaver):
    """Save episodes as individual HDF5 files with streaming support."""
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        fps: int,
        task: str = "",
        enrich_saved_frame: Optional[Callable[[Dict[str, dict]], Dict[str, dict]]] = None,
    ):
        super().__init__(output_dir, fps, task, enrich_saved_frame=enrich_saved_frame)
        if not HAS_H5PY:
            raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")
        
        # HDF5 streaming state
        self._h5_file: Optional[Any] = None  # h5py.File
        self._h5_datasets: Dict[str, Any] = {}  # key -> h5py.Dataset
        self._h5_tmp_path: Optional[Path] = None
        self._h5_timestamps: List[float] = []
        
        # Count existing episodes
        self._episode_count = self._count_existing_episodes()
    
    def _count_existing_episodes(self) -> int:
        """Count existing HDF5 episode files."""
        if not self.output_dir.exists():
            return 0
        count = sum(1 for f in self.output_dir.glob("episode_*.hdf5"))
        return count
    
    def _get_episode_path(self, episode_idx: int) -> Path:
        return self.output_dir / f"episode_{episode_idx:04d}.hdf5"
    
    def _start_streaming(
        self,
        episode_idx: int,
        device_names: List[str],
        teleop_device: Optional[str],
        first_frame: Dict[str, dict],
    ) -> bool:
        """Start streaming to HDF5 file."""
        if self._stream_thread is not None and self._stream_thread.is_alive():
            logger.warning("Streaming already in progress")
            return False
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Create temp file
            self._h5_tmp_path = self.output_dir / f"episode_{episode_idx:04d}.tmp.hdf5"
            if self._h5_tmp_path.exists():
                self._h5_tmp_path.unlink()
            
            self._stream_frame_count = 0
            self._stream_error = None
            self._h5_timestamps = []
            
            # Setup queue and stop event
            self._stream_queue = queue.Queue(maxsize=5000)
            self._stream_stop_event = threading.Event()
            
            # Start background writer thread
            self._stream_thread = threading.Thread(
                target=self._h5_streaming_writer_thread,
                args=(device_names, teleop_device, first_frame),
                daemon=True,
            )
            self._stream_thread.start()
            
            return True
            
        except Exception as e:
            logger.exception("Failed to start HDF5 streaming: {}", e)
            self._stream_error = e
            return False
    
    def _h5_streaming_writer_thread(
        self,
        device_names: List[str],
        teleop_device: Optional[str],
        first_frame: Dict[str, dict],
    ) -> None:
        """Background thread that writes frames to HDF5 file."""
        h5_file = None
        datasets: Dict[str, Any] = {}
        timestamps: List[float] = []
        CHUNK_SIZE = 25  # Resize in chunks for efficiency
        
        try:
            # Convert first frame to infer schema
            conv_first = self._convert_frame(first_frame, device_names, teleop_device)
            
            # Open HDF5 file
            h5_file = h5py.File(str(self._h5_tmp_path), "w")
            
            # Create resizable datasets based on first frame
            for key, value in conv_first.items():
                if key == "_sync_timestamp":
                    continue
                
                arr = np.asarray(value)
                shape = (0,) + arr.shape  # Start with 0 rows
                maxshape = (None,) + arr.shape  # Unlimited first dimension
                chunks = (min(CHUNK_SIZE, 100),) + arr.shape
                
                # Determine compression
                comp = "gzip" if arr.nbytes > 1000 else None
                
                ds = h5_file.create_dataset(
                    key.replace(".", "/"),
                    shape=shape,
                    maxshape=maxshape,
                    chunks=chunks,
                    dtype=arr.dtype,
                    compression=comp,
                )
                datasets[key] = ds
                
                # Track feature schema
                if key not in self._feature_schema:
                    is_image = ".images." in key or (arr.ndim == 3 and arr.shape[-1] in (1, 3, 4))
                    if is_image:
                        self._feature_schema[key] = {"dtype": "image", "shape": list(arr.shape)}
                    else:
                        self._feature_schema[key] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
            
            # Create timestamps dataset
            ts_ds = h5_file.create_dataset(
                "timestamps",
                shape=(0,),
                maxshape=(None,),
                chunks=(CHUNK_SIZE,),
                dtype=np.float64,
            )
            
            # Process frames from queue
            frame_buffer: List[Dict[str, np.ndarray]] = []
            ts_buffer: List[float] = []
            
            def flush_buffer():
                nonlocal frame_buffer, ts_buffer
                if not frame_buffer:
                    return
                
                # Resize and write each dataset
                for key, ds in datasets.items():
                    values = [f.get(key) for f in frame_buffer if key in f]
                    if values:
                        stacked = np.stack(values)
                        old_size = ds.shape[0]
                        new_size = old_size + len(values)
                        ds.resize((new_size,) + ds.shape[1:])
                        ds[old_size:new_size] = stacked
                
                # Write timestamps
                if ts_buffer:
                    old_size = ts_ds.shape[0]
                    new_size = old_size + len(ts_buffer)
                    ts_ds.resize((new_size,))
                    ts_ds[old_size:new_size] = np.array(ts_buffer, dtype=np.float64)
                
                frame_buffer = []
                ts_buffer = []
            
            while not self._stream_stop_event.is_set() or not self._stream_queue.empty():
                try:
                    frame = self._stream_queue.get(timeout=0.05)
                except queue.Empty:
                    if self._stream_stop_event.is_set() and frame_buffer:
                        flush_buffer()
                    continue
                
                # Convert frame
                conv_frame = self._convert_frame(frame, device_names, teleop_device)
                
                # Extract timestamp
                ts = conv_frame.pop("_sync_timestamp", 0.0)
                ts_buffer.append(ts)
                
                # Convert values to arrays
                frame_data = {}
                for key, value in conv_frame.items():
                    frame_data[key] = np.asarray(value)
                
                frame_buffer.append(frame_data)
                self._stream_frame_count += 1
                
                # Flush when buffer is full
                if len(frame_buffer) >= CHUNK_SIZE:
                    flush_buffer()
            
            # Final flush
            flush_buffer()
            
            # Write metadata
            h5_file.attrs["fps"] = self.fps
            h5_file.attrs["task"] = self.task
            h5_file.attrs["num_frames"] = self._stream_frame_count
            
        except Exception as e:
            self._stream_error = e
            logger.exception("HDF5 streaming writer error: {}", e)
        finally:
            if h5_file:
                h5_file.close()
    
    def _finish_streaming(self, save: bool = True) -> int:
        """Finish HDF5 streaming and optionally save."""
        if self._stream_queue is None or self._stream_thread is None:
            return -1
        
        # Signal writer thread to stop
        self._stream_stop_event.set()
        
        # Wait for writer thread to finish
        self._stream_thread.join(timeout=10.0)
        
        frame_count = self._stream_frame_count
        tmp_path = self._h5_tmp_path
        episode_idx = self._current_episode_idx
        
        # Clean up streaming state
        self._stream_queue = None
        self._stream_thread = None
        self._stream_stop_event = None
        
        if self._stream_error:
            logger.error("HDF5 streaming had errors: {}", self._stream_error)
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
            return -1
        
        if not save:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
            return frame_count
        
        if frame_count == 0:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
            return 0
        
        # Rename temp to final
        final_path = self._get_episode_path(episode_idx)
        if final_path.exists():
            final_path.unlink()
        tmp_path.rename(final_path)
        
        return frame_count
    
    def finalize(self) -> None:
        """Finalize - stop any active streaming."""
        if self.is_streaming:
            self._finish_streaming(save=False)
    
    @property
    def dataset_path(self) -> str:
        return str(self.output_dir)


# =============================================================================
# LeRobot v3.0 Data Saver
# =============================================================================

class LeRobotV30DataSaver(BaseDataSaver):
    """
    Save episodes in LeRobotDataset v3.0 format.
    Uses dtype="image" (embedded into parquet) and does NOT produce mp4 videos.
    Supports appending to existing datasets.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        fps: int,
        task: str = "",
        enrich_saved_frame: Optional[Callable[[Dict[str, dict]], Dict[str, dict]]] = None,
    ):
        super().__init__(output_dir, fps, task, enrich_saved_frame=enrich_saved_frame)
        self.dataset = None
        self.dataset_root: Optional[Path] = None
        self.repo_id: Optional[str] = None
        self._features_schema: Dict[str, dict] = {}
        self._feature_placeholders: Dict[str, np.ndarray] = {}
        
        # Count existing episodes on init (before dataset is loaded)
        self._episode_count = self._count_existing_episodes()
    
    def _ensure_lerobot_imported(self):
        # Set offline mode to prevent HuggingFace Hub access
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        project_root = Path(__file__).resolve().parent.parent
        lerobot_src = project_root / "third_party" / "lerobot" / "src"
        if str(lerobot_src) not in sys.path:
            sys.path.insert(0, str(lerobot_src))
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    
    def _is_valid_lerobot_dataset(self, path: Path) -> bool:
        """Check if path contains a valid LeRobot v3.0 dataset."""
        meta_dir = path / "meta"
        # Must have info.json and tasks.parquet at minimum
        return (
            (meta_dir / "info.json").exists() and
            (meta_dir / "tasks.parquet").exists()
        )
    
    def _count_existing_episodes(self) -> int:
        """Count existing episodes from info.json without loading the full dataset."""
        # Check if output_dir is already a valid dataset
        dataset_root = self._choose_dataset_root()
        if not self._is_valid_lerobot_dataset(dataset_root):
            return 0
        
        info_path = dataset_root / "meta" / "info.json"
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
            return info.get("total_episodes", 0)
        except Exception:
            return 0
    
    def _choose_dataset_root(self) -> Path:
        # If output_dir already looks like a valid lerobot dataset, use it directly.
        if self._is_valid_lerobot_dataset(self.output_dir):
            return self.output_dir
        # If output_dir doesn't exist, create dataset there.
        if not self.output_dir.exists():
            return self.output_dir
        # Check if output_dir is empty or only contains non-conflicting files
        if self.output_dir.is_dir():
            contents = list(self.output_dir.iterdir())
            # If empty, use it
            if not contents:
                return self.output_dir
            # If it has meta/ but is not a valid dataset, it's corrupted - use subdir
            if (self.output_dir / "meta").exists():
                return self.output_dir / "lerobotv30"
        # Otherwise, avoid mixing with existing files (e.g., .hdf5) by creating a subdir.
        return self.output_dir / "lerobotv30"
    
    def _derive_repo_id(self, dataset_root: Path) -> str:
        project_root = Path(__file__).resolve().parent.parent
        try:
            rel = dataset_root.resolve().relative_to(project_root.resolve())
            parts = rel.parts
            if len(parts) >= 2 and parts[0] == "data":
                return str(Path(*parts[1:]))
            return str(rel)
        except Exception:
            return dataset_root.name
    
    def _infer_features_from_frame(self, converted_frame: Dict[str, Any]) -> Dict[str, dict]:
        """Infer LeRobot features schema from a converted frame."""
        features: Dict[str, dict] = {}
        
        for key, value in converted_frame.items():
            if key == "_sync_timestamp":
                continue
            
            if isinstance(value, np.ndarray):
                # Image data: observation.images.xxx or observation.teleop.images.xxx
                is_image_key = ".images." in key
                is_3d_image = value.ndim == 3 and value.shape[-1] in (1, 3, 4)
                
                if is_image_key or is_3d_image:
                    if value.ndim == 3:
                        h, w, c = value.shape
                        features[key] = {
                            "dtype": "image",
                            "shape": (h, w, c),
                            "names": ["height", "width", "channels"],
                        }
                        self._feature_placeholders[key] = np.zeros((h, w, c), dtype=np.uint8)
                elif value.ndim <= 1:
                    # Vector data (state, action, etc.)
                    dim = value.size
                    features[key] = {
                        "dtype": "float32",
                        "shape": (dim,),
                        "names": [f"{key}_{i}" for i in range(dim)],
                    }
                    self._feature_placeholders[key] = np.zeros((dim,), dtype=np.float32)
        
        return features
    
    # LeRobot's default features that are auto-managed (don't add to frame manually)
    _DEFAULT_FEATURE_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    
    def _load_schema_from_existing_dataset(self) -> None:
        """Load feature schema from existing dataset."""
        if self.dataset is None:
            return
        
        self._features_schema = {}
        self._feature_placeholders = {}
        
        for key, ft in getattr(self.dataset, "features", {}).items():
            # Skip default features that are auto-managed by LeRobot
            if key in self._DEFAULT_FEATURE_KEYS:
                continue
            
            self._features_schema[key] = ft
            dtype = ft.get("dtype", "float32")
            shape = tuple(ft.get("shape", ()))
            
            if dtype in ("image", "video"):
                if len(shape) == 3:
                    self._feature_placeholders[key] = np.zeros(shape, dtype=np.uint8)
            elif dtype == "float32" and len(shape) == 1:
                self._feature_placeholders[key] = np.zeros(shape, dtype=np.float32)
    
    def _ensure_dataset_initialized(
        self,
        converted_frames: List[Dict[str, Any]],
    ) -> None:
        if self.dataset is not None:
            return
        
        self._ensure_lerobot_imported()
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        self.dataset_root = self._choose_dataset_root()
        self.repo_id = self._derive_repo_id(self.dataset_root)
        
        # Load existing or create new
        if self._is_valid_lerobot_dataset(self.dataset_root):
            logger.info("Loading existing LeRobot v3.0 dataset at: {}", str(self.dataset_root))
            self.dataset = LeRobotDataset(repo_id=self.repo_id, root=str(self.dataset_root))
            self._load_schema_from_existing_dataset()
            self._feature_schema = self._features_schema.copy()  # Expose for task config generation
            self._episode_count = self.dataset.num_episodes
            return
        
        # Infer features from first frame
        if not converted_frames:
            raise ValueError("Cannot create dataset without frames")
        
        # Debug: log the converted frame keys
        logger.debug("First converted frame keys: {}", list(converted_frames[0].keys()))
        
        self._features_schema = self._infer_features_from_frame(converted_frames[0])
        self._feature_schema = self._features_schema.copy()  # Expose for task config generation
        
        logger.info("Creating LeRobot v3.0 dataset at: {}", str(self.dataset_root))
        logger.info("Inferred features ({} total): {}", len(self._features_schema), list(self._features_schema.keys()))
        
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            robot_type=None,
            features=self._features_schema,
            root=str(self.dataset_root),
            use_videos=False,
            image_writer_processes=0,
            image_writer_threads=4,
            batch_encoding_size=1,
        )
    
    # =========================================================================
    # Streaming Implementation
    # =========================================================================
    
    def _start_streaming(
        self,
        episode_idx: int,
        device_names: List[str],
        teleop_device: Optional[str],
        first_frame: Dict[str, dict],
    ) -> bool:
        """Start streaming to LeRobot dataset."""
        if self._stream_thread is not None and self._stream_thread.is_alive():
            logger.warning("Streaming already in progress")
            return False
        
        try:
            # Convert first frame to infer schema
            conv_first = self._convert_frame(first_frame, device_names, teleop_device)
            
            # Initialize dataset if needed
            self._ensure_dataset_initialized([conv_first])
            
            self._stream_frame_count = 0
            self._stream_error = None
            
            # Setup queue and stop event
            self._stream_queue = queue.Queue(maxsize=5000)
            self._stream_stop_event = threading.Event()
            
            # Start background writer thread
            self._stream_thread = threading.Thread(
                target=self._v30_streaming_writer_thread,
                args=(device_names, teleop_device),
                daemon=True,
            )
            self._stream_thread.start()
            
            return True
            
        except Exception as e:
            logger.exception("Failed to start LeRobot v3.0 streaming: {}", e)
            self._stream_error = e
            return False
    
    def _v30_streaming_writer_thread(
        self,
        device_names: List[str],
        teleop_device: Optional[str],
    ) -> None:
        """Background thread that writes frames to LeRobot dataset."""
        try:
            required_keys = set(self._features_schema.keys())
            
            while not self._stream_stop_event.is_set() or not self._stream_queue.empty():
                try:
                    raw_frame = self._stream_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                
                # Convert frame
                conv_fr = self._convert_frame(raw_frame, device_names, teleop_device)
                
                # Build frame for LeRobot
                frame: Dict[str, Any] = {"task": self.task}
                
                for key in required_keys:
                    if key in conv_fr:
                        value = conv_fr[key]
                        if self._features_schema[key].get("dtype") == "float32":
                            if not isinstance(value, np.ndarray) or value.dtype != np.float32:
                                value = np.asarray(value, dtype=np.float32).reshape(-1)
                        frame[key] = value
                    elif key in self._feature_placeholders:
                        frame[key] = self._feature_placeholders[key]
                
                self.dataset.add_frame(frame)
                self._stream_frame_count += 1
                
        except Exception as e:
            self._stream_error = e
            logger.exception("LeRobot v3.0 streaming writer error: {}", e)
    
    def _finish_streaming(self, save: bool = True) -> int:
        """Finish LeRobot v3.0 streaming and optionally save."""
        if self._stream_queue is None or self._stream_thread is None:
            return -1
        
        # Signal writer thread to stop
        self._stream_stop_event.set()
        
        # Wait for writer thread to finish
        self._stream_thread.join(timeout=10.0)
        
        frame_count = self._stream_frame_count
        
        # Clean up streaming state
        self._stream_queue = None
        self._stream_thread = None
        self._stream_stop_event = None
        
        if self._stream_error:
            logger.error("LeRobot v3.0 streaming had errors: {}", self._stream_error)
            # Try to clear episode buffer
            if self.dataset is not None:
                try:
                    self.dataset.clear_episode_buffer()
                except Exception:
                    pass
            return -1
        
        if not save or frame_count == 0:
            # Discard: clear episode buffer
            if self.dataset is not None:
                try:
                    self.dataset.clear_episode_buffer()
                except Exception:
                    pass
            return frame_count if not save else 0
        
        # Save episode
        try:
            self.dataset.save_episode()
            return frame_count
        except Exception as e:
            logger.exception("Failed to save LeRobot v3.0 episode: {}", e)
            return -1
    
    def finalize(self) -> None:
        """Finalize - stop any active streaming and close dataset."""
        if self.is_streaming:
            self._finish_streaming(save=False)
        if self.dataset is not None:
            try:
                self.dataset.finalize()
            except Exception as e:
                logger.warning("LeRobot dataset finalize failed: {}", e)
    
    @property
    def dataset_path(self) -> str:
        if self.dataset_root is not None:
            return str(self.dataset_root)
        return str(self.output_dir)


# =============================================================================
# LeRobot v2.1 Data Saver (supports overwriting episodes)
# =============================================================================

class LeRobotV21DataSaver(BaseDataSaver):
    """
    Save episodes in LeRobot v2.1 format using HuggingFace datasets library.
    Each episode is stored as a separate parquet file, making it easy to overwrite.
    
    Structure:
        data/chunk-000/episode_XXXXXX.parquet
        meta/info.json
        meta/episodes.jsonl
        meta/tasks.jsonl
        meta/stats.json (aggregated statistics)
    
    Statistics are computed per-episode and stored in episodes.jsonl with running stats
    (sum, sum_of_squares, min, max, count) to support incremental updates when episodes
    are overwritten.
    
    Reference: lerobot-0.3.3/src/lerobot/datasets/lerobot_dataset.py
    """
    
    CODEBASE_VERSION = "v2.1"
    
    # Default features that are auto-managed (same as lerobot)
    DEFAULT_FEATURES = {
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    
    # Features to skip when computing statistics (metadata fields)
    SKIP_STATS_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        fps: int,
        task: str = "",
        enrich_saved_frame: Optional[Callable[[Dict[str, dict]], Dict[str, dict]]] = None,
    ):
        super().__init__(output_dir, fps, task, enrich_saved_frame=enrich_saved_frame)
        self._features_schema: Dict[str, dict] = {}
        self._info: Optional[Dict[str, Any]] = None
        self._episodes: Dict[int, Dict[str, Any]] = {}  # episode_index -> episode_dict
        self._tasks: Dict[int, str] = {}  # task_index -> task string
        self._task_to_idx: Dict[str, int] = {}  # task string -> task_index
        self._initialized = False
        self._total_frames = 0
        
        # Statistics state
        self._aggregated_stats: Optional[Dict[str, Dict[str, Any]]] = None  # Aggregated stats from all episodes
        
        # LeRobotV21 specific streaming state
        self._stream_episode_idx: Optional[int] = None
        self._stream_tmp_path: Optional[Path] = None
        
        # Count existing episodes on init
        self._episode_count = self._count_episodes_from_files()
    
    # =========================================================================
    # Statistics Computation Methods
    # =========================================================================
    
    def _compute_episode_stats(
        self,
        converted_frames: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistics for a single episode.
        
        Returns a dict mapping feature keys to their running stats:
        {
            "feature_key": {
                "min": np.ndarray,
                "max": np.ndarray,
                "mean": np.ndarray,
                "std": np.ndarray,
                "sum": np.ndarray,        # For running stats aggregation
                "sum_of_squares": np.ndarray,  # For variance calculation
                "count": int,
            },
            ...
        }
        """
        if not converted_frames:
            return {}
        
        # Collect data by feature key
        feature_data: Dict[str, List[np.ndarray]] = {}
        
        for conv_fr in converted_frames:
            for key, value in conv_fr.items():
                if key == "_sync_timestamp":
                    continue
                if key in self.SKIP_STATS_FEATURES:
                    continue
                # Skip image features (they have dtype "image" in schema)
                if key in self._features_schema:
                    if self._features_schema[key].get("dtype") == "image":
                        continue
                
                if key not in feature_data:
                    feature_data[key] = []
                
                if isinstance(value, np.ndarray):
                    feature_data[key].append(value.flatten().astype(np.float64))
                elif isinstance(value, (list, tuple)):
                    feature_data[key].append(np.array(value, dtype=np.float64).flatten())
                elif isinstance(value, (int, float)):
                    feature_data[key].append(np.array([value], dtype=np.float64))
        
        # Compute statistics for each feature
        episode_stats: Dict[str, Dict[str, Any]] = {}
        
        for key, values in feature_data.items():
            if not values:
                continue
            
            # Stack all values: shape (num_frames, feature_dim)
            try:
                stacked = np.stack(values)
            except ValueError:
                # Skip if shapes are inconsistent
                continue
            
            count = stacked.shape[0]
            if count == 0:
                continue
            
            # Compute running stats
            sum_vals = np.sum(stacked, axis=0)
            sum_of_squares = np.sum(stacked ** 2, axis=0)
            min_vals = np.min(stacked, axis=0)
            max_vals = np.max(stacked, axis=0)
            mean_vals = sum_vals / count
            
            # Compute variance using E[X^2] - E[X]^2
            variance = (sum_of_squares / count) - (mean_vals ** 2)
            variance = np.maximum(variance, 0)  # Ensure non-negative due to numerical errors
            std_vals = np.sqrt(variance)
            
            episode_stats[key] = {
                "min": min_vals.tolist(),
                "max": max_vals.tolist(),
                "mean": mean_vals.tolist(),
                "std": std_vals.tolist(),
                "sum": sum_vals.tolist(),
                "sum_of_squares": sum_of_squares.tolist(),
                "count": count,
            }
        
        return episode_stats
    
    def _aggregate_episode_stats(
        self,
        episode_stats_list: List[Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate statistics from multiple episodes using running stats.
        
        Uses the parallel variance algorithm:
        - Combined mean = (sum1 + sum2 + ...) / (count1 + count2 + ...)
        - Combined variance = (ss1 + ss2 + ...) / total_count - combined_mean^2
        """
        if not episode_stats_list:
            return {}
        
        # Collect all feature keys
        all_keys: set = set()
        for ep_stats in episode_stats_list:
            all_keys.update(ep_stats.keys())
        
        aggregated: Dict[str, Dict[str, Any]] = {}
        
        for key in all_keys:
            # Gather stats from all episodes that have this key
            key_stats = [ep_stats[key] for ep_stats in episode_stats_list if key in ep_stats]
            if not key_stats:
                continue
            
            # Aggregate using running stats
            total_count = sum(s["count"] for s in key_stats)
            if total_count == 0:
                continue
            
            # Stack and sum
            total_sum = np.sum([np.array(s["sum"]) for s in key_stats], axis=0)
            total_sum_of_squares = np.sum([np.array(s["sum_of_squares"]) for s in key_stats], axis=0)
            
            # Min/max across all episodes
            all_mins = np.stack([np.array(s["min"]) for s in key_stats])
            all_maxs = np.stack([np.array(s["max"]) for s in key_stats])
            
            combined_min = np.min(all_mins, axis=0)
            combined_max = np.max(all_maxs, axis=0)
            combined_mean = total_sum / total_count
            
            # Variance using E[X^2] - E[X]^2
            combined_variance = (total_sum_of_squares / total_count) - (combined_mean ** 2)
            combined_variance = np.maximum(combined_variance, 0)
            combined_std = np.sqrt(combined_variance)
            
            aggregated[key] = {
                "min": combined_min.tolist(),
                "max": combined_max.tolist(),
                "mean": combined_mean.tolist(),
                "std": combined_std.tolist(),
                "count": total_count,
            }
        
        return aggregated
    
    def _load_aggregated_stats(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Load aggregated stats from meta/stats.json if exists."""
        stats_path = self.output_dir / "meta" / "stats.json"
        if not stats_path.exists():
            return None
        try:
            with open(stats_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load stats.json: {}", e)
            return None
    
    def _save_aggregated_stats(self, stats: Dict[str, Dict[str, Any]]) -> None:
        """Save aggregated stats to meta/stats.json."""
        stats_path = self.output_dir / "meta" / "stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
    
    def _update_aggregated_stats_after_episode(
        self,
        new_episode_stats: Dict[str, Dict[str, Any]],
        old_episode_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Update aggregated stats after saving an episode.
        
        If old_episode_stats is provided (episode overwrite case), subtract old stats
        and add new stats. Otherwise, just add new stats.
        """
        # Load current aggregated stats if not cached
        if self._aggregated_stats is None:
            self._aggregated_stats = self._load_aggregated_stats() or {}
        
        # Get all episode stats from episodes.jsonl for re-aggregation
        # This is more accurate than incremental update, especially for overwrites
        all_episode_stats = []
        for ep_idx, ep_meta in self._episodes.items():
            ep_stats = ep_meta.get("stats")
            if ep_stats:
                all_episode_stats.append(ep_stats)
        
        # Re-aggregate from all episodes
        self._aggregated_stats = self._aggregate_episode_stats(all_episode_stats)
        
        # Save to disk
        self._save_aggregated_stats(self._aggregated_stats)
    
    def _compute_stats_from_parquet(
        self,
        parquet_path: Path,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistics by reading data from a parquet file.
        Used for streaming mode where we don't have all frames in memory.
        """
        import pyarrow.parquet as pq
        
        try:
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
        except Exception as e:
            logger.warning("Failed to read parquet for stats: {}", e)
            return {}
        
        episode_stats: Dict[str, Dict[str, Any]] = {}
        
        for col_name in df.columns:
            # Skip metadata columns and image columns
            if col_name in self.SKIP_STATS_FEATURES:
                continue
            if col_name in self._features_schema:
                if self._features_schema[col_name].get("dtype") == "image":
                    continue
            
            try:
                col_data = df[col_name]
                
                # Handle list/array columns
                if col_data.dtype == object:
                    # Check if it's a list column
                    first_val = col_data.iloc[0] if len(col_data) > 0 else None
                    if isinstance(first_val, (list, np.ndarray)):
                        # Stack into numpy array
                        stacked = np.stack([np.array(x, dtype=np.float64) for x in col_data])
                    elif isinstance(first_val, dict):
                        # Skip dict columns (e.g., image bytes)
                        continue
                    else:
                        continue
                else:
                    # Scalar column
                    stacked = col_data.to_numpy(dtype=np.float64).reshape(-1, 1)
                
                count = stacked.shape[0]
                if count == 0:
                    continue
                
                # Compute running stats
                sum_vals = np.sum(stacked, axis=0)
                sum_of_squares = np.sum(stacked ** 2, axis=0)
                min_vals = np.min(stacked, axis=0)
                max_vals = np.max(stacked, axis=0)
                mean_vals = sum_vals / count
                
                # Compute variance using E[X^2] - E[X]^2
                variance = (sum_of_squares / count) - (mean_vals ** 2)
                variance = np.maximum(variance, 0)
                std_vals = np.sqrt(variance)
                
                episode_stats[col_name] = {
                    "min": min_vals.tolist(),
                    "max": max_vals.tolist(),
                    "mean": mean_vals.tolist(),
                    "std": std_vals.tolist(),
                    "sum": sum_vals.tolist(),
                    "sum_of_squares": sum_of_squares.tolist(),
                    "count": count,
                }
            except Exception as e:
                logger.debug("Skipping stats for column {}: {}", col_name, e)
                continue
        
        return episode_stats
    
    def _get_data_dir(self) -> Path:
        return self.output_dir / "data" / "chunk-000"
    
    def _get_episode_path(self, episode_idx: int) -> Path:
        return self._get_data_dir() / f"episode_{episode_idx:06d}.parquet"
    
    def _load_schema_only(self) -> bool:
        """
        Load only info.json for schema. Returns True if exists.
        This is the fastest way to get dataset schema without loading episode metadata.
        """
        import json
        
        info_path = self.output_dir / "meta" / "info.json"
        if not info_path.exists():
            return False
        
        with open(info_path) as f:
            self._info = json.load(f)
        
        # Also get total_frames for global index calculation
        self._total_frames = self._info.get("total_frames", 0)
        return True
    
    def _count_episodes_from_files(self) -> int:
        """
        Get episode count from parquet files in data directory.
        Returns max_episode_index + 1, which is used to determine the next episode index.
        Searches all chunk-* directories (chunk-000, chunk-001, etc.).
        """
        import re
        data_root = self.output_dir / "data"
        if not data_root.exists():
            return 0
        
        # Find max episode index from all chunk-* directories
        max_idx = -1
        episode_pattern = re.compile(r"episode_(\d+)\.parquet$")
        for chunk_dir in data_root.iterdir():
            if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
                for f in chunk_dir.iterdir():
                    if f.is_file():
                        match = episode_pattern.match(f.name)
                        if match:
                            idx = int(match.group(1))
                            if idx > max_idx:
                                max_idx = idx
        
        return max_idx + 1 if max_idx >= 0 else 0
    
    def _infer_features_from_frame(self, converted_frame: Dict[str, Any]) -> Dict[str, dict]:
        """Infer feature schema from a converted frame (lerobot format)."""
        features: Dict[str, dict] = {}
        
        for key, value in converted_frame.items():
            if key == "_sync_timestamp":
                continue
            
            if isinstance(value, np.ndarray):
                is_image_key = ".images." in key
                is_3d_image = value.ndim == 3 and value.shape[-1] in (1, 3, 4)
                
                if is_image_key or is_3d_image:
                    if value.ndim == 3:
                        h, w, c = value.shape
                        # lerobot uses "image" dtype, shape is original HWC
                        features[key] = {
                            "dtype": "image",
                            "shape": (h, w, c),
                            "names": ["height", "width", "channels"],
                        }
                elif value.ndim <= 1:
                    dim = value.size
                    # Vector features use tuple shape
                    features[key] = {
                        "dtype": "float32",
                        "shape": (dim,),
                        "names": None,
                    }
        
        # Add default features (from lerobot)
        features.update(self.DEFAULT_FEATURES)
        
        return features
    
    def _ensure_initialized(self, converted_frames: List[Dict[str, Any]], episode_idx: Optional[int] = None) -> None:
        """
        Initialize dataset structure and metadata.
        
        Args:
            converted_frames: Frames to infer schema from (for new datasets)
            episode_idx: If provided and dataset exists, use fast mode (only load schema)
        """
        if self._initialized:
            return
        
        # Create directories
        (self.output_dir / "meta").mkdir(parents=True, exist_ok=True)
        self._get_data_dir().mkdir(parents=True, exist_ok=True)
        
        # Fast: only load schema from info.json (no episode metadata loading)
        has_existing = self._load_schema_only()
        
        if self._info is None:
            # Infer features from first frame
            if not converted_frames:
                raise ValueError("Cannot create dataset without frames")
            
            self._features_schema = self._infer_features_from_frame(converted_frames[0])
            
            self._info = {
                "codebase_version": self.CODEBASE_VERSION,
                "robot_type": None,
                "total_episodes": 0,
                "total_frames": 0,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": self.fps,
                "features": self._features_schema,
                "splits": {"train": "0:0"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": None,
            }
            
            # Add default task
            task_str = self.task if self.task else "default"
            self._tasks = {0: task_str}
            self._task_to_idx = {task_str: 0}
            
            logger.info("Creating LeRobot v2.1 dataset at: {}", str(self.output_dir))
            logger.debug("Features: {}", list(self._features_schema.keys()))
        else:
            # Existing dataset: only load schema, count episodes from files
            self._features_schema = self._info.get("features", {})
            self._episode_count = self._count_episodes_from_files()
            # Don't load episode metadata - will be handled in _save_metadata
        
        # Expose feature schema for task config generation (exclude default LeRobot features)
        self._feature_schema = {k: v for k, v in self._features_schema.items() 
                               if k not in self.DEFAULT_FEATURES}
        
        self._initialized = True
    
    def _get_task_index(self, task: str) -> int:
        """Get or create task index for a task string."""
        if task not in self._task_to_idx:
            new_idx = len(self._tasks)
            self._tasks[new_idx] = task
            self._task_to_idx[task] = new_idx
        return self._task_to_idx[task]
    
    def _save_metadata(self, updated_episode: Optional[Dict[str, Any]] = None) -> None:
        """
        Save metadata files. Optimized for incremental updates.
        
        Args:
            updated_episode: If provided, only update this episode in episodes.jsonl
        """
        import json
        
        episodes_path = self.output_dir / "meta" / "episodes.jsonl"
        tasks_path = self.output_dir / "meta" / "tasks.jsonl"
        
        # If we have a single episode update and episodes weren't fully loaded,
        # read existing episodes, merge, and write back.
        # Check: if _episodes only contains the updated_episode (or is empty), we need to load from file
        need_load_episodes = (
            updated_episode is not None and 
            (not self._episodes or 
             (len(self._episodes) == 1 and updated_episode["episode_index"] in self._episodes))
        )
        
        if need_load_episodes:
            # Read existing episodes from file
            existing_episodes = {}
            if episodes_path.exists():
                with open(episodes_path) as f:
                    for line in f:
                        if line.strip():
                            ep = json.loads(line)
                            existing_episodes[ep["episode_index"]] = ep
            
            # Update with new episode
            existing_episodes[updated_episode["episode_index"]] = updated_episode
            self._episodes = existing_episodes
            
            # Recalculate total frames
            self._total_frames = sum(ep.get("length", 0) for ep in existing_episodes.values())
        
        # If tasks weren't loaded, read them
        if not self._tasks:
            if tasks_path.exists():
                with open(tasks_path) as f:
                    for line in f:
                        if line.strip():
                            t = json.loads(line)
                            self._tasks[t["task_index"]] = t["task"]
                            self._task_to_idx[t["task"]] = t["task_index"]
            else:
                # Default task
                task_str = self.task if self.task else "default"
                self._tasks = {0: task_str}
                self._task_to_idx = {task_str: 0}
        
        # Update info
        # total_episodes should be max_episode_index + 1 (not count), because
        # the data loader uses range(total_episodes) to enumerate episode indices
        if self._episodes:
            max_episode_idx = max(self._episodes.keys())
            self._info["total_episodes"] = max_episode_idx + 1
        else:
            self._info["total_episodes"] = 0
        self._info["total_frames"] = self._total_frames
        self._info["splits"] = {"train": f"0:{self._info['total_episodes']}"}
        
        # Save info.json
        with open(self.output_dir / "meta" / "info.json", "w") as f:
            json.dump(self._info, f, indent=2)
        
        # Save episodes.jsonl (sorted by episode_index)
        with open(episodes_path, "w") as f:
            for idx in sorted(self._episodes.keys()):
                f.write(json.dumps(self._episodes[idx]) + "\n")
        
        # Save tasks.jsonl (sorted by task_index)
        with open(tasks_path, "w") as f:
            for idx in sorted(self._tasks.keys()):
                f.write(json.dumps({"task_index": idx, "task": self._tasks[idx]}) + "\n")
    
    def write_episode(
        self,
        frames: List[Dict[str, dict]],
        device_names: List[str],
        teleop_device: Optional[str],
        episode_idx: Optional[int] = None,
    ) -> bool:
        """
        Write episode using PyArrow directly for maximum speed.
        Avoids HuggingFace datasets overhead.
        """
        if not frames:
            return False
        
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            import cv2
            from concurrent.futures import ThreadPoolExecutor
            
            # Convert all frames
            converted_frames = [
                self._convert_frame(fr, device_names, teleop_device)
                for fr in frames
            ]
            
            # Initialize if needed (pass episode_idx for fast mode detection)
            self._ensure_initialized(converted_frames, episode_idx)
            
            # Determine episode index (use provided or append)
            actual_episode_idx = episode_idx if episode_idx is not None else self._episode_count
            
            # Check if this is an overwrite operation
            episode_path = self._get_episode_path(actual_episode_idx)
            is_overwrite = episode_path.exists()
            episode_length = len(frames)
            
            # For overwrite: delete old file first (fast operation)
            if is_overwrite:
                episode_path.unlink()
            
            # Get task index
            task_str = self.task if self.task else "default"
            task_index = self._get_task_index(task_str)
            
            # Calculate global index offset
            # For overwrite without full metadata, use simple estimate
            if is_overwrite and actual_episode_idx not in self._episodes:
                # Fast mode: estimate offset based on episode index and fps
                global_idx_offset = actual_episode_idx * 500  # Rough estimate
            elif is_overwrite:
                global_idx_offset = sum(
                    self._episodes[i].get("length", 0)
                    for i in sorted(self._episodes.keys())
                    if i < actual_episode_idx
                )
            else:
                global_idx_offset = self._total_frames
            
            # Use actual_episode_idx from here
            episode_idx = actual_episode_idx
            
            # Helper function for fast image encoding (cv2 is much faster than PIL)
            def encode_image_fast(img_array: np.ndarray) -> bytes:
                """Encode image to PNG bytes using cv2 (faster than PIL)."""
                if img_array is None:
                    return b""
                # cv2 expects BGR, but our images are RGB
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                success, encoded = cv2.imencode(".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])  # Fast compression
                return encoded.tobytes() if success else b""
            
            # Build data columns directly as lists
            columns: Dict[str, list] = {key: [] for key in self._features_schema}
            
            # Identify image columns for parallel encoding
            image_keys = [k for k, ft in self._features_schema.items() 
                         if ft.get("dtype") == "image" and k not in self.DEFAULT_FEATURES]
            
            # Pre-encode all images in parallel for speed
            image_data: Dict[str, List[bytes]] = {k: [] for k in image_keys}
            if image_keys:
                def encode_frame_images(conv_fr):
                    """Encode all images for one frame."""
                    result = {}
                    for key in image_keys:
                        value = conv_fr.get(key)
                        if value is not None and isinstance(value, np.ndarray):
                            result[key] = encode_image_fast(value)
                        else:
                            result[key] = b""
                    return result
                
                # Use thread pool for parallel encoding
                with ThreadPoolExecutor(max_workers=4) as executor:
                    encoded_frames = list(executor.map(encode_frame_images, converted_frames))
                
                # Organize by key
                for frame_encoded in encoded_frames:
                    for key in image_keys:
                        image_data[key].append(frame_encoded.get(key, b""))
            
            # Process each frame
            for frame_idx, conv_fr in enumerate(converted_frames):
                # Auto-managed features
                columns["timestamp"].append(float(frame_idx) / self.fps)
                columns["frame_index"].append(frame_idx)
                columns["episode_index"].append(episode_idx)
                columns["index"].append(global_idx_offset + frame_idx)
                columns["task_index"].append(task_index)
                
                # User features
                for key, ft in self._features_schema.items():
                    if key in self.DEFAULT_FEATURES:
                        continue
                    
                    dtype = ft.get("dtype", "float32")
                    shape = ft.get("shape", (1,))
                    if isinstance(shape, list):
                        shape = tuple(shape)
                    
                    if dtype == "image":
                        # Use pre-encoded image data
                        img_bytes = image_data[key][frame_idx]
                        columns[key].append({"bytes": img_bytes, "path": None})
                    else:
                        value = conv_fr.get(key)
                        if value is not None:
                            if isinstance(value, np.ndarray):
                                if shape == (1,):
                                    columns[key].append(float(value.flat[0]))
                                else:
                                    columns[key].append(value.astype(np.float32).tolist())
                            else:
                                columns[key].append(value)
                        else:
                            if shape == (1,):
                                columns[key].append(0.0)
                            else:
                                columns[key].append([0.0] * shape[0])
            
            # Build PyArrow arrays with correct types
            pa_arrays = {}
            for key, values in columns.items():
                ft = self._features_schema.get(key, {})
                dtype = ft.get("dtype", "float32")
                shape = ft.get("shape", (1,))
                if isinstance(shape, list):
                    shape = tuple(shape)
                
                if dtype == "image":
                    # Image as struct {bytes, path}
                    pa_arrays[key] = pa.array(values, type=pa.struct([
                        ("bytes", pa.binary()),
                        ("path", pa.string())
                    ]))
                elif shape == (1,):
                    # Scalar
                    if dtype == "int64":
                        pa_arrays[key] = pa.array(values, type=pa.int64())
                    else:
                        pa_arrays[key] = pa.array(values, type=pa.float32())
                else:
                    # 1D sequence
                    if dtype == "int64":
                        pa_arrays[key] = pa.array(values, type=pa.list_(pa.int64()))
                    else:
                        pa_arrays[key] = pa.array(values, type=pa.list_(pa.float32()))
            
            # Create and write table
            table = pa.table(pa_arrays)
            pq.write_table(table, episode_path)
            
            # Compute episode statistics
            episode_stats = self._compute_episode_stats(converted_frames)
            
            # Get old episode stats if overwriting (for proper stats update)
            old_episode_stats = None
            if is_overwrite and episode_idx in self._episodes:
                old_episode_stats = self._episodes[episode_idx].get("stats")
            
            # Create episode metadata with stats
            ep_meta = {
                "episode_index": episode_idx,
                "tasks": [task_str],
                "length": episode_length,
                "stats": episode_stats,  # Include per-episode statistics
            }
            
            # Update in-memory metadata (if available)
            if episode_idx in self._episodes:
                old_length = self._episodes[episode_idx].get("length", 0)
                self._total_frames = self._total_frames - old_length + episode_length
                self._episodes[episode_idx] = ep_meta
            elif not is_overwrite:
                # New episode (append mode)
                self._total_frames += episode_length
                self._episodes[episode_idx] = ep_meta
                self._episode_count = len(self._episodes)
            # else: overwrite without full metadata loaded, _save_metadata will handle it
            
            if is_overwrite:
                logger.info("Overwrote episode {} ({} frames)", episode_idx, episode_length)
            
            # Save metadata (pass ep_meta for incremental update)
            self._save_metadata(updated_episode=ep_meta)
            
            # Update aggregated statistics
            self._update_aggregated_stats_after_episode(episode_stats, old_episode_stats)
            
            return True
        except Exception as e:
            logger.exception("Failed to write LeRobot v2.1 episode: {}", e)
            return False
    
    # =========================================================================
    # Streaming Implementation
    # =========================================================================
    
    def _start_streaming(
        self,
        episode_idx: int,
        device_names: List[str],
        teleop_device: Optional[str],
        first_frame: Dict[str, dict],
    ) -> bool:
        """
        Start streaming recording to a temporary file.
        This initializes the schema and starts the background writer thread.
        
        Args:
            episode_idx: Episode index (None for append, int for specific/overwrite)
            device_names: List of device names
            teleop_device: Name of teleop device (if any)
            first_frame: First frame data (used for schema inference)
        
        Returns:
            True if streaming started successfully
        """
        if self._stream_thread is not None and self._stream_thread.is_alive():
            logger.warning("Streaming already in progress")
            return False
        
        try:
            # Convert first frame to get schema
            converted_first = self._convert_frame(first_frame, device_names, teleop_device)
            
            # Initialize dataset if needed
            self._ensure_initialized([converted_first], episode_idx)
            
            # Determine episode index
            if episode_idx is None:
                episode_idx = self._episode_count
            
            self._stream_episode_idx = episode_idx
            self._stream_frame_count = 0
            self._stream_error = None
            
            # Create temp file path
            self._stream_tmp_path = self._get_data_dir() / f"episode_{episode_idx:06d}.tmp"
            
            # Delete existing temp file if any
            if self._stream_tmp_path.exists():
                self._stream_tmp_path.unlink()
            
            # Setup queue and stop event
            # Large buffer to handle slow disk I/O (e.g., 25fps * 120s = 3000 frames)
            self._stream_queue = queue.Queue(maxsize=5000)
            self._stream_stop_event = threading.Event()
            
            # Start background writer thread
            self._stream_thread = threading.Thread(
                target=self._streaming_writer_thread,
                args=(device_names, teleop_device),
                daemon=True,
            )
            self._stream_thread.start()
            
            return True
            
        except Exception as e:
            logger.exception("Failed to start streaming: {}", e)
            self._stream_error = e
            return False
    
    def _finish_streaming(self, save: bool = True) -> int:
        """
        Finish streaming and optionally save the episode.
        
        Args:
            save: If True, rename temp to final file and update metadata.
                  If False, delete the temp file.
        
        Returns:
            Number of frames recorded, or -1 if error
        """
        if self._stream_queue is None or self._stream_thread is None:
            return -1
        
        # Signal writer thread to stop
        self._stream_stop_event.set()
        
        # Wait for writer thread to finish (with timeout)
        self._stream_thread.join(timeout=5.0)
        
        frame_count = self._stream_frame_count
        episode_idx = self._stream_episode_idx
        tmp_path = self._stream_tmp_path
        
        # Clean up streaming state
        self._stream_queue = None
        self._stream_thread = None
        self._stream_stop_event = None
        
        if self._stream_error:
            logger.error("Streaming had errors: {}", self._stream_error)
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
            return -1
        
        if not save:
            # Discard: delete temp file
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
            return frame_count
        
        if frame_count == 0:
            # No frames recorded
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
            return 0
        
        # Save: rename temp to final
        final_path = self._get_episode_path(episode_idx)
        
        # Check if overwrite and get old stats
        is_overwrite = final_path.exists()
        old_episode_stats = None
        if is_overwrite and episode_idx in self._episodes:
            old_episode_stats = self._episodes[episode_idx].get("stats")
        
        # Delete existing file if overwriting
        if final_path.exists():
            final_path.unlink()
        
        # Rename temp to final
        tmp_path.rename(final_path)
        
        # Compute episode statistics from the saved parquet file
        episode_stats = self._compute_stats_from_parquet(final_path)
        
        # Update metadata
        task_str = self.task if self.task else "default"
        ep_meta = {
            "episode_index": episode_idx,
            "tasks": [task_str],
            "length": frame_count,
            "stats": episode_stats,  # Include per-episode statistics
        }
        
        # Check if overwrite
        if episode_idx in self._episodes:
            old_length = self._episodes[episode_idx].get("length", 0)
            self._total_frames = self._total_frames - old_length + frame_count
        else:
            self._total_frames += frame_count
        
        self._episodes[episode_idx] = ep_meta
        self._episode_count = max(self._episode_count, episode_idx + 1)
        
        # Save metadata
        self._save_metadata(updated_episode=ep_meta)
        
        # Update aggregated statistics
        self._update_aggregated_stats_after_episode(episode_stats, old_episode_stats)
        
        return frame_count
    
    def _streaming_writer_thread(
        self,
        device_names: List[str],
        teleop_device: Optional[str],
    ) -> None:
        """Background thread that writes frames to temp parquet file in batches."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        import cv2
        from concurrent.futures import ThreadPoolExecutor
        
        writer = None
        schema = None
        BATCH_SIZE = 25  # Write every 25 frames (1 second at 25fps)
        
        try:
            # Helper function for image encoding (JPEG is much faster than PNG)
            def encode_image_fast(img_array) -> bytes:
                if img_array is None or not isinstance(img_array, np.ndarray):
                    return b""
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                # Use JPEG for speed (quality 90 is good balance)
                success, encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                return encoded.tobytes() if success else b""
            
            # Get task index
            task_str = self.task if self.task else "default"
            task_index = self._get_task_index(task_str)
            
            # Calculate global index offset
            global_idx_offset = self._total_frames
            
            # Identify image keys
            image_keys = [k for k, ft in self._features_schema.items() 
                        if ft.get("dtype") == "image" and k not in self.DEFAULT_FEATURES]
            
            # Batch buffer: columns -> list of values
            batch_buffer: Dict[str, list] = {key: [] for key in self._features_schema}
            
            def flush_batch():
                """Write accumulated batch to parquet."""
                nonlocal writer, schema, batch_buffer
                
                if not batch_buffer.get("timestamp"):
                    return
                
                # Create schema on first flush
                if schema is None:
                    fields = []
                    for key, ft in self._features_schema.items():
                        dtype = ft.get("dtype", "float32")
                        shape = ft.get("shape", (1,))
                        if isinstance(shape, list):
                            shape = tuple(shape)
                        
                        if dtype == "image":
                            fields.append(pa.field(key, pa.struct([
                                ("bytes", pa.binary()),
                                ("path", pa.string())
                            ])))
                        elif shape == (1,):
                            if dtype == "int64":
                                fields.append(pa.field(key, pa.int64()))
                            else:
                                fields.append(pa.field(key, pa.float32()))
                        else:
                            if dtype == "int64":
                                fields.append(pa.field(key, pa.list_(pa.int64())))
                            else:
                                fields.append(pa.field(key, pa.list_(pa.float32())))
                    
                    schema = pa.schema(fields)
                    writer = pq.ParquetWriter(str(self._stream_tmp_path), schema)
                
                # Build arrays from batch buffer
                arrays = []
                for key in self._features_schema.keys():
                    values = batch_buffer[key]
                    ft = self._features_schema[key]
                    dtype = ft.get("dtype", "float32")
                    shape = ft.get("shape", (1,))
                    if isinstance(shape, list):
                        shape = tuple(shape)
                    
                    if dtype == "image":
                        arrays.append(pa.array(values, type=pa.struct([
                            ("bytes", pa.binary()),
                            ("path", pa.string())
                        ])))
                    elif shape == (1,):
                        if dtype == "int64":
                            arrays.append(pa.array(values, type=pa.int64()))
                        else:
                            arrays.append(pa.array(values, type=pa.float32()))
                    else:
                        if dtype == "int64":
                            arrays.append(pa.array(values, type=pa.list_(pa.int64())))
                        else:
                            arrays.append(pa.array(values, type=pa.list_(pa.float32())))
                
                batch = pa.record_batch(arrays, schema=schema)
                writer.write_batch(batch)
                
                # Clear buffer
                batch_buffer = {key: [] for key in self._features_schema}
            
            # Process frames from queue
            while not self._stream_stop_event.is_set() or not self._stream_queue.empty():
                try:
                    # Get frame from queue with timeout
                    frame = self._stream_queue.get(timeout=0.05)
                except:
                    # Timeout - flush if we have pending data and stopping
                    if self._stream_stop_event.is_set() and batch_buffer.get("timestamp"):
                        flush_batch()
                    continue
                
                # Convert frame
                conv_fr = self._convert_frame(frame, device_names, teleop_device)
                frame_idx = self._stream_frame_count
                
                # Add auto-managed features to buffer
                batch_buffer["timestamp"].append(float(frame_idx) / self.fps)
                batch_buffer["frame_index"].append(frame_idx)
                batch_buffer["episode_index"].append(self._stream_episode_idx)
                batch_buffer["index"].append(global_idx_offset + frame_idx)
                batch_buffer["task_index"].append(task_index)
                
                # Add user features to buffer
                for key, ft in self._features_schema.items():
                    if key in self.DEFAULT_FEATURES:
                        continue
                    
                    value = conv_fr.get(key)
                    dtype = ft.get("dtype", "float32")
                    shape = ft.get("shape", (1,))
                    if isinstance(shape, list):
                        shape = tuple(shape)
                    
                    if dtype == "image":
                        img_bytes = encode_image_fast(value)
                        batch_buffer[key].append({"bytes": img_bytes, "path": None})
                    else:
                        if value is not None:
                            if isinstance(value, np.ndarray):
                                if shape == (1,):
                                    batch_buffer[key].append(float(value.flat[0]))
                                else:
                                    batch_buffer[key].append(value.astype(np.float32).tolist())
                            else:
                                batch_buffer[key].append(value)
                        else:
                            if shape == (1,):
                                batch_buffer[key].append(0.0)
                            else:
                                batch_buffer[key].append([0.0] * shape[0])
                
                self._stream_frame_count += 1
                
                # Flush batch when full
                if len(batch_buffer["timestamp"]) >= BATCH_SIZE:
                    flush_batch()
                    batch_buffer = {key: [] for key in self._features_schema}
            
            # Final flush
            if batch_buffer.get("timestamp"):
                flush_batch()
                
        except Exception as e:
            self._stream_error = e
            logger.exception("Streaming writer error: {}", e)
        finally:
            if writer:
                writer.close()
    
    def finalize(self) -> None:
        """Finalize - stop any active streaming and save metadata."""
        if self.is_streaming:
            self._finish_streaming(save=False)
        if self._initialized:
            self._save_metadata()
    
    @property
    def dataset_path(self) -> str:
        return str(self.output_dir)


# =============================================================================
# Factory Function
# =============================================================================

def create_data_saver(
    format: str,
    output_dir: Union[str, Path],
    fps: int,
    task: str = "",
    enrich_saved_frame: Optional[Callable[[Dict[str, dict]], Dict[str, dict]]] = None,
    **kwargs,
) -> BaseDataSaver:
    """
    Create a data saver based on format.
    
    Args:
        format: "lerobotv21", "lerobotv30", or "hdf5"
        output_dir: Output directory
        fps: Recording frequency
        task: Task description
        enrich_saved_frame: Optional hook to enrich raw frames at save/stream time
    
    Returns:
        A data saver instance
    """
    common = dict(
        output_dir=output_dir,
        fps=fps,
        task=task,
        enrich_saved_frame=enrich_saved_frame,
    )
    if format == "hdf5":
        return HDF5DataSaver(**common)
    elif format == "lerobotv30":
        return LeRobotV30DataSaver(**common)
    elif format == "lerobotv21":
        return LeRobotV21DataSaver(**common)
    else:
        raise ValueError(f"Unknown format: {format}. Supported: lerobotv21, lerobotv30, hdf5")


# =============================================================================
# Task Config Generation
# =============================================================================

def generate_task_config(
    data_saver: BaseDataSaver,
    gen_config_spec: str,
    dataset_format: str,
    task_name: Optional[str] = None,
) -> Optional[Path]:
    """
    Parse gen_config_spec to output path; if dataset has episodes and file does not
    exist, generate task config YAML. Returns the config Path if generated, None if
    skipped or failed.
    gen_config_spec: e.g. "local.task_name" -> configs/task/local/task_name.yaml.
    """
    if data_saver.episode_count == 0:
        return None
    # Parse output path: "local.task_name" -> configs/task/local/task_name.yaml
    if "/" in gen_config_spec or "\\" in gen_config_spec:
        output_path = Path(gen_config_spec)
    elif "." in gen_config_spec:
        parts = gen_config_spec.rsplit(".", 1)
        output_path = Path("configs/task") / parts[0] / f"{parts[1]}.yaml" if len(parts) == 2 else Path("configs/task") / f"{gen_config_spec}.yaml"
    else:
        output_path = Path("configs/task") / f"{gen_config_spec}.yaml"
    if output_path.exists():
        logger.warning("Task config already exists: {}, skipping generation", output_path)
        return None
    try:
        import yaml

        output_path = Path(output_path)
        dataset_path = Path(data_saver.dataset_path)
        feature_schema = data_saver.get_feature_schema()
        camera_names = sorted([k for k in feature_schema.keys() if ".images." in k])
        robot_obs_keys = sorted([
            k for k in feature_schema.keys()
            if k.startswith("observation.") and ".images." not in k and ".teleop." not in k
        ])
        state_key = None
        for candidate in ["observation.qpos", "observation.state"]:
            if candidate in robot_obs_keys:
                state_key = candidate
                break
        if state_key is None and robot_obs_keys:
            state_key = robot_obs_keys[0]
        action_key = state_key
        if task_name is None:
            task_name = dataset_path.name
        state_dim = 0
        if state_key and state_key in feature_schema:
            shape = feature_schema[state_key].get("shape", [])
            if shape:
                state_dim = shape[0] if isinstance(shape, (list, tuple)) else shape
        action_dim = state_dim
        image_size = [256, 256]
        if camera_names and camera_names[0] in feature_schema:
            shape = feature_schema[camera_names[0]].get("shape", [])
            if len(shape) >= 2:
                h, w = shape[0], shape[1]
                if shape[0] in (1, 3, 4):
                    h, w = shape[1], shape[2]
                image_size = [w, h]
        config = {}
        chunk_size = 16
        if dataset_format == "hdf5":
            dataset_type = "data_utils.datasets.HDF5Dataset"
            dataset_args = {"dataset_path_list": [str(dataset_path)], "chunk_size": chunk_size}
            if camera_names:
                dataset_args["camera_names"] = camera_names
        elif dataset_format == "lerobotv21":
            dataset_type = "data_utils.datasets.lerobotv21_wrapper.WrappedLerobotV21Dataset"
            dataset_args = {"dataset_path_list": [str(dataset_path)], "chunk_size": chunk_size}
            if camera_names:
                dataset_args["camera_names"] = camera_names
            if state_key:
                dataset_args["state_key"] = state_key
            if action_key:
                dataset_args["action_key"] = action_key
        elif dataset_format == "lerobotv30":
            dataset_type = "data_utils.datasets.lerobot_wrapper.WrappedLerobotDataset"
            dataset_args = {"dataset_path_list": [str(dataset_path)], "chunk_size": chunk_size}
            if camera_names:
                dataset_args["camera_names"] = camera_names
            if state_key:
                dataset_args["state_key"] = state_key
            if action_key:
                dataset_args["action_key"] = action_key
        else:
            raise ValueError(f"Unknown format: {dataset_format}")
        dataset_args = {k: v for k, v in dataset_args.items() if v is not None}
        config["datasets"] = [{"type": dataset_type, "name": task_name, "args": dataset_args}]
        config["meta"] = {
            "action_dim": action_dim,
            "state_dim": state_dim,
            "image_size": image_size,
            "action_normalize": "zscore",
            "state_normalize": "zscore",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info("Generated task config: {}", output_path)
        logger.info("  - Dataset type: {}", dataset_type)
        logger.info("  - Cameras: {}", camera_names)
        if state_key:
            logger.info("  - state_key: {} (dim={})", state_key, state_dim)
        if action_key:
            logger.info("  - action_key: {} (dim={})", action_key, action_dim)
        return output_path
    except Exception as e:
        logger.error("Failed to generate task config: {}", e)
        return None
