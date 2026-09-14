"""
High-performance, zero-copy shared memory communication for robotics.

- **Added Data Age Metric**: Benchmark now correctly calculates End-to-End latency.
- **Robust Auto-Connect**: Reader ensures connection stability.
"""

import numpy as np
from multiprocessing import shared_memory, resource_tracker
import struct
import pickle
import time
import warnings
import os
import sys
from typing import Callable, Deque, Dict, Any, Tuple, Optional, Union, List

# ==============================================================================
# 1. Aggressive Noise Suppression (resource_tracker KeyError spam)
# ==============================================================================
warnings.filterwarnings("ignore", category=UserWarning, module='multiprocessing.resource_tracker')


def _fix_resource_tracker():
    """
    Completely disable resource_tracker registration for shared_memory.
    This prevents all KeyError spam by not registering in the first place.
    """
    # Method 1: Make register/unregister no-ops for shared_memory
    if not hasattr(resource_tracker, '_original_register'):
        resource_tracker._original_register = resource_tracker.register
        resource_tracker._original_unregister = resource_tracker.unregister

        def _noop_register(name, rtype):
            if rtype == "shared_memory":
                return  # Don't register shared_memory at all
            return resource_tracker._original_register(name, rtype)

        def _noop_unregister(name, rtype):
            if rtype == "shared_memory":
                return  # Don't unregister shared_memory at all
            try:
                return resource_tracker._original_unregister(name, rtype)
            except KeyError:
                pass

        resource_tracker.register = _noop_register
        resource_tracker.unregister = _noop_unregister

    # Method 2: Patch the ResourceTracker class methods directly
    try:
        if hasattr(resource_tracker, 'ResourceTracker'):
            RT = resource_tracker.ResourceTracker
            if not hasattr(RT, '_original_register_method'):
                RT._original_register_method = RT.register
                RT._original_unregister_method = RT.unregister

                def _rt_register(self, name, rtype):
                    if rtype == "shared_memory":
                        return
                    return RT._original_register_method(self, name, rtype)

                def _rt_unregister(self, name, rtype):
                    if rtype == "shared_memory":
                        return
                    try:
                        return RT._original_unregister_method(self, name, rtype)
                    except KeyError:
                        pass

                RT.register = _rt_register
                RT.unregister = _rt_unregister
    except Exception:
        pass

    # Method 3: Clear any existing shared_memory entries from tracker cache
    try:
        if hasattr(resource_tracker, '_resource_tracker'):
            tracker = resource_tracker._resource_tracker
            if hasattr(tracker, '_cache') and 'shared_memory' in tracker._cache:
                tracker._cache['shared_memory'] = set()
    except Exception:
        pass


# Apply patch immediately at module import
_fix_resource_tracker()


def _remove_shm_from_tracker(name: str):
    """Remove shm from resource tracker to prevent cleanup warnings (legacy, now mostly no-op)."""
    try:
        if hasattr(resource_tracker, '_resource_tracker'):
            tracker = resource_tracker._resource_tracker
            if hasattr(tracker, '_cache') and 'shared_memory' in tracker._cache:
                tracker._cache['shared_memory'].discard(name)
    except Exception:
        pass

# ==============================================================================
# 2. Core Class
# ==============================================================================

class SharedMemoryChannel:
    """
    Triple-Buffered Shared Memory Channel.
    Layout: [Magic(8B)] [Index(8B)] [Size(8B)] [Count(8B)] ... [Buffers]
    """
    
    CONFIG_SIZE = 64
    MAGIC_NUMBER = 0x123456789ABC
    MB_UNIT = 1024 * 1024

    def __init__(self, name: str, max_size_mb: int = 64, is_writer: bool = False, timeout: float = 10.0):
        self.name = name
        self.is_writer = is_writer
        
        self._last_header_bytes = None
        self._last_schema = None
        self._local_last_idx = -1
        self._last_read_timestamp: float = 0.0  # for skip_unchanged detection
        self.shm = None

        if is_writer:
            self.buffer_count = 3
            self.per_buffer_size = int(max_size_mb * self.MB_UNIT)
            self.total_size = self.CONFIG_SIZE + (self.per_buffer_size * self.buffer_count)
            self._init_writer()
        else:
            self.buffer_count = 0
            self.per_buffer_size = 0
            self.total_size = 0
            try:
                self.connect(timeout=timeout)
            except TimeoutError:
                pass

    def _init_writer(self):
        try:
            tmp = shared_memory.SharedMemory(name=self.name)
            tmp.close()
            tmp.unlink()
        except Exception:
            pass

        self.shm = shared_memory.SharedMemory(create=True, size=self.total_size, name=self.name)
        _remove_shm_from_tracker(self.name)

        # 1. Write Config FIRST
        self.shm.buf[8:16] = struct.pack('q', 0)
        self.shm.buf[16:24] = struct.pack('q', self.per_buffer_size)
        self.shm.buf[24:32] = struct.pack('q', self.buffer_count)
        
        # 2. Write Magic LAST
        self.shm.buf[0:8] = struct.pack('q', self.MAGIC_NUMBER)
        
        self._current_buffer_idx = 0

    def connect(self, timeout: float = 5.0):
        if self.shm is not None: return

        t0 = time.time()
        while True:
            try:
                self.shm = shared_memory.SharedMemory(name=self.name)
                _remove_shm_from_tracker(self.name)
                
                magic = struct.unpack('q', self.shm.buf[0:8])[0]
                if magic != self.MAGIC_NUMBER:
                    self.shm.close()
                    self.shm = None
                    if time.time() - t0 > timeout:
                         raise TimeoutError("Writer initializing too long...")
                    time.sleep(0.01)
                    continue
                
                self.per_buffer_size = struct.unpack('q', self.shm.buf[16:24])[0]
                self.buffer_count = struct.unpack('q', self.shm.buf[24:32])[0]
                
                if self.per_buffer_size == 0:
                     self.shm.close()
                     self.shm = None
                     time.sleep(0.01); continue

                self.total_size = self.CONFIG_SIZE + (self.per_buffer_size * self.buffer_count)
                break
                
            except (FileNotFoundError, ValueError, struct.error, OSError):
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"SharedMemory '{self.name}' not found.")
                time.sleep(0.01)

    def write(self, data_dict: Dict[str, Any]):
        if not self.is_writer: raise PermissionError("Writer only.")

        next_idx = (self._current_buffer_idx + 1) % self.buffer_count
        buffer_start = self.CONFIG_SIZE + (next_idx * self.per_buffer_size)

        # Auto-inject write timestamp for change detection
        data_dict = dict(data_dict)  # shallow copy to avoid mutating caller's dict
        data_dict["__timestamp__"] = time.perf_counter()

        schema = {}
        raw_arrays = []
        payload_offset = 0

        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                schema[k] = {'t': 'np', 's': v.shape, 'd': v.dtype.str, 'o': payload_offset, 'b': v.nbytes}
                raw_arrays.append(v)
                payload_offset += v.nbytes
            else:
                schema[k] = {'t': 'obj', 'v': v}

        header_bytes = pickle.dumps(schema)
        header_len = len(header_bytes)

        if 4 + header_len + payload_offset > self.per_buffer_size:
            raise ValueError(f"Data too large! Need {payload_offset} bytes.")

        self.shm.buf[buffer_start : buffer_start+4] = struct.pack('i', header_len)
        self.shm.buf[buffer_start+4 : buffer_start+4+header_len] = header_bytes

        curr_ptr = buffer_start + 4 + header_len
        for arr in raw_arrays:
            target = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self.shm.buf[curr_ptr : curr_ptr+arr.nbytes])
            target[:] = arr
            curr_ptr += arr.nbytes

        self.shm.buf[8:16] = struct.pack('q', next_idx)
        self._current_buffer_idx = next_idx

    def read(
        self,
        blocking: bool = False,
        timeout: Optional[float] = None,
        return_index: bool = False,
        skip_unchanged: bool = False,
        copy: bool = False,
    ):
        """Read data from shared memory.

        Args:
            blocking: If True, wait until new data is available.
            timeout: Timeout in seconds for blocking read.
            return_index: If True, return (data, index) tuple.
            skip_unchanged: If True, return None when __timestamp__ hasn't changed
                            since last read (avoids returning duplicate data).
            copy: If True, numpy arrays are copied out of the shared memory
                buffer so that subsequent writes cannot corrupt them.  Use this
                when the returned data will be held across writer cycles (e.g.
                stored in an action buffer for seconds).  Default False
                (zero-copy) for performance with large payloads like images.
        """
        if not self.shm: 
            try: self.connect(timeout=0.1)
            except TimeoutError: return None if not return_index else (None, -1)

        t_start = time.time()
        
        while True:
            try:
                global_idx = struct.unpack('q', self.shm.buf[8:16])[0]
                
                if blocking:
                    if global_idx == self._local_last_idx:
                        if timeout and (time.time() - t_start > timeout):
                            return None if not return_index else (None, -1)
                        time.sleep(0.0001) 
                        continue
                
                self._local_last_idx = global_idx
                buffer_start = self.CONFIG_SIZE + (global_idx * self.per_buffer_size)
                
                hlen = struct.unpack('i', self.shm.buf[buffer_start:buffer_start+4])[0]
                if hlen == 0: 
                    if blocking:
                        time.sleep(0.001); continue
                    else:
                        return None if not return_index else (None, -1)

                h_start = buffer_start + 4
                h_bytes = self.shm.buf[h_start : h_start+hlen].tobytes()

                if h_bytes == self._last_header_bytes:
                    schema = self._last_schema
                else:
                    schema = pickle.loads(h_bytes)
                    self._last_schema = schema
                    self._last_header_bytes = h_bytes

                result = {}
                payload_base = h_start + hlen
                for k, meta in schema.items():
                    if meta['t'] == 'np':
                        ptr = payload_base + meta['o']
                        arr = np.ndarray(meta['s'], dtype=meta['d'], buffer=self.shm.buf[ptr:ptr+meta['b']])
                        result[k] = arr.copy() if copy else arr
                    else:
                        result[k] = meta['v']

                # skip_unchanged: check __timestamp__ to detect duplicate reads
                if skip_unchanged:
                    ts = result.get("__timestamp__", 0.0)
                    if ts == self._last_read_timestamp:
                        # same data as last read, skip
                        return None if not return_index else (None, -1)
                    self._last_read_timestamp = ts
                
                if return_index: return result, global_idx
                else: return result

            except Exception:
                return None if not return_index else (None, -1)

    def destroy(self):
        """Clean up shared memory. Writer will unlink, reader just closes."""
        if self.shm:
            try:
                self.shm.close()
            except Exception:
                pass
            if self.is_writer:
                _unlink_shm(self.name)
            self.shm = None

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        try:
            self.destroy()
        except Exception:
            pass


def _unlink_shm(name: str):
    """Safely unlink shared memory by name."""
    try:
        from multiprocessing import shared_memory as _shm_mod
        if hasattr(_shm_mod, '_posixshmem'):
            _shm_mod._posixshmem.shm_unlink(name)
        else:
            # Fallback: try to open and unlink
            try:
                tmp = shared_memory.SharedMemory(name=name)
                tmp.close()
                tmp.unlink()
            except FileNotFoundError:
                pass
    except FileNotFoundError:
        pass  # Already unlinked
    except Exception:
        pass


def cleanup_shm(name: str):
    """
    Public API to forcefully clean up a shared memory segment by name.
    Use this to clean up orphaned shm from previous crashed runs.
    """
    _unlink_shm(name)


def cleanup_all_shm(names: List[str]):
    """Clean up multiple shared memory segments."""
    for name in names:
        _unlink_shm(name)

# ==============================================================================
# SharedMemoryDataSynchronizer - Multi-device timestamp alignment
# ==============================================================================

from collections import deque
import bisect


class SharedMemoryDataSynchronizer:
    """
    Read from multiple SHM channels and return synchronized frames.

    Two modes:
    1. get_synced_frame(reference_time): Non-blocking, nearest-neighbor alignment at given time.
    2. get_synced_frame_blocking(): Block until ALL devices have NEW data, then return one frame.
       Each device's data is as fresh as possible; reference time = when slowest device gets new data.
    """

    def __init__(
        self,
        shm_channels: List[Tuple[str, SharedMemoryChannel]],
        buffer_maxlen: int = 100,
        max_tolerance_s: float = 0.1,
    ):
        """
        Args:
            shm_channels: List of (name, SharedMemoryChannel) to read from.
            buffer_maxlen: Max samples to keep per device (oldest dropped).
            max_tolerance_s: Max allowed timestamp deviation for nearest-neighbor (skip if exceeded).
        """
        self.shm_channels = list(shm_channels)  # [(name, ch), ...]
        self.buffer_maxlen = buffer_maxlen
        self.max_tolerance_s = max_tolerance_s
        # Per device: deque of (timestamp, data) sorted by timestamp
        self._buffers: Dict[str, deque] = {name: deque(maxlen=buffer_maxlen) for name, _ in shm_channels}
        # Per device: timestamp of last emitted sample (for blocking mode, avoid duplicates)
        self._last_emitted_ts: Dict[str, float] = {name: float("-inf") for name, _ in shm_channels}
        # Filled by get_synced_frame_blocking for diagnostics
        self.last_sync_stats: Dict[str, object] = {}

    def _get_timestamp(self, data: dict) -> float:
        """Extract timestamp from data. Uses __timestamp__ (writer) or timestamp (legacy)."""
        return data.get("__timestamp__", data.get("timestamp", 0.0))

    def read_and_buffer(self) -> None:
        """Read latest from each channel (non-blocking) and append to buffers."""
        for name, ch in self.shm_channels:
            if ch is None:
                continue
            # Zero-copy read; we copy arrays ourselves below (one copy total).
            data = ch.read(blocking=False, skip_unchanged=True, copy=False)
            if data is not None:
                ts = self._get_timestamp(data)
                data_copy = {}
                for k, v in data.items():
                    if k in ("__timestamp__", "timestamp"):
                        continue
                    data_copy[k] = v.copy() if isinstance(v, np.ndarray) else v
                data_copy["__timestamp__"] = ts
                self._buffers[name].append((ts, data_copy))

    def get_synced_frame(
        self,
        reference_time: float,
    ) -> Optional[Dict[str, dict]]:
        """
        Get one synchronized frame: for each device, find the sample with timestamp
        nearest to reference_time. Returns {device_name: data} or None if any required
        device has no valid sample within tolerance.

        Args:
            reference_time: Target timestamp for alignment.

        Returns:
            Dict mapping device name -> data dict (without __timestamp__ in nested data),
            or None if sync failed (missing device or tolerance exceeded).
        """
        result = {}
        for name, _ in self.shm_channels:
            buf = self._buffers[name]
            if len(buf) == 0:
                return None
            # Nearest-neighbor: find index where buf[i][0] >= reference_time
            timestamps = [b[0] for b in buf]
            idx = bisect.bisect_left(timestamps, reference_time)
            if idx == 0:
                best_idx = 0
            elif idx >= len(timestamps):
                best_idx = len(timestamps) - 1
            else:
                # Compare idx-1 and idx
                if abs(timestamps[idx - 1] - reference_time) <= abs(timestamps[idx] - reference_time):
                    best_idx = idx - 1
                else:
                    best_idx = idx
            ts, data = buf[best_idx]
            if abs(ts - reference_time) > self.max_tolerance_s:
                return None
            result[name] = data
        return result

    def get_synced_frame_blocking(
        self,
        stop_check: Optional[Callable[[], bool]] = None,
        poll_interval_s: float = 0.001,
        debug: bool = False,
    ) -> Optional[Dict[str, dict]]:
        """
        Block until ALL devices have NEW data (timestamp > last_emitted), then return one
        synchronized frame. Each device's data is as fresh as possible: reference time is
        when the slowest device gets its first new sample; for each device we pick the
        sample nearest to that reference (among new samples).

        Args:
            stop_check: If provided, called each loop iteration. If returns True, abort and return None.
            poll_interval_s: Sleep duration when waiting for new data.
            debug: If True, print which device is being waited for.

        Returns:
            Dict mapping device name -> data dict, or None if stop_check aborted.

        Side effect:
            Sets ``self.last_sync_stats`` with wait_ms, wait_loops, wait_counts_ms per device,
            bottleneck device, and per-device age_ms of the emitted samples.
        """
        wait_start = time.perf_counter()
        wait_count = 0
        last_waiting_for = None
        wait_counts: Dict[str, int] = {name: 0 for name, _ in self.shm_channels}
        
        while True:
            self.read_and_buffer()
            if stop_check is not None and stop_check():
                return None

            # For each device, collect timestamps of "new" samples (ts > last_emitted)
            new_ts_per_device: Dict[str, List[float]] = {}
            waiting_for = None
            for name, _ in self.shm_channels:
                buf = self._buffers[name]
                last = self._last_emitted_ts[name]
                new_ts = [b[0] for b in buf if b[0] > last]
                if not new_ts:
                    waiting_for = name
                    wait_counts[name] = wait_counts.get(name, 0) + 1
                    time.sleep(poll_interval_s)
                    break
                new_ts_per_device[name] = new_ts
            else:
                # All devices have new data
                # Reference = when slowest device gets first new sample = max_d(min(new_ts_d))
                ref = max(min(ts_list) for ts_list in new_ts_per_device.values())

                result = {}
                emitted_ts: Dict[str, float] = {}
                for name, _ in self.shm_channels:
                    buf = self._buffers[name]
                    last = self._last_emitted_ts[name]
                    candidates = [(b[0], b[1]) for b in buf if b[0] > last]
                    if not candidates:
                        waiting_for = name
                        wait_counts[name] = wait_counts.get(name, 0) + 1
                        time.sleep(poll_interval_s)
                        break
                    # Pick sample with ts closest to ref, preferring fresher (ts >= ref)
                    best_ts, best_data = min(
                        candidates,
                        key=lambda x: (abs(x[0] - ref), -x[0]),  # min |ts-ref|, then prefer larger ts
                    )
                    if abs(best_ts - ref) > self.max_tolerance_s:
                        waiting_for = f"{name}(tolerance)"
                        base = name.split("(")[0]
                        wait_counts[base] = wait_counts.get(base, 0) + 1
                        time.sleep(poll_interval_s)
                        break
                    self._last_emitted_ts[name] = best_ts
                    emitted_ts[name] = best_ts
                    result[name] = best_data
                else:
                    wait_time_ms = (time.perf_counter() - wait_start) * 1000
                    # Approximate time spent waiting on each device (poll loops * interval)
                    wait_ms_per_device = {
                        n: c * poll_interval_s * 1000.0 for n, c in wait_counts.items() if c > 0
                    }
                    bottleneck = None
                    if wait_ms_per_device:
                        bottleneck = max(wait_ms_per_device.items(), key=lambda kv: kv[1])[0]
                    elif last_waiting_for:
                        bottleneck = str(last_waiting_for).split("(")[0]
                    now = time.perf_counter()
                    age_ms = {
                        n: max(0.0, (now - ts) * 1000.0) for n, ts in emitted_ts.items()
                    }
                    self.last_sync_stats = {
                        "wait_ms": wait_time_ms,
                        "wait_loops": wait_count,
                        "wait_ms_per_device": wait_ms_per_device,
                        "bottleneck": bottleneck,
                        "last_waiting_for": last_waiting_for,
                        "age_ms": age_ms,
                        "ref_ts": ref,
                    }
                    # Success - print debug if waited
                    if debug and wait_count > 0:
                        if wait_time_ms > 5:  # Only print if waited > 5ms
                            print(
                                f"[Sync] Waited {wait_time_ms:.1f}ms ({wait_count} loops), "
                                f"bottleneck={bottleneck}, waits={wait_ms_per_device}, "
                                f"last={last_waiting_for}"
                            )
                    return result
            
            # Track what we're waiting for
            wait_count += 1
            if waiting_for:
                last_waiting_for = waiting_for
                
        return None  # unreachable

    def get_latest_frame_nonblocking(
        self,
        required_devices: Optional[List[str]] = None,
        strict_new_only: bool = True,
    ) -> Optional[Dict[str, dict]]:
        """
        Get the latest available data from all devices without blocking.
        
        Args:
            required_devices: Devices that must have new data. If None, all devices are required.
            strict_new_only: If True, ALL devices must have new (unused) data.
                            If False, only required_devices must have new data,
                            others use latest available (may be reused).
            
        Returns:
            Dict mapping device name -> data dict, or None if required conditions not met.
        """
        self.read_and_buffer()
        
        if required_devices is None:
            required_devices = [name for name, _ in self.shm_channels]
        
        result = {}
        for name, _ in self.shm_channels:
            buf = self._buffers[name]
            if len(buf) == 0:
                if name in required_devices or strict_new_only:
                    return None  # Required device has no data
                continue
            
            last = self._last_emitted_ts[name]
            
            # Get new samples (not yet emitted)
            new_samples = [(b[0], b[1]) for b in buf if b[0] > last]
            
            if name in required_devices:
                # Must have NEW data
                if not new_samples:
                    return None  # Required device has no new data
                # Use the newest new sample
                best_ts, best_data = max(new_samples, key=lambda x: x[0])
            elif strict_new_only:
                # All devices must have new data in strict mode
                if not new_samples:
                    return None
                best_ts, best_data = max(new_samples, key=lambda x: x[0])
            else:
                # Non-strict: use latest available (may be old)
                if new_samples:
                    best_ts, best_data = max(new_samples, key=lambda x: x[0])
                else:
                    best_ts, best_data = max(buf, key=lambda x: x[0])
            
            self._last_emitted_ts[name] = best_ts
            result[name] = best_data
        
        return result

    def get_synced_frame_at_time_or_none(
        self,
        reference_time: float,
        required_devices: Optional[List[str]] = None,
    ) -> Optional[Dict[str, dict]]:
        """
        Like get_synced_frame but only require certain devices. If required_devices
        is None, all devices are required.
        """
        if required_devices is None:
            required_devices = [name for name, _ in self.shm_channels]
        result = {}
        for name, _ in self.shm_channels:
            buf = self._buffers[name]
            if len(buf) == 0:
                if name in required_devices:
                    return None
                continue
            timestamps = [b[0] for b in buf]
            idx = bisect.bisect_left(timestamps, reference_time)
            if idx == 0:
                best_idx = 0
            elif idx >= len(timestamps):
                best_idx = len(timestamps) - 1
            else:
                if abs(timestamps[idx - 1] - reference_time) <= abs(timestamps[idx] - reference_time):
                    best_idx = idx - 1
                else:
                    best_idx = idx
            ts, data = buf[best_idx]
            if name in required_devices and abs(ts - reference_time) > self.max_tolerance_s:
                return None
            result[name] = data
        return result


# ==============================================================================
# Test Suite (Complete Metrics)
# ==============================================================================

if __name__ == "__main__":
    from multiprocessing import Process, Event
    import os

    TEST_NAME = "shm_benchmark_v5_3"
    IMG_SHAPE = (480, 640, 3) 
    TARGET_FREQ = 500         
    TEST_FRAMES = 1000
    WRITER_SIZE_MB = 10 
    
    evt_writer_ready = Event()
    evt_reader_done = Event()

    def writer_process(ready_evt, done_evt):
        print(f"[Writer] Started. Allocating {WRITER_SIZE_MB} MB.")
        try:
            shm = SharedMemoryChannel(TEST_NAME, max_size_mb=WRITER_SIZE_MB, is_writer=True)
            ready_evt.set()
            
            img = np.zeros(IMG_SHAPE, dtype=np.uint8)
            interval = 1.0 / TARGET_FREQ
            next_run = time.time()

            for i in range(TEST_FRAMES):
                if done_evt.is_set(): break 
                img[0, 0, 0] = i % 255
                shm.write({"seq_id": i, "image": img, "timestamp": time.time()})
                while time.time() < next_run: pass
                next_run += interval
                
        except Exception as e:
            print(f"[Writer] Error: {e}")
        finally:
            print("[Writer] Waiting for reader...")
            done_evt.wait(timeout=2.0)
            shm.destroy()
            print("[Writer] Done.")

    def reader_process(ready_evt, done_evt):
        print(f"[Reader] Started.")
        if not ready_evt.wait(timeout=5.0): return

        print("[Reader] Connecting (Auto-Config)...")
        try:
            shm = SharedMemoryChannel(TEST_NAME, is_writer=False)
            real_size = shm.per_buffer_size / (1024*1024)
            print(f"[Reader] Success! Buffer Size: {real_size:.2f} MB")
        except Exception as e:
            print(f"[Reader] Fail: {e}")
            return
        
        received_count = 0
        latencies_overhead = [] # 读取API耗时
        latencies_e2e = []      # 数据新鲜度 (Age)
        
        try:
            while received_count < TEST_FRAMES:
                t0 = time.time()
                data = shm.read(blocking=True, timeout=1.0) 
                t1 = time.time() # 读取完成时刻

                if data is None: break
                
                # 1. 记录 API Overhead
                latencies_overhead.append((t1 - t0) * 1000)
                
                # 2. 记录 E2E Latency (Data Age)
                # Age = 当前时间 - Writer写入时间
                age = (t1 - data['timestamp']) * 1000
                latencies_e2e.append(age)
                
                received_count += 1
        finally:
            done_evt.set()
        
        # 统计报告
        if received_count > 0:
            avg_overhead = np.mean(latencies_overhead)
            avg_e2e = np.mean(latencies_e2e)
            
            print("\n" + "="*40)
            print(f"   SHM V5.3 PERFORMANCE REPORT")
            print("="*40)
            print(f"Frames Processed: {received_count}")
            print("-" * 40)
            print(f"1. API Overhead: {avg_overhead:.4f} ms")
            print(f"   (Includes blocking wait time)")
            print("-" * 40)
            print(f"2. Data Age (E2E): {avg_e2e:.4f} ms")
            print(f"   (Writer -> Reader physical latency)")
            print("=" * 40)
        else:
            print("[Reader] No data received.")

    p_w = Process(target=writer_process, args=(evt_writer_ready, evt_reader_done))
    p_r = Process(target=reader_process, args=(evt_writer_ready, evt_reader_done))

    p_w.start(); p_r.start()
    p_w.join(); p_r.join()