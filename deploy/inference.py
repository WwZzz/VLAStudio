"""
Inference Utilities for ILStudio.

Provides the inference worker process and SHM communication helpers for
decoupling policy inference from the environment step loop.

Architecture (Sim):
    ┌──────────────┐   obs_shm    ┌────────────────┐   action_shm   ┌──────────────┐
    │  Environment  │ ──────────> │ Inference Proc  │ ────────────> │ ActionManager │
    │  (main proc)  │  ctrl_shm   │  (subprocess)   │               │  (main proc)  │
    │              │ ──────────> │                  │               │              │
    └──────────────┘             └────────────────┘               └──────────────┘

Architecture (Real):
    ┌──────────────┐             ┌────────────────┐   action_shm   ┌──────────────┐
    │ Device SHMs   │ ──────────>│ Inference Proc  │ ────────────> │ ActionManager │
    │ (device procs)│  (direct)  │  (subprocess)   │               │  (main proc)  │
    └──────────────┘             └────────────────┘               └──────────────┘
                                  ↑ ctrl_shm (trigger)               │
                                  └──────────────────────────────────┘

Key difference: obs source is configurable (obs_shm for sim, device SHMs for real).
Trigger mechanism is unified: ctrl_shm "infer" command from action_manager.

Usage:
    # Sim mode:
    ctx = start_inference_process(model_name_or_path, ...)
    ctx.update_obs(obs, t)        # Main process writes obs
    # action_manager sends trigger internally
    
    # Real mode:
    ctx = start_inference_process(model_name_or_path, ..., 
              device_shm_names=["cam1", "robot1"], robot_module_name="robot.so101")
    # Worker reads obs from device SHMs directly, no obs_shm needed
"""

from __future__ import annotations

import os
import time
import signal
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Optional, List

import numpy as np
import torch
from loguru import logger

from deploy.shm_utils import SharedMemoryChannel, cleanup_all_shm


def _extract_step_action(step):
    if isinstance(step, np.ndarray) and step.dtype == object and len(step) > 0:
        first = step[0]
        if isinstance(first, dict):
            return first.get("action", None)
    if isinstance(step, np.ndarray):
        return step
    return None


# ---------------------------------------------------------------------------
# Inference Process (runs in subprocess)
# ---------------------------------------------------------------------------

def inference_worker(
    model_name_or_path: str,
    action_shm_name: str,
    ctrl_shm_name: str,
    # Sim mode: read obs from obs_shm (written by main process)
    obs_shm_name: str = None,
    # Real mode: read obs from device SHMs directly
    device_shm_names: List[str] = None,
    device_configs: List[dict] = None,
    robot_module_name: str = None,
    sync_buffer_maxlen: int = 100,
    sync_max_tolerance_s: float = 0.05,
    # Common params
    action_shm_size_mb: int = 32,
    device: str = "cuda",
    dataset_id: str = "",
    chunk_size: int = -1,
    worker_ready_event: Optional[object] = None,
):
    """
    Unified inference worker for both sim and real modes.
    
    Waits for "infer" trigger from ctrl_shm, then:
    - Sim mode: reads latest obs from obs_shm
    - Real mode: reads latest obs from device SHMs via synchronizer + obs2meta
    
    Runs policy inference and writes action chunks to action_shm.
    """
    # Re-apply resource_tracker patch in subprocess
    from deploy.shm_utils import _fix_resource_tracker
    _fix_resource_tracker()

    # Spawn children often line-buffer stderr only when attached to a TTY; flush so logs
    # align with real time and users do not think inference "reconnects" in one burst.
    import sys
    logger.info(f"[InferenceWorker] subprocess pid={os.getpid()} (loading...)")
    
    import configs  # noqa: side-effects
    from data_utils.utils import set_seed
    set_seed(0)
    
    is_real_mode = device_shm_names is not None and len(device_shm_names) > 0
    mode_str = "REAL" if is_real_mode else "SIM"
    
    logger.info(f"[InferenceWorker-{mode_str}] Starting (PID={os.getpid()})")
    logger.info(f"[InferenceWorker-{mode_str}] Model: {model_name_or_path}")
    logger.info(f"[InferenceWorker-{mode_str}] Device: {device}")
    
    # Load policy
    from policy.utils import load_policy
    from types import SimpleNamespace
    args = SimpleNamespace(
        model_name_or_path=model_name_or_path,
        device=device,
        dataset_id=dataset_id,
        chunk_size=chunk_size,
        is_training=False,
    )
    policy = load_policy(args)
    logger.info(f"[InferenceWorker-{mode_str}] Policy loaded successfully")
    
    # --- Setup obs source ---
    obs_reader = None
    synchronizer = None
    obs2meta_func = None
    
    if is_real_mode:
        # Real mode: connect to device SHMs + synchronizer (parallel attach to reduce wall time)
        from deploy.shm_utils import SharedMemoryDataSynchronizer
        from deploy.base import create_obs2meta_func, default_obs2meta

        def _connect_one_device_shm(name: str):
            try:
                ch = SharedMemoryChannel(name, is_writer=False, timeout=30.0)
                return name, ch, None
            except Exception as e:
                return name, None, e

        name_to_ch = {}
        n_dev = len(device_shm_names)
        max_workers = max(1, n_dev)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_connect_one_device_shm, n): n for n in device_shm_names}
            for fut in as_completed(futures):
                name, ch, err = fut.result()
                if err is None:
                    name_to_ch[name] = ch
                    logger.info(f"[InferenceWorker-REAL] Connected to device SHM: {name}")
                else:
                    logger.warning(f"[InferenceWorker-REAL] Failed to connect to {name}: {err}")

        shm_channels = [(n, name_to_ch[n]) for n in device_shm_names if n in name_to_ch]
        
        if not shm_channels:
            logger.error("[InferenceWorker-REAL] No device SHM channels connected. Exiting.")
            return
        
        synchronizer = SharedMemoryDataSynchronizer(
            shm_channels=shm_channels,
            buffer_maxlen=sync_buffer_maxlen,
            max_tolerance_s=sync_max_tolerance_s,
        )
        
        if device_configs:
            obs2meta_func = create_obs2meta_func(device_configs)
        else:
            obs2meta_func = default_obs2meta
            logger.info("[InferenceWorker-REAL] No device_configs, using default obs2meta")
    else:
        # Sim mode: read from obs_shm
        if obs_shm_name is None:
            logger.error("[InferenceWorker-SIM] obs_shm_name is required in sim mode. Exiting.")
            return
        obs_reader = SharedMemoryChannel(name=obs_shm_name, is_writer=False, timeout=30.0)
        logger.info(f"[InferenceWorker-SIM] Connected to obs SHM: {obs_shm_name}")
    
    # --- Setup action and ctrl channels ---
    action_writer = SharedMemoryChannel(name=action_shm_name, max_size_mb=action_shm_size_mb, is_writer=True)
    logger.info(f"[InferenceWorker-{mode_str}] Created action SHM: {action_shm_name}")
    
    ctrl_reader = SharedMemoryChannel(name=ctrl_shm_name, is_writer=False, timeout=30.0)
    logger.info(f"[InferenceWorker-{mode_str}] Connected to ctrl SHM: {ctrl_shm_name}")
    
    running = True
    inference_count = 0
    
    def _handle_signal(signum, frame):
        nonlocal running
        running = False
    
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    def _get_obs_sim():
        """Read obs from obs_shm (sim mode). Returns (mobs, timestep, synced_dict_or_none)."""
        from benchmark.base import dict2meta
        obs_data = obs_reader.read(blocking=False)
        if obs_data is None:
            return None, 0, None
        t = obs_data.get("t", 0)
        mobs_dict = {k: v for k, v in obs_data.items() if not k.startswith("__") and k != "t"}
        mobs = dict2meta(mobs_dict, mtype="obs")
        return mobs, t, None
    
    def _get_obs_real():
        """Read obs from device SHMs via synchronizer (real mode). Returns (mobs, 0, synced_data)."""
        # For trigger-based real-world inference, using get_synced_frame_blocking()
        # would return the earliest unread synchronized frame since the previous
        # inference. With chunked execution this can lag by almost one whole chunk.
        # Here we instead wait until every device has at least one fresh sample and
        # then consume the newest unread sample from each device.
        synced_data = None
        wait_t0 = time.perf_counter()
        last_warn_t = 0.0
        while running and synced_data is None:
            synced_data = synchronizer.get_latest_frame_nonblocking(strict_new_only=True)
            if synced_data is None:
                now = time.perf_counter()
                if now - wait_t0 > 2.0 and now - last_warn_t > 5.0:
                    logger.warning(
                        "[InferenceWorker-REAL] Still waiting for a fresh synced observation from "
                        "ALL device SHMs (strict_new_only). If the robot or a camera stops writing "
                        "for a moment, inference will block here without publishing actions."
                    )
                    last_warn_t = now
                time.sleep(0.001)
        if synced_data is None:
            return None, 0, None
        mobs = obs2meta_func(synced_data)
        return mobs, 0, synced_data
    
    get_obs = _get_obs_real if is_real_mode else _get_obs_sim

    logger.info(f"[InferenceWorker-{mode_str}] Ready, waiting for triggers...")
    if worker_ready_event is not None:
        worker_ready_event.set()
    try:
        sys.stderr.flush()
    except Exception:
        pass

    try:
        with torch.inference_mode():
            while running:
                # Real mode: always keep synchronizer buffer updated
                if synchronizer is not None:
                    synchronizer.read_and_buffer()
                
                # Check for control signals (non-blocking)
                ctrl_data = ctrl_reader.read(blocking=False, skip_unchanged=True)
                if ctrl_data is not None:
                    cmd = ctrl_data.get("cmd", "")
                    if cmd == "stop":
                        logger.info(f"[InferenceWorker-{mode_str}] Received stop command")
                        break
                    elif cmd == "reset":
                        policy.reset()
                        logger.info(f"[InferenceWorker-{mode_str}] Policy reset")
                        continue
                    elif cmd == "infer":
                        # Carry trigger metadata through inference.
                        trigger_epoch = ctrl_data.get("epoch", 0)
                        trigger_t = int(ctrl_data.get("t", 0))
                        
                        # Get latest observation
                        mobs, obs_t, _ = get_obs()
                        if mobs is None:
                            logger.debug(f"[InferenceWorker-{mode_str}] No obs available at trigger time")
                            continue

                        t = trigger_t if is_real_mode else obs_t
                        
                        # Set timestep
                        if mobs.state is not None:
                            batch_size = mobs.state.shape[0] if mobs.state.ndim > 1 else 1
                        else:
                            batch_size = 1
                        mobs.timestep = np.array([[t] for _ in range(batch_size)])
                        
                        # Run inference
                        t_start = time.perf_counter()
                        mact_list = policy.inference(mobs)
                        t_end = time.perf_counter()
                        
                        inference_count += 1
                        latency_ms = (t_end - t_start) * 1000
                        
                        # Serialize and write (include epoch for staleness detection)
                        action_payload = serialize_mact_list(mact_list)
                        action_payload["t"] = t
                        action_payload["epoch"] = trigger_epoch
                        action_payload["latency_ms"] = latency_ms
                        action_writer.write(action_payload)

                        # Surface GPU slowdown (e.g. MX150 thermal) — sync_chunk then
                        # freezes for seconds between 16-step bursts and looks "broken".
                        if latency_ms >= 1500.0:
                            logger.warning(
                                f"[InferenceWorker-{mode_str}] SLOW inference #{inference_count}: "
                                f"latency={latency_ms:.0f}ms (chunk={len(mact_list)}). "
                                f"Likely GPU throttle; publish rate will collapse under sync_chunk."
                            )
                        elif inference_count % 50 == 0:
                            logger.info(
                                f"[InferenceWorker-{mode_str}] Inference #{inference_count}, "
                                f"latency={latency_ms:.1f}ms, chunk_size={len(mact_list)}"
                            )
                else:
                    time.sleep(0.0001)  # 0.1ms idle sleep
    
    except Exception as e:
        logger.error(f"[InferenceWorker-{mode_str}] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if obs_reader is not None:
            obs_reader.destroy()
        if synchronizer is not None:
            for _, ch in synchronizer.shm_channels:
                try:
                    ch.destroy()
                except Exception:
                    pass
        action_writer.destroy()
        ctrl_reader.destroy()
        logger.info(f"[InferenceWorker-{mode_str}] Stopped (processed {inference_count} inferences)")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def serialize_mact_list(mact_list: list) -> dict:
    """
    Serialize mact_list for SharedMemory transmission.
    
    mact_list is List[np.ndarray(dtype=object)] where each np.ndarray contains
    dicts like {'action': np.ndarray, 'ctrl_type': str, 'ctrl_space': str}.
    
    We flatten it into a dict of numpy arrays for efficient SHM transfer.
    """
    all_actions = []
    ctrl_type = "delta"
    ctrl_space = "ee"
    
    for step_arr in mact_list:
        step_actions = []
        for item in step_arr:
            if isinstance(item, dict):
                act = item.get("action", None)
                if act is not None:
                    step_actions.append(act)
                ctrl_type = item.get("ctrl_type", ctrl_type)
                ctrl_space = item.get("ctrl_space", ctrl_space)
        if step_actions:
            all_actions.append(np.stack(step_actions))
    
    if all_actions:
        actions_array = np.stack(all_actions)
    else:
        actions_array = np.zeros((0,))
    
    return {
        "actions": actions_array,
        "chunk_len": len(mact_list),
        "ctrl_type": ctrl_type,
        "ctrl_space": ctrl_space,
    }


def deserialize_mact_list(payload: dict) -> list:
    """
    Deserialize SHM payload back to mact_list format.
    
    Returns:
        List[np.ndarray(dtype=object)] matching the format from MetaPolicy.inference()
    """
    actions_array = payload.get("actions", None)
    if actions_array is None or (isinstance(actions_array, np.ndarray) and actions_array.size == 0):
        return []
    
    chunk_len = payload.get("chunk_len", 0)
    ctrl_type = payload.get("ctrl_type", "delta")
    ctrl_space = payload.get("ctrl_space", "ee")
    
    mact_list = []
    for i in range(min(chunk_len, len(actions_array))):
        step_actions = actions_array[i]
        step_dicts = []
        if step_actions.ndim == 1:
            step_dicts.append({
                "action": np.array(step_actions, copy=True),
                "ctrl_type": ctrl_type,
                "ctrl_space": ctrl_space,
            })
        else:
            for j in range(step_actions.shape[0]):
                step_dicts.append({
                    "action": np.array(step_actions[j], copy=True),
                    "ctrl_type": ctrl_type,
                    "ctrl_space": ctrl_space,
                })
        mact_list.append(np.array(step_dicts, dtype=object))
    
    return mact_list


# ---------------------------------------------------------------------------
# Process lifecycle management
# ---------------------------------------------------------------------------

class InferenceContext:
    """
    Context for managing inference process and SHM channels.
    
    Created by start_inference_process(), used by evaluate() and action_manager,
    cleaned up by stop_inference_process().
    
    Obs and trigger are decoupled:
    - update_obs(): main process writes obs to obs_shm (sim mode only)
    - send_trigger(): action_manager requests inference via ctrl_shm
    """
    def __init__(
        self,
        process: mp.Process,
        obs_writer: Optional[SharedMemoryChannel],  # None in real mode
        action_reader: SharedMemoryChannel,
        ctrl_writer: SharedMemoryChannel,
        shm_names: tuple,
    ):
        self.process = process
        self.obs_writer = obs_writer
        self.action_reader = action_reader
        self.ctrl_writer = ctrl_writer
        self.shm_names = shm_names
        self._epoch = 0  # Incremented on reset; used to discard stale chunks
    
    def update_obs(self, mobs, t: int) -> None:
        """
        Write observation to obs_shm (sim mode only).
        
        In real mode (obs_writer is None), this is a no-op because
        the inference worker reads from device SHMs directly.
        """
        if self.obs_writer is None:
            return
        obs_dict = asdict(mobs)
        obs_dict["t"] = t
        self.obs_writer.write(obs_dict)
    
    def send_trigger(self, t: int = 0) -> None:
        """Send inference trigger to worker via ctrl_shm."""
        self.ctrl_writer.write({"cmd": "infer", "t": t, "epoch": self._epoch})
    
    def poll_action_chunks(self) -> Optional[list]:
        """
        Non-blocking check for new action chunks from inference process.
        
        Automatically discards stale chunks from previous epochs (rollouts).
        
        Returns:
            mact_list if new chunk available, None otherwise.
        """
        action_data = self.action_reader.read(blocking=False, skip_unchanged=True)
        if action_data is None:
            return None
        
        # Epoch check: discard stale chunks from previous rollouts
        chunk_epoch = action_data.get("epoch", 0)
        if chunk_epoch != self._epoch:
            logger.debug(f"Discarding stale action chunk (epoch {chunk_epoch} != current {self._epoch})")
            return None
        
        return deserialize_mact_list(action_data)
    
    def send_reset(self) -> None:
        """Send reset command to inference process and increment epoch."""
        self._epoch += 1
        self.ctrl_writer.write({"cmd": "reset", "epoch": self._epoch})
    
    def send_stop(self) -> None:
        """Send stop command to inference process."""
        self.ctrl_writer.write({"cmd": "stop"})
    
    def is_alive(self) -> bool:
        """Check if inference process is still running."""
        return self.process is not None and self.process.is_alive()


def start_inference_process(
    model_name_or_path: str,
    device: str = "cuda",
    dataset_id: str = "",
    chunk_size: int = -1,
    # Sim mode params
    obs_shm_size_mb: int = 64,
    # Real mode params
    device_shm_names: List[str] = None,
    device_configs: List[dict] = None,
    robot_module_name: str = None,
    sync_buffer_maxlen: int = 100,
    sync_max_tolerance_s: float = 0.05,
    # Common params
    action_shm_size_mb: int = 32,
    ctrl_shm_size_mb: int = 1,
    shm_prefix: str = "infer",
) -> InferenceContext:
    """
    Start the inference subprocess and create SHM channels.
    
    Args:
        model_name_or_path: Path to model checkpoint.
        device: Device for inference.
        dataset_id: Dataset ID for normalizer loading.
        chunk_size: Passed through to policy loading / MetaPolicy for compatibility;
                    MetaPolicy no longer truncates chunks (action_manager handles it).
        obs_shm_size_mb: Size of obs SHM in MB (sim mode only).
        device_shm_names: Device SHM names (real mode). If provided, enables real mode.
        device_configs: Device config dicts (real mode). Used to build composite obs2meta.
        robot_module_name: Robot module name for obs2meta (real mode, legacy fallback).
        sync_buffer_maxlen: Synchronizer buffer size (real mode).
        sync_max_tolerance_s: Synchronizer tolerance (real mode).
        action_shm_size_mb: Size of action SHM in MB.
        ctrl_shm_size_mb: Size of control SHM in MB.
        shm_prefix: Prefix for SHM channel names.
    
    Returns:
        InferenceContext for use by action_manager and main process.
    """
    is_real_mode = device_shm_names is not None and len(device_shm_names) > 0
    mode_str = "REAL" if is_real_mode else "SIM"
    
    logger.info(f"🚀 Starting inference process ({mode_str} mode)")
    logger.info(f"   Model: {model_name_or_path}")
    logger.info(f"   Device: {device}")
    
    # Generate SHM names
    pid = os.getpid()
    obs_shm_name = f"{shm_prefix}_{pid}_obs" if not is_real_mode else None
    action_shm_name = f"{shm_prefix}_{pid}_action"
    ctrl_shm_name = f"{shm_prefix}_{pid}_ctrl"
    
    all_shm_names = [action_shm_name, ctrl_shm_name]
    if obs_shm_name:
        all_shm_names.append(obs_shm_name)
    
    # Create SHM channels in main process
    obs_writer = None
    if not is_real_mode:
        obs_writer = SharedMemoryChannel(
            name=obs_shm_name,
            max_size_mb=obs_shm_size_mb,
            is_writer=True,
        )
    
    ctrl_writer = SharedMemoryChannel(
        name=ctrl_shm_name,
        max_size_mb=ctrl_shm_size_mb,
        is_writer=True,
    )
    
    # Start inference subprocess
    worker_kwargs = dict(
        model_name_or_path=model_name_or_path,
        action_shm_name=action_shm_name,
        ctrl_shm_name=ctrl_shm_name,
        action_shm_size_mb=action_shm_size_mb,
        device=device,
        dataset_id=dataset_id,
        chunk_size=chunk_size,
    )
    
    if is_real_mode:
        worker_kwargs.update(
            device_shm_names=device_shm_names,
            device_configs=device_configs,
            robot_module_name=robot_module_name,
            sync_buffer_maxlen=sync_buffer_maxlen,
            sync_max_tolerance_s=sync_max_tolerance_s,
        )
    else:
        worker_kwargs["obs_shm_name"] = obs_shm_name
    
    # Use 'spawn' context for the inference process: CUDA cannot be
    # re-initialized in a forked subprocess (the main process may have
    # already touched CUDA via set_seed / torch.cuda.is_available).
    # Only the inference process needs spawn; device processes stay on fork.
    ctx = mp.get_context("spawn")
    worker_ready_event = ctx.Event()
    worker_kwargs["worker_ready_event"] = worker_ready_event
    process = ctx.Process(
        target=inference_worker,
        kwargs=worker_kwargs,
        daemon=True,
    )
    process.start()
    logger.info(f"   Inference process started (PID={process.pid})")
    
    # Connect to action SHM (created by inference process)
    action_reader = None
    for attempt in range(1800):  # Up to ~180s (heavy policy load / slow disk)
        try:
            action_reader = SharedMemoryChannel(
                name=action_shm_name,
                is_writer=False,
                timeout=0.1,
            )
            break
        except (TimeoutError, FileNotFoundError):
            if not process.is_alive():
                # Cleanup on failure
                if obs_writer is not None:
                    obs_writer.destroy()
                ctrl_writer.destroy()
                cleanup_all_shm(all_shm_names)
                raise RuntimeError("Inference process died during startup")
            time.sleep(0.1)
    else:
        if obs_writer is not None:
            obs_writer.destroy()
        ctrl_writer.destroy()
        cleanup_all_shm(all_shm_names)
        raise RuntimeError("Timeout waiting for inference process to create action SHM")

    # Wait until worker has bound get_obs and entered the main loop (avoids racing triggers).
    if not worker_ready_event.wait(timeout=300.0):
        if not process.is_alive():
            if obs_writer is not None:
                obs_writer.destroy()
            ctrl_writer.destroy()
            cleanup_all_shm(all_shm_names)
            raise RuntimeError("Inference process died before signaling ready")
        if obs_writer is not None:
            obs_writer.destroy()
        ctrl_writer.destroy()
        cleanup_all_shm(all_shm_names)
        raise RuntimeError(
            "Timeout waiting for inference worker ready signal (policy load / SHM init too slow?)"
        )
    
    logger.info(f"✓ Inference process ready ({mode_str} mode)")
    
    return InferenceContext(
        process=process,
        obs_writer=obs_writer,
        action_reader=action_reader,
        ctrl_writer=ctrl_writer,
        shm_names=tuple(all_shm_names),
    )


def stop_inference_process(ctx: InferenceContext) -> None:
    """
    Stop the inference process and cleanup SHM.
    """
    if ctx is None:
        return
    
    logger.info("Stopping inference process...")
    
    # Send stop command
    try:
        ctx.send_stop()
    except Exception:
        pass
    
    # Wait for process to finish
    if ctx.process is not None and ctx.process.is_alive():
        ctx.process.join(timeout=5.0)
        if ctx.process.is_alive():
            logger.warning("Inference process did not stop gracefully, terminating...")
            ctx.process.terminate()
            ctx.process.join(timeout=2.0)
    
    # Cleanup SHM
    if ctx.obs_writer is not None:
        ctx.obs_writer.destroy()
    
    if ctx.ctrl_writer is not None:
        ctx.ctrl_writer.destroy()
    
    if ctx.action_reader is not None:
        ctx.action_reader.destroy()
    
    # Clean up any residual SHM
    cleanup_all_shm(list(ctx.shm_names))
    
    logger.info("✓ Inference process stopped")
