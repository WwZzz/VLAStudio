#!/usr/bin/env python
"""
vlastudio collect

Data collection for teleoperation:
1. Start all devices (teleop optional, robot + cameras required)
2. Read from device SHM buffers, sync by timestamp (nearest-neighbor)
3. Enter to START recording, Enter to STOP, Enter to SAVE (or any input + Enter to DISCARD)
4. Save each episode as either:
   - LeRobotDataset v2.1 (default, supports episode overwriting)
   - LeRobotDataset v3.0 (append only)
   - HDF5 file (legacy)
5. Optionally auto-generate a task config YAML for training

Usage:
    vlastudio collect --runtime current --robot configs/robot/bi_so101_follower.yaml [--teleop configs/teleop/bi_so101_leader.yaml] -o data/teleop_recordings
    
    # With visualization (if robot module provides a Visualizer class):
    vlastudio collect --runtime current --robot configs/robot/so101_sim_qpos.yaml --teleop configs/teleop/so101_leader.yaml --visualize
    
    # Overwrite episode 0 (lerobotv21 only):
    vlastudio collect --runtime current --robot <...> -o data/teleop_recordings --start-idx 0
    
    # Save as HDF5 (legacy):
    vlastudio collect --runtime current --robot <...> --teleop <...> -o data/teleop_recordings --dataset-format hdf5
    
    # Auto-generate task config after collection:
    vlastudio collect --runtime current --robot <...> -o data/teleop_recordings -g local.my_task
    # Or with full path:
    vlastudio collect --runtime current --robot <...> -o data/teleop_recordings -g configs/task/my_task.yaml
    
    # Enable streaming mode (faster save, writes to disk in real-time):
    vlastudio collect --runtime current --robot <...> -o data/teleop_recordings --streaming
"""

# CRITICAL: Import shm_utils FIRST to patch resource_tracker before multiprocessing
import vlastudio.deploy.shm_utils; from vlastudio import deploy  # noqa: F401 - applies _fix_resource_tracker() on import

import argparse
from contextlib import ExitStack
from pathlib import Path
import signal
import time
from typing import List, Optional, Tuple
from loguru import logger
from vlastudio.deploy.base import create_enrich_saved_frame_func, is_camera_config
from vlastudio.deploy.runtime import KBHit
from vlastudio.deploy.shm_utils import SharedMemoryChannel, SharedMemoryDataSynchronizer, cleanup_all_shm
from vlastudio.deploy.utils import (
    RateLimiter,
    get_shm_name,
    load_device_configs,
    start_devices,
)
from vlastudio.deploy.visualizer.base import start_all_visualizers, stop_all_visualizers
from vlastudio.data_utils.data_saver import create_data_saver, generate_task_config
from vlastudio.utils.extensions import resolve
from vlastudio.webui.bridge import keyboard, phase


def _stop_device_processes(workers):
    with ExitStack() as signals:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(sig, signal.SIG_IGN)
            signals.callback(signal.signal, sig, previous)
        _join_devices(workers)


def _join_devices(workers):
    # A device may own a multi-stage shutdown and require an unbounded join.
    failures = []
    for _, process, _ in workers:
        if process.is_alive():
            process.terminate()
    for name, process, timeout in workers:
        if timeout is None:
            logger.info("Waiting for {} to confirm shutdown...", name)
        process.join(timeout=timeout)
        if timeout is None:
            if process.exitcode != 0:
                failures.append(f"{name}: exit {process.exitcode}")
        elif process.is_alive():
            logger.warning("Device {} did not exit within {}s, killing...", name, timeout)
            process.kill()
            process.join(timeout=1.0)
    if failures:
        raise RuntimeError("Device shutdown unconfirmed: " + ", ".join(failures))


def main():
    parser = argparse.ArgumentParser(description="Collect teleoperation data with timestamp sync")
    parser.add_argument("-r", "--robot", type=str, required=True, 
                        help="Robot config (name or path, e.g., 'so101_sim_qpos' or 'configs/robot/xxx.yaml')")
    parser.add_argument("-t", "--teleop", type=str, default="", 
                        help="Teleop config (name or path, optional)")
    parser.add_argument("-o", "--output-dir", type=str, default="data/teleop_recordings", 
                        help="Output directory for episodes / dataset")
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="lerobotv21",
        choices=["lerobotv21", "lerobotv30", "hdf5"],
        help="Dataset format to save. Default: lerobotv21 (supports overwriting episodes)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Optional natural-language task description stored in dataset (default: empty string).",
    )
    parser.add_argument("--hdf5-compression", choices=("gzip", "lzf"), default="gzip",
                        help="Lossless HDF5 compression (lzf favors throughput).")
    parser.add_argument("-f", "--frequency", type=float, default=25.0, 
                        help="Target max Hz; actual = min(-f, slowest sensor)")
    parser.add_argument(
        "-s", "--start-idx", "--episode",
        type=int,
        default=None,
        dest="start_idx",
        help="Starting episode index to continue/overwrite from "
             "(lerobotv21/HDF5 can overwrite; lerobotv30 append-only). "
             "Alias: --episode",
    )
    parser.add_argument("-v", "--visualize", action="store_true", 
                        help="Enable visualization (if robot module provides Visualizer)")
    parser.add_argument(
        "--no-record-teleop",
        action="store_true",
        help="Run teleop for control only: exclude teleop from sync and saved observations "
             "(avoids recording FPS being gated by sparse teleop SHM updates).",
    )
    parser.add_argument(
        "-g", "--gen-config",
        type=str,
        default=None,
        metavar="NAME",
        help="Generate task config YAML. Supports: 'local.my_task' -> configs/task/local/my_task.yaml, "
             "'xxx.task' -> configs/task/xxx/task.yaml, or full path. Won't overwrite existing files.",
    )
    sync_args = parser.add_mutually_exclusive_group()
    sync_args.add_argument("--sync-config", metavar="YAML",
                           help="Recording groups: required channels set the frame rate; missing optional samples are zero-filled.")
    sync_args.add_argument("--sample-quest-actions", action="store_true",
                           help="Use the single-arm Quest zero-fill recording preset.")
    args, unknown_args = parser.parse_known_args()

    all_configs, all_shm_names, teleop_configs, visualizer_configs = load_device_configs(
        args, unknown_args
    )
    preset_name = None
    sync_path = args.sync_config
    if args.sample_quest_actions and not args.no_record_teleop:
        cfg = teleop_configs[0] if len(teleop_configs) == 1 else {}
        opts = cfg.get("args", {})
        if (cfg.get("type") != "vlastudio.deploy.teleoperator.quest3.Quest3Teleop"
                or opts.get("arm_mode") not in ("left", "right") or opts.get("mode") != "rel_ee"
                or opts.get("publish_thumbsticks")):
            parser.error("--sample-quest-actions requires a single-arm relative-EE Quest profile")
        sync_path = Path(__file__).resolve().parents[1] / "configs/recording/rebot_b601rs_head.yaml"
        preset_name = get_shm_name(cfg)
    optional_inputs = {}
    if sync_path:
        from vlastudio.deploy.recording_sync import load_optional_inputs
        recorded = set(all_shm_names)
        if args.no_record_teleop:
            recorded -= {get_shm_name(cfg) for cfg in teleop_configs}
        optional_inputs = load_optional_inputs(sync_path, recorded, single_quest_name=preset_name)

    # Clean orphaned SHM
    cleanup_all_shm(all_shm_names)
    logger.info("Cleaned up orphaned SHM segments.")

    # Start all devices
    logger.info("Starting all devices...")
    workers = []
    with ExitStack() as devices:
        def on_term(signum, frame):
            raise KeyboardInterrupt
        old_term = signal.signal(signal.SIGTERM, on_term)
        devices.callback(signal.signal, signal.SIGTERM, old_term)
        devices.callback(_stop_device_processes, workers)
        for cfg in all_configs:
            timeout = getattr(resolve(cfg["type"]), "process_shutdown_timeout_s", 8.0)
            process, = start_devices([cfg])
            workers.append((get_shm_name(cfg), process, timeout))
        _collect(args, all_configs, all_shm_names, teleop_configs, visualizer_configs, workers, devices, optional_inputs)


def _collect(args, all_configs, all_shm_names, teleop_configs, visualizer_configs, workers, devices, optional_inputs):
    time.sleep(1.0)
    
    # Start visualizer subprocesses if requested
    viz_procs = []
    if args.visualize:
        viz_procs = start_all_visualizers(
            device_configs=all_configs,
            get_shm_name_func=get_shm_name,
            is_camera_config_func=is_camera_config,
            visualizer_configs=visualizer_configs,
        )

    # Connect to all SHM channels
    shm_channels: List[Tuple[str, Optional[SharedMemoryChannel]]] = []
    for cfg in all_configs:
        name = get_shm_name(cfg)
        try:
            # Quest creates its channel after the headset joins the VR session.
            timeout = 10.0 + float(cfg.get("args", {}).get("connect_timeout", 0.0))
            ch = SharedMemoryChannel(name, is_writer=False, timeout=timeout)
            shm_channels.append((name, ch))
            logger.info("Connected: {}", name)
        except Exception as e:
            logger.warning("Failed {}: {}", name, e)
            shm_channels.append((name, None))

    # Create synchronizer
    valid_channels = [(n, ch) for n, ch in shm_channels if ch is not None]
    if args.sync_config or optional_inputs or any(timeout is None for _, _, timeout in workers):
        missing = [name for name, channel in shm_channels if channel is None]
        if missing:
            raise RuntimeError(f"Required device channels unavailable: {missing}")
    teleop_names = {get_shm_name(cfg) for cfg in teleop_configs}
    if args.no_record_teleop and teleop_names:
        excluded = [n for n, _ in valid_channels if n in teleop_names]
        valid_channels = [(n, ch) for n, ch in valid_channels if n not in teleop_names]
        if excluded:
            logger.info(
                "Not recording teleop (control-only): excluded from sync/save: {}",
                ", ".join(excluded),
            )
    if not valid_channels:
        logger.error("No SHM channels connected. Exiting.")
        return

    sync_options = dict(shm_channels=valid_channels, buffer_maxlen=200, max_tolerance_s=0.05)
    if optional_inputs:
        from vlastudio.deploy.recording_sync import RecordingSynchronizer
        teleop = get_shm_name(teleop_configs[0]) if teleop_configs else None
        sync = RecordingSynchronizer(optional=optional_inputs, teleop_name=teleop, **sync_options)
        logger.info("Optional recording inputs (zero-filled when missing): {}", sorted(optional_inputs))
    else:
        sync = SharedMemoryDataSynchronizer(**sync_options)

    # Create data saver (it will count existing episodes automatically).
    # Per-device enrich_saved_frame hooks run at stream/save time (e.g. AliciaD state_ee FK).
    enrich_saved_frame = create_enrich_saved_frame_func(all_configs)
    data_saver = create_data_saver(
        format=args.dataset_format,
        hdf5_compression=args.hdf5_compression,
        output_dir=args.output_dir,
        fps=int(round(args.frequency)),
        task=args.task,
        enrich_saved_frame=enrich_saved_frame,
    )
    
    # Determine starting episode index
    if args.start_idx is not None:
        episode_idx = args.start_idx
    else:
        episode_idx = data_saver.episode_count
    
    if data_saver.episode_count > 0:
        logger.info("Found existing dataset with {} episodes, starting from episode {}", 
                    data_saver.episode_count, episode_idx)
    
    kb = keyboard(KBHit)
    kb.set_curses_term()
    
    teleop_name = (
        None
        if args.no_record_teleop
        else (get_shm_name(teleop_configs[0]) if teleop_configs else None)
    )
    device_names_list = [n for n, _ in valid_channels]

    def read_input():
        for name, process, _ in workers:
            if process.exitcode is not None:
                raise RuntimeError(f"Device {name} exited during collection (code {process.exitcode})")
        return kb.get_input()

    try:
        while True:
            # Wait for Enter to START
            logger.info("Press Enter to START episode {}...", episode_idx)
            phase("ready")
            while read_input() is None:
                sync.read_and_buffer()
                time.sleep(0.01)
            # Consume any extra input
            while read_input() is not None:
                pass

            # Start recording episode
            actual_idx = data_saver.start_episode(
                episode_idx=episode_idx,
                device_names=device_names_list,
                teleop_device=teleop_name,
            )
            logger.info("Recording episode {}. Press Enter to STOP...", actual_idx)
            phase("recording")
            
            rate_limiter = RateLimiter()

            def stop_requested() -> bool:
                return read_input() is not None

            recording_start_time = time.perf_counter()
            sync.begin_episode(recording_start_time)
            frame_count = 0
            last_debug_time = recording_start_time
            debug_interval = 1.0  # Sync bottleneck stats every 1s
            # Aggregate bottleneck hits over the episode
            bottleneck_hits: dict[str, int] = {}
            slow_sync_count = 0  # sync_wait > 50ms
            sum_sync_ms = 0.0
            
            while not stop_requested():
                loop_start = time.perf_counter()
                frame = sync.get_synced_frame_blocking(stop_check=stop_requested, debug=False)
                sync_time = time.perf_counter()
                
                if frame is None:
                    break
                
                # Use actual recording time as sync reference
                ref_ts = time.perf_counter()
                frame["_sync_timestamp"] = ref_ts
                
                # Add frame using unified interface (non-blocking for streaming savers)
                data_saver.add_frame(frame)
                frame_count += 1

                sync_delay_ms = (sync_time - loop_start) * 1000
                sum_sync_ms += sync_delay_ms
                stats = getattr(sync, "last_sync_stats", {}) or {}
                bn = stats.get("bottleneck")
                if bn:
                    bottleneck_hits[str(bn)] = bottleneck_hits.get(str(bn), 0) + 1
                if sync_delay_ms > 50:
                    slow_sync_count += 1
                    wait_map = stats.get("wait_ms_per_device") or {}
                    age_map = stats.get("age_ms") or {}
                    wait_str = ", ".join(f"{k}={v:.0f}ms" for k, v in sorted(wait_map.items(), key=lambda kv: -kv[1]))
                    age_str = ", ".join(f"{k}={v:.0f}ms" for k, v in sorted(age_map.items(), key=lambda kv: -kv[1]))
                    logger.warning(
                        "[SyncSlow] frame={} sync_wait={:.1f}ms bottleneck={} | wait_share=[{}] | sample_age=[{}]",
                        frame_count, sync_delay_ms, bn, wait_str or "-", age_str or "-",
                    )
                
                # Periodic summary
                if ref_ts - last_debug_time >= debug_interval:
                    elapsed_since_start = ref_ts - recording_start_time
                    current_fps = frame_count / elapsed_since_start if elapsed_since_start > 0 else 0
                    avg_sync = sum_sync_ms / frame_count if frame_count else 0
                    top_bn = sorted(bottleneck_hits.items(), key=lambda kv: -kv[1])[:4]
                    bn_str = ", ".join(f"{k}:{v}" for k, v in top_bn) or "-"
                    wait_map = stats.get("wait_ms_per_device") or {}
                    age_map = stats.get("age_ms") or {}
                    wait_str = ", ".join(f"{k}={v:.0f}ms" for k, v in sorted(wait_map.items(), key=lambda kv: -kv[1]))
                    age_str = ", ".join(f"{k}={v:.0f}ms" for k, v in sorted(age_map.items(), key=lambda kv: -kv[1]))
                    logger.info(
                        "Frame {}: fps={:.1f}, sync_wait={:.1f}ms (avg={:.1f}ms, slow>50ms={}), "
                        "bottleneck_hits=[{}], last_wait=[{}], age=[{}]",
                        frame_count, current_fps, sync_delay_ms, avg_sync, slow_sync_count,
                        bn_str, wait_str or "-", age_str or "-",
                    )
                    last_debug_time = ref_ts
                
                rate_limiter.sleep(args.frequency)

            # Consume stop Enter
            while read_input() is not None:
                pass

            recording_end_time = time.perf_counter()
            
            # Show actual frame rate (use actual wall-clock time)
            if frame_count > 0:
                actual_elapsed = recording_end_time - recording_start_time
                actual_fps = frame_count / actual_elapsed if actual_elapsed > 0 else 0
                avg_sync = sum_sync_ms / frame_count
                top_bn = sorted(bottleneck_hits.items(), key=lambda kv: -kv[1])
                bn_str = ", ".join(f"{k}:{v}" for k, v in top_bn) or "-"
                logger.info(
                    "Episode {}: {} frames, {:.2f}s, {:.1f} Hz (target max {} Hz); "
                    "avg_sync_wait={:.1f}ms, slow_frames={}/{}, bottleneck_hits=[{}]",
                    episode_idx, frame_count, actual_elapsed, actual_fps, args.frequency,
                    avg_sync, slow_sync_count, frame_count, bn_str,
                )

            # Save or discard
            logger.info("Save? (Enter=SAVE, any key+Enter=DISCARD)")
            phase("save")
            prompt = None
            while prompt is None:
                prompt = read_input()
                time.sleep(0.05)
            
            if len(prompt.strip()) == 0:
                # User wants to SAVE
                if frame_count == 0:
                    logger.warning("No frames to save.")
                    data_saver.finish_episode(save=False)
                else:
                    saved_frames = data_saver.finish_episode(save=True)
                    if saved_frames > 0:
                        logger.success("Saved {} frames to {} (episode {})", saved_frames, data_saver.dataset_path, actual_idx)
                        saved_any = True
                        episode_idx += 1
                    else:
                        logger.error("Failed to save episode {}", actual_idx)
            else:
                # User discarded
                data_saver.finish_episode(save=False)
                logger.info("Discarded.")

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        shutdown_error = None
        try:
            devices.close()
        except RuntimeError as exc:
            shutdown_error = exc
        kb.set_normal_term()
        
        # Stop visualizers
        stop_all_visualizers(viz_procs)
        
        for _, ch in shm_channels:
            if ch is not None:
                try:
                    ch.destroy()
                except Exception:
                    pass

        # Belt-and-suspenders: if teleop was killed mid-adb, restore Quest sleep from parent.
        try:
            from vlastudio.deploy.teleoperator.quest_keep_awake import restore_quest_proximity_best_effort
            restore_quest_proximity_best_effort()
        except Exception as e:
            logger.debug("Parent Quest proximity restore skipped: {}", e)
        
        # Finalize data saver (important for LeRobot to close parquet writers)
        if data_saver is not None:
            data_saver.finalize()
            if args.gen_config:
                generate_task_config(
                    data_saver=data_saver,
                    gen_config_spec=args.gen_config,
                    dataset_format=args.dataset_format,
                    task_name=args.task if args.task else None,
                )
        
        cleanup_all_shm(all_shm_names)
        if shutdown_error is not None:
            raise shutdown_error
        logger.info("Done.")


if __name__ == "__main__":
    main()
