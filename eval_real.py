#!/usr/bin/env python
"""
eval_real.py

Real robot evaluation with policy inference:
1. Start robot/camera devices in subprocesses (like collect_data.py)
2. Run inference process to produce action chunks
3. Use action manager to select and publish actions
4. After start: Enter pauses policy publishing; Enter again resumes (same idea as collect_data.py)

Usage:
    # Local model evaluation
    python eval_real.py -r robot/so101_follower -m /path/to/checkpoint
    
    # Remote policy server evaluation
    python eval_real.py -r robot/so101_follower -m localhost:5000
    
    # Dummy policy for testing pipeline
    python eval_real.py -r robot/so101_follower -m __dummy-random_7x16

    # With -o: manifest + optional dataset (--save-as-dataset) and/or mosaic MP4 under video/
    # (recorder subprocess only; main process does not read device SHM for recording).
"""

# CRITICAL: Import shm_utils FIRST to patch resource_tracker before multiprocessing
import deploy.shm_utils  # noqa: F401 - applies _fix_resource_tracker() on import

import os
import traceback
import time
import numpy as np
from loguru import logger

from data_utils.utils import set_seed
from deploy.base import is_camera_config
from deploy.shm_utils import SharedMemoryChannel, cleanup_all_shm
from deploy.utils import (
    RateLimiter,
    get_shm_name,
    load_device_configs,
    start_devices,
)
from deploy.action_manager import load_action_manager
from deploy.inference import (
    _extract_step_action,
    start_inference_process,
    stop_inference_process,
)
from deploy.runtime import (
    EvalRealRuntime,
    KBHit,
    eval_real_pause_menu_step,
    setup_eval_real_output_dir,
    resolve_eval_real_output_dir,
    try_enter_eval_real_pause_menu,
    validate_eval_real_save_as_dataset,
)
from deploy.visualizer.base import start_all_visualizers
from configs.loader import ConfigLoader


def parse_param():
    """
    Parse command line arguments using simple argparse.
    
    Returns:
        args: Parsed arguments namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a policy model on real robot')
    
    # Same primary flags as collect_data.py (-r / --robot); long --robot_config kept as alias.
    parser.add_argument(
        '-r', '--robot', '--robot-config', '--robot_config',
        type=str,
        default='robot/so101_follower',
        help="Robot config (name under configs/robot or path to yaml; same as collect_data.py -r/--robot)",
    )
    parser.add_argument('-pr', '--publish_rate', type=float, default=25,
                       help='Action publishing rate (Hz)')
    parser.add_argument('-sr', '--sensing_rate', type=float, default=20,
                       help='Sensing rate (Hz) - used by synchronizer')
    
    # Model arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    
    # Direct checkpoint loading
    parser.add_argument('-m', '--model_name_or_path', type=str, 
                       default='localhost:5000',
                       help='Path to model checkpoint OR server address (host:port) for remote policy server')
    parser.add_argument(
        '-o', '--output_dir', type=str, default='',
        help='If set: create dir, write eval_real_manifest.json, and spawn a dedicated process that '
             'records synchronized device data (using DataSaver) plus one mosaic MP4 per episode '
             '(video/real_eval_episode_XXXX.mp4) from device SHM only; main process does not read '
             'device SHM for recording, so control / inference path stays unchanged.',
    )
    parser.add_argument(
        '-i', '-s', '--episode_id', '--start-idx', '--episode',
        type=int,
        default=-1,
        dest='episode_id',
        help='Starting episode index for eval recording (-1 = append after existing episodes). '
             'Aliases: -s / --start-idx / --episode (same as collect_data.py).',
    )
    parser.add_argument(
        '--save-as-dataset',
        type=str,
        default=None,
        metavar='FORMAT',
        help='With -o: save synced data in this format (lerobotv21, lerobotv30, hdf5). '
             'If -o is set but this is omitted or empty, only mosaic MP4 under video/ is written.',
    )
    parser.add_argument(
        '--task',
        type=str,
        default='',
        help='Optional task description stored in eval recording dataset.',
    )

    # Action manager
    parser.add_argument('-am', '--action_manager', type=str, default='basic',
                       help='Action manager config name or path to config file')
    
    # Visualization
    parser.add_argument('-v', '--visualize', action='store_true',
                       help='Enable visualization (if robot module provides Visualizer)')

    # Optional: leave home before policy starts (abs EE hold-at-home experiments).
    parser.add_argument(
        '--prestart_dataset',
        type=str,
        default='',
        help='LeRobot dataset root with data/**/episode_XXXXXX.parquet (for --prestart_frame).',
    )
    parser.add_argument(
        '--prestart_episode',
        type=int,
        default=0,
        help='Episode index for --prestart_frame (default: 0).',
    )
    parser.add_argument(
        '--prestart_frame',
        type=int,
        default=-1,
        help='If >=0: before Enter, interpolate joints to observation.qpos[frame] then settle. '
             'Use to escape abs-EE home hold (e.g. 150).',
    )
    parser.add_argument(
        '--prestart_move_s',
        type=float,
        default=3.0,
        help='Seconds to interpolate home→prestart qpos (default: 3).',
    )
    parser.add_argument(
        '--prestart_settle_s',
        type=float,
        default=1.0,
        help='Hold prestart qpos after move (default: 1).',
    )
    
    # Parse arguments (allow unknown for dotted overrides)
    args, unknown = parser.parse_known_args()
    validate_eval_real_save_as_dataset(args, parser)

    # Store unknown args for ConfigLoader to process
    setattr(args, 'unknown_args', unknown)
    return args


def _load_prestart_qpos(dataset_root: str, episode: int, frame: int) -> np.ndarray:
    """Load observation.qpos[frame] from a LeRobot v2.1 episode parquet."""
    from pathlib import Path
    import pyarrow.parquet as pq

    root = Path(dataset_root)
    matches = sorted(root.glob(f"data/**/episode_{episode:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No episode_{episode:06d}.parquet under {root}/data"
        )
    path = matches[0]
    table = pq.read_table(path, columns=["observation.qpos"])
    n = table.num_rows
    if frame < 0 or frame >= n:
        raise IndexError(f"prestart_frame={frame} out of range for {path} (T={n})")
    q = np.asarray(table.column("observation.qpos")[frame].as_py(), dtype=np.float64).reshape(-1)
    if q.size < 7:
        raise ValueError(f"observation.qpos dim={q.size} < 7 in {path}")
    return q[:7].copy()


def _prestart_move_away_from_home(
    control_shm: SharedMemoryChannel,
    target_qpos: np.ndarray,
    *,
    publish_hz: float,
    move_s: float,
    settle_s: float,
) -> None:
    """Publish interpolated joint abs commands so the arm leaves home before policy."""
    target = np.asarray(target_qpos, dtype=np.float64).reshape(-1)[:7]
    # Connect always homes to zeros; start interpolate from that.
    q0 = np.zeros(7, dtype=np.float64)
    q0[6] = float(target[6])  # open/close with target gripper during move
    n_move = max(1, int(round(float(move_s) * float(publish_hz))))
    n_hold = max(1, int(round(float(settle_s) * float(publish_hz))))
    dt = 1.0 / max(float(publish_hz), 1.0)

    logger.info(
        "[Prestart] Moving away from home → qpos={} (deg={}, g={:.3f}) over {:.1f}s + settle {:.1f}s",
        np.array2string(target[:6], precision=3),
        np.array2string(np.rad2deg(target[:6]), precision=1),
        float(target[6]),
        move_s,
        settle_s,
    )
    t0 = time.perf_counter()
    for i in range(n_move):
        a = (i + 1) / n_move
        q = (1.0 - a) * q0 + a * target
        control_shm.write({"action": q.astype(np.float64), "timestamp": time.perf_counter()})
        target_t = t0 + (i + 1) * dt
        slack = target_t - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
    for i in range(n_hold):
        control_shm.write({"action": target.copy(), "timestamp": time.perf_counter()})
        time.sleep(dt)
    logger.info("[Prestart] At non-home pose. Press Enter when ready to start policy.")


def main():
    set_seed(0)
    args = parse_param()
    args.is_training = False

    output_dir = resolve_eval_real_output_dir(args)
    rt = EvalRealRuntime()
    rt.register_exit_hooks()

    # Create shared ConfigLoader instance
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, 'unknown_args', []))
    
    # Same loader as collect_data; eval forces policy_control_shm on robots.
    robot_configs, all_shm_names, _teleop_configs, visualizer_configs = load_device_configs(
        args,
        getattr(args, "unknown_args", []),
        cfg_loader=cfg_loader,
        force_robot_control_shm="policy_control_shm",
    )
    
    # Clean orphaned SHM
    shm_to_clean = all_shm_names + ['policy_control_shm']
    cleanup_all_shm(shm_to_clean)
    logger.info("Cleaned up orphaned SHM segments.")
    
    # Create policy control SHM (main process writes actions here)
    control_shm = SharedMemoryChannel(name='policy_control_shm', max_size_mb=1, is_writer=True)
    rt.all_shm.append(control_shm)
    logger.info("Created policy_control_shm for action publishing.")

    def send_robot_reset() -> None:
        """Reset policy state and command robot subprocess to go home."""
        control_shm.write({
            "cmd": "reset",
            "go_home": True,
            "timestamp": time.perf_counter(),
        })
        logger.info("[Pause] Sent go_home through policy_control_shm.")
    
    # Start all devices in subprocesses
    logger.info("Starting all devices...")
    device_procs = start_devices(robot_configs)
    rt.all_procs.extend(device_procs)
    
    # Start visualizer subprocesses if requested
    viz_procs = []
    if args.visualize:
        viz_procs = start_all_visualizers(
            device_configs=robot_configs,
            get_shm_name_func=get_shm_name,
            is_camera_config_func=is_camera_config,
            visualizer_configs=visualizer_configs,
        )
        rt.all_procs.extend(viz_procs)

    time.sleep(1.0)

    # Start inference process via deploy/inference.py (trigger-based)
    inference_ctx = start_inference_process(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        device_shm_names=all_shm_names,
        device_configs=robot_configs,
    )
    rt.all_procs.append(inference_ctx.process)
    
    # Load action manager and bind to inference context
    logger.info("Loading action manager: {}", args.action_manager)
    try:
        manager_cfg, manager_cfg_path = cfg_loader.load_manager(args.action_manager)
        logger.info("Loaded action manager config from: {}", manager_cfg_path)
    except Exception as e:
        logger.info("Using legacy loading for: {}", args.action_manager)
        manager_cfg = {'manager_name': args.action_manager}
    
    action_manager = load_action_manager(config=manager_cfg)
    action_manager.set_inference_context(inference_ctx)
    logger.info("ActionManager: {} (bound to InferenceContext)", action_manager.__class__.__name__)

    rt.start_session_recorder(
        output_dir=output_dir,
        device_shm_names=list(all_shm_names),
        dataset_format=getattr(args, "save_as_dataset", None),
        task=args.task,
        episode_id=args.episode_id,
        sensing_rate=float(getattr(args, "sensing_rate", 20.0)),
        initial_recording_active=False,
        wait_until_ready=False,
    )

    # Inference subprocess is already in its main loop (start_inference_process waits for
    # worker ready after policy load + device SHM attach + action/ctrl channels).
    # Recorder child logs SHM connect/skips asynchronously; brief pause so that noise
    # appears before the Enter prompt (otherwise users miss the instruction).
    time.sleep(2.0)

    if int(getattr(args, "prestart_frame", -1)) >= 0:
        ds = str(getattr(args, "prestart_dataset", "") or "").strip()
        if not ds:
            raise ValueError("--prestart_frame requires --prestart_dataset")
        pre_q = _load_prestart_qpos(
            ds,
            int(args.prestart_episode),
            int(args.prestart_frame),
        )
        _prestart_move_away_from_home(
            control_shm,
            pre_q,
            publish_hz=float(args.publish_rate),
            move_s=float(args.prestart_move_s),
            settle_s=float(args.prestart_settle_s),
        )

    kb = None

    try:
        kb = KBHit()
        kb.set_curses_term()

        logger.info(
            "\n{}\n  Press Enter to start the control loop\n{}\n",
            "=" * 60,
            "=" * 60,
        )
        while kb.get_input() is None:
            time.sleep(0.001)
        while kb.get_input() is not None:
            pass
        # Must wait for ack: a fire-and-forget post leaves a stale "resumed" ack that
        # desyncs later pause/save send_recorder_cmd calls (false "0 frames" saves).
        rt.send_recorder_cmd("resume_recording")

        # Main control loop
        logger.info("="*60)
        logger.info(
            "[Main Control Loop] Started. Publish rate: {} Hz. "
            "Enter=pause (>>> menu: save/reset then plain Enter=resume), Ctrl+C=stop.",
            args.publish_rate,
        )
        logger.info("="*60)

        rate_limiter = RateLimiter()
        action_count = 0
        last_log_time = time.perf_counter()
        paused = False

        while True:
            if not paused and try_enter_eval_real_pause_menu(kb, rt):
                paused = True

            if paused:
                outcome = eval_real_pause_menu_step(
                    kb,
                    rt,
                    action_manager,
                    on_reset=send_robot_reset,
                )
                if outcome == "idle":
                    time.sleep(0.01)
                    continue
                if outcome == "resumed":
                    paused = False
                    continue
                if outcome == "exit":
                    break
                continue

            t = time.perf_counter()

            # select_action() handles everything:
            # 1. Poll for new action chunks from inference subprocess
            # 2. If buffer empty or should_infer(): send trigger
            # 3. If buffer empty: block-wait for inference result
            # 4. Return single-step action, auto-increment t
            step = action_manager.select_action()
            # One step from BasicActionManager: np.ndarray(dtype=object) of MetaAction dicts
            # (same layout as MetaPolicy.inference); see deserialize_mact_list / _extract_step_action.
            action_arr = _extract_step_action(step)

            if action_arr is not None:
                action_arr = np.asarray(action_arr)
                if action_arr.size > 0:
                    control_shm.write({
                        'action': action_arr,
                        'timestamp': t,
                    })
                    action_count += 1
            
            if t - last_log_time > 5.0:
                logger.info("[Control Loop] Published {} actions in last 5s ({:.1f} Hz)", 
                           action_count, action_count / 5.0)
                action_count = 0
                last_log_time = t
            
            rate_limiter.sleep(args.publish_rate)

    except KeyboardInterrupt:
        logger.info("[Main Control Loop] Exit by KeyboardInterrupt Ctrl+C")
    except Exception as e:
        logger.info("[Main Control Loop] Error: {}", e)
        traceback.print_exc()
    finally:
        if kb is not None:
            try:
                kb.set_normal_term()
            except Exception:
                pass
        stop_inference_process(inference_ctx)
        rt.cleanup_all()
        cleanup_all_shm(shm_to_clean)


if __name__ == '__main__':
    main()
