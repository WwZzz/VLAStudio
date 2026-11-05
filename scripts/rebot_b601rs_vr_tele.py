"""Single B601-RS Quest launcher. --check and --smoke-seconds never enable/move."""
import argparse
import fcntl
import json
import multiprocessing
import signal
from datetime import datetime
from pathlib import Path
import time
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def stop_devices(rows, procs):
    """Allow arm parking to finish; never SIGKILL an active robot controller."""
    for p in procs:
        if p.is_alive():
            p.terminate()  # Rebot's handler requests cooperative shutdown only
    failures = []
    for row,p in zip(rows,procs):
        if 'RebotB601RS' in row['type']:
            begin = time.monotonic()
            warned = False
            while p.is_alive():
                p.join(timeout=.5)
                if time.monotonic()-begin>45 and not warned:
                    print('Arm shutdown still pending; controller will not be force-killed. Inspect arm and use local stop if needed.',flush=True)
                    warned = True
            if getattr(p,'exitcode',0) not in (0,None):
                failures.append(f"Arm process exit code {p.exitcode}; safe shutdown was not confirmed")
        else:
            p.join(timeout=4)
            if p.is_alive():
                p.kill()
                p.join()
    if failures:
        raise RuntimeError('; '.join(failures))


def check(config, seconds):
    from vlastudio.deploy.robot.rebot_b601rs import RebotB601RS
    robot = RebotB601RS(**dict(config[0]['args'],read_only=True))
    try:
        robot.connect()
        obs = robot.get_observation()
        solver = robot.ik_solver
        q = np.clip(obs['joint_angles'],solver.lower,solver.upper)
        solver.reset(q)
        result = solver.solve(solver.compute_fk(q))
        assert result[0],result
        test_q = q.copy()
        test_q[0] += .03
        result = solver.solve(solver.compute_fk(test_q))
        assert result[0],result
        ages = []
        start = time.monotonic()
        while time.monotonic()-start<seconds:
            obs = robot.get_observation()
            ages.append(float(time.monotonic()-min(obs['joint_timestamps'])))
            time.sleep(.01)
        print(json.dumps({'channel':robot.channel,'qpos':obs['qpos'].tolist(),
                          'motor_positions':obs['motor_positions'].tolist(),
                          'fk_xyz':solver.compute_fk(obs['joint_angles'])[:3,3].tolist(),
                          'ik_roundtrip_ok':True,'max_feedback_age_s':max(ages),
                          'motor_enable_sent':False,'motion_sent':False,'zero_written':False}),flush=True)
    finally:
        robot.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',default=str(ROOT/'src/vlastudio/configs/robot/rebot_b601rs_rel_ee.yaml'))
    parser.add_argument('--teleop',default=str(ROOT/'src/vlastudio/configs/teleop/quest3_rebot_b601rs_rel_ee.yaml'))
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--check-seconds',type=float,default=3.)
    parser.add_argument('--smoke-seconds',type=float,default=0.)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.robot).read_text())
    (Path.home()/'.cache').mkdir(exist_ok=True)
    with (Path.home()/'.cache/vlastudio-rebot-session.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.check:
            check(config,args.check_seconds)
            return
        from vlastudio.deploy.utils import load_device_configs,start_devices
        from vlastudio.deploy.shm_utils import cleanup_all_shm
        from vlastudio.utils.logger import logger
        rows,names,_,_ = load_device_configs(args,[])
        # Jetson forks its device workers. Each inherits this append-only sink;
        # retain warnings, faults and the final confirmed stop.
        log_path = ROOT/'logs'/('rebot_vr_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.log')
        logger.add(log_path, level='INFO',
                   filter=lambda record: record['level'].no >= 30 or record['extra'].get('stopped_confirmed', False),
                   format='{time:YYYY-MM-DD HH:mm:ss.SSS} {process.id} {level} {message}',
                   backtrace=False, diagnose=False)
        print(f'Session fault log: {log_path}', flush=True)
        if args.smoke_seconds:
            for row in rows:
                if 'RebotB601RS' in row['type']:
                    row['args']['read_only']=True
        cleanup_all_shm(names)
        procs=[]
        exit_event = multiprocessing.Event()
        for row in rows:
            if 'RebotB601RS' in row['type']:
                row['args']['exit_event'] = exit_event
        stop_requested = False
        def request_stop(signum,frame):
            nonlocal stop_requested
            stop_requested = True
        signal.signal(signal.SIGINT,request_stop)
        signal.signal(signal.SIGTERM,request_stop)
        try:
            procs=start_devices(rows)
            start=time.monotonic()
            print('Quest: https://<Jetson-IP>:8443 ; right grip=teleop, B=existing motor zero. Ctrl+C parks then disables.',flush=True)
            while not stop_requested:
                exited = [f"{row['type']} pid={p.pid} exit={p.exitcode}"
                          for row,p in zip(rows,procs) if not p.is_alive()]
                if exited:
                    detail = ', '.join(exited)
                    logger.error('Observed exited devices before stopping peers: {}', detail)
                    raise RuntimeError(f'Unexpected device exit: {detail}; see {log_path}')
                if args.smoke_seconds and time.monotonic()-start>=args.smoke_seconds:
                    print('Read-only VR startup check passed; no enable/motion/zero writes.',flush=True)
                    break
                time.sleep(.5)
        except KeyboardInterrupt:
            pass
        finally:
            exit_event.set()  # survives signals arriving before the child handler is installed
            try:
                stop_devices(rows,procs)
            finally:
                cleanup_all_shm(names)


if __name__=='__main__':
    main()
