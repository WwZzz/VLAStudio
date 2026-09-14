"""Experimental process-isolated IK. Only XLerobotPPAsync owns hardware."""
import os
import pickle
import socket
import subprocess
import sys
import threading
import time

import numpy as np
from loguru import logger

from deploy.robot.xlerobot_pp.robot import XLerobotPP
from deploy.robot.so101_plus.robot import So101Plus
from deploy.utils import RateLimiter


def send(sock, data):
    try:
        sock.send(pickle.dumps(data, protocol=5))
        return True
    except BlockingIOError:
        return False


def latest(sock):
    result = None
    while True:
        try:
            packet = sock.recv(131072)
            if not packet:
                raise EOFError("IK channel closed")
            result = pickle.loads(packet)
        except BlockingIOError:
            return result


def plain(value):
    if value is None or isinstance(value, (str, bool, int, float, np.ndarray, np.generic)):
        return True
    if isinstance(value, (list, tuple)):
        return all(plain(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and plain(v) for k, v in value.items())
    return False


def snapshot(arm):
    # Never serialize a backend, descriptor, lock, solver or SHM handle.
    excluded = {"_robot", "_lock", "_ik_solver", "shm", "control_shm", "_diag_fp"}
    state = {}
    for key, value in vars(arm).items():
        if key in excluded:
            continue
        if not plain(value):
            raise TypeError(f"Unexpected IK state field: {key}: {type(value)}")
        state[key] = value
    state["diag_log"] = False
    return state


class ComputeArm(So101Plus):
    def get_observation(self):
        q = self._measured.copy()
        self._current_q_ik = self._hw7_norm_to_ik6(q)
        self._current_gripper = float(q[6])
        return {"qpos": q}

    def publish_action(self, action):
        raise RuntimeError("IK worker cannot write motors")

    def connect(self):
        raise RuntimeError("IK worker cannot connect hardware")

    def close(self):
        pass


def worker(fd):
    owner_pid = os.getppid()
    sock = socket.socket(fileno=fd)
    init = pickle.loads(sock.recv(131072))
    arms = []
    for state in init:
        arm = object.__new__(ComputeArm)
        arm.__dict__.update(state)
        arm._lock = threading.RLock()
        arm._ik_solver = None
        arm._robot = None
        arm._diag_fp = None
        arms.append(arm)
    sock.setblocking(False)
    arm_epochs = [None, None]
    while True:
        if os.getppid() != owner_pid:
            return
        msg = latest(sock)
        if msg is None:
            time.sleep(.001)
            continue
        if msg.get("stop"):
            return
        output = np.array(msg["hold"], copy=True)
        for i, arm in enumerate(arms):
            arm._measured = np.array(msg["qpos"][i*7:(i+1)*7], copy=True)
            arm.get_observation()
            edge = arm_epochs[i] != msg["arm_epochs"][i]
            if edge:
                arm._sync_state_to_qpos(msg["hold"][i*7:(i+1)*7])
                arm._rel_ee_anchor_pose = None
                arm._rel_ee_vr_unsqueeze_active = False
            active = msg["gates"][i]
            if active:
                raw = np.asarray(msg["action"][i*7:(i+1)*7])
                if arm.gripper_absolute:
                    arm._target_gripper = float(np.clip(raw[6]*arm.gripper_scale*100, 0, 100))
                else:
                    raise ValueError("Experiment requires absolute gripper control")
                result = arm.process_action(dict(action=raw, unsqueeze_active=True, anchor_just_set=edge))
                if result.get("action") is not None:
                    output[i*7:(i+1)*7] = result["action"]
                output[i*7+6] = arm._target_gripper
            arm_epochs[i] = msg["arm_epochs"][i]
        send(sock, dict(epoch=msg["epoch"], seq=msg["seq"], sent=msg["sent"], action=output))


def valid_result(result, epoch, seq, now, timeout):
    if not result or result["epoch"] != epoch or result["seq"] <= seq:
        return False
    target = np.asarray(result["action"])
    return now-result["sent"] <= timeout and target.shape == (14,) and np.isfinite(target).all()


class XLerobotPPAsync(XLerobotPP):
    def __init__(self, *args, **kwargs):
        self._ik_process = None
        self._ik_socket = None
        super().__init__(*args, **kwargs)
        if self.control_mode != "rel_ee" or not all(a.gripper_absolute for a in (self.left, self.right)):
            raise ValueError("Async experiment supports rel_ee with absolute grippers only")

    def _start_worker(self):
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._ik_socket = parent
        try:
            self._ik_process = subprocess.Popen(
                [sys.executable, "-m", "deploy.robot.xlerobot_pp.async_ik", str(child.fileno())],
                pass_fds=(child.fileno(),), close_fds=True,
            )
            parent.send(pickle.dumps([snapshot(self.left), snapshot(self.right)], protocol=5))
            parent.setblocking(False)
        finally:
            child.close()

    def start(self):
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)
        self.control_shm = self.connect_to_existing_shm(self.control_shm_name)
        obs = self.get_observation()
        if obs is None:
            raise RuntimeError("Cannot seed async control without measured qpos")
        hold = np.asarray(obs["qpos"], dtype=float).copy()
        self._start_worker()
        self.is_running = True
        limiter = RateLimiter()
        epoch = seq = 0
        arm_epochs = [0, 0]
        applied = -1
        gates = (False, False)
        last_input = time.monotonic()
        timeout = .3
        home = None
        require_release = False
        logger.info("[AsyncIK] Hardware owner PID={}, IK PID={}; state loop={} Hz", os.getpid(), self._ik_process.pid, self.fps)
        try:
            while self.is_running:
                if self._ik_process.poll() is not None:
                    raise RuntimeError("IK worker exited; stopping robot")
                now = time.monotonic()
                obs = self.get_observation()
                if obs is None:
                    epoch += 1
                    limiter.sleep(self.fps)
                    continue
                self.write_data({"qpos": obs["qpos"]})
                action = self.read_action()
                if action is not None:
                    last_input = now
                    if action.get("cmd") == "reset" or action.get("go_home", False):
                        epoch += 1
                        arm_epochs = [v+1 for v in arm_epochs]
                        gates = (False, False)
                        require_release = True
                        self._stop_base()
                        target = np.concatenate([self.left._home_qpos, self.right._home_qpos])
                        home = (now, np.array(obs["qpos"], copy=True), target)
                    elif home is None:
                        incoming = tuple(bool(action.get(k, action.get("unsqueeze_active", False))) for k in ("left_unsqueeze_active", "right_unsqueeze_active"))
                        if require_release:
                            require_release = any(incoming)
                            incoming = (False, False)
                        if incoming != gates:
                            epoch += 1
                            arm_epochs = [v+int(a != b) for v, a, b in zip(arm_epochs, incoming, gates)]
                            gates = incoming
                        for key, command in (("left_thumbstick", self._command_head_from_stick), ("right_thumbstick", self._command_base_from_stick)):
                            stick = self._stick(action, key)
                            if stick is not None:
                                command(stick)
                        raw = action.get("action")
                        if raw is not None and np.asarray(raw).shape == (14,) and np.isfinite(raw).all():
                            seq += 1
                            send(self._ik_socket, dict(epoch=epoch, arm_epochs=arm_epochs, seq=seq, sent=now, gates=gates, action=raw, qpos=obs["qpos"], hold=hold))
                if now-last_input > timeout and any(gates):
                    epoch += 1
                    arm_epochs = [v+1 for v in arm_epochs]
                    gates = (False, False)
                    require_release = True
                result = latest(self._ik_socket)
                if home is not None:
                    begin, initial, target = home
                    alpha = min(1.0, (now-begin)/self.home_duration_s)
                    hold = initial + alpha*(target-initial)
                    self.publish_action(hold)
                    if alpha == 1:
                        home = None
                        epoch += 1
                        if self._head_ready:
                            self.head_bus.sync_write("Goal_Position", dict(zip(self.HEAD_MOTOR_IDS, self._head_home_raw.astype(int).tolist())), normalize=False)
                elif valid_result(result, epoch, applied, now, timeout) and any(gates):
                    target = np.asarray(result["action"], dtype=float)
                    for i, active in enumerate(gates):
                        if active:
                            hold[i*7:(i+1)*7] = target[i*7:(i+1)*7]
                    self.publish_action(hold)
                    applied = result["seq"]
                elif result and now-result["sent"] > timeout and any(gates):
                    epoch += 1
                    arm_epochs = [v+1 for v in arm_epochs]
                    gates = (False, False)
                    require_release = True
                    logger.warning("[AsyncIK] Expired IK target discarded; release grips to re-arm")
                limiter.sleep(self.fps)
        finally:
            self._stop_worker()
            self._stop_base()

    def _stop_worker(self):
        proc, sock = self._ik_process, self._ik_socket
        self._ik_process = self._ik_socket = None
        if proc is not None:
            proc.terminate() if proc.poll() is None else None
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if sock is not None:
            sock.close()

    def shutdown(self):
        self._stop_worker()
        super().shutdown()


if __name__ == "__main__":
    worker(int(sys.argv[1]))
