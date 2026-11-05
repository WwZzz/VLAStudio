"""Two serial-verified Alicia-D arms, using the single-arm IK and smoothing."""
import time
import numpy as np
from loguru import logger
from vlastudio.deploy.robot.base import BaseRobot
from vlastudio.deploy.utils import RateLimiter
from .body_binding import BodySerialAliciaD


class BiAliciaD(BaseRobot):
    ARM_CLS = BodySerialAliciaD

    def __init__(self, name="bi_alicia_d", max_size_mb=4, fps=100.,
                 control_shm_name=None, left_body_serial="ADFX260423AFNTM0",
                 right_body_serial="ADFX2604211G8E8N", control_mode="rel_ee",
                 home_duration_s=3., **arm_args):
        super().__init__(name, max_size_mb, fps, control_shm_name)
        if left_body_serial == right_body_serial:
            raise ValueError("Left and right body serials must differ")
        if control_mode not in ("qpos", "rel_ee", "delta_ee"):
            raise ValueError("Invalid control_mode")
        self.control_mode = control_mode
        self.home_duration_s = max(1., float(home_duration_s))
        shared = dict(arm_args, fps=fps, control_mode=control_mode)
        self.left = self.ARM_CLS(name=name + "_left", body_serial=left_body_serial, **shared)
        self.right = self.ARM_CLS(name=name + "_right", body_serial=right_body_serial, **shared)
        self._active = [False, False]
        self._home = None

    def connect(self):
        try:
            if not self.left.connect() or not self.right.connect():
                raise ConnectionError("Both verified Alicia-D arms are required")
            if self.left.port == self.right.port:
                raise ConnectionError("Both arms resolved to the same port")
            obs = self.get_observation()
            if obs is None:
                raise ConnectionError("Missing measured startup feedback")
            self._home = obs["qpos"].copy()
            return True
        except Exception:
            self.shutdown()
            raise

    def get_action_dim(self):
        return 14

    def get_observation(self):
        left, right = self.left.get_observation(), self.right.get_observation()
        if left is None or right is None:
            return None
        return dict(qpos=np.r_[left["qpos"], right["qpos"]],
                    left_qpos=left["qpos"], right_qpos=right["qpos"],
                    left=left, right=right, timestamp=time.monotonic())

    def get_data(self):
        return self.get_observation()

    def obs2meta(self, device_data):
        return {"state": np.asarray(device_data["qpos"], dtype=np.float32)}

    def meta2act(self, mact):
        return mact["action"]

    def is_running(self):
        return self.left._connected and self.right._connected

    @staticmethod
    def _vector(action):
        a = np.asarray(action, dtype=float)
        if a.shape != (14,) or not np.all(np.isfinite(a)):
            raise ValueError("Expected 14 finite values [left7, right7]")
        return a

    def process_action(self, message):
        """Route each hand's clutch and anchor independently; never invent hold targets."""
        if message.get("action") is None:
            return dict(message, action=None)
        raw = self._vector(message["action"])
        outputs = []
        for i, (side, arm) in enumerate((("left", self.left), ("right", self.right))):
            active = bool(message.get(side + "_unsqueeze_active", False))
            edge = active and not self._active[i]
            self._active[i] = active
            part = raw[i * 7:(i + 1) * 7].copy()
            if self.control_mode == "qpos":
                outputs.append(part)
                continue
            before = arm._target_gripper
            if active:
                g = part[6] * arm.gripper_scale * 1000.
                arm._target_gripper = float(np.clip(g if arm.gripper_absolute else before + g, 0, 1000))
            # Match AliciaD.start(): gripper writes need not wait for IK.
            if active and abs(before - arm._target_gripper) > .5:
                hold = arm._smooth_qpos if arm._smooth_qpos is not None else arm._target_qpos
                if hold is not None:
                    arm.publish_action(np.r_[hold[:6], arm._target_gripper / 1000.], from_ik_teleop=True)
            out = arm.process_action(dict(action=part, unsqueeze_active=active,
                                          anchor_just_set=edge, go_home=False))
            cmd = out.get("action")
            if cmd is not None:
                cmd = np.asarray(cmd, dtype=float).copy()
                cmd[6] = arm._target_gripper / 1000.
            outputs.append(cmd)
        return dict(message, action=None, arm_actions=outputs)

    def publish_action(self, action):
        a = self._vector(action)
        self.left.publish_action(a[:7])
        self.right.publish_action(a[7:])

    def dispatch(self, message):
        if message.get("cmd") == "reset" or message.get("go_home", False):
            self.set_home()
            return
        processed = self.process_action(message)
        for arm, cmd in zip((self.left, self.right), processed.get("arm_actions", [])):
            if cmd is not None:
                arm.publish_action(cmd, from_ik_teleop=self.control_mode != "qpos")

    def set_home(self):
        """B/Y returns slowly to measured startup pose, never a synthetic zero pose."""
        last_obs = None
        try:
            if self._home is None:
                raise RuntimeError("No measured home available")
            last_obs = self._read_home_state()
            start = last_obs["qpos"].copy()
            steps = max(1, int(self.home_duration_s / .03))
            logger.info("[BiAliciaD] Returning to measured startup pose")
            for alpha in np.linspace(0, 1, steps + 1)[1:]:
                # The normal observation loop is blocked by this method. Keep
                # querying BOTH arms here, or the 0.5s freshness guard will trip.
                last_obs = self._read_home_state()
                target = start + alpha * (self._home - start)
                for arm, cmd in ((self.left, target[:7]), (self.right, target[7:])):
                    arm.publish_action(cmd)
                    reason = arm._driver.serial_comm.thermal_policy.reason()
                    if reason:
                        raise RuntimeError(f"{arm.body_serial}: {reason}")
                    # AliciaD.publish_action logs exceptions instead of raising.
                    # Do not keep advancing the interpolation after a failed send.
                    sent = arm._last_sent_qpos
                    if (sent is None or not np.allclose(sent, cmd[:6], atol=1e-7, rtol=0)
                            or arm._last_sent_gripper != int(cmd[6] * 1000)):
                        raise RuntimeError(f"{arm.body_serial}: home target was not sent")
                time.sleep(self.home_duration_s / steps)
            # Sending the final target is not proof of arrival. Keep feedback and
            # recording alive while waiting, without issuing additional targets.
            deadline = time.monotonic() + 2.
            while True:
                last_obs = self._read_home_state()
                q = last_obs["qpos"]
                joint_idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
                if (np.max(np.abs(q[joint_idx] - self._home[joint_idx])) <= np.deg2rad(1.)
                        and np.max(np.abs(q[[6, 13]] - self._home[[6, 13]])) <= .015):
                    logger.info("[BiAliciaD] Home reached (measured feedback)")
                    return True
                if time.monotonic() >= deadline:
                    raise RuntimeError("Final home position not confirmed within settling time")
                time.sleep(.03)
        except RuntimeError as exc:
            logger.error("[BiAliciaD] Home stopped: {}", exc)
            return False
        finally:
            self._active = [False, False]
            for i, arm in enumerate((self.left, self.right)):
                arm._rel_ee_vr_unsqueeze_active = False
                arm._rel_ee_anchor_pose = None
                arm._rel_ee_pos_offset = np.zeros(3)
                arm._rel_ee_rot_offset = np.zeros(3)
                arm._prev_ik_qpos = None
                arm._still_count = 0
                arm._teleop_holding_still = False
                if last_obs is not None:
                    measured = last_obs["qpos"][i * 7:(i + 1) * 7]
                    arm._target_qpos = measured[:6].copy()
                    arm._smooth_qpos = measured[:6].copy()
                    arm._target_gripper = float(measured[6] * 1000)
                    if arm._ik_solver is not None:
                        arm._ik_solver.reset(measured[:6])

    def _read_home_state(self):
        """Fresh replies, unchanged fault checks, and live observations during home."""
        for arm in (self.left, self.right):
            if not arm._driver.acquire_info("joint_gripper", wait=True, timeout=.2):
                raise RuntimeError(f"{arm.body_serial}: joint feedback timed out")
            reason = arm._driver.serial_comm.thermal_policy.reason()
            if reason:
                raise RuntimeError(f"{arm.body_serial}: {reason}")
        obs = self.get_observation()
        if obs is None:
            raise RuntimeError("No measured joint feedback during home")
        if self.shm is not None:
            self.write_data(obs)
        return obs

    def start(self):
        try:
            self.connect()
            if self.control_mode != "qpos":
                for arm in (self.left, self.right):
                    arm.ik_solver.reset(arm._current_joint_angles)
            # Only the pair owns SHM; child arms are never started as devices.
            self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)
            if self.control_shm_name and self.control_shm is None:
                for attempt in range(20):
                    try:
                        self.control_shm = self.connect_to_existing_shm(self.control_shm_name)
                        break
                    except ValueError:
                        if attempt == 19:
                            raise
                        time.sleep(.5)
            self.is_running = True
            rate = RateLimiter()
            while self.is_running:
                message = self.read_action()
                if message is not None:
                    self.dispatch(message)
                obs = self.get_data()
                if obs is not None:
                    self.write_data(obs)
                rate.sleep(self.fps)
        finally:
            self.shutdown()

    def shutdown(self):
        self.is_running = False
        try:
            self.left.shutdown()
        finally:
            self.right.shutdown()

    def close(self):
        self.shutdown()
        super().close()
