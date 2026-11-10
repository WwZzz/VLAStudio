"""MotorBridge transport. Connection is read-only; enable is a separate operation."""
import fcntl
from pathlib import Path
import threading
import time
import numpy as np
from loguru import logger
from .feedback import ActiveReports


class FeedbackTimeout(RuntimeError):
    def __init__(self, ids, stopped):
        self.ids = ids
        super().__init__(f'Fresh {"stopped " if stopped else ""}reports missing for IDs {ids}')


class RebotBus:
    REPORTS_CLS = ActiveReports

    def __init__(self, channel, *, read_only=False, rate=100., velocity=.5,
                 gripper_range=4.71238898038, gripper_velocity=1., gripper_damping=1.5,
                 gripper_deadband=.03, gripper_travel_damping=None,
                 controller_factory=None):
        if controller_factory is None:
            from motorbridge import Controller
            controller_factory = Controller
        self.channel, self.read_only = channel, read_only
        self.factory, self.rate, self.velocity = controller_factory, rate, velocity
        self.gripper_range = gripper_range
        self.gripper_velocity = float(gripper_velocity)
        if not np.isfinite(self.gripper_velocity) or self.gripper_velocity<=0:
            raise ValueError('Gripper velocity must be finite and positive')
        if not np.isfinite(gripper_damping) or gripper_damping<0:
            raise ValueError('Gripper damping must be finite and nonnegative')
        self.gripper_travel_damping = gripper_travel_damping
        if gripper_travel_damping is not None and not (
                np.isfinite(gripper_travel_damping) and 0 < gripper_travel_damping <= gripper_damping):
            raise ValueError('Travel damping must be positive and at most the holding damping')
        self._homing = False
        self._gripper_measured_speed = 0.
        self.gripper_deadband = float(gripper_deadband)
        if not np.isfinite(self.gripper_deadband) or self.gripper_deadband<0:
            raise ValueError('Gripper deadband must be finite and nonnegative')
        self._gripper_diagnostics = None
        self._lock, self._io = threading.RLock(), threading.RLock()
        self._stop = threading.Event()
        self.controller, self.motors, self.thread = None, [], None
        self.q = np.zeros(7)
        self.times = np.zeros(7)
        self.report_modes = np.full(7, -1)
        self.reports = None
        self._started_reports = []
        self._report_intervals = []
        self.target = self.command = None
        self.fault, self.armed = None, False
        self.output_started = False
        self.kp = [50., 150., 150., 50., 50., 50., 12.]
        self.kd = [3., 10., 10., 5., 4., 4., float(gripper_damping)]

    def connect(self):
        cache = Path.home()/'.cache'
        cache.mkdir(exist_ok=True)
        if not self.channel.replace('_','').isalnum():
            raise ValueError('Invalid CAN interface name')
        self._file = (cache/('vlastudio-rebot-'+self.channel+'.lock')).open('a')
        fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            self.controller = self.factory(self.channel)
            for i in range(1,8):
                m = self.controller.add_robstride_motor(i,0xfd,'rs-06' if i<=3 else 'rs-00')
                self.motors.append(m)
                mid,_ = m.robstride_ping_host_id(0xfd,300)
                if mid != i:
                    raise RuntimeError(f'Motor identity mismatch for ID {i}')
                self._check_fault(i-1)
            self.reports = self.REPORTS_CLS(self.channel)
            time.sleep(.06)
            self._receive_reports()
            for i,m in enumerate(self.motors):
                if self.times[i] == 0:
                    self._started_reports.append(m)
                    m.robstride_set_active_report(True)
            try:
                self._verify_positions()
            except FeedbackTimeout as exc:
                logger.warning("Active feedback missing for IDs {}; retrying report setup once", exc.ids)
                for mid in exc.ids:
                    m = self.motors[mid-1]
                    interval = m.robstride_get_param_u16(0x7026, 300)
                    self._report_intervals.append((m, interval))
                    if m not in self._started_reports:
                        self._started_reports.append(m)
                    m.robstride_set_active_report(False)
                    m.robstride_write_param_u16(0x7026, 1)
                    if m.robstride_get_param_u16(0x7026, 300) != 1:
                        raise RuntimeError(f'Motor {mid} report interval was not applied')
                    m.robstride_set_active_report(True)
                self._verify_positions()
            self.target, self.command = self.q.copy(), self.q.copy()
            self.thread = threading.Thread(target=self._loop,daemon=True,name='rebot-can')
            self.thread.start()
            return True
        except BaseException:
            self.close()
            raise

    def _read_position(self, i):
        try:
            return self.motors[i].robstride_get_param_f32_host_id(0x7019,0xfd,150)
        except Exception as exc:
            raise RuntimeError(f'{self.channel} motor ID {i+1} position read: {exc}') from exc

    def _check_fault(self, i):
        fault, warning = self.motors[i].robstride_get_fault_report()
        if fault:
            raise RuntimeError(f'Motor {i+1} firmware fault 0x{fault:x} warning=0x{warning:x}')

    def _receive_reports(self):
        with self._lock:
            for i, stamp, pos, mode, faults in self.reports.drain():
                if faults:
                    raise RuntimeError(f'Motor {i+1} status fault bits 0x{faults:02x}')
                if stamp > time.monotonic() + .001:
                    raise RuntimeError('CAN receive timestamp in the future')
                if stamp <= self.times[i]:
                    continue
                if i == 6:
                    elapsed = stamp - self.times[i]
                    self._gripper_measured_speed = (
                        (pos-self.q[i])/elapsed if .001 <= elapsed <= .12 else 0.)
                self.q[i], self.times[i], self.report_modes[i] = pos, stamp, mode

    def _await_reports(self, after, *, stopped=False):
        deadline = time.monotonic() + .3
        while True:
            self._receive_reports()
            missing = self.times <= after
            if stopped:
                missing |= self.report_modes != 0
            if not missing.any():
                return
            if time.monotonic() >= deadline:
                ids = (np.flatnonzero(missing) + 1).tolist()
                raise FeedbackTimeout(ids, stopped)
            time.sleep(.002)

    def _verify_positions(self):
        # Align each comparison with a new report for a stationary startup.
        refs = np.array([self._read_position(i) for i in range(7)])
        self._await_reports(time.monotonic())
        errors = np.abs(self.q - refs)
        if not np.isfinite(errors).all() or np.any(errors > np.deg2rad(1)):
            raise RuntimeError(f'Active report disagrees with startup position: {errors} rad')

    def snapshot(self):
        with self._lock:
            if self.fault:
                raise RuntimeError(self.fault)
            stale = time.monotonic() - self.times > .3
            if stale.any():
                raise RuntimeError(f'Rebot joint feedback older than 0.3s for IDs {(np.flatnonzero(stale)+1).tolist()}')
            return self.q.copy(), self.times.copy()

    def enable(self):
        if self.read_only:
            raise RuntimeError('Read-only connection cannot enable motors')
        from motorbridge import Mode
        with self._io:
            # Read actual positions immediately before enabling; never seed zero.
            self.snapshot()
            for i in range(7):
                self._check_fault(i)
            for m in self.motors:
                m.ensure_mode(Mode.MIT)
            self._verify_positions()
            q, _ = self.snapshot()
            with self._lock:
                self.target, self.command = q.copy(), q.copy()
                self._gripper_measured_speed = 0.
            for i,m in enumerate(self.motors):
                self.output_started = True
                for after_enable in (False,True):
                    if after_enable:
                        m.enable()
                    if i==6:
                        self._send_gripper(m,float(q[i]),float(q[i]))
                    else:
                        m.send_mit(float(q[i]),0.,self.kp[i],self.kd[i],0.)
            self.armed = True

    def cancel_pending_target(self):
        """Hold the last streamed command, cancelling the remaining target travel."""
        if self.read_only or not self.armed:
            raise RuntimeError('Motor output is not enabled')
        with self._io:
            self.snapshot()
            with self._lock:
                self.target = self.command.copy()

    def disable(self, *, closing=False):
        """Stop output before disable; attempt every motor and verify stop feedback."""
        if self.read_only:
            raise RuntimeError('Read-only connection cannot disable motors')
        with self._io:
            self.armed = False  # worker must never stream another MIT after disable
            if closing:
                self._stop.set()  # the shutdown thread now owns the final status reads
            errors = []
            for i,m in enumerate(self.motors,1):
                try:
                    m.disable()
                except Exception as exc:
                    errors.append(f'ID {i}: {exc}')
            for i, m in enumerate(self.motors, 1):
                try:
                    for _ in range(3):
                        m.request_feedback()
                        time.sleep(.04)
                        state = m.get_state()
                        if state is not None and state.status_code == 0:
                            break
                    else:
                        errors.append(f'ID {i}: stopped feedback not confirmed')
                except Exception as exc:
                    errors.append(f'ID {i} verification: {exc}')
            try:
                self._await_reports(time.monotonic(), stopped=True)
            except Exception as exc:
                errors.append(str(exc))
            if errors:
                raise RuntimeError('Disable incomplete: '+'; '.join(errors))
            self.output_started = False
            return True

    def set_target(self, q):
        if self.read_only or not self.armed:
            raise RuntimeError('Motor output is not enabled')
        self.snapshot()
        with self._lock:
            self.target = np.asarray(q,dtype=float).copy()

    def set_velocity_limit(self, velocity, *, homing=False):
        velocity = float(velocity)
        if not np.isfinite(velocity) or velocity<=0:
            raise ValueError('Velocity limit must be finite and positive')
        with self._io:
            self.velocity = velocity
            self._homing = homing

    def _advance_command(self, goal, dt):
        arm_step = self.velocity * dt
        self.command[:6] += np.clip(goal[:6] - self.command[:6], -arm_step, arm_step)
        grip_speed = min(self.gripper_velocity, 1.) if self._homing else self.gripper_velocity
        grip = self.command[6]
        measured = self.q[6]
        travel = self.gripper_travel_damping is not None and not self._homing
        if travel:
            # If the operator reverses across the fingers, start from the fingers.
            # A blocked grasp must not leave seconds of old command to unwind.
            grip = np.clip(grip, min(measured, goal[6]), max(measured, goal[6]))
        grip += np.clip(goal[6] - grip, -grip_speed * dt, grip_speed * dt)
        if travel:
            # Enough lead to reach the 1 Nm drive cap, even with a static object.
            lead = self.gripper_deadband + 1. / self.kp[6] + .05
            grip = np.clip(grip, measured - lead, measured + lead)
        self.command[6] = grip

    def _send_gripper(self, motor, commanded, measured, *, goal=None):
        goal = commanded if goal is None else goal
        now = time.monotonic()
        age = now - self.times[6] if self.times[6] else None
        damping = self.kd[6]
        if self.gripper_travel_damping is not None and not self._homing and age is not None and age <= .12:
            remaining = goal - measured
            # Require observed motion toward the operator's target. Blend back to
            # holding damping over the final 0.65 rad (fully restored by 0.25).
            distance_mix = np.clip((abs(remaining) - .25) / .40, 0., 1.)
            progress = np.sign(remaining) * self._gripper_measured_speed
            motion_mix = np.clip((progress - .10) / .25, 0., 1.)
            damping += (self.gripper_travel_damping - damping) * distance_mix * motion_mix
        error = commanded-measured
        # Continuous dead zone: encoder noise around the target must not keep
        # reversing the host torque. Native velocity damping remains active.
        error = np.sign(error)*max(0.,abs(error)-self.gripper_deadband)
        torque = float(np.clip(self.kp[6]*error,-1.,1.))
        motor.send_mit(0.,0.,0.,float(damping),torque)
        with self._lock:
            self._gripper_diagnostics = dict(timestamp=now,
                target_rad=float(goal),
                command_rad=float(commanded),measured_rad=float(measured),
                feedback_age_s=age,measured_speed_rad_s=self._gripper_measured_speed,
                torque_term_nm=torque,damping=float(damping))

    def gripper_diagnostics(self):
        with self._lock:
            return None if self._gripper_diagnostics is None else dict(self._gripper_diagnostics)

    def _loop(self):
        index, previous = 0, time.monotonic()
        try:
            while not self._stop.is_set():
                begin = time.monotonic()
                dt = min(max(begin-previous,.001),2/self.rate)
                previous = begin
                with self._io:
                    if self._stop.is_set():  # shutdown may have held the lock during disable
                        break
                    self._receive_reports()
                    self._check_fault(index % 7)
                    q,_ = self.snapshot()
                    if self.armed:
                        with self._lock:
                            goal = self.target.copy()
                        self._advance_command(goal,dt)
                        for j,m in enumerate(self.motors):
                            if j<6:
                                m.send_mit(float(self.command[j]),0.,self.kp[j],self.kd[j],0.)
                            else:
                                # Gripper impedance as bounded torque (official LeRobot approach).
                                self._send_gripper(m,self.command[j],q[j],goal=goal[j])
                index += 1
                self._stop.wait(max(0.,1/self.rate-(time.monotonic()-begin)))
        except Exception as exc:
            with self._lock:
                self.fault = str(exc)
            logger.error('[RebotB601RS] CAN worker stopped: {}', exc)

    def close(self):
        self._stop.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise RuntimeError('CAN worker did not stop; bus left untouched')
        errors = []
        for m in self._started_reports:
            try:
                m.robstride_set_active_report(False)
            except Exception as exc:
                errors.append(str(exc))
        self._started_reports.clear()
        for m, interval in self._report_intervals:
            try:
                m.robstride_write_param_u16(0x7026, interval)
            except Exception as exc:
                errors.append(str(exc))
        self._report_intervals.clear()
        if self.reports is not None:
            self.reports.close()
            self.reports = None
        # No calibration, homing, clear-error, store, or torque-off on disconnect.
        for m in self.motors:
            m.close()
        self.motors = []
        if self.controller:
            self.controller.close_bus()
            self.controller.close()
            self.controller = None
        if getattr(self,'_file',None):
            self._file.close()
        if errors:
            raise RuntimeError('Failed to restore CAN reporting: '+'; '.join(errors))
