"""B601-RS VLAStudio robot: existing motor zero, motor-coordinate radians, Pinocchio IK."""
from pathlib import Path
import signal
import time
import numpy as np
from loguru import logger
from vlastudio.deploy.robot.base import BaseRobot
from vlastudio.deploy.robot.alicia_d.robot import (
    AliciaD, _transform_matrix_from_xyz_rpy, _xyz_rpy_from_transform_matrix)
from vlastudio.deploy.utils import RateLimiter
from .kinematics import RebotIK
from .bus import RebotBus

URDF = Path(__file__).parent/'assets/ReBot_Arm_RS.urdf'


class RebotB601RS(AliciaD):
    """Reuse VLAStudio's tested relative-EE/EMA math, not its Alicia hardware SDK."""
    BUS_CLS = RebotBus
    # Only shutdown() can decide when parking and motor disable are complete.
    process_shutdown_timeout_s = None

    def __init__(self, name='rebot_b601rs', channel='can1', control_mode='qpos',
                 read_only=False, velocity_limit_rad_s=.5, home_duration_s=4.,
                 gripper_open_rad=4.71238898038, exit_event=None, home_velocity_rad_s=.5,
                 gripper_velocity_rad_s=1., gripper_damping=1.5, gripper_deadband_rad=.03,
                 home_gripper_tolerance_rad=.05, gripper_travel_damping=None, **kwargs):
        if control_mode not in ('qpos','ee','delta_ee','rel_ee'):
            raise ValueError('Expected qpos, ee, delta_ee, or rel_ee')
        kwargs.setdefault('fps',100.)
        kwargs.setdefault('joint_smooth_alpha',.15)
        kwargs.setdefault('joint_stop_hold',True)
        kwargs.setdefault('gripper_absolute',True)
        kwargs.setdefault('max_joint_delta_deg',30.)
        kwargs.setdefault('ik_end_frame','link6')
        self.ik_end_frame = kwargs['ik_end_frame']
        super().__init__(name=name,urdf_path=str(URDF),
                         control_mode='qpos' if control_mode=='ee' else control_mode,**kwargs)
        self.control_mode = control_mode
        self.channel, self.read_only = channel, read_only
        self.velocity_limit = float(velocity_limit_rad_s)
        self.home_velocity = min(float(home_velocity_rad_s),self.velocity_limit)
        if self.home_velocity <= 0:
            raise ValueError('Joint/home velocity limits must be positive')
        self.home_duration = float(home_duration_s)
        self.gripper_open_rad = float(gripper_open_rad)
        self.gripper_velocity = float(gripper_velocity_rad_s)
        self.gripper_damping = float(gripper_damping)
        self.gripper_travel_damping = gripper_travel_damping
        self.gripper_deadband = float(gripper_deadband_rad)
        self.home_gripper_tolerance=float(home_gripper_tolerance_rad)
        if not np.isfinite(self.home_gripper_tolerance) or self.home_gripper_tolerance<=self.gripper_deadband:
            raise ValueError('Gripper home tolerance must exceed its control deadband')
        self._tracking_time=None
        self._delta_direction = np.zeros(6)
        self._ik_tracking_status=None
        self._home_status=None
        self.bus = None
        self.exit_event = exit_event
        self._exit_requested = False
        self._shutting_down = False
        self.shutdown_ok = None

    def _make_ik_solver(self, **kwargs):
        return RebotIK(self.urdf_path,end_frame=self.ik_end_frame,
                       max_iter=80,tolerance=.001,budget_s=.022)

    def set_joints(self, joint_angles, gripper=None):
        if gripper is None:
            gripper = float(self.get_observation()['qpos'][6])
        self.publish_action(np.r_[joint_angles,gripper])

    def set_pose(self, target_pose, gripper=None):
        q = self.get_observation()['joint_angles']
        ok, joints, _, _ = self.ik_solver.solve(target_pose,q_init=q)
        if ok:
            self.set_joints(joints,gripper)
        return ok

    def set_state_ee(self, state_ee, gripper=None, q_init=None):
        a = np.asarray(state_ee,dtype=float)
        if a.shape not in ((6,),(7,)) or not np.isfinite(a).all():
            raise ValueError('Expected six or seven finite EE values')
        if gripper is None and len(a)==7:
            gripper = float(a[6])
        if q_init is None:
            q_init = self.get_observation()['joint_angles']
        ok,joints,_,_ = self.ik_solver.solve(
            _transform_matrix_from_xyz_rpy(a[:3],a[3:6]),q_init=q_init)
        if ok:
            self.set_joints(joints,gripper)
        return ok

    @property
    def ik_solver(self):
        if self._ik_solver is None:
            self._ik_solver = self._make_ik_solver()
            logger.info('[RebotB601RS] Pinocchio LOCAL DLS IK initialized, frame={}',self.ik_end_frame)
        return self._ik_solver

    def connect(self):
        if self._connected:
            return True
        self.bus = self.BUS_CLS(self.channel,read_only=self.read_only,rate=self.fps,
                               velocity=self.velocity_limit,gripper_range=self.gripper_open_rad,
                               gripper_velocity=self.gripper_velocity,gripper_damping=self.gripper_damping,
                               gripper_deadband=self.gripper_deadband,
                               gripper_travel_damping=self.gripper_travel_damping)
        try:
            self.bus.connect()
            self._connected = True
            obs = self.get_observation()
            self._seed(obs)
            # Only check arm-coordinate plausibility; do not change any stored zero.
            q = obs['joint_angles']
            if np.any(q < self.ik_solver.lower-np.deg2rad(1)) or np.any(q > self.ik_solver.upper+np.deg2rad(1)):
                raise RuntimeError('Measured arm pose outside official limits; verify existing zero/model')
            return True
        except BaseException:
            self.shutdown()
            raise

    def _seed(self, obs):
        self._current_joint_angles = np.asarray(obs['joint_angles'],dtype=float).copy()
        self._target_qpos = self._current_joint_angles.copy()
        self._smooth_qpos = self._current_joint_angles.copy()
        self._current_gripper = self._target_gripper = float(obs['qpos'][6]*1000)
        self._ik_initialized = False
        self._delta_direction.fill(0)
        self._tracking_time = None

    def enable(self):
        if not self._connected:
            raise RuntimeError('Connect first')
        self.bus.enable()
        self._seed(self.get_observation())

    def get_observation(self):
        if not self._connected:
            return None
        q, stamps = self.bus.snapshot()
        grip = float(np.clip(q[6]/self.gripper_open_rad,0,1))
        self._current_joint_angles = q[:6].copy()
        self._current_gripper = grip*1000
        return dict(qpos=np.r_[q[:6],grip].astype(np.float32),
                    joint_angles=q[:6].astype(np.float32),motor_positions=q,
                    joint_timestamps=stamps,timestamp=float(min(stamps)),gripper=grip*1000,
                    ee_frame=self.ik_end_frame,gripper_control=self.bus.gripper_diagnostics(),
                    ik_tracking=self._ik_tracking_status,home_status=self._home_status)

    def refresh_rel_ee_anchor(self, use_actual_joints=True):
        with self._lock:
            obs = self.get_observation()
            self._seed(obs)
            self._rel_ee_anchor_pose = self.ik_solver.compute_fk(obs['joint_angles'])
            self.ik_solver.reset(obs['joint_angles'])
            self._rel_ee_anchor_initialized = True
            self._tracking_time=None
            return True

    def _solve_teleop_ik(self, target_pose):
        now=time.monotonic()
        dt=1/60 if self._tracking_time is None else np.clip(now-self._tracking_time,.001,1/30)
        self._tracking_time=now
        measured=self.get_observation()['joint_angles']
        reference=self._smooth_qpos if self._smooth_qpos is not None else measured
        q,status=self.ik_solver.track(target_pose,reference,self.velocity_limit*dt,measured=measured)
        self._ik_tracking_status=dict(status,dt_s=float(dt))
        # Here success means a bounded, feasible tracking step, not convergence.
        # Holding at a boundary must not trigger Alicia's failed-solve re-anchoring.
        return True,q,1,status['residual']

    def process_action(self, message):
        if message.get('action') is None:
            return message
        a = np.asarray(message['action'],dtype=float)
        if a.shape!=(7,) or not np.isfinite(a).all():
            raise ValueError('Expected seven finite values')
        if self.control_mode=='qpos':
            return message
        if self.control_mode=='ee':
            target = _transform_matrix_from_xyz_rpy(a[:3],a[3:6])
            ok,q,_,_ = self.ik_solver.solve(target,q_init=self.get_observation()['joint_angles'])
            return dict(message,action=np.r_[q,a[6]] if ok else None)
        if self.control_mode == 'delta_ee':
            # Keep accumulated deltas during a held key. On direction changes
            # the raw IK pose may be a whole step ahead of the filtered command.
            # Drop that outstanding step before composing the new increment.
            direction = np.sign(a[:6])
            if not np.array_equal(direction, self._delta_direction):
                self._delta_direction = direction
                if self._smooth_qpos is not None:
                    self.ik_solver.set_reference(self._smooth_qpos[:6])
                self._prev_ik_qpos = None
                self._still_count = 0
                self._teleop_holding_still = False
            # Keyboard sends normalized opening increments. Unlike Alicia's loop,
            # Rebot.start() does not update the gripper before process_action().
            self._target_gripper = float(np.clip(self._target_gripper + a[6]*1000, 0, 1000))
        if message.get('unsqueeze_active',False) and self.control_mode != 'delta_ee':
            if self.gripper_absolute:
                self._target_gripper = float(np.clip(a[6]*1000,0,1000))
            else:
                self._target_gripper = float(np.clip(self._target_gripper+a[6]*1000,0,1000))
            # Alicia's delta-action filter drops an all-zero frame. With the
            # absolute Quest trigger it also means "close fully at the anchor".
            if (self.control_mode == 'rel_ee' and self.gripper_absolute
                    and self._rel_ee_vr_unsqueeze_active and not message.get('anchor_just_set')
                    and np.allclose(a, 0., atol=1e-6)):
                if self._last_sent_gripper == 0:
                    return dict(message, action=None)
                hold = self._last_sent_qpos if self._last_sent_qpos is not None else self._smooth_qpos
                return dict(message, action=np.r_[hold, 0.])
        # This method is pure EE/IK/EMA logic in the shared VLAStudio adapter.
        return super().process_action(dict(message))

    def publish_action(self, action, **kwargs):
        a = np.asarray(action,dtype=float)
        if a.shape!=(7,) or not np.isfinite(a).all():
            raise ValueError('Expected seven finite values')
        solver = self.ik_solver
        if np.any(a[:6]<solver.lower-np.deg2rad(1)) or np.any(a[:6]>solver.upper+np.deg2rad(1)) or not 0<=a[6]<=1:
            raise ValueError('Target outside B601-RS joint/gripper limits')
        a = a.copy()
        a[:6] = np.clip(a[:6],solver.lower,solver.upper)
        raw = np.r_[a[:6],a[6]*self.gripper_open_rad]
        self.bus.set_target(raw)
        self._last_sent_qpos = a[:6].copy()
        self._last_sent_gripper = int(a[6]*1000)

    def _write_observation(self):
        obs = self.get_observation()
        if self.shm is not None:
            self.write_data(obs)
        return obs

    def _home_reached(self, obs, goal, include_gripper):
        raw=obs['motor_positions']
        arm_errors=np.rad2deg(np.abs(raw[:6]-goal[:6]))
        grip_error=abs(raw[6]-goal[6]*self.gripper_open_rad)
        self._home_status=dict(joint_error_deg=arm_errors.tolist(),
            gripper_error_rad=float(grip_error),gripper_tolerance_rad=self.home_gripper_tolerance)
        return bool(max(arm_errors)<1 and
                    (not include_gripper or grip_error<self.home_gripper_tolerance))

    def set_home(self, include_gripper=True, staged=False, **kwargs):
        """Explicit B/reset: minimum-jerk return to existing motor zero (all seven)."""
        start = self.get_observation()['qpos'].astype(float)
        previous_velocity = self.bus.velocity
        self.bus.set_velocity_limit(self.home_velocity, homing=True)
        goals = []
        if staged:
            intermediate = start.copy()
            intermediate[[0,3,4,5]] = 0
            goals.append(intermediate)
        goal = start.copy()
        goal[:6] = 0
        if include_gripper:
            goal[6] = 0
        goals.append(goal)
        try:
            for goal in goals:
                # A normal exit during an in-flight B trajectory interrupts it;
                # shutdown starts a fresh staged park from measured feedback.
                duration = max(self.home_duration,
                    1.9*max(abs(goal[:6]-start[:6]))/self.home_velocity,
                    1.9*abs(goal[6]-start[6])*self.gripper_open_rad)
                steps = max(2,int(duration*50))
                for s in np.linspace(0,1,steps+1)[1:]:
                    if self._exit_pending() and not self._shutting_down:
                        return False
                    blend = 10*s**3-15*s**4+6*s**5
                    target = start+blend*(goal-start)
                    target[:6] = np.clip(target[:6],self.ik_solver.lower,self.ik_solver.upper)
                    self.publish_action(target)
                    self._write_observation()
                    time.sleep(duration/steps)
                deadline = time.monotonic()+3
                while True:
                    obs = self._write_observation()
                    if self._home_reached(obs,goal,include_gripper):
                        break
                    if time.monotonic()>deadline:
                        logger.error('[RebotB601RS] Home stage not confirmed: joint errors deg={}, gripper error={:.4f} rad (tol {:.4f}); aborting remaining stages',
                            np.round(self._home_status['joint_error_deg'],3).tolist(),
                            self._home_status['gripper_error_rad'],self.home_gripper_tolerance)
                        return False
                    time.sleep(.02)
                start = obs['qpos'].astype(float)
            logger.info('[RebotB601RS] Home reached: existing motor zero')
            return True
        finally:
            # Keep the slow backend limit through shutdown settling/disable.
            # A normal B return restores the operator's VR speed afterwards.
            if not self._shutting_down:
                self.bus.set_velocity_limit(previous_velocity)
            if self._connected:
                try:
                    self._seed(self.get_observation())
                except Exception:
                    pass
            self._rel_ee_anchor_pose = None
            self._rel_ee_vr_unsqueeze_active = False
            self._prev_ik_qpos = None
            self._still_count = 0
            self._teleop_holding_still = False

    def reset_for_teleoperation(self):
        return self.set_home()

    def start(self):
        # Override BaseDevice's immediate-close signal handler. Repeated signals
        # request the same shutdown; they must not interrupt parking or disable.
        signal.signal(signal.SIGINT,self._request_exit)
        signal.signal(signal.SIGTERM,self._request_exit)
        try:
            if self._exit_pending():
                return
            self.connect()
            if self._exit_pending():
                return
            if self.control_shm_name:
                for attempt in range(30):
                    try:
                        self.control_shm = self.connect_to_existing_shm(self.control_shm_name)
                        break
                    except (ValueError, TimeoutError):
                        if self._exit_pending():
                            return
                        if attempt==29: raise
                        time.sleep(.5)
            if self._exit_pending():
                return
            if not self.read_only:
                self.enable()
            self.shm = self.create_shm(name=self.name,max_size_mb=self.max_size_mb,is_writer=True)
            self.is_running = True
            rate = RateLimiter()
            while self.is_running and not self._exit_pending():
                message = self.read_action()
                if message and not self.read_only:
                    if message.get('go_home') or message.get('cmd')=='reset':
                        self.set_home()
                    else:
                        out = self.process_action(message).get('action')
                        if out is not None:
                            self.publish_action(out)
                self._write_observation()
                rate.sleep(self.fps)
        except Exception as exc:
            # Cleanup can itself fail and raise SystemExit(2). Keep the first error.
            logger.opt(exception=True).error('[RebotB601RS] Device error before shutdown: {}', exc)
            raise
        finally:
            if not self.shutdown():
                # BaseDevice catches Exception and otherwise exits with code 0.
                # SystemExit preserves a failed shutdown status for the supervisor.
                raise SystemExit(2)

    def _request_exit(self, signum=None, frame=None):
        self._exit_requested = True
        self.is_running = False

    def _exit_pending(self):
        return self._exit_requested or (self.exit_event is not None and self.exit_event.is_set())

    def _confirm_parked_and_still(self):
        samples = []
        last_stamp = -1.
        deadline = time.monotonic()+3
        while time.monotonic()<deadline:
            obs = self._write_observation()
            stamp = min(obs['joint_timestamps'])
            if stamp>last_stamp:
                last_stamp = stamp
                samples.append(obs['motor_positions'][:6].copy())
                samples = samples[-3:]
                if len(samples)==3 and np.max(np.abs(samples))<np.deg2rad(1) and np.max(np.ptp(samples,axis=0))<.003:
                    return True
            time.sleep(.1)
        return False

    def shutdown(self):
        self.is_running = False
        self._exit_requested = True
        if self._shutting_down:
            return False
        if self.bus is None:
            return self.shutdown_ok is not False
        self._shutting_down = True
        bus = self.bus
        self.shutdown_ok = False
        stage = 'check enable state'
        try:
            if not self.read_only and bus.armed:
                logger.warning('[RebotB601RS] Exit: cancel target, staged park, verify, then disable')
                stage = 'cancel pending target'
                bus.cancel_pending_target()
                time.sleep(.3)
                stage = 'park arm'
                if not self.set_home(include_gripper=False,staged=True):
                    raise RuntimeError('Park trajectory did not reach zero')
                stage = 'confirm parked and still'
                if not self._confirm_parked_and_still():
                    raise RuntimeError('Zero pose / settling not confirmed')
                stage = 'disable and verify motors'
                bus.disable(closing=True)
                logger.bind(stopped_confirmed=True).info('[RebotB601RS] Exit complete: all 7 motors confirmed stopped')
            elif not self.read_only and getattr(bus,'output_started',False):
                raise RuntimeError('Partial enable failure: automatic parking is unavailable')
            self.shutdown_ok = True
        except Exception as exc:
            # A fault/stale pose cannot justify an unverified path or torque-off.
            # Do not claim that closing the CAN socket powers down the motors.
            logger.critical('[RebotB601RS] SAFE EXIT FAILED during {}: {}. Motors may remain enabled; keep clear and use the local stop procedure.',stage,exc)
            try:
                if bus.armed:
                    bus.cancel_pending_target()
            except Exception:
                pass
        finally:
            try:
                bus.close()
            finally:
                self.bus = None
                self._connected = False
                self._shutting_down = False
        return self.shutdown_ok

    def close(self):
        self.shutdown()
        BaseRobot.close(self)

    @classmethod
    def enrich_saved_frame(cls, device_data, device_args=None):
        if not device_data or 'qpos' not in device_data:
            return device_data
        frame = device_data.get('ee_frame') or (device_args or {}).get('ik_end_frame','link6')
        if getattr(cls,'_rebot_save_solver',None) is None or getattr(cls,'_rebot_save_frame',None)!=frame:
            cls._rebot_save_solver = RebotIK(URDF,end_frame=frame)
            cls._rebot_save_frame = frame
        solver = cls._rebot_save_solver
        q = np.asarray(device_data['qpos'])
        return dict(device_data,ee_frame=frame,
                    state_ee=np.r_[_xyz_rpy_from_transform_matrix(solver.compute_fk(q[:6])),q[6]])

    @staticmethod
    def obs2meta(device_data):
        if device_data is None:
            return {}
        enriched = RebotB601RS.enrich_saved_frame(device_data)
        return dict(state=enriched['qpos'],state_joint=enriched['qpos'],state_ee=enriched['state_ee'])
