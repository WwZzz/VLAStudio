"""No hardware motion: kinematics, command semantics, and transport isolation."""
import time
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from vlastudio.deploy.robot.rebot_b601rs.robot import RebotB601RS, URDF
from vlastudio.deploy.robot.rebot_b601rs.kinematics import RebotIK
from vlastudio.deploy.robot.rebot_b601rs.bus import RebotBus


class FakeBus:
    def __init__(self, channel, read_only=False, **kwargs):
        self.read_only, self.armed = read_only, False
        self.q = np.array([.1,.3,.6,.1,-.2,.1,.5])
        self.targets = []
        self.closed = False
        self.events = []
        self.output_started = False
        self.velocity = kwargs.get('velocity',.5)
        self.target_velocities = []

    def connect(self): return True
    def snapshot(self): return self.q.copy(), np.full(7,time.monotonic())
    def enable(self):
        if self.read_only: raise RuntimeError('read only')
        self.armed = True
        self.output_started = True
    def set_target(self,q):
        if not self.armed or self.read_only: raise RuntimeError('not enabled')
        self.targets.append(np.array(q).copy())
        self.target_velocities.append(self.velocity)
        self.q = np.array(q).copy()
    def cancel_pending_target(self): self.events.append('cancel')
    def set_velocity_limit(self,velocity,*,homing=False):self.velocity=velocity
    def disable(self, *, closing=False):
        self.events.append(('disable',self.q.copy()))
        self.armed = False
        self.output_started = False
        return True
    def close(self): self.closed=True
    def gripper_diagnostics(self):return None


class RebotTests(unittest.TestCase):
    def setUp(self):
        # Fail before opening a real CAN bus even if the fake injection regresses.
        guard = patch.object(RebotBus, 'connect', side_effect=AssertionError('Real CAN forbidden in unit tests'))
        guard.start()
        self.addCleanup(guard.stop)

    def make(self, **kwargs):
        with patch.object(RebotB601RS,'BUS_CLS',FakeBus):
            robot = RebotB601RS(**kwargs)
            robot.connect()
        self.assertIsInstance(robot.bus, FakeBus)
        def cleanup():
            with patch('vlastudio.deploy.robot.rebot_b601rs.robot.time.sleep',lambda _:None):
                robot.shutdown()
        self.addCleanup(cleanup)
        return robot

    def test_connect_never_enables_or_moves(self):
        r = self.make()
        self.assertFalse(r.bus.armed)
        self.assertFalse(r.bus.targets)

    def test_read_only_cannot_enable_or_publish(self):
        r = self.make(read_only=True)
        with self.assertRaises(RuntimeError): r.enable()
        with self.assertRaises(RuntimeError): r.publish_action(r.get_observation()['qpos'])

    def test_qpos_uses_measured_radians_and_normalized_gripper(self):
        r = self.make()
        r.enable()
        q = np.array([.2,.3,.4,.1,-.1,.2,.75])
        r.publish_action(q)
        np.testing.assert_allclose(r.bus.targets[-1][:6],q[:6])
        self.assertAlmostEqual(r.bus.targets[-1][6],.75*r.gripper_open_rad)
        np.testing.assert_allclose(r.get_observation()['qpos'],q,atol=1e-7)

    def test_invalid_or_out_of_range_target_is_rejected(self):
        r = self.make()
        r.enable()
        for q in (np.zeros(6), np.full(7,np.nan), np.array([8,0,0,0,0,0,0])):
            with self.assertRaises(ValueError): r.publish_action(q)
        self.assertFalse(r.bus.targets)

    def test_home_is_all_motor_zeros_not_startup_pose(self):
        r = self.make()
        r.enable()
        with patch('vlastudio.deploy.robot.rebot_b601rs.robot.time.sleep',lambda _:None):
            self.assertTrue(r.set_home())
        np.testing.assert_allclose(r.bus.q,np.zeros(7),atol=1e-8)
        self.assertGreater(len(r.bus.targets),100)
        self.assertIsNone(r._rel_ee_anchor_pose)

    def test_absolute_ee_converts_to_joint_command(self):
        from vlastudio.deploy.robot.alicia_d.robot import _xyz_rpy_from_transform_matrix
        r = self.make(control_mode='ee')
        q = r.get_observation()['qpos'][:6].astype(float)
        r.ik_solver.reset(q)
        ee = np.r_[_xyz_rpy_from_transform_matrix(r.ik_solver.compute_fk(q)),.4]
        out = r.process_action({'action':ee})['action']
        self.assertIsNotNone(out)
        self.assertLess(np.linalg.norm(r.ik_solver.compute_fk(out[:6])-r.ik_solver.compute_fk(q)),.005)
        self.assertEqual(out[6],.4)

    def test_relative_vr_anchor_and_freeze(self):
        r = self.make(control_mode='rel_ee')
        out = r.process_action(dict(action=np.zeros(7),unsqueeze_active=True,anchor_just_set=True))
        self.assertIsNone(out['action'])
        self.assertIsNotNone(r._rel_ee_anchor_pose)
        out = r.process_action(dict(action=np.zeros(7),unsqueeze_active=False,anchor_just_set=False))
        self.assertIsNone(out['action'])
        self.assertFalse(r.bus.targets)

    def test_absolute_trigger_fully_closes_at_stationary_anchor(self):
        r = self.make(control_mode='rel_ee', gripper_absolute=True)
        r.enable()
        msg = dict(action=np.zeros(7), unsqueeze_active=True)
        self.assertIsNone(r.process_action(dict(msg, anchor_just_set=True))['action'])
        joints = r.get_observation()['joint_angles']
        r.publish_action(np.r_[joints, .7])
        out = r.process_action(msg)['action']
        r.publish_action(out)
        np.testing.assert_array_equal(r.bus.targets[-1], np.r_[joints, 0.])
        self.assertIsNone(r.process_action(dict(msg, unsqueeze_active=False))['action'])

    def test_saved_ee_uses_rebot_geometry(self):
        r=self.make()
        obs=r.get_observation()
        out=r.enrich_saved_frame(obs)
        self.assertEqual(out['state_ee'].shape,(7,))
        np.testing.assert_allclose(out['state_ee'][:3],r.ik_solver.compute_fk(obs['joint_angles'])[:3,3],atol=1e-7)
        self.assertEqual(obs['ee_frame'],'link6')
        self.assertEqual(out['ee_frame'],'link6')

    def test_recording_keeps_the_configured_frame_without_cache_leak(self):
        wrist=self.make()
        tip=self.make(ik_end_frame='gripper_end')
        for robot in (wrist,tip,wrist):
            obs=robot.get_observation()
            out=robot.enrich_saved_frame(obs)
            self.assertEqual(out['ee_frame'],robot.ik_end_frame)
            np.testing.assert_allclose(out['state_ee'][:3],
                robot.ik_solver.compute_fk(obs['joint_angles'])[:3,3],atol=1e-7)

    def test_direct_control_keeps_closed_gripper_and_failed_ik_sends_nothing(self):
        r=self.make()
        r.enable()
        r.bus.q[6]=0
        q=r.get_observation()['joint_angles']
        r.set_joints(q)
        self.assertEqual(r.bus.targets[-1][6],0)
        T=r.ik_solver.compute_fk(q)
        self.assertTrue(r.set_pose(T))
        from vlastudio.deploy.robot.alicia_d.robot import _xyz_rpy_from_transform_matrix
        self.assertTrue(r.set_state_ee(np.r_[_xyz_rpy_from_transform_matrix(T),.2]))
        self.assertAlmostEqual(r.bus.targets[-1][6],.2*r.gripper_open_rad)
        count=len(r.bus.targets)
        T[0,3]=20
        self.assertFalse(r.set_pose(T))
        self.assertFalse(r.set_state_ee([20,0,0,0,0,0,.3]))
        self.assertEqual(len(r.bus.targets),count)

    def test_exit_parks_arm_before_disable_and_preserves_gripper(self):
        r=self.make()
        r.enable()
        bus=r.bus
        before=bus.q.copy()
        with patch('vlastudio.deploy.robot.rebot_b601rs.robot.time.sleep',lambda _:None):
            self.assertTrue(r.shutdown())
        self.assertEqual(bus.events[0],'cancel')
        kind, at_disable=bus.events[-1]
        self.assertEqual(kind,'disable')
        np.testing.assert_allclose(at_disable[:6],0,atol=1e-7)
        self.assertAlmostEqual(at_disable[6],before[6],places=6)
        np.testing.assert_allclose(bus.targets[199][[1,2]],before[[1,2]],atol=1e-7)
        self.assertTrue(bus.closed)
        self.assertTrue(r.shutdown())  # idempotent; no second trajectory

    def test_failed_park_never_disables(self):
        r=self.make()
        r.enable()
        bus=r.bus
        with patch.object(r,'set_home',return_value=False), patch('vlastudio.deploy.robot.rebot_b601rs.robot.time.sleep',lambda _:None):
            self.assertFalse(r.shutdown())
        self.assertFalse(any(isinstance(e,tuple) and e[0]=='disable' for e in bus.events))
        self.assertTrue(bus.closed)
        self.assertFalse(r.shutdown())

    def test_stale_feedback_never_homes_or_disables(self):
        r=self.make()
        r.enable()
        bus=r.bus
        with patch.object(bus,'cancel_pending_target',side_effect=RuntimeError('stale')), patch.object(r,'set_home') as home:
            self.assertFalse(r.shutdown())
            home.assert_not_called()
        self.assertFalse(any(isinstance(e,tuple) for e in bus.events))

    def test_first_error_survives_failed_shutdown(self):
        from loguru import logger
        r = self.make()
        r.control_shm_name = None
        bus = r.bus
        messages = []
        sink = logger.add(messages.append, format='{message}', level='ERROR', diagnose=False)
        self.addCleanup(logger.remove, sink)
        with patch.object(r, 'create_shm'), patch.object(r, 'read_action', side_effect=ValueError('input failed')), \
                patch.object(bus, 'cancel_pending_target', side_effect=RuntimeError('CAN timeout')), \
                patch('signal.signal'), self.assertRaises(SystemExit) as stopped:
            r.start()
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn('Device error before shutdown: input failed', messages[0])
        self.assertIn('SAFE EXIT FAILED during cancel pending target: CAN timeout', messages[-1])
        self.assertFalse(bus.targets)

    def test_repeated_exit_signals_only_request_stop(self):
        r=self.make()
        r._request_exit(2,None)
        r._request_exit(15,None)
        self.assertTrue(r._exit_requested)
        self.assertFalse(r.bus.closed)
        self.assertFalse(r.bus.targets)

    def test_supervisor_exit_event_is_seen_before_enable(self):
        event=SimpleNamespace(is_set=lambda:True)
        r=self.make(exit_event=event)
        self.assertTrue(r._exit_pending())
        self.assertFalse(r.bus.armed)
        with patch('vlastudio.deploy.robot.rebot_b601rs.robot.signal.signal'), patch.object(r,'connect') as connect:
            r.start()
            connect.assert_not_called()

    def test_stop_hold_waits_for_smoothing_to_catch_up(self):
        r=self.make(control_mode='rel_ee',joint_smooth_alpha=.15)
        r.process_action(dict(action=np.zeros(7),unsqueeze_active=True,anchor_just_set=True))
        goal=r.get_observation()['joint_angles'].astype(float)
        goal[0]+=.08
        with patch.object(r,'_solve_teleop_ik',return_value=(True,goal,1,0)):
            for _ in range(70):
                out=r.process_action(dict(action=np.array([.01,0,0,0,0,0,.5]),unsqueeze_active=True))
        self.assertLess(max(abs(out['action'][:6]-goal)),r.joint_still_rad)

    def test_stop_hold_resumes_on_cumulative_slow_motion(self):
        r=self.make(control_mode='rel_ee',joint_smooth_alpha=.5)
        r.process_action(dict(action=np.zeros(7),unsqueeze_active=True,anchor_just_set=True))
        q=r.get_observation()['joint_angles'].astype(float)
        r._smooth_qpos=q.copy()
        r._prev_ik_qpos=q.copy()
        r._teleop_holding_still=True
        r._still_count=100
        def solve(*args,**kwargs):
            q[0]+=np.deg2rad(.1)
            return True,q.copy(),1,0
        with patch.object(r,'_solve_teleop_ik',side_effect=solve):
            for _ in range(20):
                out=r.process_action(dict(action=np.array([.01,0,0,0,0,0,.5]),unsqueeze_active=True))
        self.assertGreater(out['action'][0]-r.get_observation()['joint_angles'][0],np.deg2rad(1.))

    def test_faster_vr_does_not_increase_home_trajectory_speed(self):
        r=self.make(velocity_limit_rad_s=1.,home_velocity_rad_s=.5,home_duration_s=.1)
        r.enable()
        initial=r.get_observation()['qpos'].astype(float)
        delays=[]
        with patch('vlastudio.deploy.robot.rebot_b601rs.robot.time.sleep',side_effect=delays.append):
            self.assertTrue(r.set_home())
        points=np.vstack([initial[:6]]+[q[:6] for q in r.bus.targets])
        steps=np.max(np.abs(np.diff(points,axis=0)),axis=1)
        self.assertLessEqual(max(steps/np.asarray(delays)),.5+1e-6)
        self.assertEqual(set(r.bus.target_velocities),{.5})
        self.assertEqual(r.bus.velocity,1.)

    def test_home_accepts_observed_closed_gripper_without_loosening_arm_guard(self):
        r=self.make()
        obs=r.get_observation()
        raw=np.array([.008863,.001747,-.000341,.001035,-.004487,.002800,.036931])
        obs['motor_positions']=raw
        self.assertTrue(r._home_reached(obs,np.zeros(7),True))
        raw[0]=np.deg2rad(1.1)
        self.assertFalse(r._home_reached(obs,np.zeros(7),True))
        raw[:6]=0
        raw[6]=.08
        self.assertFalse(r._home_reached(obs,np.zeros(7),True))
        self.assertTrue(r._home_reached(obs,np.zeros(7),False))

    def test_vr_far_target_has_bounded_steps_and_never_reanchors(self):
        r=self.make(control_mode='rel_ee',velocity_limit_rad_s=1.,joint_smooth_alpha=.5)
        r.enable()
        r.process_action(dict(action=np.zeros(7),unsqueeze_active=True,anchor_just_set=True))
        anchor=r._rel_ee_anchor_pose.copy()
        previous=r.get_observation()['joint_angles'].astype(float)
        clock=[time.monotonic()]
        with patch('vlastudio.deploy.robot.rebot_b601rs.robot.time.monotonic',side_effect=lambda:clock[0]), \
             patch.object(r,'_reanchor_to_last_good_pose',side_effect=AssertionError('No automatic reanchor')), \
             patch.object(r,'_warning_throttled_ik',side_effect=AssertionError('No jump/failure loop')):
            for distance in ([2.]*60+[0.]*60):
                clock[0]+=1/60
                out=r.process_action(dict(action=np.array([distance,0,0,0,0,0,.5]),unsqueeze_active=True))['action']
                self.assertIsNotNone(out)
                self.assertLessEqual(max(abs(out[:6]-previous)),1/30+1e-6)
                r.publish_action(out)
                previous=out[:6].copy()
        np.testing.assert_array_equal(r._rel_ee_anchor_pose,anchor)
        np.testing.assert_array_equal(r._rel_ee_pos_offset,np.zeros(3))


class IKTests(unittest.TestCase):
    def test_tracking_converges_on_nearby_reachable_pose(self):
        import pinocchio as pin
        s=RebotIK(URDF)
        q=np.array([.1,.4,.6,.2,.3,.1])
        goal=q+np.array([.06,.02,-.03,.04,.01,-.02])
        target=s.compute_fk(goal)
        for _ in range(60):
            q,status=s.track(target,q,1/60,measured=q)
        error=pin.log6(pin.SE3(s.compute_fk(q)).inverse()*pin.SE3(target)).vector
        self.assertLessEqual(np.linalg.norm(error),s.tol)

    def test_tracking_does_not_build_more_lead_when_motor_feedback_lags(self):
        s=RebotIK(URDF)
        q=np.array([.5,.4,.6,.2,.3,.1])
        measured=q.copy();measured[0]-=.3
        goal=q.copy();goal[0]+=.5
        result,_=s.track(s.compute_fk(goal),q,.02,measured=measured)
        self.assertLessEqual(result[0],q[0]+1e-10)

    def test_tracking_log_jacobian_matches_finite_difference(self):
        import pinocchio as pin
        s=RebotIK(URDF)
        q=np.array([.1,.4,.6,.2,.3,.1])
        target=pin.SE3(s.compute_fk(q+np.array([.1,.02,-.03,.04,.01,-.02])))
        relative=pin.SE3(s.compute_fk(q)).inverse()*target
        e=pin.log6(relative).vector
        pin.computeJointJacobians(s.reduced_model,s.reduced_data,q)
        J=pin.Jlog6(relative.inverse())@pin.getFrameJacobian(s.reduced_model,s.reduced_data,s.frame,pin.LOCAL)
        dq=np.array([1.,-.5,.2,.3,-.1,.4])*1e-6
        trial=pin.log6(pin.SE3(s.compute_fk(q+dq)).inverse()*target).vector
        np.testing.assert_allclose(trial-e,-J@dq,atol=1e-11,rtol=1e-4)

    def test_tracking_far_pose_improves_error_with_small_joint_step(self):
        import pinocchio as pin
        s=RebotIK(URDF)
        q=np.array([.1,.4,.6,.2,.3,.1])
        goal=q.copy();goal[0]+=.75
        target=s.compute_fk(goal)
        before=np.linalg.norm(pin.log6(pin.SE3(s.compute_fk(q)).inverse()*pin.SE3(target)).vector)
        result,status=s.track(target,q,1/60,measured=q)
        self.assertLessEqual(max(abs(result-q)),1/60+1e-8)
        self.assertLess(status['residual'],before)

    def test_tracking_unreachable_pose_stays_finite_and_within_limits(self):
        s=RebotIK(URDF)
        q=np.array([.1,.4,.6,.2,.3,.1])
        target=s.compute_fk(q);target[0,3]=20
        for _ in range(80):
            result,status=s.track(target,q,.015,measured=q)
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(np.all(result>=s.lower-1e-9))
            self.assertTrue(np.all(result<=s.upper+1e-9))
            self.assertLessEqual(max(abs(result-q)),.015+1e-8)
            q=result
    def test_default_tcp_is_link6_and_old_tip_is_distinct(self):
        wrist=RebotIK(URDF)
        tip=RebotIK(URDF,end_frame='gripper_end')
        self.assertEqual(wrist.end_frame,'link6')
        self.assertEqual(wrist.frame,wrist.reduced_model.getFrameId('link6'))
        for q in (np.zeros(6),np.array([.1,.4,.6,.2,-.3,.1])):
            A=wrist.compute_fk(q)
            B=tip.compute_fk(q)
            # Test against the URDF's known fixed link6 -> gripper_end transform.
            np.testing.assert_allclose((np.linalg.inv(A)@B)[:3,3],[0,0,.16621],atol=1e-8)

    def test_official_model_chain_and_small_pose_roundtrips(self):
        s=RebotIK(URDF,budget_s=.1)
        rng=np.random.default_rng(123)
        for _ in range(20):
            q=s.lower+(s.upper-s.lower)*rng.uniform(.2,.8,6)
            goal=np.clip(q+rng.uniform(-.01,.01,6),s.lower,s.upper)
            s.reset(q)
            ok,result,_,error=s.solve(s.compute_fk(goal))
            self.assertTrue(ok,error)
            self.assertLess(np.linalg.norm(s.compute_fk(result)-s.compute_fk(goal)),.004)

    def test_unreachable_target_never_changes_warmstart(self):
        s=RebotIK(URDF)
        q=np.array([0,.4,.6,0,0,0])
        s.reset(q)
        T=s.compute_fk(q);T[0,3]=20
        self.assertFalse(s.solve(T)[0])
        np.testing.assert_array_equal(s.init_data,q)


class FakeMotor:
    def __init__(self,mid):
        self.mid=mid;self.pos=.2;self.calls=[]
        self.reporting=False
        self.status=0
    def robstride_set_active_report(self, enabled): self.reporting=enabled
    def robstride_ping_host_id(self,*args): return self.mid,254
    def robstride_get_param_f32_host_id(self,*args): return self.pos
    def robstride_get_fault_report(self): return 0,0
    def close(self): pass
    def enable(self): self.calls.append('enable');self.status=2
    def ensure_mode(self,*a): self.calls.append('mode')
    def send_mit(self,*a): self.calls.append('motion')
    def disable(self): self.calls.append('disable');self.status=0
    def request_feedback(self): pass
    def get_state(self): return SimpleNamespace(status_code=0)
    def set_zero_position(self): raise AssertionError('Calibration is forbidden')


class FakeController:
    def __init__(self,channel): self.motors=[]
    def add_robstride_motor(self,mid,*args):
        m=FakeMotor(mid);self.motors.append(m);return m
    def close_bus(self): pass
    def close(self): pass


class ActiveReportTests(unittest.TestCase):
    def test_linux_data_filters_and_kernel_timestamp(self):
        import socket
        import struct
        from unittest.mock import Mock
        from vlastudio.deploy.robot.rebot_b601rs.feedback import ActiveReports
        sock=Mock()
        with patch.object(socket,'socket',return_value=sock):
            reader=ActiveReports('testrebot')
        filters=sock.setsockopt.call_args_list[0].args[2]
        actual=list(struct.iter_unpack('=II',filters))
        self.assertEqual(actual,[(0x980000fd | (mid<<8),0xdf00ffff) for mid in range(1,8)])
        # Position zero from motor 3, mode 2, fault bit 1, timestamp 100 ms ago.
        frame=struct.pack('=IB3x8s',0x988203fd,8,bytes.fromhex('7fff000000000000'))
        ancillary=[(socket.SOL_SOCKET,reader.timestamp_option,struct.pack('@ll',20,0))]
        sock.recvmsg.side_effect=[(frame,ancillary,0,None),BlockingIOError()]
        with patch.object(time,'monotonic',return_value=10.1),patch.object(time,'time',return_value=20.1):
            reports=list(reader.drain())
        self.assertEqual(len(reports),1)
        np.testing.assert_allclose(reports[0],[2,10,0,2,2])
        reader.close()


class TransportTests(unittest.TestCase):
    def setUp(self):
        def receiver(bus, channel):
            def drain():
                return [(i,time.monotonic(),m.pos,m.status,0) for i,m in
                        enumerate(bus.motors) if m.reporting]
            return SimpleNamespace(drain=drain,close=lambda:None)
        factory=patch.object(RebotBus,'REPORTS_CLS',receiver)
        factory.start()
        self.addCleanup(factory.stop)

    def test_startup_mismatch_restores_only_new_streams(self):
        bus=RebotBus('testrebot',controller_factory=FakeController)
        controller=FakeController('testrebot')
        def add(mid,*args):
            m=FakeMotor(mid)
            m.reporting=(mid==1)
            controller.motors.append(m)
            return m
        controller.add_robstride_motor=add
        bus.factory=lambda channel:controller
        with patch.object(bus,'_read_position',return_value=.25):
            with self.assertRaisesRegex(RuntimeError,'disagrees'):
                bus.connect()
        self.assertEqual([m.reporting for m in controller.motors],[True]+[False]*6)
        self.assertTrue(all(not m.calls for m in controller.motors))

    def test_worker_waiting_for_disable_lock_exits_without_polling(self):
        from threading import Event, Thread
        bus = RebotBus('testrebot', controller_factory=FakeController)
        bus.motors = [FakeMotor(i) for i in range(1, 8)]
        bus.armed = True
        bus.reports=bus.REPORTS_CLS(bus.channel)
        for motor in bus.motors: motor.reporting=True
        entered = Event()
        lock = bus._io
        class ObservedLock:
            def __enter__(self):
                entered.set()
                lock.acquire()
            def __exit__(self, *args):
                lock.release()
        bus._io = ObservedLock()
        with lock:
            worker = Thread(target=bus._loop, daemon=True)
            worker.start()
            self.assertTrue(entered.wait(1))
            self.assertTrue(bus.disable(closing=True))
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(bus.fault)
        self.assertTrue(all(m.calls == ['disable'] for m in bus.motors))

    def test_old_queued_report_only_stops_the_worker_for_the_stale_id(self):
        bus = RebotBus('testrebot', controller_factory=FakeController)
        bus.motors = [FakeMotor(i) for i in range(1, 8)]
        now=time.monotonic()
        bus.times.fill(now-.5)
        bus.reports=SimpleNamespace(drain=lambda:[
            (i,now-.4 if i==2 else now,.2,2,0) for i in range(7)])
        bus.armed=True
        bus._loop()
        with self.assertRaisesRegex(RuntimeError, r'older than 0.3s for IDs \[3\]'):
            bus.snapshot()
        self.assertTrue(all(not m.calls for m in bus.motors))

    def make_vr_bus(self):
        import yaml
        path = Path(__file__).resolve().parents[1] / 'src/vlastudio/configs/robot/rebot_b601rs_rel_ee.yaml'
        robot = RebotB601RS(**yaml.safe_load(path.read_text())[0]['args'])
        with patch.object(robot, 'BUS_CLS', side_effect=lambda *args, **kwargs:
                RebotBus(*args, controller_factory=FakeController, **kwargs)), \
                patch.object(RebotBus, 'connect', return_value=True), \
                patch.object(robot, 'get_observation', return_value={
                    'joint_angles':np.zeros(6), 'qpos':np.zeros(7)}):
            robot.connect()
        return robot.bus

    def test_vr_clamp_reverses_from_blocked_position_and_brakes_on_stall(self):
        bus = self.make_vr_bus()
        bus.q = bus.command = np.full(7, 2.)
        bus.command = bus.q.copy()
        motor = SimpleNamespace(send_mit=lambda *args: None)
        goal = bus.q.copy()
        goal[6] = 0.
        for _ in range(300):
            bus._advance_command(goal, .01)
        self.assertLess(abs(bus.command[6] - bus.q[6]), .17)
        bus.times[6] = time.monotonic()
        bus._send_gripper(motor, bus.command[6], bus.q[6], goal=0.)
        self.assertEqual(bus.gripper_diagnostics()['torque_term_nm'], -1.)
        self.assertEqual(bus.gripper_diagnostics()['damping'], 1.5)
        goal[6] = 4.7
        for tick in range(3):
            bus._advance_command(goal, .01)
            bus._send_gripper(motor, bus.command[6], bus.q[6], goal=goal[6])
            self.assertGreaterEqual(bus.gripper_diagnostics()['torque_term_nm'], 0.)
        self.assertGreater(bus.gripper_diagnostics()['torque_term_nm'], 0.)

    def test_fast_damping_only_for_fresh_motion_toward_a_distant_goal(self):
        bus = self.make_vr_bus()
        calls = []
        motor = SimpleNamespace(send_mit=lambda *args: calls.append(args))
        for goal, speed, age, kd in (
                (4., 1., .02, .4), (0., -1., .02, .4), (1.8, -1., .02, 1.5),
                (4., -1., .02, 1.5), (4., 0., .02, 1.5), (4., 1., .15, 1.5)):
            with self.subTest(goal=goal, speed=speed, age=age):
                bus.times[6] = time.monotonic() - age
                bus._gripper_measured_speed = speed
                bus._send_gripper(motor, goal, 2., goal=goal)
                self.assertAlmostEqual(calls[-1][3], kd)
                self.assertLessEqual(abs(calls[-1][4]), 1.)
        bus.set_velocity_limit(.5, homing=True)
        bus.times[6] = time.monotonic()
        bus.q = np.zeros(7)
        bus.command = bus.q.copy()
        bus._advance_command(np.ones(7), .01)
        np.testing.assert_allclose(bus.command, [.005]*6 + [.01])
        bus._send_gripper(motor, .4, 0., goal=4.)
        self.assertEqual(calls[-1][3], 1.5)

    def test_gripper_full_travel_ramp_is_independent_of_arm_speed(self):
        bus=RebotBus('testrebot',velocity=.5,gripper_velocity=4.,controller_factory=FakeController)
        bus.command=np.zeros(7)
        goal=np.r_[np.ones(6),4.71238898038]
        for _ in range(120):bus._advance_command(goal,.01)
        self.assertAlmostEqual(bus.command[6],goal[6])
        np.testing.assert_allclose(bus.command[:6],.6)

    def test_gripper_keeps_one_nm_cap_and_restores_damping(self):
        bus=RebotBus('testrebot',controller_factory=FakeController)
        motor=SimpleNamespace(send_mit=lambda *args:calls.append(args))
        calls=[]
        bus._send_gripper(motor,4.7,0.)
        bus._send_gripper(motor,0.,4.7)
        bus._send_gripper(motor,.5,.5)
        self.assertEqual([args[4] for args in calls],[1.,-1.,0.])
        for args in calls:self.assertEqual(args[:4],(0.,0.,0.,1.5))

    def test_gripper_target_noise_has_no_alternating_torque(self):
        bus=RebotBus('testrebot',controller_factory=FakeController)
        calls=[]
        motor=SimpleNamespace(send_mit=lambda *args:calls.append(args))
        for measured in (.5,.501,.49,.52,.48):
            bus._send_gripper(motor,.5,measured)
        self.assertTrue(all(a[4]==0 for a in calls))
        bus._send_gripper(motor,.5,.469)
        self.assertAlmostEqual(calls[-1][4],.012)
        diagnostics=bus.gripper_diagnostics()
        self.assertAlmostEqual(diagnostics['measured_rad'],.469)
        self.assertAlmostEqual(diagnostics['torque_term_nm'],.012)

    def test_cancel_then_disable_stops_all_further_mit_output(self):
        bus=RebotBus('testrebot',controller_factory=FakeController)
        try:
            bus.connect()
            bus.enable()
            bus.set_target(np.full(7,.8))
            time.sleep(.025)
            bus.cancel_pending_target()
            with bus._io:
                np.testing.assert_array_equal(bus.target,bus.command)
            with patch('vlastudio.deploy.robot.rebot_b601rs.bus.time.sleep',lambda _:None):
                self.assertTrue(bus.disable())
            counts=[len(m.calls) for m in bus.motors]
            time.sleep(.04)
            self.assertEqual(counts,[len(m.calls) for m in bus.motors])
            for m in bus.motors:self.assertEqual(m.calls[-1],'disable')
            self.assertFalse(bus.armed)
            with self.assertRaises(RuntimeError):bus.set_target(np.zeros(7))
        finally:bus.close()

    def test_disable_attempts_other_motors_after_one_failure(self):
        bus=RebotBus('testrebot',controller_factory=FakeController)
        try:
            bus.connect()
            with patch.object(bus.motors[0],'disable',side_effect=RuntimeError('lost ack')), patch('vlastudio.deploy.robot.rebot_b601rs.bus.time.sleep',lambda _:None):
                with self.assertRaisesRegex(RuntimeError,'ID 1'):bus.disable()
            for m in bus.motors[1:]:self.assertIn('disable',m.calls)
        finally:bus.close()

    def test_real_transport_read_only_has_no_enable_mode_or_motion_calls(self):
        bus=RebotBus('testrebot',read_only=True,controller_factory=FakeController)
        try:
            bus.connect()
            time.sleep(.09)
            self.assertTrue(np.isfinite(bus.snapshot()[0]).all())
            for m in bus.motors: self.assertFalse(m.calls)
            with self.assertRaises(RuntimeError): bus.enable()
            with self.assertRaises(RuntimeError): bus.set_target(np.zeros(7))
        finally: bus.close()


class SupervisorTests(unittest.TestCase):
    def test_arm_is_not_killed_while_parking(self):
        path=Path(__file__).resolve().parents[1]/'scripts/rebot_b601rs_vr_tele.py'
        spec=importlib.util.spec_from_file_location('rebot_launch',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Process:
            def __init__(self):self.joins=0;self.killed=False;self.stops=0
            def is_alive(self):return self.joins<12
            def terminate(self):self.stops+=1
            def join(self,timeout=None):self.joins+=1
            def kill(self):self.killed=True
        p=Process()
        module.stop_devices([{'type':'vlastudio.deploy.robot.rebot_b601rs.RebotB601RS'}],[p])
        self.assertEqual(p.stops,1)
        self.assertFalse(p.killed)
        self.assertEqual(p.joins,12)


if __name__=='__main__': unittest.main()
