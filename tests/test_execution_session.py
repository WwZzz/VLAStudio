"""Acceptance tests exercise native VLAStudio subprocesses, SHM and scheduling.

Run with CUDA_VISIBLE_DEVICES='' and PYTHONPATH=src:tests. Set
VLA_NATIVE_ACCEPTANCE_OUTPUT to a checkpoint subdirectory to retain videos.
No physical robot, model checkpoint, network service or GPU is used.
"""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from vlastudio.deploy.execution_session import ExecutionSession


def records(path):
    path = Path(path)
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A concurrently appended final line may not have been flushed yet.
            continue
    return result


def eventually(predicate, *, timeout=10, description='condition'):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError(f'Timed out waiting for {description}')


class NativeExecutionTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '',
                         "Run native acceptance with CUDA_VISIBLE_DEVICES='' (CPU only)")
        keep = os.environ.get('VLA_NATIVE_ACCEPTANCE_OUTPUT')
        if keep:
            self.root = Path(keep) / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            directory = tempfile.TemporaryDirectory(prefix='native-execution-')
            self.addCleanup(directory.cleanup)
            self.root = Path(directory.name)
        self.sessions = []
        self.addCleanup(self._close_all)

    def _close_all(self):
        for session in reversed(self.sessions):
            session.close()

    def create_session(self, backend='sim', **overrides):
        fixture = self.root / backend
        fixture.mkdir()
        name = f'acceptance_{uuid.uuid4().hex[:12]}'
        if backend == 'sim':
            config = {'type': 'execution_session_fixtures.CounterEnvironment',
                      'args': {'fixture_dir': str(fixture),
                               **overrides.pop('environment_args', {})}}
        else:
            config = [{'type': 'execution_session_fixtures.CounterRobot',
                       'args': {'fixture_dir': str(fixture), 'name': name,
                                'fps': 50, **overrides.pop('environment_args', {})}}]
        configuration = {
            'backend': {'type': backend, 'config': config},
            'policy': {'checkpoint': str(fixture), 'device': 'cpu', 'dataset_id': '',
                       'loader': 'execution_session_fixtures:load_policy',
                       **overrides.pop('policy_args', {})},
            'action_manager': 'basic', 'control_hz': 20, 'record_hz': 20,
            'startup_timeout_s': 60, 'inference_timeout_s': 3,
            'observation_timeout_s': 3, 'action_shape': [3], **overrides,
        }
        session = ExecutionSession(configuration, output_dir=fixture / 'runtime')
        self.sessions.append(session)
        return session, fixture

    @staticmethod
    def intent(code, **extra):
        return {'skill_id': f'skill_{code}', 'skill_index': code,
                'instruction': f'acceptance intent {code}', **extra}

    def published(self, root):
        return records(root / 'published.jsonl')

    def test_generic_context_without_language_or_skill_fields(self):
        session, fixture = self.create_session()
        receipt = session.update_context({'target': 9}, limits={'max_steps': 3})
        result = session.wait(receipt['version'], timeout_s=10)
        self.assertEqual(result['state'], 'exhausted', result)
        self.assertEqual(len(self.published(fixture)), 3)
        self.assertTrue(all(row['action'][0] == 9 for row in self.published(fixture)))
        applied = [row['context'] for row in records(fixture/'policy.jsonl') if row['event'] == 'context']
        self.assertEqual(applied[-1]['target'], 9)
        self.assertNotIn('skill_latent', applied[-1])

    def assert_video(self, status):
        import imageio.v2 as imageio
        video = Path(status['video'])
        self.assertTrue(video.is_file(), status)
        self.assertGreater(video.stat().st_size, 0)
        reader = imageio.get_reader(str(video))
        try:
            frame = reader.get_data(0)
            self.assertEqual(frame.ndim, 3)
            self.assertGreater(frame.size, 0)
        finally:
            reader.close()

    def test_same_api_runs_native_sim_and_device_pipelines(self):
        for backend in ('sim', 'robot'):
            with self.subTest(backend=backend):
                session, fixture = self.create_session(backend)
                initial = session.status()
                self.assertEqual(initial['state'], 'idle')
                time.sleep(.15)
                self.assertEqual(self.published(fixture), [])
                receipt = session.set_intent(self.intent(7), limits={'max_steps': 4})
                result = session.wait(receipt['version'], timeout_s=10)
                self.assertEqual(result['state'], 'exhausted', result)
                self.assertEqual(result['control_cycles'], 4)
                emitted = eventually(lambda: len(self.published(fixture)) >= 4,
                                     description=f'{backend} publications')
                self.assertTrue(emitted)
                actions = self.published(fixture)
                self.assertEqual(len(actions), 4)
                self.assertTrue(all(event['action'][0] == 7 for event in actions))
                policy = records(fixture / 'policy.jsonl')
                policy_pids = {event['pid'] for event in policy}
                device_pids = {event['pid'] for event in actions}
                self.assertNotIn(os.getpid(), policy_pids)
                self.assertNotIn(os.getpid(), device_pids)
                self.assertTrue(policy_pids.isdisjoint(device_pids))
                inference = [event for event in policy if event['event'] == 'inference_started']
                self.assertTrue(inference)
                self.assertEqual(inference[0]['raw_lang'], 'acceptance intent 7')
                self.assertGreaterEqual(inference[0]['reset_count'], 1)
                snapshot = session.observe()
                self.assertIn('image', snapshot['observation'])
                self.assertGreater(snapshot['sequence'], 0)
                self.assertEqual(snapshot['version'], receipt['version'])
                session.close()
                self.assert_video(result)
                runtime = Path(result['output_dir'])
                for relative in ('events.jsonl', 'result.json', 'traces/actions.jsonl'):
                    self.assertTrue((runtime / relative).is_file(), relative)

    def test_intent_switch_discards_inflight_old_actions(self):
        session, fixture = self.create_session()
        old = session.set_intent(self.intent(11, inference_delay_s=.65))
        eventually(lambda: any(e['event'] == 'inference_started' and e['code'] == 11
                               for e in records(fixture / 'policy.jsonl')),
                   description='old inference started')
        switched = session.set_intent(self.intent(22), limits={'max_steps': 4}, timeout_s=3)
        self.assertGreater(switched['version'], old['version'])
        after_ack = time.monotonic()
        result = session.wait(switched['version'], timeout_s=10)
        self.assertEqual(result['state'], 'exhausted', result)
        actions = self.published(fixture)
        self.assertTrue(actions)
        self.assertTrue(all(event['action'][0] == 22 for event in actions
                            if event['at'] >= after_ack))
        self.assertFalse(any(event['action'][0] == 11 for event in actions))
        log = records(fixture / 'policy.jsonl')
        self.assertTrue(any(e['event'] == 'inference_finished' and e['code'] == 11 for e in log))
        started = [e for e in log if e['event'] == 'inference_started' and e['code'] == 22]
        self.assertTrue(started)
        self.assertGreaterEqual(started[0]['reset_count'], 2)

    def test_explicit_continuation_reports_env_done_and_preserves_stage_limits(self):
        session, fixture = self.create_session(environment_args={'done_after':2},
                                                stop_on_environment_done=False)
        first = session.set_intent(self.intent(11),limits={'max_steps':4})
        result = session.wait(first['version'],timeout_s=10)
        self.assertEqual(result['state'],'exhausted')
        self.assertEqual(result['control_cycles'],4)
        self.assertTrue(result['environment_result']['done'])
        second = session.set_intent(self.intent(22),limits={'max_steps':3})
        result = session.wait(second['version'],timeout_s=10)
        self.assertEqual(result['state'],'exhausted')
        self.assertEqual(result['control_cycles'],3)
        self.assertEqual(len(self.published(fixture)),7)
        self.assertEqual(sum(e['event']=='reset' for e in records(fixture/'environment.jsonl')),1)
        events = records(fixture/'runtime/events.jsonl')
        self.assertTrue(any(e['event']=='environment_done' and not e['stopping'] for e in events))

    def test_environment_done_stops_by_default(self):
        session, fixture = self.create_session(environment_args={'done_after':2})
        receipt = session.set_intent(self.intent(11),limits={'max_steps':4})
        result = session.wait(receipt['version'],timeout_s=10)
        self.assertEqual(result['state'],'completed')
        self.assertEqual(result['stop_reason'],'environment_terminated')
        self.assertEqual(result['control_cycles'],2)

    def test_slow_intent_context_persists_across_fast_sensor_updates(self):
        session, fixture = self.create_session('robot')
        first_context = self.intent(31, skill_latent=[.125, -.75], chunk_size=1)
        first = session.set_intent(first_context, limits={'max_steps': 6})
        self.assertEqual(session.wait(first['version'], timeout_s=10)['state'], 'exhausted')
        second_context = self.intent(32, skill_latent=[-.25, .5], chunk_size=1)
        second = session.set_intent(second_context, limits={'max_steps': 6})
        acknowledged_at = time.monotonic()
        self.assertEqual(session.wait(second['version'], timeout_s=10)['state'], 'exhausted')
        snapshot = session.observe()
        # Intent is a separate control context, not sensor data injected into SHM.
        self.assertNotIn('skill_id', snapshot['observation'])
        self.assertNotIn('skill_latent', snapshot['observation'])
        calls = [event for event in records(fixture / 'policy.jsonl')
                 if event['event'] == 'inference_started']
        for context in (first_context, second_context):
            selected = [event for event in calls if event['code'] == context['skill_index']]
            self.assertGreaterEqual(len(selected), 3)
            self.assertTrue(all(event['raw_lang'] == context['instruction'] for event in selected))
            self.assertTrue(all(event['latent'] == context['skill_latent'] for event in selected))
            self.assertEqual(len({event['reset_count'] for event in selected}), 1,
                             'fast observation updates must not reset the policy')
            self.assertGreaterEqual(len({event['observation_seq'] for event in selected}), 3)
        actions = self.published(fixture)
        # An already published command can drain in BaseRobot after the ACK.
        # The guarantee is no NEW publication with the old context, and no old
        # device action after the device has started consuming the new context.
        published = records(fixture / 'runtime/traces/actions.jsonl')
        self.assertTrue(all(event['action'][0]['action'][0] == 32 for event in published
                            if event['at'] >= acknowledged_at))
        changed = next(i for i,event in enumerate(actions) if event['action'][0] == 32)
        self.assertTrue(all(event['action'][0] == 32 for event in actions[changed:]))

    def test_pause_is_bounded_during_inference_and_resume_uses_fresh_device_obs(self):
        session, fixture = self.create_session('robot')
        receipt = session.set_intent(self.intent(3, inference_delay_s=.7))
        waiter_result = {}
        def wait_in_parallel():
            try:
                waiter_result['status'] = session.wait(receipt['version'], timeout_s=15)
            except BaseException as error:
                waiter_result['error'] = error
        waiter = threading.Thread(target=wait_in_parallel, daemon=True)
        waiter.start()
        eventually(lambda: any(e['event'] == 'inference_started'
                               for e in records(fixture / 'policy.jsonl')),
                   description='slow inference started')
        started = time.monotonic()
        session.pause(timeout_s=1)
        self.assertLess(time.monotonic() - started, .6)
        self.assertEqual(session.status()['state'], 'paused')
        paused = session.observe()
        trace = Path(session.status()['output_dir']) / 'traces/actions.jsonl'
        publication_count = len(records(trace))
        # The software ACK stops new control-SHM writes. A device may still consume
        # the last pre-ACK command, so let that command drain before inspecting it.
        time.sleep(.06)
        count = len(self.published(fixture))
        time.sleep(.8)
        self.assertEqual(len(records(trace)), publication_count)
        self.assertEqual(len(self.published(fixture)), count)
        session.resume(timeout_s=3)
        eventually(lambda: len(self.published(fixture)) > count,
                   description='post-resume publication')
        action = self.published(fixture)[count]
        previous_sensor = np.asarray(paused['observation']['state']).reshape(-1)[0]
        self.assertGreater(action['action'][1], float(previous_sensor))
        session.cancel(timeout_s=2)
        self.assertEqual(session.status()['state'], 'cancelled')
        publication_count = len(records(trace))
        time.sleep(.06)
        count = len(self.published(fixture))
        time.sleep(.15)
        self.assertEqual(len(records(trace)), publication_count)
        self.assertEqual(len(self.published(fixture)), count)
        waiter.join(timeout=2)
        self.assertFalse(waiter.is_alive(), 'wait() blocked concurrent pause/cancel control')
        self.assertNotIn('error', waiter_result, waiter_result)
        self.assertEqual(waiter_result['status']['state'], 'cancelled')

    def test_timeout_interrupts_inference_without_publishing_late_result(self):
        session, fixture = self.create_session()
        receipt = session.set_intent(self.intent(4, inference_delay_s=.8),
                                     limits={'timeout_s': .2})
        result = session.wait(receipt['version'], timeout_s=3)
        self.assertIn(result['state'], ('exhausted', 'cancelled'))
        self.assertIn('timeout', str(result['stop_reason']).lower())
        time.sleep(.9)
        self.assertEqual(self.published(fixture), [])

    def test_environment_termination_stops_remaining_chunk(self):
        session, fixture = self.create_session(environment_args={'done_after': 2})
        receipt = session.set_intent(self.intent(5), limits={'max_steps': 100})
        result = session.wait(receipt['version'], timeout_s=10)
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(result['control_cycles'], 2)
        self.assertEqual(len(self.published(fixture)), 2)
        time.sleep(.15)
        self.assertEqual(len(self.published(fixture)), 2)

    def test_terminated_environment_rejects_next_intent_without_reset(self):
        session, fixture = self.create_session(environment_args={'done_after': 2})
        first = session.set_intent(self.intent(14), limits={'max_steps': 100})
        result = session.wait(first['version'], timeout_s=10)
        self.assertEqual(result['state'], 'completed', result)
        with self.assertRaisesRegex(RuntimeError, '(?i)environment|reset|terminat'):
            session.set_intent(self.intent(15), limits={'max_steps': 100})
        self.assertEqual(session.status()['version'], first['version'])
        self.assertEqual(len(self.published(fixture)), 2)
        environment = records(fixture / 'environment.jsonl')
        self.assertEqual(sum(event['event'] == 'reset' for event in environment), 1,
                         'a new intent must not silently reset a completed scene')

    def test_failed_policy_startup_cleans_its_processes_and_shared_memory(self):
        shm_root = Path('/dev/shm')
        before = {path.name for path in shm_root.iterdir()} if shm_root.exists() else set()
        with self.assertRaisesRegex(RuntimeError, '(?i)loader|startup|died|deliberate'):
            self.create_session(policy_args={'loader': 'execution_session_fixtures:load_failing_policy'})
        fixture = self.root / 'sim'
        processes = {event['pid'] for filename in ('policy.jsonl', 'environment.jsonl')
                     for event in records(fixture / filename)}
        self.assertGreaterEqual(len(processes), 2)
        for pid in processes:
            def stopped():
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return True
                return False
            eventually(stopped, description=f'failed startup process {pid} exit')
        if shm_root.exists():
            created = {path.name for path in shm_root.iterdir()} - before
            own_channels = {name for name in created
                            if any(f'_{pid}_' in name for pid in processes)}
            result = json.loads((fixture/'runtime/result.json').read_text())
            names = set(result.get('shared_memory', []))
            if result.get('observation_shm'):
                names.add(result['observation_shm'])
            own_channels.update(name for name in names if Path('/dev/shm', name).exists())
            self.assertFalse(own_channels, f'failed startup leaked native SHM: {own_channels}')
        self.assertTrue(any(event['event'] == 'closed'
                            for event in records(fixture / 'environment.jsonl')))

    def test_slow_startup_timeout_reclaims_native_children_and_channels(self):
        original_root = self.root
        # A short budget exercises startup cancellation during worker import;
        # the longer budget guarantees reaching the deliberately slow loader.
        for budget in (4, 8):
            with self.subTest(startup_timeout_s=budget):
                self.root = original_root / f'budget_{budget}'
                self.root.mkdir()
                before = {path.name for path in Path('/dev/shm').iterdir()}
                started = time.monotonic()
                with self.assertRaisesRegex((RuntimeError, TimeoutError),
                                             '(?i)timeout|timed out|startup|starting'):
                    self.create_session(startup_timeout_s=budget,
                        policy_args={'loader': 'execution_session_fixtures:slow_load_policy'})
                self.assertLess(time.monotonic()-started, budget+8,
                                'startup timeout did not bound native cleanup')
                fixture = self.root/'sim'
                policy = records(fixture/'policy.jsonl')
                if budget == 8:
                    self.assertTrue(any(event['event'] == 'slow_loader_started' for event in policy),
                                    'longer startup case must exercise the loader itself')
                processes = {event['pid'] for filename in ('policy.jsonl', 'environment.jsonl')
                             for event in records(fixture/filename)}
                self.assertTrue(processes)
                for pid in processes:
                    def stopped():
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            return True
                        return False
                    eventually(stopped, description=f'timed-out startup process {pid} exit')
                created = {path.name for path in Path('/dev/shm').iterdir()} - before
                own = {name for name in created if any(f'_{pid}_' in name for pid in processes)}
                result_path = fixture/'runtime/result.json'
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    names = set(result.get('shared_memory', []))
                    if result.get('observation_shm'):
                        names.add(result['observation_shm'])
                    own.update(name for name in names if Path('/dev/shm', name).exists())
                self.assertFalse(own, f'timed-out startup leaked channels: {own}')
                self.assertTrue(any(event['event'] == 'closed'
                                    for event in records(fixture/'environment.jsonl')))

    def test_recorder_reports_writer_finalization_failure(self):
        from vlastudio.deploy.execution_session.recording import Recorder
        class Writer:
            def __init__(self):
                self.appended = False
            def append_data(self, frame):
                self.appended = True
            def close(self):
                raise OSError('deliberate encoder finalization failure')
        writer = Writer()
        with patch('imageio.v2.get_writer', return_value=writer):
            recorder = Recorder(self.root/'recording', fps=20)
            recorder.frame(SimpleNamespace(image=np.zeros((1, 3, 8, 8), dtype=np.uint8)))
            eventually(lambda: writer.appended, description='encoder frame append')
            with self.assertRaisesRegex(RuntimeError, 'finalization.*deliberate'):
                recorder.close()
        self.assertFalse(recorder.thread.is_alive())
        self.assertTrue(recorder.events.closed)
        self.assertTrue(recorder.actions.closed)
        self.assertIn('deliberate encoder finalization failure', recorder.error)

    def test_recorder_snapshots_reused_image_buffer_before_async_encoding(self):
        from vlastudio.deploy.execution_session.recording import Recorder, observation_frame
        gate = threading.Event()
        frames = []
        class Writer:
            def append_data(self, frame):
                frames.append(np.array(frame, copy=True))
            def close(self):
                pass
        def delayed_conversion(observation):
            self.assertTrue(gate.wait(timeout=3))
            return observation_frame(observation)
        buffer = np.zeros((1, 3, 8, 8), dtype=np.uint8)
        with patch('imageio.v2.get_writer', return_value=Writer()), \
                patch('vlastudio.deploy.execution_session.recording.observation_frame',
                      side_effect=delayed_conversion):
            recorder = Recorder(self.root/'recording', fps=20)
            try:
                recorder.frame(SimpleNamespace(image=buffer))
                buffer.fill(255)
                gate.set()
                eventually(lambda: bool(frames), description='snapshot frame encoding')
            finally:
                gate.set()
                recorder.close()
        self.assertTrue(np.all(frames[0] == 0),
                        'async encoder saw a buffer reused by the next observation')

    def test_failed_policy_and_nonfinite_action_do_not_publish(self):
        for fault in ('fail_inference', 'malformed_action'):
            with self.subTest(fault=fault):
                # Each failure uses its own backend subdirectory.
                self.root = self.root / fault
                self.root.mkdir()
                session, fixture = self.create_session()
                receipt = session.set_intent(self.intent(9, **{fault: True}))
                result = session.wait(receipt['version'], timeout_s=10)
                self.assertEqual(result['state'], 'failed', result)
                self.assertTrue(result['stop_reason'])
                self.assertEqual(self.published(fixture), [])
                session.close()

    def test_environment_error_stops_publication(self):
        session, fixture = self.create_session(environment_args={'fail_after': 2})
        receipt = session.set_intent(self.intent(10), limits={'max_steps': 100})
        result = session.wait(receipt['version'], timeout_s=10)
        self.assertEqual(result['state'], 'failed', result)
        self.assertIn('deliberate acceptance environment failure', result['stop_reason'])
        self.assertEqual(len(self.published(fixture)), 2)
        time.sleep(.15)
        self.assertEqual(len(self.published(fixture)), 2)

    def test_frozen_robot_observations_fail_without_reusing_stale_sensor_data(self):
        session, fixture = self.create_session('robot', observation_timeout_s=.25)
        receipt = session.set_intent(self.intent(12, chunk_size=1))
        eventually(lambda: len(self.published(fixture)) >= 2,
                   description='fresh robot observation/action loop')
        (fixture / 'freeze_observations').touch()
        result = session.wait(receipt['version'], timeout_s=5)
        self.assertEqual(result['state'], 'failed', result)
        self.assertIn('observation', result['stop_reason'].lower())
        trace = Path(result['output_dir']) / 'traces/actions.jsonl'
        count = len(records(trace))
        time.sleep(.2)
        self.assertEqual(len(records(trace)), count)

    def test_close_cleans_native_processes_and_shared_memory(self):
        shm_root = Path('/dev/shm')
        before = {path.name for path in shm_root.iterdir()} if shm_root.exists() else set()
        session, fixture = self.create_session('robot')
        receipt = session.set_intent(self.intent(6), limits={'max_steps': 3})
        session.wait(receipt['version'], timeout_s=10)
        during = {path.name for path in shm_root.iterdir()} if shm_root.exists() else set()
        owned = set(session.status().get('shared_memory', []))
        for event in records(fixture / 'robot.jsonl'):
            if event['event'] == 'created':
                owned.update((event['name'], event['control_shm_name']))
        created = (during - before) & owned
        self.assertTrue(created, 'native execution did not create real shared memory')
        processes = {event['pid'] for filename in ('policy.jsonl', 'robot.jsonl')
                     for event in records(fixture / filename)}
        self.assertGreaterEqual(len(processes), 2)
        session.close()
        session.close()
        for pid in processes:
            def stopped():
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return True
                return False
            eventually(stopped, description=f'process {pid} exit')
        if shm_root.exists():
            self.assertFalse(created & {path.name for path in shm_root.iterdir()},
                             'native session leaked SHM')
        self.assertEqual(session.status()['state'], 'closed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
