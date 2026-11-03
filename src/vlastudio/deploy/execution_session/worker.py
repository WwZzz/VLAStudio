"""Native persistent runtime: the sole authority for action publication.

Observations, low-rate intent snapshots and control requests are independent.
This module deliberately does not invoke eval_sim/eval_real entrypoint loops.
"""
from dataclasses import asdict
from copy import deepcopy
import math
import os
from pathlib import Path
import signal
import time
import traceback
import uuid

import numpy as np

from .backends import create_backend
from .recording import Recorder


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return value


class Runtime:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output).resolve()
        self.backend = None
        self.context = None
        self.manager = None
        self.recorder = None
        self.episode_sink = None
        self.observation_writer = None
        self.intent = {}
        self.limits = {}
        self.observation = None
        self.captured_at = 0.0
        self.sequence = 0
        self.started_at = None
        self.waiting_since = None
        self.next_control = 0.0
        self.next_record = 0.0
        self.closed = False
        self.state = {'state': 'idle', 'version': 0, 'control_cycles': 0,
                      'context_applied': False, 'stop_reason': None,
                      'observation_seq': 0, 'output_dir': str(self.output),
                      'video': str(self.output / 'rollout.mp4'), 'runtime_pid': os.getpid()}

    def open(self):
        startup_deadline = time.perf_counter() + positive(self.config.get('startup_timeout_s', 60), 'startup_timeout_s')
        from vlastudio.deploy.inference import start_inference_process
        from vlastudio.deploy.action_manager import load_action_manager
        from vlastudio.deploy.shm_utils import SharedMemoryChannel
        self.control_period = 1 / positive(self.config.get('control_hz', 20), 'control_hz')
        self.record_period = 1 / positive(self.config.get('record_hz', 20), 'record_hz')
        self.inference_timeout = positive(self.config.get('inference_timeout_s', 30), 'inference_timeout_s')
        self.observation_timeout = positive(self.config.get('observation_timeout_s', 5), 'observation_timeout_s')
        self.stop_on_environment_done = self.config.get('stop_on_environment_done', True)
        if type(self.stop_on_environment_done) is not bool:
            raise ValueError('stop_on_environment_done must be a boolean')
        self.state['stop_on_environment_done'] = self.stop_on_environment_done
        self.recorder = Recorder(self.output, 1 / self.record_period)
        sink = self.config.get('episode_sink')
        if sink:
            from vlastudio.extensions import resolve
            self.episode_sink = resolve(sink['type'], kind='episode_sink')(
                output=self.output, **sink.get('args', {}))
        self.backend = create_backend(self.config['backend'], uuid.uuid4().hex[:12])
        self.backend.open()
        self.state['environment_result'] = getattr(self.backend, 'initial_result', {})
        snapshot_name = 'session_' + uuid.uuid4().hex[:12] + '_snapshot'
        self.observation_writer = SharedMemoryChannel(name=snapshot_name, max_size_mb=64, is_writer=True)
        self.state['observation_shm'] = snapshot_name
        policy = self.config['policy']
        self.context = start_inference_process(
            model_name_or_path=policy['checkpoint'], device=policy.get('device', 'cpu'),
            dataset_id=policy.get('dataset_id', ''), chunk_size=policy.get('chunk_size', -1),
            device_shm_names=self.backend.device_shm_names or None,
            device_configs=self.backend.device_configs or None,
            shm_prefix='session_' + uuid.uuid4().hex[:12],
            policy_loader=policy.get('loader'), policy_config_path=policy.get('config_path'),
            observation_timeout_s=self.observation_timeout, enable_execution_context=True,
            startup_timeout_s=max(.1, startup_deadline-time.perf_counter()-3))
        selection = self.config.get('action_manager', 'basic')
        self.manager = load_action_manager(config=selection) if isinstance(selection, dict) else load_action_manager(selection)
        if not callable(getattr(self.manager, 'select_action_nonblocking', None)):
            raise TypeError('Execution requires an ActionManager with native nonblocking selection')
        self.manager.set_inference_context(self.context)
        self.sample()
        self.recorder.event('ready', backend=self.backend.mode, inference_pid=self.context.process.pid,
                            shared_memory=list(self.context.shm_names))
        self.state['inference_pid'] = self.context.process.pid
        self.state['shared_memory'] = [*self.context.shm_names, snapshot_name]

    def sample(self):
        observation, captured_at = self.backend.observe()
        if observation is not None:
            if captured_at != self.captured_at:
                self.sequence += 1
                self.context.update_obs(observation, self.sequence, captured_at=captured_at)
            self.observation, self.captured_at = observation, captured_at
            self.state['observation_seq'] = self.sequence
            if self.sequence != getattr(self, '_snapshot_sequence', None):
                self.write_observation()

    def write_observation(self):
        if self.observation_writer is not None and self.observation is not None:
            self.observation_writer.write({'observation': asdict(self.observation),
                'sequence': self.sequence, 'captured_at': self.captured_at,
                'version': self.state['version'], 'epoch': self.state.get('epoch', 0)})
            self._snapshot_sequence = self.sequence

    def snapshot(self):
        return dict(self.state)

    def gate(self, active):
        # Epoch invalidation and buffer reset are serialized with publication.
        self.manager.reset()
        snapshot = {**self.intent, 'version': self.state['version'], 'active': active}
        self.context.set_execution_context(
            snapshot, min_obs_timestamp=time.perf_counter() if self.backend.mode == 'robot' and active else 0,
            max_observation_age_s=self.observation_timeout)
        self.state['epoch'] = self.context._epoch
        self.write_observation()
        self.waiting_since = None

    def stop(self, state, reason):
        self.gate(False)
        self.state.update(state=state, stop_reason=reason)
        self.recorder.event(state, version=self.state['version'], reason=reason)
        self.recorder.result(self.snapshot())

    def command(self, message):
        cmd = message['cmd']
        if cmd == 'status':
            return self.snapshot()
        if cmd == 'observe':
            self.sample()
            return {'observation': asdict(self.observation) if self.observation is not None else {},
                    'sequence': self.sequence, 'captured_at': self.captured_at,
                    'version': self.state['version']}
        if cmd == 'complete':
            if (message['version'] != self.state['version'] or
                    message['epoch'] != self.state.get('epoch') or self.state['state'] != 'running'):
                return {**self.snapshot(), 'accepted': False}
            evidence = dict(message.get('evidence') or {})
            sequence = evidence.get('observation_seq', self.sequence)
            if not self.state.get('stage_start_sequence', 0) <= sequence <= self.sequence:
                return {**self.snapshot(), 'accepted': False}
            reason = message.get('reason', 'external_completion')
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError('Completion reason must be a nonempty string')
            self.state['verification'] = evidence
            self.stop('completed', reason)
            return {**self.snapshot(), 'accepted': True}
        if cmd == 'set_intent':
            deadline = self.config.get('task_deadline')
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError('Task deadline reached before intent submission')
            if self.stop_on_environment_done and self.state.get('environment_result', {}).get('done'):
                raise RuntimeError('Environment episode terminated; open a new session for a new episode')
            if self.state['state'] == 'failed':
                raise RuntimeError('Execution failed; open a new session before submitting another intent')
            if not isinstance(message.get('context'), dict):
                raise TypeError('Policy context must be a mapping')
            context = dict(message['context'])
            # Payload meaning belongs to the policy. Language is optional; for
            # example, a navigation policy may accept only a goal coordinate.
            language = context.get('language', context.get('instruction'))
            if language is not None:
                if not isinstance(language, str):
                    raise ValueError('Context language must be a string')
                context['language'] = context['instruction'] = language
            limits = dict(message.get('limits') or {})
            steps = limits.get('max_control_steps', limits.get('max_steps'))
            if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0):
                raise ValueError('max_control_steps must be a positive integer')
            if 'timeout_s' in limits:
                positive(limits['timeout_s'], 'timeout_s')
            self.intent = context
            self.limits = {'max_control_steps': steps, 'timeout_s': limits.get('timeout_s')}
            self.state.pop('verification', None)
            self.state['limits'] = dict(self.limits)
            self.state.update(version=self.state['version']+1, state='running',
                              control_cycles=0, context_applied=False, stop_reason=None)
            self.gate(True)
            self.started_at = time.perf_counter()
            self.state['started_at'] = self.started_at
            self.state['stage_start_sequence'] = self.sequence
            self.next_control = self.started_at
            self.recorder.event('intent_accepted', version=self.state['version'], context=context,
                                limits=self.limits)
            return self.snapshot()
        if cmd == 'pause':
            if self.state['state'] == 'running':
                self.gate(False)
                self.state.update(state='paused', context_applied=False)
                self.recorder.event('paused', version=self.state['version'])
            return self.snapshot()
        if cmd == 'resume':
            if self.state['state'] != 'paused':
                raise ValueError('Only a paused intent can resume')
            self.gate(True)
            self.state.update(state='running', context_applied=False)
            self.next_control = time.perf_counter()
            self.recorder.event('resumed', version=self.state['version'])
            return self.snapshot()
        if cmd == 'cancel':
            self.stop('cancelled', 'cancelled_by_client')
            return self.snapshot()
        if cmd == 'close':
            self.close()
            return self.snapshot()
        raise ValueError(f'Unknown execution command: {cmd}')

    def tick(self):
        if self.state['state'] == 'failed':
            return
        now = time.perf_counter()
        self.sample()
        if self.state['version'] and now >= self.next_record:
            self.recorder.frame(self.observation, context={
                'version':self.state['version'], 'label':self.intent.get('label',''),
                'state':('waiting_policy' if self.state['state']=='running' and
                         not self.state['context_applied'] else self.state['state'])})
            self.next_record = now + self.record_period
        if self.recorder.error:
            raise RuntimeError(f'Video recording failed: {self.recorder.error}')
        if self.state['state'] != 'running':
            return
        deadline = self.config.get('task_deadline')
        if deadline is not None and now >= deadline:
            self.stop('exhausted', 'total_timeout')
            return
        timeout = self.limits.get('timeout_s')
        if timeout is not None and now-self.started_at >= timeout:
            self.stop('exhausted', 'intent_timeout')
            return
        if not self.context.is_alive():
            raise RuntimeError('Policy inference process exited')
        if self.backend.mode == 'robot' and now-self.captured_at > self.observation_timeout:
            raise TimeoutError('Robot observation freshness timeout')
        if now < self.next_control:
            return
        action = self.manager.select_action_nonblocking()
        if action is None:
            if self.waiting_since is None:
                self.waiting_since = now
            if now-self.waiting_since > self.inference_timeout:
                raise TimeoutError('Policy inference timeout')
            return
        self.waiting_since = None
        # Validate at the native publication boundary; Agent never sees/slices chunks.
        for item in action:
            value = np.asarray(item['action'])
            if value.size == 0 or not np.isfinite(value).all():
                raise ValueError('Policy returned an empty or nonfinite action')
            shape = self.config.get('action_shape')
            if shape is not None and tuple(value.shape) != tuple(shape):
                raise ValueError(f'Action shape {value.shape} does not match {shape}')
        # A simulator can mutate its observation buffers during step().
        if self.episode_sink is not None:
            observation = asdict(self.observation)
            published_action = deepcopy(action)
        result = self.backend.publish(action)
        if self.episode_sink is not None:
            self.episode_sink.append(observation, published_action, dict(self.intent),
                version=self.state['version'], epoch=self.state.get('epoch', 0),
                captured_at=self.captured_at, published_at=time.perf_counter(),
                observation_seq=self.sequence, policy_observed_at=self.context.last_observed_at)
        self.state['context_applied'] = self.context.last_context_version == self.state['version']
        self.state['control_cycles'] += 1
        self.recorder.action(version=self.state['version'], observation_seq=self.sequence,
                             observed_at=self.context.last_observed_at, action=action)
        self.next_control = now + self.control_period
        self.sample()
        steps = self.limits.get('max_control_steps')
        if result.get('done'):
            previous = self.state.get('environment_result') or {}
            self.state['environment_result'] = result
            if not previous.get('done'):
                self.recorder.event('environment_done', version=self.state['version'],
                                    stopping=self.stop_on_environment_done, result=result)
            if self.stop_on_environment_done:
                self.stop('completed', 'environment_terminated')
                return
        else:
            self.state['environment_result'] = result
        if steps is not None and self.state['control_cycles'] >= steps:
            self.stop('exhausted', 'control_budget_exhausted')

    def close(self):
        if self.closed:
            return
        from vlastudio.deploy.inference import stop_inference_process
        self.closed = True
        errors = []
        if self.manager is not None and self.context is not None:
            try:
                self.gate(False)
            except Exception as exc:
                errors.append(str(exc))
        if self.context is not None:
            try:
                stop_inference_process(self.context)
            except Exception as exc:
                errors.append(str(exc))
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception as exc:
                errors.append(str(exc))
        if self.observation_writer is not None:
            try:
                self.observation_writer.destroy()
            except Exception as exc:
                errors.append(str(exc))
        self.state['state'] = 'closed'
        if self.recorder is not None:
            try:
                self.recorder.event('closed', errors=errors)
                self.recorder.close()
            except Exception as exc:
                errors.append(str(exc))
            self.state['recording_dropped_frames'] = self.recorder.dropped_frames
            if errors:
                self.state['close_errors'] = errors
            if self.episode_sink is not None:
                try:
                    self.state['episode'] = self.episode_sink.close(self.snapshot())
                except Exception as exc:
                    errors.append(str(exc))
                    self.state['close_errors'] = errors
            self.recorder.result(self.snapshot())


def run_session(config, output_dir, connection):
    if os.name == 'posix':
        os.setsid()  # A client timeout may stop only this session's process group.
    def interrupted(signum, frame):
        # Unwind native startup's BaseException cleanup before any kill fallback.
        raise RuntimeError('Execution session terminated')
    signal.signal(signal.SIGTERM, interrupted)
    runtime = Runtime(config, output_dir)
    try:
        runtime.open()
        connection.send({'ready': runtime.snapshot()})
        while not runtime.closed:
            for _ in range(8):
                if not connection.poll():
                    break
                message = connection.recv()
                try:
                    result = runtime.command(message)
                    connection.send({'id': message['id'], 'result': result})
                except Exception as exc:
                    connection.send({'id': message['id'], 'error': f'{type(exc).__name__}: {exc}'})
                if runtime.closed:
                    break
            if runtime.closed:
                break
            try:
                runtime.tick()
            except Exception as exc:
                runtime.recorder.event('error', error=str(exc), traceback=traceback.format_exc())
                runtime.stop('failed', f'{type(exc).__name__}: {exc}')
            time.sleep(.001)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({'error': f'{type(exc).__name__}: {exc}'})
        except (EOFError, BrokenPipeError):
            pass
        traceback.print_exc()
    finally:
        runtime.close()
        connection.close()
