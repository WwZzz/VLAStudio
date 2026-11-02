"""Deterministic CPU fixtures; production workers and SHM remain unmodified."""
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import time

import numpy as np

from vlastudio.benchmark.base import MetaObs
from vlastudio.deploy.robot.base import BaseRobot


def append_event(path, event, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps({'event': event, 'at': time.monotonic(),
                               'pid': os.getpid(), **fields}) + '\n')


def _setting(config, key, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def make_observation(sequence):
    image = np.zeros((1, 3, 64, 96), dtype=np.uint8)
    image[0, 0] = sequence % 255
    image[0, 1, :, :((sequence * 3) % 96)] = 220
    image[0, 2, 20:44, 20:76] = 150
    state = np.array([sequence, 0, 0], dtype=np.float32)
    return MetaObs(state=state, state_ee=state.copy(), state_joint=state.copy(), image=image)


class EchoPolicy:
    """Encode intent, sensor sequence and reset count in every action."""

    def __init__(self, root):
        self.log = Path(root) / 'policy.jsonl'
        self.context = {}
        self.reset_count = 0
        append_event(self.log, 'loaded')

    def reset(self):
        self.reset_count += 1
        append_event(self.log, 'reset', reset_count=self.reset_count)

    def set_execution_context(self, context):
        self.context = dict(context)
        append_event(self.log, 'context', context=self.context)

    def inference(self, observation):
        context = dict(self.context)
        code = int(context.get('skill_index', context.get('target', 0)))
        state = observation.state if hasattr(observation, 'state') else observation['state']
        sequence = float(np.asarray(state).reshape(-1)[0])
        raw_lang = observation.raw_lang if hasattr(observation, 'raw_lang') else observation.get('raw_lang')
        append_event(self.log, 'inference_started', code=code, observation_seq=sequence,
                     raw_lang=raw_lang, latent=context.get('skill_latent'),
                     reset_count=self.reset_count)
        time.sleep(float(context.get('inference_delay_s', 0)))
        if context.get('fail_inference'):
            append_event(self.log, 'inference_failed', code=code)
            raise RuntimeError('deliberate acceptance policy failure')
        value = np.array([code, sequence, self.reset_count], dtype=np.float32)
        if context.get('malformed_action'):
            value[0] = np.nan
        result = [np.array([{'action': value.copy(), 'ctrl_space': 'joint',
                             'ctrl_type': 'abs'}], dtype=object)
                  for _ in range(int(context.get('chunk_size', 8)))]
        append_event(self.log, 'inference_finished', code=code, observation_seq=sequence,
                     reset_count=self.reset_count)
        return result


def load_policy(args):
    """Native inference_worker loader hook: only the neural model is a fixture."""
    return EchoPolicy(args.model_name_or_path)


def load_failing_policy(args):
    append_event(Path(args.model_name_or_path) / 'policy.jsonl', 'loader_failed')
    raise RuntimeError('deliberate acceptance loader failure')


def slow_load_policy(args):
    append_event(Path(args.model_name_or_path) / 'policy.jsonl', 'slow_loader_started')
    time.sleep(5)
    return EchoPolicy(args.model_name_or_path)


class CounterEnvironment:
    """Minimal MetaEnv-compatible simulation with inspectable actuation."""

    def __init__(self, config):
        self.root = Path(_setting(config, 'fixture_dir'))
        self.done_after = int(_setting(config, 'done_after', 0))
        self.fail_after = int(_setting(config, 'fail_after', 0))
        self.sequence = 0
        self.closed = False
        append_event(self.root / 'environment.jsonl', 'created')

    def reset(self):
        self.sequence = 0
        append_event(self.root / 'environment.jsonl', 'reset')
        return make_observation(self.sequence)

    def step(self, action):
        if self.fail_after and self.sequence >= self.fail_after:
            raise RuntimeError('deliberate acceptance environment failure')
        if is_dataclass(action):
            action = asdict(action)
        values = np.asarray(action['action']).reshape(-1)
        append_event(self.root / 'published.jsonl', 'published', action=values.tolist(),
                     sensor_sequence=self.sequence)
        self.sequence += 1
        done = bool(self.done_after and self.sequence >= self.done_after)
        return make_observation(self.sequence), 0.0, done, {'fixture': True}

    def close(self):
        self.closed = True
        append_event(self.root / 'environment.jsonl', 'closed')


class CounterRobot(BaseRobot):
    """Uses the actual BaseRobot loop; publish_action only appends a test record."""

    def __init__(self, *, fixture_dir, name, control_shm_name=None, fps=50,
                 max_size_mb=2, stop_observations_after=0, **kwargs):
        super().__init__(name=name, control_shm_name=control_shm_name,
                         fps=fps, max_size_mb=max_size_mb)
        self.root = Path(fixture_dir)
        self.sequence = 0
        self.stop_observations_after = int(stop_observations_after)
        append_event(self.root / 'robot.jsonl', 'created',
                     name=name, control_shm_name=control_shm_name)

    def connect(self):
        return True

    def get_observation(self):
        self.sequence += 1
        if ((self.root / 'freeze_observations').exists() or
                (self.stop_observations_after and self.sequence > self.stop_observations_after)):
            return None
        observation = asdict(make_observation(self.sequence))
        observation['qpos'] = observation['state']
        observation['timestamp'] = time.time()
        return observation

    @staticmethod
    def obs2meta(device_data):
        return {key: device_data[key] for key in ('state', 'state_ee', 'state_joint', 'image')}

    def get_action_dim(self):
        return 3

    def is_running(self):
        # BaseRobot.start() replaces this with its loop flag, as upstream devices do.
        return bool(self.__dict__.get('is_running', False))

    def publish_action(self, action):
        append_event(self.root / 'published.jsonl', 'published',
                     action=np.asarray(action).reshape(-1).tolist(),
                     sensor_sequence=self.sequence)

    def shutdown(self):
        self.is_running = False

    def close(self):
        super().close()
        if self.control_shm is not None:
            self.control_shm.destroy()
            self.control_shm = None
        append_event(self.root / 'robot.jsonl', 'closed')
