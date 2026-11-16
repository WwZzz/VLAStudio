"""CARLA and Bench2Drive adapters (native imports are deferred until use)."""
from dataclasses import asdict
import copy
import multiprocessing as mp
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from vlastudio.benchmark.base import MetaEnv, MetaObs


def route_metrics(record):
    infractions = record.get('infractions', {})
    clean = all(not values for name, values in infractions.items() if name != 'min_speed_infractions')
    return {'episode_id': record['route_id'],
            'success': bool(record['status'] in ['Completed', 'Perfect'] and clean),
            'driving_score': float(record['scores']['score_composed']),
            'route_completion': float(record['scores']['score_route']),
            'status': record['status'], 'infractions': infractions, 'official_record': record}


class Bench2DriveEnv(MetaEnv):
    """Official scenario dynamics and scoring, with ordinary reset/step semantics."""
    def __init__(self, config):
        from vlastudio.benchmark.carla.install import cache_root, source
        if getattr(config, 'ctrl_space', 'vehicle') != 'vehicle' or getattr(config, 'ctrl_type', 'control') != 'control':
            raise ValueError('Bench2Drive control actions are [steer, throttle, brake, reverse]')
        if list(getattr(config, 'action_order', ['steer', 'throttle', 'brake', 'reverse'])) != ['steer', 'throttle', 'brake', 'reverse']:
            raise ValueError('action_order must be [steer, throttle, brake, reverse]')
        self.config = config
        if not isinstance(getattr(config, 'remove_map_parked_vehicles', True), bool):
            raise ValueError('remove_map_parked_vehicles must be a boolean')
        if not isinstance(getattr(config, 'repair_bicycle_flow', False), bool):
            raise ValueError('repair_bicycle_flow must be a boolean')
        original = Path(getattr(config, 'routes', '') or source()/'leaderboard/data/bench2drive_0.0.4_val.xml')
        tree = ET.parse(original)
        routes = list(tree.getroot().findall('route'))
        requested = getattr(config, 'route_ids', None)
        if requested:
            ids = {str(value) for value in requested}
            routes = [route for route in routes if route.attrib['id'] in ids]
            if {route.attrib['id'] for route in routes} != ids:
                raise ValueError('Requested route IDs are missing from the official XML')
        start = int(getattr(config, 'rollout_index', 0))
        routes = routes[start:]
        if getattr(config, 'max_routes', None):
            routes = routes[:config.max_routes]
        if not routes:
            raise ValueError('No Bench2Drive evaluation routes selected')
        self.catalog = [{'route_id': route.attrib['id'], 'town': route.attrib['town']} for route in routes]
        root = ET.Element('routes')
        root.extend(copy.deepcopy(routes))
        cache_root().mkdir(parents=True, exist_ok=True)
        self.work_dir = Path(tempfile.mkdtemp(prefix='evaluation-', dir=cache_root()))
        self.work_dir.chmod(0o755)
        selected = self.work_dir/'routes.xml'
        ET.ElementTree(root).write(selected, encoding='utf-8', xml_declaration=True)
        self.worker_config = {key: getattr(config, key) for key in ['carla_root', 'graphics_runtime', 'gpu',
            'startup_timeout', 'step_timeout', 'seed', 'remove_map_parked_vehicles', 'repair_bicycle_flow'] if hasattr(config, key)}
        self.worker_config.update(routes=str(selected), work_dir=str(self.work_dir))
        self.process = self.pipe = None
        self.done, self.completed = True, 0
        self.prev_obs, self.last_info = None, {}

    def _receive(self):
        timeout = float(getattr(self.config, 'startup_timeout', 180.))+float(getattr(self.config, 'step_timeout', 600.))
        if not self.pipe.poll(timeout):
            raise TimeoutError(f'Bench2Drive worker did not respond; see {self.work_dir}')
        kind, value = self.pipe.recv()
        if kind == 'error':
            raise RuntimeError(value)
        return kind, value

    def reset(self):
        if self.completed >= len(self.catalog):
            raise IndexError('All selected Bench2Drive routes have been evaluated')
        if self.process is None:
            from .bridge import worker
            context = mp.get_context('spawn')
            self.pipe, child = context.Pipe()
            self.process = context.Process(target=worker, args=(child, self.worker_config), daemon=False)
            self.process.start()
            child.close()
        else:
            if not self.done:
                raise RuntimeError('A driving route must finish before resetting')
            self.pipe.send(('reset', None))
        kind, value = self._receive()
        if kind != 'observation':
            raise RuntimeError(f'Route did not provide its first observation: {kind}: {value}')
        self.done, self.steps = False, 0
        self.prev_obs = MetaObs(**value)
        self.last_info = {'success': False, 'episode_id': self.catalog[self.completed]['route_id']}
        return self.prev_obs

    def meta2act(self, action):
        if isinstance(action, np.ndarray) and action.dtype == object:
            action = action.reshape(-1)[0]
        if isinstance(action, dict):
            if action.get('ctrl_space', 'vehicle') != 'vehicle' or action.get('ctrl_type', 'control') != 'control':
                raise ValueError('Vehicle control configuration does not match the environment')
            action = action['action']
        elif hasattr(action, 'action'):
            if getattr(action, 'ctrl_space', 'vehicle') != 'vehicle' or getattr(action, 'ctrl_type', 'control') != 'control':
                raise ValueError('Vehicle control configuration does not match the environment')
            action = action.action
        values = np.asarray(action, dtype=float).reshape(-1)
        if len(values) != 4 or not np.isfinite(values).all():
            raise ValueError('Vehicle control must contain four finite values')
        if not (-1 <= values[0] <= 1 and 0 <= values[1] <= 1 and 0 <= values[2] <= 1 and values[3] in [0., 1.]):
            raise ValueError('Vehicle controls are outside the declared ranges')
        return values.tolist()

    def step(self, action):
        if self.done:
            return asdict(self.prev_obs), 0., True, dict(self.last_info)
        self.pipe.send(('action', self.meta2act(action)))
        kind, value = self._receive()
        self.steps += 1
        if kind == 'observation':
            self.prev_obs = MetaObs(**value)
        elif kind == 'done':
            self.done = True
            source_id = self.catalog[self.completed]['route_id']
            self.completed += 1
            self.last_info = {**route_metrics(value['record']), 'initialization': value['initialization'],
                              'source_route_id': source_id, 'steps': self.steps}
        else:
            raise RuntimeError(f'Unexpected driving worker message: {kind}')
        return asdict(self.prev_obs), float(self.last_info['success']), self.done, dict(self.last_info)

    def close(self):
        if self.process is not None:
            try:
                self.pipe.send(('close', None))
            except (BrokenPipeError, EOFError):
                pass
            self.process.join(timeout=20)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=10)
            from .server import close_owned_server
            close_owned_server(self.work_dir/'server')
            self.pipe.close()
            self.process = self.pipe = None


def create_env(config):
    return Bench2DriveEnv(config)


def evaluate(args, env, action_manager, inference_ctx=None, video_writer=None, save_example_dir=None):
    """Use official terminal success; timeouts and route completion alone do not pass."""
    import time
    from vlastudio.benchmark.utils import organize_obs
    started, count = time.time(), len(env)
    finished = np.zeros(count, dtype=bool)
    results = [{} for _ in range(count)]
    horizons = np.full(count, args.max_timesteps, dtype=int)
    try:
        observation = organize_obs(env.reset())
        for step in range(args.max_timesteps):
            if inference_ctx is not None:
                inference_ctx.update_obs(observation, step)
            raw, _, done, information = env.step(action_manager.select_action())
            for index in range(count):
                if finished[index]:
                    continue
                results[index] = dict(information[index])
                if done[index]:
                    finished[index], horizons[index] = True, step+1
            if video_writer is not None:
                for row in raw:
                    images = row.image if hasattr(row, 'image') else row.get('image')
                    if images is not None:
                        video_writer.append_data(images[0].transpose(1, 2, 0))
            observation = organize_obs(raw)
            if finished.all():
                break
    finally:
        if not getattr(args, 'reuse_env', False):
            env.close()
    success = np.asarray([bool(row.get('success', False)) for row in results])
    return {'success': success.tolist(), 'total': count, 'total_success': int(success.sum()),
            'success_rate': float(success.mean()), 'horizon': horizons.tolist(),
            'horizon_success': float(horizons[success].mean()) if success.any() else 0.,
            'avg_driving_score': float(np.mean([row.get('driving_score', 0.) for row in results])),
            'avg_route_completion': float(np.mean([row.get('route_completion', 0.) for row in results])),
            'episodes': results, 'eval_time': time.time()-started}
