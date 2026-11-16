"""Habitat-Sim PointNav and Habitat-Lab/VLN-CE adapters (optional imports)."""
import importlib
from pathlib import Path

import numpy as np

from vlastudio.benchmark.base import MetaEnv, MetaObs
from vlastudio.benchmark.navigation import NavigationEpisode, discrete_action, evaluate, navigation_options


def create_env(config):
    if getattr(config, 'backend', 'sim') == 'lab':
        return HabitatLabEnv(config)
    return HabitatNavigationEnv(config)


class HabitatNavigationEnv(NavigationEpisode, MetaEnv):
    """0 STOP, 1 FORWARD, 2 LEFT, 3 RIGHT. State = distance, bearing."""
    def __init__(self, config):
        self.navigation_options = navigation_options(config)
        import habitat_sim
        import magnum as mn
        self.hs = habitat_sim
        self.config = config
        scene = Path(config.scene).expanduser().resolve()
        if not scene.is_file():
            raise FileNotFoundError(f'Habitat scene not found: {scene}. See docs/navigation.md')
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene)
        sim_cfg.gpu_device_id = getattr(config, 'gpu_device_id', 0)
        sim_cfg.enable_physics = False
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.radius = 0.1
        agent_cfg.height = 1.5
        sensors = []
        self.render = getattr(config, 'render', True)
        sim_cfg.create_renderer = self.render
        if self.render:
            for name, kind in [('rgb', habitat_sim.SensorType.COLOR),
                               ('depth', habitat_sim.SensorType.DEPTH)]:
                sensor = habitat_sim.CameraSensorSpec()
                sensor.uuid = name
                sensor.sensor_type = kind
                sensor.resolution = [getattr(config, 'image_size', 128)] * 2
                sensor.position = mn.Vector3(0., 1.25, 0.)
                sensors.append(sensor)
        agent_cfg.sensor_specifications = sensors
        agent_cfg.action_space = {
            'move_forward': habitat_sim.agent.ActionSpec('move_forward', habitat_sim.agent.ActuationSpec(amount=0.25)),
            'turn_left': habitat_sim.agent.ActionSpec('turn_left', habitat_sim.agent.ActuationSpec(amount=30)),
            'turn_right': habitat_sim.agent.ActionSpec('turn_right', habitat_sim.agent.ActuationSpec(amount=30)),
        }
        super().__init__(habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg])))
        if not self.env.pathfinder.is_loaded:
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_radius = agent_cfg.radius
            settings.agent_height = agent_cfg.height
            if not self.env.recompute_navmesh(self.env.pathfinder, settings):
                self.close()
                raise RuntimeError('Unable to build Habitat navigation mesh')
        self.max_steps = int(getattr(config, 'max_timesteps', 100))
        self.success_distance = float(getattr(config, 'success_distance', 0.2))
        self.seed = int(getattr(config, 'seed', 0)) + int(getattr(config, 'rollout_index', 0))
        self.episode_counter = 0

    def distance(self, start=None):
        path = self.hs.ShortestPath()
        path.requested_start = self.env.get_agent(0).get_state().position if start is None else start
        path.requested_end = self.goal
        if not self.env.pathfinder.find_path(path):
            return float('inf')
        return float(path.geodesic_distance)

    def reset(self, episode=None):
        from habitat_sim.utils.common import quat_from_angle_axis
        seed = self.seed + self.episode_counter
        self.env.seed(seed)
        self.env.pathfinder.seed(seed)
        rng = np.random.default_rng(seed)
        if episode is None:
            low = getattr(self.config, 'min_distance', 1.0)
            high = getattr(self.config, 'max_distance', 5.0)
            for _ in range(10000):
                start = np.asarray(self.env.pathfinder.get_random_navigable_point(), dtype=np.float32)
                self.goal = np.asarray(self.env.pathfinder.get_random_navigable_point(), dtype=np.float32)
                distance = self.distance(start)
                if low <= distance <= high:
                    break
            else:
                raise RuntimeError('Could not sample a connected PointNav episode in the distance range')
            episode = {'start': start.tolist(), 'goal': self.goal.tolist(),
                       'yaw': float(rng.uniform(-np.pi, np.pi)), 'episode_id': str(seed)}
        self.goal = np.asarray(episode['goal'], dtype=np.float32)
        state = self.hs.AgentState()
        state.position = np.asarray(episode['start'], dtype=np.float32)
        state.rotation = quat_from_angle_axis(episode['yaw'], np.array([0., 1., 0.]))
        self.env.get_agent(0).set_state(state)
        self.episode_id = episode.get('episode_id', str(seed))
        self.episode_counter += 1
        self.episode = dict(episode)
        distance = self.distance()
        if not np.isfinite(distance):
            raise ValueError('Episode start and goal must be connected by the navigation mesh')
        self._begin(distance)
        self.follower = self.hs.GreedyGeodesicFollower(self.env.pathfinder, self.env.get_agent(0),
                    self.success_distance, stop_key=0, forward_key=1, left_key=2, right_key=3)
        self.prev_obs = self.obs2meta(self.env.get_sensor_observations())
        return self.prev_obs

    def obs2meta(self, obs):
        from habitat_sim.utils.common import quat_rotate_vector
        state = self.env.get_agent(0).get_state()
        relative = quat_rotate_vector(state.rotation.inverse(), self.goal - state.position)
        # Positive bearing is a left turn, matching Habitat's +Y rotation.
        pointgoal = np.array([np.linalg.norm(relative[[0, 2]]),
                              np.arctan2(-relative[0], -relative[2])], dtype=np.float32)
        return MetaObs(state=pointgoal,
                       image=obs['rgb'][None, ..., :3].transpose(0, 3, 1, 2).copy() if 'rgb' in obs else None,
                       depth=obs['depth'][None].copy() if 'depth' in obs else None,
                       raw_lang='Navigate to the point goal and stop.', timestep=self.steps,
                       reasoning={'navigation': dict(self.navigation_options), 'image_type': 'rgb'})

    def meta2act(self, action):
        return discrete_action(action, 4)

    def step(self, action):
        if self.done:
            from dataclasses import asdict
            return asdict(self.prev_obs), 0.0, True, dict(self.last_info)
        action = self.meta2act(action)
        before = self.env.get_agent(0).get_state().position.copy()
        obs = self.env.get_sensor_observations() if action == 0 else self.env.step(
            ('stop', 'move_forward', 'turn_left', 'turn_right')[action])
        after = self.env.get_agent(0).get_state().position
        self.travelled += float(np.linalg.norm(after - before))
        self.prev_obs = self.obs2meta(obs)
        self.prev_obs.timestep = self.steps + 1
        return self._finish_step(self.distance(), action == 0,
                                 action == 1 and np.linalg.norm(after - before) < 1e-3)

    def expert_action(self):
        """Shortest-path supervision; never called by the learned evaluator."""
        return int(self.follower.next_action_along(self.goal))


class HabitatLabEnv(MetaEnv):
    """Use a user-supplied, version-matched Habitat-Lab/VLN-CE task config."""
    def __init__(self, config):
        self.navigation_options = navigation_options(config)
        import habitat
        for module in getattr(config, 'registration_modules', []):
            importlib.import_module(module)
        self.config = config
        self.steps = 0
        self.done = False
        self.index = int(getattr(config, 'rollout_index', 0))
        self.stride = int(getattr(config, 'episode_stride', 1))
        self.actions = list(getattr(config, 'actions', ['stop', 'move_forward', 'turn_left', 'turn_right']))
        if getattr(config, 'legacy_config', False):
            self.actions = [action.upper() for action in self.actions]
        kwargs = {'config_paths': config.habitat_config} if getattr(config, 'legacy_config', False) else {
            'config_path': config.habitat_config}
        overrides = list(getattr(config, 'habitat_overrides', []) or [])
        if getattr(config, 'auto_download', False) and not getattr(config, 'legacy_config', False):
            values = dict(x.split('=', 1) for x in overrides if '=' in x)
            data_path = values.get('habitat.dataset.data_path', '')
            if not data_path or not Path(data_path).is_file():
                from vlastudio.data_utils.datasets.navigation import vlnce_annotations
                split = values.get('habitat.dataset.split', 'val_unseen')
                annotation = vlnce_annotations(split)
                overrides = [x for x in overrides if not x.startswith('habitat.dataset.data_path=')]
                overrides.append(f'habitat.dataset.data_path={annotation.resolve()}')
        if getattr(config, 'max_timesteps', None) is not None:
            key = 'ENVIRONMENT.MAX_EPISODE_STEPS' if getattr(config, 'legacy_config', False) else 'habitat.environment.max_episode_steps'
            overrides += [key, str(int(config.max_timesteps))] if getattr(config, 'legacy_config', False) else [f'{key}={int(config.max_timesteps)}']
        if overrides:
            kwargs['opts' if getattr(config, 'legacy_config', False) else 'overrides'] = overrides
        task = habitat.get_config(**kwargs)
        super().__init__(habitat.Env(config=task))

    def reset(self):
        self.steps = 0
        self.done = False
        self.last_info = {}
        # Honor framework rollout offsets even when each rollout creates a new
        # native environment. Habitat's default seeded shuffle otherwise repeats
        # the same first episode in every independently created vector slot.
        self.env.current_episode = self.env.episodes[self.index % len(self.env.episodes)]
        self.index += self.stride
        self.prev_obs = self.obs2meta(self.env.reset())
        return self.prev_obs

    def obs2meta(self, obs):
        instruction = getattr(self.env.current_episode, 'instruction', None)
        text = getattr(instruction, 'instruction_text', '')
        if isinstance(instruction, dict):
            text = instruction.get('instruction_text', '')
        state = np.asarray(obs.get('pointgoal_with_gps_compass', obs.get('pointgoal', [0., 0.])), dtype=np.float32)
        if getattr(self.config, 'state_source', None) == 'none':
            state = np.empty(0, dtype=np.float32)
        depth = np.asarray(obs['depth']) if 'depth' in obs else None
        if depth is not None and depth.ndim == 3:
            depth = depth[..., 0]
        navigation = dict(self.navigation_options)
        if getattr(self.config, 'return_instruction_tokens', False):
            tokens = instruction.get('instruction_tokens') if isinstance(instruction, dict) else getattr(instruction, 'instruction_tokens', None)
            if tokens is None:
                raise ValueError('Configured instruction tokens are missing from this episode')
            navigation['instruction_tokens'] = list(tokens)
        return MetaObs(state=state, raw_lang=text, timestep=self.steps,
                       reasoning={'navigation': navigation, 'image_type': 'rgb'},
                       image=np.asarray(obs['rgb'])[None, ..., :3].transpose(0, 3, 1, 2).copy() if 'rgb' in obs else None,
                       depth=depth[None] if depth is not None else None)

    def meta2act(self, action):
        return discrete_action(action, len(self.actions))

    def step(self, action):
        from dataclasses import asdict
        if self.done:
            return asdict(self.prev_obs), 0.0, True, dict(self.last_info)
        action = self.meta2act(action)
        self.steps += 1
        obs = self.env.step(self.actions[action])
        self.done = bool(self.env.episode_over)
        metrics = dict(self.env.get_metrics())
        # Require the configured task to expose true success; never infer it from done.
        if 'success' not in metrics:
            raise RuntimeError('Habitat task must configure a success measurement')
        self.last_info = {**metrics, 'success': bool(metrics['success']),
                          'episode_id': str(self.env.current_episode.episode_id)}
        self.prev_obs = self.obs2meta(obs)
        return asdict(self.prev_obs), float(metrics['success']), self.done, self.last_info
