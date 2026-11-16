"""Classic R2R on the real MatterSim renderer and connectivity graphs."""
import gzip
import heapq
import json
from pathlib import Path

import numpy as np

from vlastudio.benchmark.base import MetaEnv, MetaObs
from vlastudio.benchmark.navigation import NavigationEpisode, discrete_action, evaluate, navigation_options


def create_env(config):
    return MatterportNavigationEnv(config)


def load_graph(path):
    nodes = json.loads(Path(path).read_text())
    positions = {n['image_id']: np.asarray(n['pose'])[ [3, 7, 11] ].astype(float)
                 for n in nodes if n['included']}
    graph = {key: {} for key in positions}
    for node in nodes:
        source = node['image_id']
        if source not in graph:
            continue
        for i, connected in enumerate(node['unobstructed']):
            target = nodes[i]['image_id']
            if connected and target in graph:
                graph[source][target] = float(np.linalg.norm(positions[source] - positions[target]))
    return graph


def graph_distances(graph, goal):
    distance = {goal: 0.0}
    queue = [(0.0, goal)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > distance[node]:
            continue
        for target, weight in graph[node].items():
            candidate = cost + weight
            if candidate < distance.get(target, float('inf')):
                distance[target] = candidate
                heapq.heappush(queue, (candidate, target))
    return distance


class MatterportNavigationEnv(NavigationEpisode, MetaEnv):
    """Primitive actions: STOP=0 FORWARD=1 LEFT=2 RIGHT=3 UP=4 DOWN=5.

    FORWARD selects the closest angular navigable viewpoint in the current FOV,
    matching MatterSim index 1. Optional candidate mode selects a stable
    viewpoint action_id. Goal/teacher/path are never exposed to policy.
    """
    def __init__(self, config):
        if getattr(config, 'auto_download', False) and not Path(config.episodes).is_file():
            import copy
            from vlastudio.data_utils.datasets.navigation import r2r_annotations
            config = copy.copy(config)
            split = Path(config.episodes).stem.removeprefix('R2R_')
            config.episodes = str(r2r_annotations(split))
        self.navigation_options = navigation_options(config, supports_candidates=True)
        import MatterSim
        self.config = config
        self.max_steps = int(getattr(config, 'max_timesteps', 100))
        self.success_distance = float(getattr(config, 'success_distance', 3.0))
        self.success_requires_stop = bool(getattr(config, 'success_requires_stop', True))
        self.depth_enabled = bool(getattr(config, 'depth_enabled', True))
        self.rendering_enabled = bool(getattr(config, 'rendering_enabled', True))
        self.candidate_view_alignment = getattr(config, 'candidate_view_alignment', 'preserve')
        if self.candidate_view_alignment not in ('preserve', 'panoramic'):
            raise ValueError('candidate_view_alignment must be preserve or panoramic')
        if self.candidate_view_alignment == 'panoramic' and not (
                self.navigation_options['action_mode'] == 'candidate_viewpoint' and
                self.navigation_options['candidate_scope'] == 'all_neighbors'):
            raise ValueError('Panoramic alignment requires candidate_viewpoint and all_neighbors')
        if not self.rendering_enabled and self.depth_enabled:
            raise ValueError('Depth requires rendering_enabled=true')
        self.panorama_cache = {}
        self.panorama_sim = None
        opener = gzip.open if str(config.episodes).endswith('.gz') else open
        with opener(config.episodes, 'rt', encoding='utf-8') as stream:
            raw = json.load(stream)
        self.episodes = []
        for item in raw:
            for i, instruction in enumerate(item['instructions']):
                self.episodes.append({**item, 'instruction': instruction,
                                      'episode_id': f"{item['path_id']}_{i}"})
        if not self.episodes:
            raise ValueError('R2R episode file is empty')
        self.index = int(getattr(config, 'rollout_index', 0))
        self.stride = int(getattr(config, 'episode_stride', 1))
        self.graphs = {}
        sim = MatterSim.Simulator()
        sim.setDatasetPath(str(Path(config.dataset_path).expanduser().resolve()))
        sim.setNavGraphPath(str(Path(config.connectivity_path).expanduser().resolve()))
        sim.setBatchSize(1)
        sim.setRenderingEnabled(self.rendering_enabled)
        sim.setCameraResolution(getattr(config, 'image_size', 256), getattr(config, 'image_size', 256))
        sim.setCameraVFOV(np.deg2rad(60))
        sim.setDiscretizedViewingAngles(True)
        sim.setDepthEnabled(self.depth_enabled)
        if self.navigation_options['candidate_source'] == 'simulator':
            sim.setRestrictedNavigation(self.navigation_options['candidate_scope'] == 'visible')
        sim.initialize()
        if self.candidate_view_alignment == 'panoramic':
            self.panorama_sim = MatterSim.Simulator()
            self.panorama_sim.setDatasetPath(str(Path(config.dataset_path).expanduser().resolve()))
            self.panorama_sim.setNavGraphPath(str(Path(config.connectivity_path).expanduser().resolve()))
            self.panorama_sim.setBatchSize(1)
            self.panorama_sim.setRenderingEnabled(False)
            self.panorama_sim.setDepthEnabled(False)
            self.panorama_sim.setCameraResolution(640, 480)
            self.panorama_sim.setCameraVFOV(np.deg2rad(60))
            self.panorama_sim.setDiscretizedViewingAngles(True)
            self.panorama_sim.setRestrictedNavigation(True)
            self.panorama_sim.initialize()
        super().__init__(sim)

    def _panoramic_candidates(self, state):
        """Official R2R 36-view sweep, using only current-viewpoint geometry."""
        key = (state.scanId, state.location.viewpointId)
        if key not in self.panorama_cache:
            adjacent = {}
            for view in range(36):
                if view == 0:
                    self.panorama_sim.newEpisode([key[0]], [key[1]], [0.], [-np.pi/6])
                else:
                    self.panorama_sim.makeAction([0], [1.], [1. if view % 12 == 0 else 0.])
                pose = self.panorama_sim.getState()[0]
                if pose.viewIndex != view:
                    raise RuntimeError('MatterSim panoramic sweep view mismatch')
                for location in pose.navigableLocations[1:]:
                    distance = float(np.hypot(location.rel_heading, location.rel_elevation))
                    old = adjacent.get(location.viewpointId)
                    if old is None or distance < old['angular_distance']:
                        adjacent[location.viewpointId] = {
                            'viewpoint_id': location.viewpointId, 'source_view_index': view,
                            'heading': float(pose.heading+location.rel_heading),
                            'elevation': float(pose.elevation+location.rel_elevation),
                            'angular_distance': distance, 'distance': float(location.rel_distance)}
            self.panorama_cache[key] = list(adjacent.values())
        base_heading = (state.viewIndex % 12)*np.pi/6
        return [{**row, 'action_id': self.viewpoint_actions[row['viewpoint_id']],
            'relative_heading': row['heading']-base_heading,
            'relative_elevation': row['elevation']-float(state.elevation)}
            for row in self.panorama_cache[key]]

    def reset(self):
        item = self.episodes[self.index % len(self.episodes)]
        self.index += self.stride
        self.episode = item
        self.episode_id = item['episode_id']
        scan = item['scan']
        if scan not in self.graphs:
            self.graphs[scan] = load_graph(Path(self.config.connectivity_path) / f'{scan}_connectivity.json')
        self.graph = self.graphs[scan]
        # Stable for a scan, irrespective of native candidate ordering or FOV.
        self.viewpoint_actions = {viewpoint: i + 1 for i, viewpoint in enumerate(sorted(self.graph))}
        self.action_viewpoints = {action: viewpoint for viewpoint, action in self.viewpoint_actions.items()}
        self.distances = graph_distances(self.graph, item['path'][-1])
        self.env.newEpisode([scan], [item['path'][0]], [item['heading']], [0.0])
        self._begin(self.distances[item['path'][0]])
        self.prev_obs = self.obs2meta(self.env.getState()[0])
        return self.prev_obs

    def obs2meta(self, state):
        # Heading/elevation and current visible degree are observable, unlike goal distance.
        features = np.asarray([state.heading, state.elevation, len(state.navigableLocations) - 1], dtype=np.float32)
        image = None
        if getattr(self, 'rendering_enabled', True):
            image = np.asarray(state.rgb)[..., ::-1].copy()  # MatterSim uses BGR.
        depth = None
        if self.depth_enabled:
            depth = np.asarray(state.depth, dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth = depth[None] / 4000.
        options = getattr(self, 'navigation_options', navigation_options(None, supports_candidates=True))
        navigation = dict(options)
        if hasattr(state, 'scanId') and hasattr(state, 'viewIndex'):
            navigation.update(scan_id=state.scanId, viewpoint_id=state.location.viewpointId,
                              view_index=int(state.viewIndex))
        if options['candidate_source'] == 'simulator':
            navigation['candidates'] = [
                {'action_id': self.viewpoint_actions[location.viewpointId],
                 'viewpoint_id': location.viewpointId,
                 'relative_heading': float(location.rel_heading),
                 'relative_elevation': float(location.rel_elevation),
                 'distance': float(location.rel_distance)}
                for location in state.navigableLocations[1:]]
            if getattr(self, 'candidate_view_alignment', 'preserve') == 'panoramic':
                navigation['candidates'] = self._panoramic_candidates(state)
                navigation['candidate_view_alignment'] = 'panoramic'
        return MetaObs(state=features, image=None if image is None else image[None].transpose(0, 3, 1, 2),
                       depth=depth, raw_lang=self.episode['instruction'], timestep=self.steps,
                       reasoning={'navigation': navigation, 'image_type': 'rgb'})

    def _candidate_location(self, action, state):
        target = self.action_viewpoints[action]
        if getattr(self, 'candidate_view_alignment', 'preserve') == 'panoramic' and not any(
                row['action_id'] == action for row in self.prev_obs.reasoning['navigation']['candidates']):
            raise ValueError('Selected viewpoint is absent from the latest panoramic candidates')
        for index, location in enumerate(state.navigableLocations[1:], 1):
            if location.viewpointId == target:
                return index
        raise ValueError('Selected viewpoint is not currently navigable; choose action_id from the latest candidates')

    def meta2act(self, action):
        options = getattr(self, 'navigation_options', navigation_options(None, supports_candidates=True))
        if options['action_mode'] == 'candidate_viewpoint':
            value = discrete_action(action, len(self.action_viewpoints) + 1, ctrl_type='candidate_viewpoint')
            if value:
                self._candidate_location(value, self.env.getState()[0])
            return value
        return discrete_action(action, 6)

    def step(self, action):
        from dataclasses import asdict
        if self.done:
            return asdict(self.prev_obs), 0., True, dict(self.last_info)
        action = self.meta2act(action)
        before = self.env.getState()[0]
        # Native SimState is mutated in-place by makeAction. Copy the ID first.
        source = before.location.viewpointId
        location, heading, elevation = 0, 0., 0.
        collision = False
        if getattr(self, 'navigation_options', {}).get('action_mode') == 'candidate_viewpoint':
            if action:
                if getattr(self, 'candidate_view_alignment', 'preserve') == 'panoramic':
                    candidate = next(row for row in self.prev_obs.reasoning['navigation']['candidates'] if row['action_id'] == action)
                    view = candidate['source_view_index']
                    while self.env.getState()[0].viewIndex // 12 != view // 12:
                        current_level = self.env.getState()[0].viewIndex // 12
                        self.env.makeAction([0], [0.], [1. if current_level < view // 12 else -1.])
                    while self.env.getState()[0].viewIndex != view:
                        self.env.makeAction([0], [1.], [0.])
                    before = self.env.getState()[0]
                location = self._candidate_location(action, before)
        elif action == 1:
            if len(before.navigableLocations) > 1:
                location = 1
            else:
                collision = True
        elif action in (2, 3):
            # Discretized MatterSim uses +/-1 increments, not radians.
            heading = -1. if action == 2 else 1.
        elif action in (4, 5):
            elevation = 1. if action == 4 else -1.
        if action != 0:
            self.env.makeAction([location], [heading], [elevation])
        after = self.env.getState()[0]
        target = after.location.viewpointId
        if source != target:
            self.travelled += self.graph[source][target]
        self.prev_obs = self.obs2meta(after)
        self.prev_obs.timestep = self.steps + 1
        return self._finish_step(self.distances.get(target, float('inf')), action == 0, collision)

    def close(self):
        if getattr(self, 'panorama_sim', None) is not None:
            self.panorama_sim.close()
            self.panorama_sim = None
        if self.env is not None:
            self.env.close()
            self.env = None
