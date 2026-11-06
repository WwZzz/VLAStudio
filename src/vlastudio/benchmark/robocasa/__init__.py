"""RoboCasa native-action environments. Simulator imports are deferred until creation."""
from dataclasses import asdict
import copy
import time

import numpy as np

from vlastudio.benchmark.base import MetaEnv, MetaObs
from .observations import CAMERAS, STATE_KEYS, instruction, metadata, rgb_image, state_vector


def create_env(config):
    return RoboCasaEnv(config)


class RoboCasaEnv(MetaEnv):
    def __init__(self, config):
        self.config = config
        self.ctrl_space = getattr(config, 'ctrl_space', 'ee')
        self.ctrl_type = getattr(config, 'ctrl_type', 'delta')
        self.camera_names = tuple(getattr(config, 'camera_names', CAMERAS))
        self.state_keys = tuple(getattr(config, 'state_keys', STATE_KEYS))
        self.image_flip = bool(getattr(config, 'image_flip', True))
        self.max_steps = int(getattr(config, 'max_timesteps', 1000))
        if self.max_steps < 1 or not self.camera_names:
            raise ValueError('Positive max_timesteps and at least one camera are required')
        self.seed = int(getattr(config, 'seed', 10000)) + int(getattr(config, 'rollout_index', 0))
        self.episode_index = 0
        env_kwargs = copy.deepcopy(getattr(config, 'env_kwargs', {}))
        path = getattr(config, 'dataset_path', None)
        if path:
            import h5py
            with h5py.File(path, 'r') as file:
                args = metadata(file['data'].attrs['env_args'])
            env_kwargs = {**args['env_kwargs'], **env_kwargs}
            recorded_task = args['env_name']
            if recorded_task != config.task:
                raise ValueError(f'Dataset task {recorded_task} differs from {config.task}')
        if not env_kwargs:
            raise ValueError('Supply dataset_path with env_args or explicit env_kwargs')
        # Registration must happen before robosuite.make. Keep recorded controller settings.
        import robocasa  # noqa: F401
        import robosuite
        height, width = getattr(config, 'image_size', (128, 128))
        env_kwargs.update(env_name=config.task, has_renderer=False,
                          has_offscreen_renderer=True, use_camera_obs=True,
                          camera_names=list(self.camera_names), camera_heights=height,
                          camera_widths=width, ignore_done=True, horizon=self.max_steps,
                          seed=self.seed)
        super().__init__(robosuite.make(**env_kwargs))
        self.min_action, self.max_action = map(np.asarray, self.env.action_spec)

    def reset(self):
        # Rebuild the episode RNG deterministically; vector workers receive distinct config seeds.
        if hasattr(self.env, 'set_seed'):
            self.env.set_seed(self.seed + self.episode_index)
        else:
            self.env.seed = self.seed + self.episode_index
            self.env.rng = np.random.default_rng(self.seed + self.episode_index)
        self.episode_index += 1
        obs = self.env.reset()
        self.raw_lang = instruction(self.env.get_ep_meta())
        self.steps = 0
        self.done = False
        self.last_info = {}
        self.prev_obs = self.obs2meta(obs)
        return self.prev_obs

    def obs2meta(self, obs):
        state = state_vector(obs, self.state_keys)
        images = np.stack([rgb_image(obs[name + '_image'], self.image_flip)
                           for name in self.camera_names]).transpose(0, 3, 1, 2)
        return MetaObs(state=state, state_ee=state, image=images, raw_lang=self.raw_lang,
                       timestep=self.steps)

    def meta2act(self, action):
        if action['ctrl_space'] != self.ctrl_space or action['ctrl_type'] != self.ctrl_type:
            raise ValueError('RoboCasa action control tags do not match the configured controller')
        values = np.asarray(action['action'], dtype=np.float32)
        if values.shape != self.min_action.shape or not np.isfinite(values).all():
            raise ValueError(f'Expected finite native RoboCasa action {self.min_action.shape}')
        return np.clip(values, self.min_action, self.max_action)

    def step(self, action):
        if self.done:
            return asdict(self.prev_obs), 0., True, dict(self.last_info)
        obs, reward, terminated, info = self.env.step(self.meta2act(action))
        self.steps += 1
        success = bool(self.env._check_success())
        timeout = self.steps >= self.max_steps and not success
        self.done = bool(success or timeout or terminated)
        self.last_info = {**info, 'success': success, 'timeout': timeout,
                          'steps': self.steps, 'seed': self.seed + self.episode_index - 1}
        self.prev_obs = self.obs2meta(obs)
        return asdict(self.prev_obs), reward, self.done, dict(self.last_info)


def evaluate(args, env, action_manager, inference_ctx=None, video_writer=None,
             save_example_dir=None):
    """Score the task success predicate, never timeout or generic termination."""
    from vlastudio.benchmark.utils import organize_obs
    started = time.monotonic()
    finished = np.zeros(len(env), dtype=bool)
    success = np.zeros(len(env), dtype=bool)
    horizons = np.full(len(env), args.max_timesteps, dtype=int)
    try:
        obs = organize_obs(env.reset())
        for step in range(args.max_timesteps):
            if inference_ctx is not None:
                inference_ctx.update_obs(obs, step)
            actions = action_manager.select_action()
            if step == 0 and save_example_dir is not None:
                from vlastudio.benchmark.utils import _save_example_batch
                _save_example_batch(obs, actions, save_example_dir)
            raw_obs, _, done, infos = env.step(actions)
            for i in range(len(env)):
                if finished[i]:
                    continue
                success[i] = bool(infos[i].get('success', False))
                if done[i]:
                    finished[i] = True
                    horizons[i] = step + 1
                if video_writer is not None:
                    item = raw_obs[i]
                    images = item.image if isinstance(item, MetaObs) else item['image']
                    video_writer.append_data(np.concatenate(images.transpose(0, 2, 3, 1), axis=1))
            obs = organize_obs(raw_obs)
            if finished.all():
                break
    finally:
        if not getattr(args, 'reuse_env', False):
            env.close()
    return {'success': success.tolist(), 'total_success': int(success.sum()),
            'total': len(env), 'success_rate': float(success.mean()),
            'horizon': horizons.tolist(),
            'horizon_success': float(horizons[success].mean()) if success.any() else 0.,
            'eval_time': time.monotonic() - started}
