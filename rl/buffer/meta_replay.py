"""
Meta Replay Buffer (rewritten).

Stores raw MetaObs/MetaAction/MetaObs(next) in env-first layout:
    (n_envs, capacity, ...)

Sampling for training returns normalized and processor-aligned data.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
import torch

from benchmark.base import MetaObs, MetaAction, dict2meta
from rl.utils import RunningMeanStd
from .base_replay import BaseReplay
from .transition import RLTransition


CTRL_SPACE_MAP = {'ee': 0, 'joint': 1}
CTRL_SPACE_INV_MAP = {v: k for k, v in CTRL_SPACE_MAP.items()}

CTRL_TYPE_MAP = {'delta': 0, 'absolute': 1, 'relative': 2}
CTRL_TYPE_INV_MAP = {v: k for k, v in CTRL_TYPE_MAP.items()}

GRIPPER_CONTINUOUS_MAP = {False: 0, True: 1}
GRIPPER_CONTINUOUS_INV_MAP = {v: k for k, v in GRIPPER_CONTINUOUS_MAP.items()}


class _MetaObsStorage:
    def __init__(
        self,
        n_envs: int,
        capacity: int,
        state_dim: Optional[int],
        state_ee_dim: Optional[int],
        state_joint_dim: Optional[int],
        state_obj_dim: Optional[int],
        image_shape: Optional[tuple],
        depth_shape: Optional[tuple],
        pc_shape: Optional[tuple],
        store_images: bool,
        store_depth: bool,
        store_pc: bool,
        store_lang: bool,
    ) -> None:
        self.n_envs = n_envs
        self.capacity = capacity
        self.store_images = store_images
        self.store_depth = store_depth
        self.store_pc = store_pc
        self.store_lang = store_lang

        self.state = np.zeros((n_envs, capacity, state_dim), dtype=np.float32) if state_dim else None
        self.state_ee = np.zeros((n_envs, capacity, state_ee_dim), dtype=np.float32) if state_ee_dim else None
        self.state_joint = np.zeros((n_envs, capacity, state_joint_dim), dtype=np.float32) if state_joint_dim else None
        self.state_obj = np.zeros((n_envs, capacity, state_obj_dim), dtype=np.float32) if state_obj_dim else None
        self.image = np.zeros((n_envs, capacity, *image_shape), dtype=np.uint8) if (store_images and image_shape) else None
        self.depth = np.zeros((n_envs, capacity, *depth_shape), dtype=np.float32) if (store_depth and depth_shape) else None
        self.pc = np.zeros((n_envs, capacity, *pc_shape), dtype=np.float32) if (store_pc and pc_shape) else None
        self.raw_lang: Optional[List[List[str]]] = [['' for _ in range(capacity)] for _ in range(n_envs)] if store_lang else None
        self.timestep = np.zeros((n_envs, capacity), dtype=np.int32)

    def set(self, idx: int, obs: Union[MetaObs, Dict[str, Any], List[Any]]) -> None:
        obs_dict = self._coerce_obs(obs)
        if obs_dict.get('state') is not None and self.state is not None:
            self.state[:, idx] = self._ensure_env_batch(obs_dict['state'])
        if obs_dict.get('state_ee') is not None and self.state_ee is not None:
            self.state_ee[:, idx] = self._ensure_env_batch(obs_dict['state_ee'])
        if obs_dict.get('state_joint') is not None and self.state_joint is not None:
            self.state_joint[:, idx] = self._ensure_env_batch(obs_dict['state_joint'])
        if obs_dict.get('state_obj') is not None and self.state_obj is not None:
            self.state_obj[:, idx] = self._ensure_env_batch(obs_dict['state_obj'])
        if obs_dict.get('image') is not None and self.image is not None:
            self.image[:, idx] = self._ensure_env_batch(obs_dict['image'])
        if obs_dict.get('depth') is not None and self.depth is not None:
            self.depth[:, idx] = self._ensure_env_batch(obs_dict['depth'])
        if obs_dict.get('pc') is not None and self.pc is not None:
            self.pc[:, idx] = self._ensure_env_batch(obs_dict['pc'])

        if self.store_lang and self.raw_lang is not None and obs_dict.get('raw_lang') is not None:
            raw_lang = obs_dict['raw_lang']
            if isinstance(raw_lang, str):
                raw_lang = [raw_lang] * self.n_envs
            for env_i, value in enumerate(raw_lang):
                self.raw_lang[env_i][idx] = value

        if obs_dict.get('timestep') is not None:
            timestep = np.asarray(obs_dict['timestep'])
            if timestep.ndim == 0:
                timestep = np.array([timestep])
            self.timestep[:, idx] = timestep

    def get(self, time_indices: np.ndarray, env_indices: np.ndarray, keys: Set[str]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        if 'state' in keys and self.state is not None:
            batch['state'] = self.state[env_indices, time_indices].copy()
        if 'state_ee' in keys and self.state_ee is not None:
            batch['state_ee'] = self.state_ee[env_indices, time_indices].copy()
        if 'state_joint' in keys and self.state_joint is not None:
            batch['state_joint'] = self.state_joint[env_indices, time_indices].copy()
        if 'state_obj' in keys and self.state_obj is not None:
            batch['state_obj'] = self.state_obj[env_indices, time_indices].copy()
        if 'image' in keys and self.image is not None:
            batch['image'] = self.image[env_indices, time_indices].copy()
        if 'depth' in keys and self.depth is not None:
            batch['depth'] = self.depth[env_indices, time_indices].copy()
        if 'pc' in keys and self.pc is not None:
            batch['pc'] = self.pc[env_indices, time_indices].copy()
        if 'raw_lang' in keys and self.raw_lang is not None:
            batch['raw_lang'] = [self.raw_lang[e][t] for t, e in zip(time_indices, env_indices)]
        if 'timestep' in keys:
            batch['timestep'] = self.timestep[env_indices, time_indices].copy()
        return batch

    def _coerce_obs(self, obs: Union[MetaObs, Dict[str, Any], List[Any]]) -> Dict[str, Any]:
        if isinstance(obs, list):
            obs_dicts = [self._coerce_obs(o) for o in obs]
            return self._stack_dicts(obs_dicts)
        if hasattr(obs, '__dataclass_fields__'):
            return asdict(obs)
        if isinstance(obs, dict):
            return obs
        if hasattr(obs, '__dict__'):
            return vars(obs)
        return {}

    def _stack_dicts(self, dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not dict_list:
            return {}
        result = {}
        for key in dict_list[0].keys():
            values = [d.get(key) for d in dict_list]
            if values[0] is None:
                result[key] = None
            elif isinstance(values[0], np.ndarray):
                result[key] = np.stack(values)
            elif isinstance(values[0], (int, float, bool)):
                result[key] = np.array(values)
            elif isinstance(values[0], str):
                result[key] = values
            else:
                result[key] = values
        return result

    def _ensure_env_batch(self, data: Any) -> np.ndarray:
        arr = np.asarray(data)
        if self.n_envs == 1 and arr.ndim >= 1 and arr.shape[0] != 1:
            arr = arr[np.newaxis, ...]
        return arr


class _MetaActionStorage:
    def __init__(self, n_envs: int, capacity: int, action_dim: Optional[int]) -> None:
        self.n_envs = n_envs
        self.capacity = capacity
        self.action = np.zeros((n_envs, capacity, action_dim), dtype=np.float32) if action_dim else None
        self.ctrl_space = np.zeros((n_envs, capacity), dtype=np.int8)
        self.ctrl_type = np.zeros((n_envs, capacity), dtype=np.int8)
        self.gripper_continuous = np.zeros((n_envs, capacity), dtype=np.int8)

    def set(self, idx: int, action: Union[MetaAction, Dict[str, Any], List[Any]]) -> None:
        action_dict = self._coerce_action(action)
        if action_dict.get('action') is not None and self.action is not None:
            self.action[:, idx] = self._ensure_env_batch(action_dict['action'])

        ctrl_space = action_dict.get('ctrl_space', 'ee')
        if isinstance(ctrl_space, str):
            ctrl_space = [ctrl_space] * self.n_envs
        self.ctrl_space[:, idx] = np.array([CTRL_SPACE_MAP.get(cs, 0) for cs in ctrl_space], dtype=np.int8)

        ctrl_type = action_dict.get('ctrl_type', 'delta')
        if isinstance(ctrl_type, str):
            ctrl_type = [ctrl_type] * self.n_envs
        self.ctrl_type[:, idx] = np.array([CTRL_TYPE_MAP.get(ct, 0) for ct in ctrl_type], dtype=np.int8)

        gripper_continuous = action_dict.get('gripper_continuous', False)
        if isinstance(gripper_continuous, bool):
            gripper_continuous = [gripper_continuous] * self.n_envs
        self.gripper_continuous[:, idx] = np.array([GRIPPER_CONTINUOUS_MAP.get(gc, 0) for gc in gripper_continuous], dtype=np.int8)

    def get(self, time_indices: np.ndarray, env_indices: np.ndarray, keys: Set[str]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        if 'action' in keys and self.action is not None:
            batch['action'] = self.action[env_indices, time_indices].copy()
        if 'ctrl_space' in keys:
            batch['ctrl_space'] = self.ctrl_space[env_indices, time_indices].copy()
            batch['ctrl_space_str'] = [CTRL_SPACE_INV_MAP[v] for v in batch['ctrl_space']]
        if 'ctrl_type' in keys:
            batch['ctrl_type'] = self.ctrl_type[env_indices, time_indices].copy()
            batch['ctrl_type_str'] = [CTRL_TYPE_INV_MAP[v] for v in batch['ctrl_type']]
        if 'gripper_continuous' in keys:
            batch['gripper_continuous'] = self.gripper_continuous[env_indices, time_indices].copy()
            batch['gripper_continuous_bool'] = [GRIPPER_CONTINUOUS_INV_MAP[v] for v in batch['gripper_continuous']]
        return batch

    def _coerce_action(self, action: Union[MetaAction, Dict[str, Any], List[Any]]) -> Dict[str, Any]:
        if isinstance(action, list):
            action_dicts = [self._coerce_action(a) for a in action]
            return self._stack_dicts(action_dicts)
        if hasattr(action, '__dataclass_fields__'):
            return asdict(action)
        if isinstance(action, dict):
            return action
        if hasattr(action, '__dict__'):
            return vars(action)
        return {}

    def _stack_dicts(self, dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not dict_list:
            return {}
        result = {}
        for key in dict_list[0].keys():
            values = [d.get(key) for d in dict_list]
            if values[0] is None:
                result[key] = None
            elif isinstance(values[0], np.ndarray):
                result[key] = np.stack(values)
            elif isinstance(values[0], (int, float, bool)):
                result[key] = np.array(values)
            elif isinstance(values[0], str):
                result[key] = values
            else:
                result[key] = values
        return result

    def _ensure_env_batch(self, data: Any) -> np.ndarray:
        arr = np.asarray(data)
        if self.n_envs == 1 and arr.ndim >= 1 and arr.shape[0] != 1:
            arr = arr[np.newaxis, ...]
        return arr


class MetaReplay(BaseReplay):
    """
    Replay buffer for MetaObs/MetaAction with env-first storage.
    """

    DEFAULT_SAMPLE_KEYS = {'state', 'action', 'next_state', 'reward', 'done', 'truncated'}
    ALL_SAMPLE_KEYS = {
        'state', 'state_ee', 'state_joint', 'state_obj', 'image', 'depth', 'pc', 'raw_lang', 'timestep',
        'next_state', 'next_state_ee', 'next_state_joint', 'next_state_obj', 'next_image', 'next_depth',
        'next_pc', 'next_raw_lang', 'next_timestep',
        'action', 'ctrl_space', 'ctrl_type', 'gripper_continuous',
        'reward', 'done', 'truncated', 'trajectory_id',
    }

    def __init__(
        self,
        capacity: int = 100000,
        device: Union[str, torch.device] = 'cpu',
        env_type: Optional[str] = None,
        n_envs: int = 1,
        state_dim: Optional[int] = None,
        state_ee_dim: Optional[int] = None,
        state_joint_dim: Optional[int] = None,
        state_obj_dim: Optional[int] = None,
        image_shape: Optional[tuple] = None,
        depth_shape: Optional[tuple] = None,
        pc_shape: Optional[tuple] = None,
        action_dim: Optional[int] = None,
        store_images: bool = True,
        store_depth: bool = False,
        store_pc: bool = False,
        store_lang: bool = False,
        data_processor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        data_collator: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
        state_normalizer: Optional[RunningMeanStd] = None,
        action_normalizer: Optional[RunningMeanStd] = None,
        update_normalizers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(capacity=capacity, device=device, env_type=env_type, n_envs=n_envs, **kwargs)

        self.state_dim = state_dim
        self.state_ee_dim = state_ee_dim
        self.state_joint_dim = state_joint_dim
        self.state_obj_dim = state_obj_dim
        self.image_shape = image_shape
        self.depth_shape = depth_shape
        self.pc_shape = pc_shape
        self.action_dim = action_dim
        self.store_images = store_images
        self.store_depth = store_depth
        self.store_pc = store_pc
        self.store_lang = store_lang

        self.data_processor = data_processor
        self.data_collator = data_collator
        self.update_normalizers = update_normalizers
        self.state_normalizer = state_normalizer or (RunningMeanStd((state_dim,)) if state_dim else None)
        self.action_normalizer = action_normalizer or (RunningMeanStd((action_dim,)) if action_dim else None)

        self._obs = _MetaObsStorage(
            n_envs=n_envs,
            capacity=capacity,
            state_dim=state_dim,
            state_ee_dim=state_ee_dim,
            state_joint_dim=state_joint_dim,
            state_obj_dim=state_obj_dim,
            image_shape=image_shape,
            depth_shape=depth_shape,
            pc_shape=pc_shape,
            store_images=store_images,
            store_depth=store_depth,
            store_pc=store_pc,
            store_lang=store_lang,
        )
        self._next_obs = _MetaObsStorage(
            n_envs=n_envs,
            capacity=capacity,
            state_dim=state_dim,
            state_ee_dim=state_ee_dim,
            state_joint_dim=state_joint_dim,
            state_obj_dim=state_obj_dim,
            image_shape=image_shape,
            depth_shape=depth_shape,
            pc_shape=pc_shape,
            store_images=store_images,
            store_depth=store_depth,
            store_pc=store_pc,
            store_lang=store_lang,
        )
        self._action = _MetaActionStorage(n_envs=n_envs, capacity=capacity, action_dim=action_dim)
        self._reward = np.zeros((n_envs, capacity), dtype=np.float32)
        self._done = np.zeros((n_envs, capacity), dtype=np.bool_)
        self._truncated = np.zeros((n_envs, capacity), dtype=np.bool_)
        self._trajectory_id = np.zeros((n_envs, capacity), dtype=np.int32)

    def add(self, transition: Union[RLTransition, Dict[str, Any]]) -> None:
        if not isinstance(transition, RLTransition):
            transition = self._coerce_transition(transition)

        idx = self._position


        self._obs.set(idx, transition.obs)
        self._action.set(idx, transition.action)
        self._next_obs.set(idx, transition.next_obs)

        reward = np.asarray(transition.reward)
        done = np.asarray(transition.done)
        truncated = np.asarray(transition.truncated) if transition.truncated is not None else np.zeros(self.n_envs, dtype=bool)
        if self.n_envs == 1:
            reward = np.atleast_1d(reward)
            done = np.atleast_1d(done)
            truncated = np.atleast_1d(truncated)

        self._reward[:, idx] = reward
        self._done[:, idx] = done
        self._truncated[:, idx] = truncated

        if isinstance(transition.info, dict) and 'trajectory_id' in transition.info:
            traj_id = transition.info['trajectory_id']
            if self.n_envs == 1:
                traj_id = np.atleast_1d(traj_id)
            self._trajectory_id[:, idx] = traj_id

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        env_indices: Optional[Union[List[int], np.ndarray]] = None,
        keys: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        if self._size == 0:
            return {}

        if keys is None:
            keys = self.DEFAULT_SAMPLE_KEYS

        if env_indices is None:
            env_indices = np.arange(self.n_envs)
        env_indices = np.asarray(env_indices)

        time_indices = np.random.randint(0, self._size, size=batch_size)
        env_sample_indices = np.random.choice(env_indices, size=batch_size)

        batch = {}
        obs_keys = {k for k in keys if not k.startswith('next_')}
        next_obs_keys = {k[5:] for k in keys if k.startswith('next_')}

        batch.update(self._obs.get(time_indices, env_sample_indices, obs_keys))
        next_obs_batch = self._next_obs.get(time_indices, env_sample_indices, next_obs_keys)
        for key, value in next_obs_batch.items():
            batch[f"next_{key}"] = value
        batch.update(self._action.get(time_indices, env_sample_indices, keys))

        if 'reward' in keys:
            batch['reward'] = self._reward[env_sample_indices, time_indices].copy()
        if 'done' in keys:
            batch['done'] = self._done[env_sample_indices, time_indices].copy()
        if 'truncated' in keys:
            batch['truncated'] = self._truncated[env_sample_indices, time_indices].copy()
        if 'trajectory_id' in keys:
            batch['trajectory_id'] = self._trajectory_id[env_sample_indices, time_indices].copy()

        batch['time_indices'] = time_indices.copy()
        batch['env_indices'] = env_sample_indices.copy()
        return batch

    def sample_as_tensor(
        self,
        batch_size: int,
        env_indices: Optional[Union[List[int], np.ndarray]] = None,
        keys: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        batch = self.sample(batch_size, env_indices=env_indices, keys=keys)
        if not batch:
            return {}
        tensor_batch: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                tensor_batch[key] = torch.from_numpy(value).to(self.device)
            else:
                tensor_batch[key] = value
        return tensor_batch

    def sample_for_training(
        self,
        batch_size: int,
        env_indices: Optional[Union[List[int], np.ndarray]] = None,
        keys: Optional[Set[str]] = None,
        data_processor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        data_collator: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
        apply_normalization: bool = True,
    ) -> Dict[str, Any]:
        batch = self.sample(batch_size, env_indices=env_indices, keys=keys)
        if not batch:
            return {}

        if apply_normalization:
            batch = self._apply_normalization(batch)

        processor = data_processor or self.data_processor
        collator = data_collator or self.data_collator
        if processor is not None or collator is not None:
            obs_samples = self._build_samples(batch, prefix='')
            next_obs_samples = self._build_samples(batch, prefix='next_')

            if processor is not None:
                obs_samples = [processor(sample) for sample in obs_samples]
                next_obs_samples = [processor(sample) for sample in next_obs_samples]

            if collator is not None:
                batch['processed_obs'] = collator(obs_samples)
                batch['processed_next_obs'] = collator(next_obs_samples)
            else:
                batch['processed_obs'] = self._default_collate(obs_samples)
                batch['processed_next_obs'] = self._default_collate(next_obs_samples)

        return batch

    def clear(self) -> None:
        self._size = 0
        self._position = 0
        self.__init__(
            capacity=self.capacity,
            device=self.device,
            env_type=self.env_type,
            n_envs=self.n_envs,
            state_dim=self.state_dim,
            state_ee_dim=self.state_ee_dim,
            state_joint_dim=self.state_joint_dim,
            state_obj_dim=self.state_obj_dim,
            image_shape=self.image_shape,
            depth_shape=self.depth_shape,
            pc_shape=self.pc_shape,
            action_dim=self.action_dim,
            store_images=self.store_images,
            store_depth=self.store_depth,
            store_pc=self.store_pc,
            store_lang=self.store_lang,
            data_processor=self.data_processor,
            data_collator=self.data_collator,
            state_normalizer=self.state_normalizer,
            action_normalizer=self.action_normalizer,
            update_normalizers=self.update_normalizers,
        )

    def save(self, path: str, **kwargs) -> None:
        fmt = kwargs.get('format', 'pkl')
        if path.endswith('.npz'):
            fmt = 'npz'

        data = {
            'size': self._size,
            'position': self._position,
            'capacity': self.capacity,
            'n_envs': self.n_envs,
            'env_type': self.env_type,
            'state_dim': self.state_dim,
            'state_ee_dim': self.state_ee_dim,
            'state_joint_dim': self.state_joint_dim,
            'state_obj_dim': self.state_obj_dim,
            'image_shape': self.image_shape,
            'depth_shape': self.depth_shape,
            'pc_shape': self.pc_shape,
            'action_dim': self.action_dim,
            'store_images': self.store_images,
            'store_depth': self.store_depth,
            'store_pc': self.store_pc,
            'store_lang': self.store_lang,
            'storage_layout': 'env_first',
        }

        size = self._size
        data.update({
            '_state': None if self._obs.state is None else self._obs.state[:, :size],
            '_state_ee': None if self._obs.state_ee is None else self._obs.state_ee[:, :size],
            '_state_joint': None if self._obs.state_joint is None else self._obs.state_joint[:, :size],
            '_state_obj': None if self._obs.state_obj is None else self._obs.state_obj[:, :size],
            '_image': None if self._obs.image is None else self._obs.image[:, :size],
            '_depth': None if self._obs.depth is None else self._obs.depth[:, :size],
            '_pc': None if self._obs.pc is None else self._obs.pc[:, :size],
            '_timestep': self._obs.timestep[:, :size],
            '_next_state': None if self._next_obs.state is None else self._next_obs.state[:, :size],
            '_next_state_ee': None if self._next_obs.state_ee is None else self._next_obs.state_ee[:, :size],
            '_next_state_joint': None if self._next_obs.state_joint is None else self._next_obs.state_joint[:, :size],
            '_next_state_obj': None if self._next_obs.state_obj is None else self._next_obs.state_obj[:, :size],
            '_next_image': None if self._next_obs.image is None else self._next_obs.image[:, :size],
            '_next_depth': None if self._next_obs.depth is None else self._next_obs.depth[:, :size],
            '_next_pc': None if self._next_obs.pc is None else self._next_obs.pc[:, :size],
            '_next_timestep': self._next_obs.timestep[:, :size],
            '_action': None if self._action.action is None else self._action.action[:, :size],
            '_ctrl_space': self._action.ctrl_space[:, :size],
            '_ctrl_type': self._action.ctrl_type[:, :size],
            '_gripper_continuous': self._action.gripper_continuous[:, :size],
            '_reward': self._reward[:, :size],
            '_done': self._done[:, :size],
            '_truncated': self._truncated[:, :size],
            '_trajectory_id': self._trajectory_id[:, :size],
        })
        if self.store_lang and self._obs.raw_lang is not None:
            data['_raw_lang'] = [row[:size] for row in self._obs.raw_lang]
        if self.store_lang and self._next_obs.raw_lang is not None:
            data['_next_raw_lang'] = [row[:size] for row in self._next_obs.raw_lang]

        if fmt == 'npz':
            np_data = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
            if '_raw_lang' in data:
                np_data['_raw_lang'] = np.array(data['_raw_lang'], dtype=object)
            if '_next_raw_lang' in data:
                np_data['_next_raw_lang'] = np.array(data['_next_raw_lang'], dtype=object)
            np_data['_metadata'] = np.array([data['size'], data['position'], data['capacity'], data['n_envs']])
            np_data['_env_type'] = np.array([data['env_type']], dtype=object)
            np.savez_compressed(path, **np_data)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)

    def load(self, path: str, **kwargs) -> None:
        append = kwargs.get('append', False)
        if not append:
            self.clear()

        if path.endswith('.npz'):
            data = np.load(path, allow_pickle=True)
            size = int(data['_metadata'][0])
            for i in range(size):
                transition = self._extract_transition_from_npz(data, i)
                self.add(transition)
            return

        with open(path, 'rb') as f:
            data = pickle.load(f)
        size = data['size']
        for i in range(size):
            transition = self._extract_transition_from_data(data, i)
            self.add(transition)

    def _apply_normalization(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.state_normalizer is not None:
            if 'state' in batch:
                batch['state'] = self.state_normalizer.normalize(batch['state'])
            if 'next_state' in batch:
                batch['next_state'] = self.state_normalizer.normalize(batch['next_state'])
        if self.action_normalizer is not None and 'action' in batch:
            batch['action'] = self.action_normalizer.normalize(batch['action'])
        return batch

    def _build_samples(self, batch: Dict[str, Any], prefix: str = '') -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        state_key = f'{prefix}state'
        image_key = f'{prefix}image'
        raw_lang_key = f'{prefix}raw_lang'
        timestep_key = f'{prefix}timestep'

        state_arr = batch.get(state_key, None)
        batch_size = state_arr.shape[0] if isinstance(state_arr, np.ndarray) and state_arr.ndim > 1 else len(batch.get(raw_lang_key, []))
        if batch_size == 0:
            return samples

        for i in range(batch_size):
            sample = {}
            if image_key in batch:
                sample['image'] = batch[image_key][i]
            if state_key in batch:
                sample['state'] = batch[state_key][i] if isinstance(batch[state_key], np.ndarray) else batch[state_key]
            if raw_lang_key in batch:
                sample['raw_lang'] = batch[raw_lang_key][i]
            if timestep_key in batch:
                sample['timestamp'] = batch[timestep_key][i]
            samples.append(sample)
        return samples

    def _default_collate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            return {}
        batch: Dict[str, Any] = {}
        keys = samples[0].keys()
        for key in keys:
            values = [s[key] for s in samples]
            if values[0] is None:
                batch[key] = None
            elif isinstance(values[0], np.ndarray):
                batch[key] = torch.from_numpy(np.stack(values))
            elif isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            elif isinstance(values[0], str):
                batch[key] = values
            elif isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            else:
                batch[key] = values
        return batch

    def _maybe_update_normalizers(self, obs: MetaObs, action: MetaAction) -> None:
        if not self.update_normalizers:
            return
        if self.state_normalizer is not None and obs is not None and getattr(obs, 'state', None) is not None:
            state_arr = np.asarray(getattr(obs, 'state'))
            if self.n_envs == 1 and state_arr.ndim == 1:
                state_arr = state_arr[np.newaxis, :]
            if state_arr.ndim >= 2:
                self.state_normalizer.update(state_arr)
        if self.action_normalizer is not None and action is not None and getattr(action, 'action', None) is not None:
            action_arr = np.asarray(getattr(action, 'action'))
            if self.n_envs == 1 and action_arr.ndim == 1:
                action_arr = action_arr[np.newaxis, :]
            if action_arr.ndim >= 2:
                self.action_normalizer.update(action_arr)

    def _coerce_transition(self, transition: Dict[str, Any]) -> RLTransition:
        obs = transition.get('state') or transition.get('obs')
        action = transition.get('action')
        next_obs = transition.get('next_state') or transition.get('next_obs')
        reward = transition.get('reward')
        done = transition.get('done')
        truncated = transition.get('truncated', None)
        info = transition.get('info', None)

        if not isinstance(obs, MetaObs):
            obs = dict2meta(obs or {}, mtype='obs')
        if not isinstance(action, MetaAction):
            action = dict2meta(action or {}, mtype='act')
        if not isinstance(next_obs, MetaObs):
            next_obs = dict2meta(next_obs or {}, mtype='obs')

        return RLTransition(obs=obs, action=action, next_obs=next_obs, reward=reward, done=done, truncated=truncated, info=info)

    def _extract_transition_from_data(self, data: Dict[str, Any], idx: int) -> RLTransition:
        obs = MetaObs(
            state=data.get('_state')[:, idx] if data.get('_state') is not None else None,
            state_ee=data.get('_state_ee')[:, idx] if data.get('_state_ee') is not None else None,
            state_joint=data.get('_state_joint')[:, idx] if data.get('_state_joint') is not None else None,
            state_obj=data.get('_state_obj')[:, idx] if data.get('_state_obj') is not None else None,
            image=data.get('_image')[:, idx] if data.get('_image') is not None else None,
            depth=data.get('_depth')[:, idx] if data.get('_depth') is not None else None,
            pc=data.get('_pc')[:, idx] if data.get('_pc') is not None else None,
            raw_lang=[row[idx] for row in data.get('_raw_lang', [])] if data.get('_raw_lang') is not None else None,
            timestep=data.get('_timestep')[:, idx] if data.get('_timestep') is not None else None,
        )
        next_obs = MetaObs(
            state=data.get('_next_state')[:, idx] if data.get('_next_state') is not None else None,
            state_ee=data.get('_next_state_ee')[:, idx] if data.get('_next_state_ee') is not None else None,
            state_joint=data.get('_next_state_joint')[:, idx] if data.get('_next_state_joint') is not None else None,
            state_obj=data.get('_next_state_obj')[:, idx] if data.get('_next_state_obj') is not None else None,
            image=data.get('_next_image')[:, idx] if data.get('_next_image') is not None else None,
            depth=data.get('_next_depth')[:, idx] if data.get('_next_depth') is not None else None,
            pc=data.get('_next_pc')[:, idx] if data.get('_next_pc') is not None else None,
            raw_lang=[row[idx] for row in data.get('_next_raw_lang', [])] if data.get('_next_raw_lang') is not None else None,
            timestep=data.get('_next_timestep')[:, idx] if data.get('_next_timestep') is not None else None,
        )
        action = MetaAction(
            action=data.get('_action')[:, idx] if data.get('_action') is not None else None,
            ctrl_space=[CTRL_SPACE_INV_MAP[int(v)] for v in np.atleast_1d(data.get('_ctrl_space')[:, idx])],
            ctrl_type=[CTRL_TYPE_INV_MAP[int(v)] for v in np.atleast_1d(data.get('_ctrl_type')[:, idx])],
            gripper_continuous=[GRIPPER_CONTINUOUS_INV_MAP[int(v)] for v in np.atleast_1d(data.get('_gripper_continuous')[:, idx])],
        )
        return RLTransition(
            obs=obs,
            action=action,
            next_obs=next_obs,
            reward=data.get('_reward')[:, idx],
            done=data.get('_done')[:, idx],
            truncated=data.get('_truncated')[:, idx] if data.get('_truncated') is not None else None,
        )

    def _extract_transition_from_npz(self, npz_data, idx: int) -> RLTransition:
        data = {k: npz_data[k] for k in npz_data.files}
        return self._extract_transition_from_data(data, idx)

    def get_trajectory(self, trajectory_id: int, env_idx: int = 0) -> Dict[str, Any]:
        mask = self._trajectory_id[env_idx, :self._size] == trajectory_id
        time_indices = np.where(mask)[0]
        if len(time_indices) == 0:
            return {}
        env_indices = np.full(len(time_indices), env_idx, dtype=np.int64)
        return self.sample(len(time_indices), env_indices=env_indices)

    def __repr__(self) -> str:
        return (f"MetaReplay(capacity={self.capacity}, size={self._size}, "
                f"n_envs={self.n_envs}, total_transitions={self.total_transitions}, "
                f"env_type='{self.env_type}', state_dim={self.state_dim}, "
                f"action_dim={self.action_dim}, store_images={self.store_images}, "
                f"store_lang={self.store_lang})")


