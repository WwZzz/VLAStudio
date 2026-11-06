"""Native RoboCasa image HDF5 demonstrations, without simulator dependencies."""
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from .base import EpisodicDataset
from vlastudio.benchmark.robocasa.observations import (
    CAMERAS, STATE_KEYS, instruction, metadata, rgb_image, state_vector,
)


class RoboCasaDataset(EpisodicDataset):
    """Read data/demo_N/{obs,actions}; preserve the full native action vector.

    Files are opened per read, so workers never inherit live h5py handles.
    image_flip defaults to False: official stored images are already upright.
    """
    def __init__(self, dataset_path_list, camera_names=CAMERAS, chunk_size=50,
                 ctrl_space='ee', ctrl_type='delta', image_size=None,
                 preload_data=False, filter_by_attribute=None,
                 state_keys=STATE_KEYS, image_flip=False):
        torch.utils.data.Dataset.__init__(self)
        if not dataset_path_list or chunk_size < 1:
            raise ValueError('Nonempty dataset_path_list and positive chunk_size are required')
        if image_size is not None:
            raise ValueError('Resize in the policy processor; pass image_size=None')
        self.chunk_size = int(chunk_size)
        self.ctrl_space, self.ctrl_type = ctrl_space, ctrl_type
        self.camera_names = list(camera_names)
        self.state_keys = tuple(state_keys)
        self.image_flip = bool(image_flip)
        self.preload_data = bool(preload_data)
        self.freq = -1
        files = set()
        for value in dataset_path_list:
            path = Path(os.path.expandvars(str(value))).expanduser()
            cache = os.environ.get('VLASTUDIO_DATA_CACHE_DIR')
            if cache and not path.is_absolute():
                path = Path(cache) / 'datasets' / path.name
            if path.is_dir():
                files.update(p.resolve() for p in path.rglob('*') if p.suffix in ('.h5', '.hdf5'))
            elif path.is_file():
                files.add(path.resolve())
            else:
                raise FileNotFoundError(path)
        if not files:
            raise ValueError('No RoboCasa HDF5 files found')
        self.dataset_dir = os.path.commonpath([str(p.parent) for p in files])
        self.records, self.episode_len, self.languages = [], [], []
        self.action_dim = self.state_dim = None
        for path in sorted(files):
            with h5py.File(path, 'r') as file:
                data = file['data']
                env_args = metadata(data.attrs['env_args'])
                freq = env_args.get('env_kwargs', {}).get('control_freq', 20)
                if self.freq != -1 and self.freq != freq:
                    raise ValueError('RoboCasa files must have matching control frequencies')
                self.freq = freq
                demos = sorted(data, key=lambda n: (int(n[5:]) if n.startswith('demo_') and n[5:].isdigit() else -1, n))
                if filter_by_attribute is not None:
                    mask = file['mask'][filter_by_attribute].asstr()[()].tolist()
                    if any(name not in data for name in mask):
                        raise ValueError('RoboCasa mask references an unknown demonstration')
                    demos = [name for name in demos if name in set(mask)]
                for name in demos:
                    demo = data[name]
                    actions = demo['actions']
                    if actions.ndim != 2 or len(actions) == 0:
                        raise ValueError(f'Invalid actions in {path}:{name}')
                    for key in (*self.state_keys, *(c + '_image' for c in self.camera_names)):
                        if len(demo['obs'][key]) != len(actions):
                            raise ValueError(f'Observation/action length mismatch: {name}/{key}')
                    dim = state_vector({k: demo['obs'][k][0] for k in self.state_keys}, self.state_keys).size
                    if self.action_dim is not None and (self.action_dim != actions.shape[1] or self.state_dim != dim):
                        raise ValueError('RoboCasa files must have matching state/action dimensions')
                    self.action_dim, self.state_dim = actions.shape[1], dim
                    self.records.append((str(path), name))
                    self.episode_len.append(len(actions))
                    self.languages.append(instruction(demo.attrs['ep_meta']))
        if not self.records:
            raise ValueError('No demonstrations selected by the RoboCasa dataset filter')
        self.episode_len = np.asarray(self.episode_len, dtype=np.int64)
        self.episode_ids = np.arange(len(self.records))
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = int(self.episode_len.max())
        self.dataset_path_list = list(range(len(self.records)))
        self._cache = {}
        if preload_data:
            for i in self.dataset_path_list:
                with h5py.File(self.records[i][0], 'r') as file:
                    demo = file['data'][self.records[i][1]]
                    self._cache[i] = {'actions': demo['actions'][()], 'obs': {
                        k: demo['obs'][k][()] for k in (*self.state_keys, *(c + '_image' for c in self.camera_names))}}

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return super().__getitem__(index)

    def _read(self, episode, start, stop, feats):
        def take(demo):
            result = {}
            if 'action' in feats:
                result['action'] = np.asarray(demo['actions'][start:stop], dtype=np.float32)
                if not np.isfinite(result['action']).all():
                    raise ValueError('RoboCasa actions contain non-finite values')
            if 'state' in feats:
                result['state'] = state_vector({k: demo['obs'][k][start:stop] for k in self.state_keys}, self.state_keys)
            if 'image' in feats:
                result['image'] = {c: rgb_image(demo['obs'][c + '_image'][start], self.image_flip) for c in self.camera_names}
            return result
        if episode in self._cache:
            return take(self._cache[episode])
        path, name = self.records[episode]
        with h5py.File(path, 'r') as file:
            return take(file['data'][name])

    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        start = 0 if start_ts is None else int(start_ts)
        result = self._read(dataset_path, start, start + self.chunk_size, ('action', 'state', 'image'))
        result['state'] = result['state'][0]
        result['language_instruction'] = self.languages[dataset_path]
        return result

    def load_feat_from_episode(self, dataset_path, feats=()):
        feats = (feats,) if isinstance(feats, str) else feats
        result = self._read(dataset_path, 0, int(self.episode_len[dataset_path]), feats or ('action', 'state'))
        if not feats or 'language_instruction' in feats:
            result['language_instruction'] = self.languages[dataset_path]
        return result
