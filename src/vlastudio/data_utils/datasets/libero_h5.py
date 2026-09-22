from vlastudio.data_utils.datasets.base import EpisodicDataset
import numpy as np 
import os 
import h5py
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List

class LiberoHDF5(EpisodicDataset):
    def __init__(self, root:str="", split:List[str]=['object'], camera_names: List[str]=['primary'], 
                chunk_size: int = 16, 
                ctrl_space: str = 'ee', ctrl_type: str = 'delta',
                image_size: tuple = (256, 256), preload_data: bool = False):
        super(EpisodicDataset).__init__()
        if isinstance(split, str): split = [split]
        root = self._download_dataset_if_needed(dataset_dir=root, split=split)
        self.root = root
        dataset_path_list = [os.path.join(root, f'libero_{s}') for s in split]
        if isinstance(dataset_path_list, str): dataset_path_list = [dataset_path_list]
            # New behavior: dataset_path_list contains a single directory path
        all_h5_files = []
        for dataset_dir in dataset_path_list:
            if not os.path.isdir(dataset_dir):
                raise ValueError(f"Expected a directory path, but got: {dataset_dir}")
            crt_h5_files = self._find_all_hdf5(dataset_dir)
            all_h5_files.extend(crt_h5_files)
        self.dataset_path_list = all_h5_files
        self.dataset_dir = dataset_path_list[0] 
        self.episode_ids = None
        self.chunk_size = chunk_size
        self.camera_names = camera_names
        self.ctrl_space = ctrl_space  # ['ee', 'joint', 'other']
        self.ctrl_type = ctrl_type  # ['abs', 'rel', 'delta']
        self.image_size = image_size
        self.preload_data = preload_data
        self.freq = -1
        self.max_workers = 8
        self.initialize()
    
    def _download_dataset_if_needed(self, dataset_dir="", split:List[str]=['object']):
        """
        Download dataset from HuggingFace if dataset_dir is empty or doesn't exist.
        
        Returns:
            Updated dataset_dir path
        """
        if not os.path.exists(dataset_dir):
            min_episodes = 0
        else:
            min_episodes = 100
            for s in split:
                p = os.path.join(dataset_dir, f'libero_{s}')
                num_episodes = len(self._find_all_hdf5(p))
                if num_episodes < min_episodes: min_episodes = num_episodes
        if min_episodes >= 10:
            return dataset_dir
        task_name = dataset_dir
        # Determine which HuggingFace repo to download from based on task name
        repo_id = "yifengzhu-hf/LIBERO-datasets"
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download datasets. "
                "Please install it with: pip install huggingface_hub"
            )
        if not dataset_dir:
            dataset_dir = os.environ.get('ILSTD_CACHE', os.path.expanduser('~/.cache/ilstd'))
            os.makedirs(dataset_dir, exist_ok=True)
        # Download the dataset using HuggingFace's default cache
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dataset_dir,
            local_dir_use_symlinks=False
        )
        return downloaded_path or dataset_dir
    
    def _get_h5_len(self, datapath):
        try:
            with h5py.File(datapath, 'r') as root:
                num_episodes = len(root['data'])
                raw_lang = eval(root['data'].attrs["problem_info"])["language_instruction"]
                episode_lens = [root[f'data/demo_{i}/actions'].shape[0] for i in range(num_episodes)]
            return {datapath: {'num_episodes': num_episodes, "episode_lens": episode_lens, 'raw_lang': raw_lang}}
        except Exception as e:
            print(f'Error loading {datapath} in _get_h5_len due to {e}')
            quit()
    
    def _get_episode_len(self):
        all_data = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit parallel tasks
            results = executor.map(self._get_h5_len, self.dataset_path_list)
            # Collect results
            for result in results:
                all_data.update(result)
        return all_data
    
    def get_dataset_dir(self):
        return self.root
    
    def get_raw_lang(self, data_path):
        return ""
    
    def get_episode_len(self):
        """
        Get the length of each episode in the dataset.
        
        Returns:
            List of episode lengths
        """
        self._languages = dict()
        res = []
        
        if self.loaded_data is not None:
            tmp = self.loaded_data[list(self.loaded_data.keys())[0]]
            all_episode_len = [
                [self.loaded_data[pi][f'/data/demo_{i}/actions'].shape[0] for i in range(50)]
                for pi in self.dataset_path_list
            ]
            for lens in all_episode_len:
                res.extend(lens)
            for dataset_path in self.loaded_data.keys():
                self._languages[dataset_path] = self.loaded_data[dataset_path].pop('raw_lang')
        else:
            all_episode_len = self._get_episode_len()
            key = "actions"
            for dataset_path in tqdm(self.dataset_path_list):
                res.extend(all_episode_len[dataset_path]['episode_lens'])
                self._languages[dataset_path] = all_episode_len[dataset_path]['raw_lang']
        return res

    def flatten_dataset_path_list(self):
        flattened_list = []
        for h5file in tqdm(self.dataset_path_list):
            with h5py.File(h5file, 'r') as root:
                dataset_paths = [f"{h5file}:demo_{i}" for i in range(len(root['data']))]
                flattened_list.extend(dataset_paths)
        return flattened_list
    
    def initialize(self):
        """Initialize the dataset by loading data and computing episode lengths."""
        self.loaded_data = self._load_all_episodes_into_memory() if self.preload_data else None
        self.episode_len = self.get_episode_len()  # Get length of each episode
        self.dataset_path_list = self.flatten_dataset_path_list()
        self.episode_ids = [i for i in range(len(self.dataset_path_list))]
        self.cumulative_len = np.cumsum(self.episode_len)  # Compute cumulative lengths
        self.max_episode_len = max(self.episode_len)  # Get maximum episode length

    def _load_all_episodes_into_memory(self):
        """
        Load all HDF5 files into memory in parallel.
        
        Returns:
            Dictionary containing all loaded data
        """
        print("Pre-Loading all data into memory...")
        memory_data = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit parallel tasks
            results = executor.map(self._load_file_into_memory, self.dataset_path_list)
            # Collect results
            for result in results:
                memory_data.update(result)
        print("Pre-Loading Done")
        return memory_data

    def _load_file_into_memory(self, dataset_path):
        """
        Load a single HDF5 file and flatten its content.
        
        Args:
            dataset_path: Path to the HDF5 file
            
        Returns:
            Dictionary containing flattened data from the file
        """
        flattened_data = {}
        with h5py.File(dataset_path, 'r') as f:
            def recursive_load(group, current_path=""):
                for key, item in group.items():
                    full_path = f"{current_path}/{key}" if current_path else f"/{key}"
                    if isinstance(item, h5py.Group):
                        recursive_load(item, full_path)
                    elif isinstance(item, h5py.Dataset):
                        flattened_data[full_path] = item[()]
            recursive_load(f)
            flattened_data['raw_lang'] = eval(f['data'].attrs["problem_info"])["language_instruction"]
        return {dataset_path: flattened_data}
    
    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """
        Load one-step data at start_ts from the episode specified by dataset_path.
        
        Args:
            dataset_path: Path to the dataset file
            start_ts: Starting timestep
            
        Returns:
            Dictionary containing the loaded data
        """
        dataset_path, ep = dataset_path.split(':')
        root = self.loaded_data[dataset_path] if self.loaded_data is not None else h5py.File(dataset_path, 'r')
        prefix = f'/data/{ep}/'
        # Load language 
        raw_lang = self._languages[dataset_path]
        # Load state
        state = np.concatenate([root[prefix+'obs/ee_states'][start_ts],root[prefix+'obs/gripper_states'][start_ts]], axis=0)
        # Load action
        action = root[prefix+'actions'][start_ts:min(start_ts+self.chunk_size, root[prefix+'actions'].shape[0])]
        # Load image
        image_dict = dict(
            image_primary = cv2.resize(root[prefix+'obs/agentview_rgb'][start_ts], self.image_size)
        )
        if 'image_wrist' in self.camera_names:
            image_dict['image_wrist'] = cv2.resize(root[prefix+'obs/eye_in_hand_rgb'][start_ts], self.image_size)
        return dict(
            action=action,
            state=state,
            image=image_dict,
            language_instruction=raw_lang,
            reasoning="",
        )
       
    def load_feat_from_episode(self, dataset_path, feats=[]):
        """
        Load all steps data from the episode specified by dataset_path.
        
        Args:
            dataset_path: Path to the dataset file
            feats: List of features to load
            
        Returns:
            Dictionary containing the loaded data
        """
        dataset_path, ep = dataset_path.split(':')
        root = self.loaded_data[dataset_path] if self.loaded_data is not None else h5py.File(dataset_path, 'r')
        prefix = f'/data/{ep}/'
        data_dict = {}
        if 'language_instruction' in feats or len(feats) == 0: 
            data_dict['language_instruction'] = self._languages[dataset_path]
        if 'state' in feats or len(feats) == 0: 
            data_dict['state'] = np.concatenate([root[prefix+'obs/ee_states'][:],root[prefix+'obs/gripper_states'][:]], axis=-1)
        if 'action' in feats or len(feats) == 0:  # Load action
            data_dict['action'] = root[prefix+'actions'][()]
        if 'image_primary' in feats or len(feats) == 0:
            images = []
            for i in range(root[prefix+'obs/agentview_rgb'].shape[0]):
                img = cv2.resize(root[prefix+'obs/agentview_rgb'][i], self.image_size)
                images.append(img)
            data_dict['image_primary'] = np.stack(images, axis=0)
        if 'image_wrist' in feats or len(feats) == 0 and 'wrist' in self.camera_names:
            images = []
            for i in range(root[prefix+'obs/eye_in_hand_rgb'].shape[0]):
                img = cv2.resize(root[prefix+'obs/eye_in_hand_rgb'][i], self.image_size)
                images.append(img)
            data_dict['image_wrist'] = np.stack(images, axis=0)
        return data_dict


if __name__=='__main__':
    ds = LiberoHDF5(root='/inspire/hdd/project/robot-action/public/data/libero/', split=['object'], camera_names=['image_primary'], image_size=(256, 256), preload_data=True)
    d = ds[0]
    f = ds.extract_from_episode(0, keyname=['state', 'action'])
    import torch.utils.data as tud
    loader = tud.DataLoader(ds, batch_size=512)
    for batch in loader:
        continue
    print('ok')