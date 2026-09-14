from .base import EpisodicDataset
import numpy as np 
import os 

class D4RLDataset(EpisodicDataset):
    def __init__(self, dataset_path_list: list, camera_names: list=[], 
                chunk_size: int = 16, 
                ctrl_space: str = 'ee', ctrl_type: str = 'delta',
                image_size: tuple = (480, 640), preload_data: bool = False):
        super(EpisodicDataset).__init__()
        self.episode_ids = np.arange(len(dataset_path_list))
        self.dataset_path_list = dataset_path_list
        self.chunk_size = chunk_size
        self.camera_names = camera_names
        self.ctrl_space = ctrl_space  # ['ee', 'joint', 'other']
        self.ctrl_type = ctrl_type  # ['abs', 'rel', 'delta']
        self.image_size = image_size
        self.preload_data = preload_data
        self.freq = -1
        self.max_workers = 8
        self.initialize()
    
    def get_dataset_dir(self):
        return self._dataset_dir
    
    def get_raw_lang(self, data_path):
        return ""
    
    def initialize(self):
        import minari
        dataset_dir = os.environ.get("MINARI_DATASETS_PATH")
        dataset_dir = dataset_dir if dataset_dir is not None else os.path.join(os.path.expanduser("~"), ".minari", "datasets")
        os.makedirs(dataset_dir, exist_ok=True)
        datasets = [minari.load_dataset(di, download=True) for di in self.dataset_path_list]
        self._languages = [self.get_raw_lang(di) for di in self.dataset_path_list]
        all_episodes = []
        for dataset in datasets:
            episodes = []
            for episode_data in dataset.iterate_episodes():
                episode_dict = dict(                
                    observations = episode_data.observations,
                    actions = episode_data.actions,
                )
                episodes.append(episode_dict)
            all_episodes.append(episodes)
        self._datasets = all_episodes
        
        self._dataset_dir = os.path.join(dataset_dir, self.dataset_path_list[0])
        self.episode_ids = np.arange(sum([len(ei) for ei in self._datasets]))
        self.dataset_path_list = sum([[f"{i}:{j}" for j in range(len(self._datasets[i]))] for i in range(len(self._datasets))], [])
        self.episode_len = sum([[episode['actions'].shape[0] for episode in episodes] for episodes in self._datasets], [])
        self.cumulative_len = np.cumsum(self.episode_len)  # Compute cumulative lengths
        self.max_episode_len = max(self.episode_len)  # Get maximum episode length
      
    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """
        Load one-step data at start_ts from the episode specified by dataset_path.
        
        Args:
            dataset_path: Path to the dataset file
            start_ts: Starting timestep
            
        Returns:
            Dictionary containing the loaded data
        """
        dataset_idx, ep = dataset_path.split(':')
        dataset = self._datasets[eval(dataset_idx)]
        data = dataset[eval(ep)]
        # Load language 
        raw_lang = self._languages[eval(dataset_idx)]
        # Load state
        state = data['observations'][start_ts]
        # Load action
        action = data['actions'][start_ts:min(start_ts+self.chunk_size, data['actions'].shape[0])]
        # Load image
        image_dict = dict(
        )
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
        dataset_idx, ep = dataset_path.split(':')
        dataset = self._datasets[eval(dataset_idx)]
        data = dataset[eval(ep)]
        data_dict = {}
        if isinstance(feats, str): 
            feats = [feats]
        if 'language_instruction' in feats or len(feats) == 0: 
            data_dict['language_instruction'] = self._languages[dataset_idx]
        if 'state' in feats or len(feats) == 0: 
            data_dict['state'] = data['observations']
        if 'action' in feats or len(feats) == 0:  # Load action
            data_dict['action'] = data['actions']
        return data_dict
