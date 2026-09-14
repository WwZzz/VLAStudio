from vlastudio.data_utils.datasets.lerobotv21_wrapper import WrappedLerobotV21Dataset
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np

ALL_EMBODIMENTS = [
    "franka_3rgb",
    "franka_1rgb",
    "agilex_3rgb", 
    "tienkung_gello_1rgb", 
    "tienkung_xsens_1rgb", 
    "ur_1rgb",
    "franka_fr3_dual", 
    "tienkung_prod1_gello_1rgb",
]

EE_EFFECTIVE_DIMS = {
    'franka_3rgb': [0,1,2,3,4,5,13],
    'franka_1rgb': [0,1,2,3,4,5,13],
    'agilex_3rgb': [0,1,2,3,4,5,6,13,14,15,16,17,18,19,20,27],
    'ur_1rgb': [0,1,2,3,4,5,12],
    'franka_fr3_dual': [0,1,2,3,4,5,19, 6,7,8,9,10,11,25],
}

class RoboMindDataset(WrappedLerobotV21Dataset):
    def __init__(
        self,
        dataset_path_list: List[str],
        camera_names: List[str] = [],
        root: Optional[str] = None,
        chunk_size: int = 16,
        ctrl_space: str = 'ee',
        ctrl_type: str = 'delta',
        image_size: Optional[Tuple[int, int]] = None,
        tolerance_s: float = 0.1,
        state_key: Union[str, List[str]] = 'observation.state',
        action_key: Union[str, List[str]] = 'action',
        episode_filter: Optional[dict] = None,
        download_videos: bool = True,
        filter_invalid_videos: bool = False,
        video_backend: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.robot = None
        for robot_name in ALL_EMBODIMENTS:
            if robot_name in dataset_path_list[0]:
                self.robot = robot_name
                break
        assert self.robot is not None, f"Unknown robot type in dataset path: {dataset_path_list[0]}"
        assert all(self.robot in d for d in dataset_path_list), f"Mixed robot types in dataset paths: {dataset_path_list} are not supported"
        state_key, action_key, self.is_dual = self.get_dataset_info(self.robot, ctrl_space=ctrl_space)
        self.effective_dim = EE_EFFECTIVE_DIMS.get(self.robot, None) if ctrl_space=='ee' else None
        super().__init__(
            dataset_path_list=dataset_path_list, 
            camera_names=camera_names,
            root=root,
            chunk_size=chunk_size,
            ctrl_space=ctrl_space,
            ctrl_type=ctrl_type,
            image_size=image_size,
            tolerance_s=tolerance_s,
            state_key=state_key,
            action_key=action_key,
            episode_filter=episode_filter,
            download_videos=download_videos,
            filter_invalid_videos=filter_invalid_videos,
            video_backend=video_backend,
            *args, 
            **kwargs,
        )
        
        # Update state_dim and action_dim after dimension reduction
        if self.effective_dim is not None:
            self.state_dim = len(self.effective_dim)
            self.action_dim = len(self.effective_dim)

    
    def reduce_dim(self, x, eff_idxs):
        if self.ctrl_space=='joint': return x
        if len(x.shape)==1:
            return x[eff_idxs]
        else:
            return x[:,eff_idxs]
    
    def get_dataset_info(self, robot: str, ctrl_space: str = 'ee'):
        """Get state/action keys for each robot type.
        
        Design principle: state[t] = action[t], i.e. both state and action
        use the same observation keys so that action represents the absolute
        state at the current timestep.
        """
        if robot=='franka_3rgb' or robot=='franka_1rgb':
            is_dual=False
            if ctrl_space=='ee': # dim 7 (after effective_dim reduction from 14)
                state_key = ['observation.states.end_effector', 'observation.states.joint_position']
                action_key = state_key  # state[t] = action[t]
            elif ctrl_space=='joint':  # dim 8
                state_key = 'observation.states.joint_position'
                action_key = state_key  # state[t] = action[t]
        elif robot=='agilex_3rgb':
            is_dual = True
            if ctrl_space=='ee': # dim 16 (after effective_dim reduction from 28)
                state_key = ['observation.states.end_effector_left', 'observation.states.joint_effort_left', 'observation.states.end_effector_right',  'observation.states.joint_effort_right',]
                action_key = state_key  # state[t] = action[t]
            elif ctrl_space=='joint': # dim 14
                state_key = ['observation.states.joint_position_left', 'observation.states.joint_position_right']
                action_key = state_key  # state[t] = action[t]
        elif robot=='tienkung_gello_1rgb' or robot=='tienkung_prod1_gello_1rgb': 
            is_dual = True 
            if ctrl_space=='ee':
                raise ValueError("EE control not supported for tienkung_gello_1rgb / tienkung_prod1_gello_1rgb")
            elif ctrl_space=='joint': # dim 16
                state_key = 'observation.states.joint_position'
                action_key = state_key  # state[t] = action[t]
        elif robot=='tienkung_xsens_1rgb': 
            is_dual = True
            if ctrl_space=='ee': # dim 12
                state_key = 'observation.states.end_effector'
                action_key = state_key  # state[t] = action[t]
            elif ctrl_space=='joint': # dim 14
                state_key = 'observation.states.joint_position'
                action_key = state_key  # state[t] = action[t]
        elif robot=='ur_1rgb':
            is_dual = False
            if ctrl_space=='ee': # dim 7 (after effective_dim reduction from 13)
                state_key = ['observation.states.end_effector', 'observation.states.joint_position']
                action_key = state_key  # state[t] = action[t]
            elif ctrl_space=='joint': # dim 7
                state_key = 'observation.states.joint_position'
                action_key = state_key  # state[t] = action[t]
        elif robot=='franka_fr3_dual':
            is_dual = True
            if ctrl_space=='ee': # dim 14 (after effective_dim reduction from 28)
                state_key = ['observation.states.end_effector', 'observation.states.joint_position']
                action_key = state_key  # state[t] = action[t]
            elif ctrl_space=='joint': # dim 16
                state_key = 'observation.states.joint_position'
                action_key = state_key  # state[t] = action[t]
        return state_key, action_key, is_dual

    def __getitem__(self, index):
        res = super().__getitem__(index)
        if self.effective_dim is not None:
            res['action'] = self.reduce_dim(res['action'], self.effective_dim)
            res['state'] = self.reduce_dim(res['state'], self.effective_dim)

        # For franka_3rgb: randomly select one camera from the 3 views each time.
        # This reduces multi-view to single-view while keeping config simple.
        if self.robot == 'franka_3rgb' and 'image' in res and res['image'] is not None:
            img = res['image']
            if hasattr(img, 'ndim') and img.ndim == 4 and img.shape[0] == 3:
                # Deterministic-ish randomness per sample & worker for reproducibility.
                try:
                    from torch.utils.data import get_worker_info
                    wi = get_worker_info()
                    worker_id = wi.id if wi is not None else 0
                except Exception:
                    worker_id = 0

                seed = (int(index) * 1000003 + worker_id * 101) & 0xFFFFFFFF
                rng = np.random.RandomState(seed)
                k = int(rng.randint(0, 3))
                res['image'] = img[k:k+1]
        return res
    
    def _reduce_stats(self, stats: Dict[str, np.ndarray], eff_idxs: List[int]) -> Dict[str, np.ndarray]:
        """Apply dimension reduction to statistics.
        
        Args:
            stats: Dictionary containing 'mean', 'std', 'min', 'max', 'q01', 'q99' arrays
            eff_idxs: Indices of effective dimensions to keep
            
        Returns:
            Dictionary with reduced statistics
        """
        if not stats:
            return stats
        
        reduced = {}
        for key, value in stats.items():
            if isinstance(value, np.ndarray):
                # Handle both 1D (state) and potentially 2D arrays
                if value.ndim == 1:
                    reduced[key] = value[eff_idxs]
                else:
                    # For multi-dimensional, reduce along the last axis
                    reduced[key] = value[..., eff_idxs]
            else:
                reduced[key] = value
        return reduced
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics with dimension reduction applied.
        
        Overrides parent method to apply the same dimension reduction
        that is applied to state and action in __getitem__.
        """
        # Get original statistics from parent
        stats = super().get_dataset_statistics()
        
        # Apply dimension reduction if effective_dim is set
        if self.effective_dim is not None:
            if 'state' in stats and stats['state']:
                stats['state'] = self._reduce_stats(stats['state'], self.effective_dim)
            if 'action' in stats and stats['action']:
                stats['action'] = self._reduce_stats(stats['action'], self.effective_dim)
        
        return stats
    
    def extract_from_episode(self, episode_idx: int, keyname: List[str] = []) -> Dict[str, np.ndarray]:
        """Extract specific features from an episode with dimension reduction applied."""
        result = super().extract_from_episode(episode_idx, keyname)
        
        # Apply dimension reduction if effective_dim is set
        if self.effective_dim is not None:
            if 'state' in result:
                result['state'] = self.reduce_dim(result['state'], self.effective_dim)
            if 'action' in result:
                result['action'] = self.reduce_dim(result['action'], self.effective_dim)
        
        return result
