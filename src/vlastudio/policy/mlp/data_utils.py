import torch
import numpy as np
from typing import Dict, List, Any


def data_collator(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Data collator for MLP policy.
    
    MLP policy only uses state and image modalities, so text modalities are ignored.
    
    Args:
        batch: List of dictionaries containing individual samples
        
    Returns:
        Dictionary with batched tensors (only state, action, image)
    """
    if not batch:
        return {}
    
    # Text modalities that should be ignored for MLP
    text_modalities = [
        'raw_lang', 'lang', 'language', 'text', 'instruction', 'task_description',
        'task', 'episode_id', 'trajectory_id', 'dataset_name'
    ]
    
    # Only process relevant keys for MLP
    relevant_keys = ['state', 'action', 'image', 'is_pad']
    batched = {}
    
    for key in relevant_keys:
        if key not in batch[0]:
            continue  # Skip if key doesn't exist in the batch
            
        # Collect all values for this key
        values = [sample[key] for sample in batch if key in sample]
        if not values:  # Skip if no values found
            continue
            
        # Handle different data types
        if key in ['state', 'action', 'is_pad']:
            # Convert to tensors and stack
            if isinstance(values[0], np.ndarray):
                values = [torch.FloatTensor(v) for v in values]
            elif isinstance(values[0], (int, float)):
                values = [torch.FloatTensor([v]) for v in values]
            else:
                values = values
            # Stack all tensors
            batched[key] = torch.stack(values)
            
        elif key == 'image':
            # Handle image data for camera modality
            if isinstance(values[0], np.ndarray):
                values = [torch.FloatTensor(v) for v in values]
                batched[key] = torch.stack(values)
            # Skip non-numpy image data
    
    return batched


class MLPDataProcessor:
    """
    Data processor for MLP policy that handles state-based observations.
    """
    def __init__(self, state_dim=None, use_camera=False):
        self.state_dim = state_dim
        self.use_camera = use_camera
        
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sample.
        
        Args:
            sample: Dictionary containing sample data
            
        Returns:
            Processed sample dictionary (only state, action, image)
        """
        # Text modalities that should be ignored for MLP
        text_modalities = [
            'raw_lang', 'lang', 'language', 'text', 'instruction', 'task_description',
            'task', 'episode_id', 'trajectory_id', 'dataset_name'
        ]
        
        # Start with empty processed sample
        processed_sample = {}
        # Ensure state is in the right format
        if 'state' not in sample:
            raise ValueError("Sample must contain 'state' key")
            
        state = sample['state']
        
        # Convert to numpy if needed
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        elif isinstance(state, list):
            state = np.array(state)
        
        # Ensure float32 dtype
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32)
        
        # Optionally check dimension
        if self.state_dim is not None and len(state) != self.state_dim:
            print(f"Warning: Expected state_dim={self.state_dim}, got {len(state)}")
        
        processed_sample['state'] = state
        
        # Handle action data
        if 'action' in sample:
            action = sample['action']
            
            # Convert to numpy if needed
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            elif isinstance(action, list):
                action = np.array(action)
            
            # Ensure float32 dtype
            if isinstance(action, np.ndarray):
                action = action.astype(np.float32)
            
            processed_sample['action'] = action
        
        # Handle image data if using camera
        if self.use_camera and 'image' in sample:
            image = sample['image']
            
            # Convert to numpy if needed
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            elif isinstance(image, list):
                image = np.array(image)
            
            # Ensure float32 dtype
            if isinstance(image, np.ndarray):
                image = image.astype(np.float32)
            
            processed_sample['image'] = image
        
        # Explicitly ignore text modalities - they are not added to processed_sample
        # This ensures MLP only gets the data it needs
        processed_sample['is_pad'] = sample['is_pad']
        return processed_sample