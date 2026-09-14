"""
Data utilities for Flow Matching Policy.

This module contains the data processing pipeline for Flow Matching Policy.
"""

import torch
import numpy as np
from typing import Dict, List, Any


class FlowMatchingDataProcessor:
    """
    Data processor for Flow Matching Policy.
    
    Processes individual samples from dataset __getitem__ or __iter__ to
    the format required by the Flow Matching model.
    """
    
    def __init__(self, model_config):
        """
        Initialize the data processor.
        
        Args:
            model_config: Model configuration containing state_dim, action_dim, chunk_size
        """
        self.state_dim = model_config.state_dim
        self.action_dim = model_config.action_dim
        self.chunk_size = model_config.chunk_size
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sample from the dataset.
        
        Expected input sample format:
            - state: numpy array or torch tensor [state_dim] or [seq_len, state_dim]
            - action: numpy array or torch tensor [action_dim] or [chunk_size, action_dim]
            - (optional) image: for future multimodal support
            - (optional) raw_lang: for future language conditioning
        
        Returns:
            Processed sample dict with tensors
        """
        processed = {}
        
        # Process state
        state = sample.get('state', sample.get('observation', None))
        if state is not None:
            state = self._process_state(state)
            processed['state'] = state
        
        # Process action
        action = sample.get('action', None)
        if action is not None:
            action = self._process_action(action)
            processed['action'] = action
        
        # Copy is_pad if exists
        if 'is_pad' in sample:
            processed['is_pad'] = sample['is_pad']
        
        return processed
    
    def _process_state(self, state):
        """Process state to tensor format."""
        # Convert to tensor
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Handle different state shapes
        if state.dim() == 1:
            # Single state [state_dim]
            return state
        elif state.dim() == 2:
            # Sequence of states [seq_len, state_dim], use the last one
            return state[-1]
        else:
            return state
    
    def _process_action(self, action):
        """Process action to tensor format."""
        # Convert to tensor
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        
        # Handle different action shapes
        if action.dim() == 1:
            # Single action [action_dim]
            # Reshape to [1, action_dim] for chunk_size=1
            if self.chunk_size == 1:
                action = action.unsqueeze(0)
        elif action.dim() == 2:
            # Action sequence [chunk_size, action_dim]
            pass
        
        return action


def flow_matching_collator(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Data collator for Flow Matching Policy.
    
    Collates multiple processed samples into a batch.
    
    Args:
        batch: List of processed sample dicts
        
    Returns:
        Batch dict with tensors
    """
    if not batch:
        return {}
    
    batched = {}
    
    # Collect states
    if 'state' in batch[0]:
        states = [s['state'] for s in batch]
        batched['state'] = torch.stack(states, dim=0)
    
    # Collect actions
    if 'action' in batch[0]:
        actions = [s['action'] for s in batch]
        batched['action'] = torch.stack(actions, dim=0)
    
    # Collect is_pad if exists
    if 'is_pad' in batch[0]:
        is_pads = [s['is_pad'] for s in batch]
        if isinstance(is_pads[0], (int, bool)):
            batched['is_pad'] = torch.tensor(is_pads, dtype=torch.bool)
        else:
            batched['is_pad'] = torch.stack([torch.tensor(p, dtype=torch.bool) for p in is_pads])
    
    return batched

