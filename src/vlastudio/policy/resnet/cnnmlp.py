from transformers import PreTrainedModel
import torch.nn as nn
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from .detr_vae import build_cnnmlp
from transformers import PretrainedConfig

class CNNMLPPolicyConfig(PretrainedConfig):
    """
    Configuration class for CNNMLPPolicy, inheriting from transformers' PretrainedConfig.
    This class includes all arguments defined in the provided argparse parser.
    """
    def __init__(
        self,
        # Training-related hyperparameters
        lr_backbone=1e-5,
        # Model parameters
        backbone="resnet18",
        dilation=False,
        position_embedding="sine",
        camera_names=['primary'],  # Expected to be a list
        chunk_size=8,
        masks=False,
        hidden_dim=512,
        # Policy-specific arguments
        state_dim=14,
        action_dim=14,
        **kwargs,  # Allow for future extensions
    ):
        super().__init__(**kwargs)
        
        # Store all arguments as instance attributes
        self.lr_backbone = lr_backbone
        # Backbone settings
        self.hidden_dim = hidden_dim
        self.backbone = backbone
        self.dilation = dilation
        self.position_embedding = position_embedding
        self.camera_names = camera_names if camera_names is not None else []
        self.chunk_size = chunk_size
        self.masks = masks
        # Task-related arguments
        self.state_dim = state_dim
        self.action_dim = action_dim
    

class CNNMLPPolicy(PreTrainedModel):
    config_class = CNNMLPPolicyConfig  # Configuration class

    def __init__(self, config):
        super().__init__(config)
        # Build model and optimizer
        self.model = build_cnnmlp(config)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        Forward method for training and inference. Trainer calls this method automatically.
        
        Args:
            qpos: Tensor, shape (batch_size, state_dim), robot state.
            image: Tensor, shape (batch_size, C, H, W), normalized visual inputs.
            actions: Tensor, shape (batch_size, num_queries, action_dim), action sequences for training.
            is_pad: Tensor, shape (batch_size, num_queries), padding mask.
        Returns:
            Loss dictionary during training; sampled actions during inference.
        """
        env_state = None
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = self.normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.chunk_size]
            is_pad = is_pad[:, :self.model.chunk_size]
            a_hat = self.model(qpos, image, env_state, actions)
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict = {'l1': l1, 'loss': l1}
            return loss_dict
        else: # inference time
            a_hat  = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def select_action(self, batch_obs):
        """
        Inference action from processed batch observation.
        
        Args:
            batch_obs: Collated batch from meta2obs (via data_collator or default_collate)
        
        Returns:
            Action predictions
        """
        # Move tensors to device and get data
        device = next(self.parameters()).device
        image = batch_obs['image'].to(device)
        state = batch_obs['qpos'].to(device)
        # Normalize image (ResNet-specific normalization)
        image = self.normalize(image)
        # Forward pass
        a_hat = self.model(state, image, None)
        return a_hat
