import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
import numpy as np
from torch.nn import functional as F

class MLPPolicyConfig(PretrainedConfig):
    """
    Configuration class for MLPPolicy, inheriting from transformers' PretrainedConfig.
    """
    def __init__(
        self,
        # Model architecture parameters
        state_dim=14,
        action_dim=14,
        num_layers=3,
        hidden_dim=256,
        activation="relu",
        dropout=0.0,
        use_camera=False,
        image_dim=None,  # Flattened image dimension (H*W*C)
        
        # Training parameters
        chunk_size=1,  # For consistency with other policies
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Store all arguments as instance attributes
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout
        self.use_camera = use_camera
        self.image_dim = image_dim
        self.chunk_size = chunk_size
        
        # Calculate input dimension
        if self.use_camera and self.image_dim is not None:
            self.input_dim = self.state_dim + self.image_dim
        else:
            self.input_dim = self.state_dim


class MLPPolicy(PreTrainedModel):
    """
    Simple Multi-Layer Perceptron (MLP) policy for imitation learning.
    
    This policy takes state observations as input and outputs actions directly.
    It's designed for environments that primarily use state-based observations.
    """
    config_class = MLPPolicyConfig
    
    def __init__(self, config: MLPPolicyConfig):
        super().__init__(config)
        self.config = config
        
        # Build the MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        layers.append(self._get_activation(config.activation))
        
        # Hidden layers
        for _ in range(config.num_layers - 2):
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(self._get_activation(config.activation))
        
        # Output layer - outputs chunk_size * action_dim
        if config.dropout > 0 and config.num_layers > 1:
            layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(config.hidden_dim, config.chunk_size * config.action_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation_name):
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "gelu": nn.GELU(),
        }
        return activations.get(activation_name.lower(), nn.ReLU())
    
    def _init_weights(self, *args, **kwargs):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state, image=None, action=None, is_pad=None, **kwargs):
        """
        Forward pass of the MLP policy.
        
        Args:
            state: Tensor of shape (batch_size, state_dim) containing state observations
            image: Tensor of shape (batch_size, ...) containing image observations (optional)
            **kwargs: Additional keyword arguments (for compatibility with other policies)
            
        Returns:
            dict: Dictionary containing 'action' key with predicted actions
        """
        # Ensure state is the right shape
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension if missing
        
        # Prepare input
        if self.config.use_camera and image is not None:
            # Flatten image and concatenate with state
            batch_size = state.shape[0]
            image_flat = image.reshape(batch_size, -1)  # Flatten image
            input_tensor = torch.cat([state, image_flat], dim=-1)
        else:
            input_tensor = state
        
        # Forward through MLP
        action_flat = self.mlp(input_tensor)
        # Reshape to (batch_size, chunk_size, action_dim)
        batch_size = action_flat.shape[0]
        pred_action = action_flat.view(batch_size, self.config.chunk_size, self.config.action_dim)
        if action is None:
            # infer Phase
            return {'action':pred_action}
        loss =  F.mse_loss(action, pred_action, reduction='none')
        # all_l1 = F.l1_loss(action, pred_action, reduction='none')
        loss = (loss * ~is_pad.unsqueeze(-1)).mean()
        return {"loss": loss}
    
    def select_action(self, batch_obs):
        """
        Inference action from processed batch observation.
        
        Args:
            batch_obs: Processed and collated batch from meta2obs
            
        Returns:
            numpy array: Predicted actions
        """
        # Get model's device
        device = next(self.parameters()).device
        
        # Move tensors to device
        for k, v in batch_obs.items():
            if isinstance(v, torch.Tensor):
                batch_obs[k] = v.to(device)
        
        # Extract state
        if 'state' not in batch_obs:
            raise ValueError("No 'state' found in batch observation")
        state = batch_obs['state']
        
        # Extract image if using camera
        image = None
        if self.config.use_camera and 'image' in batch_obs:
            image = batch_obs['image']
            # Normalize image if needed
            if image.dtype == torch.uint8 or image.max() > 1.0:
                image = image.float() / 255.0
        
        # Forward pass
        with torch.no_grad():
            result = self.forward(state, image=image)
            action = result['action']
        
        # Convert back to numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        return action


def count_parameters(model):
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage and testing
    config = MLPPolicyConfig(
        state_dim=14,
        action_dim=14,
        num_layers=3,
        hidden_dim=256,
        activation="relu",
        dropout=0.1
    )
    
    model = MLPPolicy(config)
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    batch_size = 4
    dummy_state = torch.randn(batch_size, config.state_dim)
    
    output = model(dummy_state)
    print(f"Input shape: {dummy_state.shape}")
    print(f"Output shape: {output['action'].shape}")
    
    # Test predict_action method
    obs_dict = {'state': dummy_state[0].numpy()}
    action = model.predict_action(obs_dict)
    print(f"Single prediction shape: {action.shape}")
