"""
Flow Matching Policy Implementation

A simple and efficient implementation of Flow Matching for robot policy learning.
Based on Conditional Flow Matching (Lipman et al., 2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Dict, Any
import numpy as np

# Try to import torchdiffeq for ODE solving
try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("Warning: torchdiffeq not available, will use Euler method for sampling")


# ============================================================================
# Configuration
# ============================================================================

class FlowMatchingConfig(PretrainedConfig):
    """Configuration for Flow Matching Policy."""
    
    model_type = "flow_matching_policy"
    
    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 7,
        time_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        learning_rate: float = 1e-4,
        # Chunk settings (for action chunking)
        chunk_size: int = 1,
        # Sampling settings
        num_sampling_steps: int = 100,
        use_ode_solver: bool = True,
        ode_atol: float = 1e-5,
        ode_rtol: float = 1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.chunk_size = chunk_size
        self.num_sampling_steps = num_sampling_steps
        self.use_ode_solver = use_ode_solver and HAS_TORCHDIFFEQ
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol


# ============================================================================
# Time Embedding
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for encoding time steps."""
    
    def __init__(self, dim: int, max_time: float = 1.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension {dim} must be even.")
        self.dim = dim
        self.max_time = max_time
        
        # Pre-compute frequency terms
        half_dim = dim // 2
        div_term = torch.exp(
            torch.arange(half_dim, dtype=torch.float32) *
            -(torch.log(torch.tensor(10000.0)) / (half_dim - 1.0))
        )
        self.register_buffer('div_term', div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor of shape [B] or [B, 1]
        
        Returns:
            Time embedding of shape [B, dim]
        """
        t = t.float().view(-1, 1) / self.max_time
        pe = torch.zeros(t.shape[0], self.dim, device=t.device, dtype=t.dtype)
        pe[:, 0::2] = torch.sin(t * self.div_term)
        pe[:, 1::2] = torch.cos(t * self.div_term)
        return pe


# ============================================================================
# Velocity Model
# ============================================================================

class VelocityModel(nn.Module):
    """
    Neural network that predicts velocity field v_theta(a_t, t, s).
    
    This is the core network that learns to transform noise to actions
    conditioned on the state.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_dim = time_dim
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        
        # Input: concatenate (action_t, time_embedding, state)
        input_dim = action_dim + time_dim + state_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SiLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, a_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity at time t.
        
        Args:
            a_t: Action at time t, shape [B, action_dim]
            t: Time step(s), shape [B] or scalar
            s: State condition, shape [B, state_dim]
        
        Returns:
            Predicted velocity, shape [B, action_dim]
        """
        # Handle scalar time
        if t.dim() == 0:
            t = t.expand(a_t.shape[0])
        
        # Get time embedding
        t_emb = self.time_embed(t)
        
        # Concatenate all inputs
        x = torch.cat([a_t, t_emb, s], dim=-1)
        
        # Predict velocity
        v = self.net(x)
        
        return v


# ============================================================================
# Flow Matching Policy
# ============================================================================

class FlowMatchingPolicy(PreTrainedModel):
    """
    Flow Matching Policy for Imitation Learning.
    
    Uses Conditional Flow Matching to learn a policy that maps states to actions
    by learning to transform Gaussian noise to expert actions.
    """
    
    config_class = FlowMatchingConfig
    
    def __init__(self, config: FlowMatchingConfig):
        super().__init__(config)
        
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.chunk_size = config.chunk_size
        
        # Initialize velocity model
        self.velocity_model = VelocityModel(
            state_dim=config.state_dim,
            action_dim=config.action_dim * config.chunk_size,  # Support action chunking
            time_dim=config.time_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        )
        
        # Initialize optimizer (can be overridden by Trainer)
        self.optimizer = None
    
    def forward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.
        
        Args:
            state: State tensor, shape [B, state_dim]
            action: Ground truth action (for training), shape [B, action_dim] or [B, chunk_size, action_dim]
            
        Returns:
            Dictionary containing:
                - loss: Training loss (if action is provided)
                - action: Sampled action (if action is not provided)
        """
        if action is not None:
            # Training mode
            if action.dim() == 3:  # [B, chunk_size, action_dim]
                action = action.reshape(action.shape[0], -1)  # [B, chunk_size * action_dim]
            
            loss = self._compute_flow_matching_loss(state, action)
            return {'loss': loss}
        else:
            # Inference mode
            sampled_action = self.select_action(state)
            return {'action': sampled_action}
    
    def _compute_flow_matching_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Flow Matching loss.
        
        The loss is: E_{t, z} ||v_theta(a_t, t, s) - (a_1 - a_0)||^2
        where a_t = t * a_1 + (1-t) * a_0, a_0 ~ N(0, I), a_1 = action
        
        Args:
            state: State tensor, shape [B, state_dim]
            action: Target action, shape [B, action_dim * chunk_size]
        
        Returns:
            Scalar loss
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Sample time uniformly from [epsilon, 1] for numerical stability
        epsilon = 1e-5
        t = torch.rand(batch_size, device=device) * (1.0 - epsilon) + epsilon
        
        # Sample noise (initial state a_0)
        noise = torch.randn_like(action)
        
        # Interpolate: a_t = t * action + (1-t) * noise
        t_expanded = t.view(-1, 1)
        a_t = t_expanded * action + (1.0 - t_expanded) * noise
        
        # Target velocity is the difference: u_t = action - noise
        u_t = action - noise
        
        # Predict velocity
        v_pred = self.velocity_model(a_t, t, state)
        
        # Compute MSE loss
        loss = F.mse_loss(v_pred, u_t)
        
        return loss
    
    @torch.no_grad()
    def select_action(self, batch_obs) -> np.ndarray:
        """
        Sample action using the learned flow.
        
        Args:
            batch_obs: Processed and collated batch from meta2obs
        
        Returns:
            Sampled action as numpy array, shape [B, chunk_size, action_dim]
        """
        self.eval()
        
        # Extract state and move to device
        state = batch_obs['state']
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = state.to('cuda')
        
        # Sample action using flow
        if self.config.use_ode_solver and HAS_TORCHDIFFEQ:
            action = self._sample_ode(state)
        else:
            action = self._sample_euler(state)
        
        self.train()
        
        # Reshape to [B, chunk_size, action_dim]
        batch_size = state.shape[0]
        action = action.reshape(batch_size, self.chunk_size, self.action_dim)
        
        return action.cpu().numpy()
    
    def _sample_ode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Sample action using ODE solver (more accurate).
        
        Args:
            state: State tensor, shape [B, state_dim]
        
        Returns:
            Sampled action, shape [B, action_dim * chunk_size]
        """
        device = state.device
        batch_size = state.shape[0]
        action_dim_total = self.action_dim * self.chunk_size
        
        # Define ODE function
        class ODEFunc(nn.Module):
            def __init__(self, velocity_model, state):
                super().__init__()
                self.velocity_model = velocity_model
                self.state = state
            
            def forward(self, t, a_t):
                return self.velocity_model(a_t, t, self.state)
        
        ode_func = ODEFunc(self.velocity_model, state)
        
        # Initial condition: sample from N(0, I)
        a_0 = torch.randn(batch_size, action_dim_total, device=device)
        
        # Time span [0, 1]
        t_span = torch.tensor([0.0, 1.0], device=device)
        
        # Solve ODE
        solution = odeint(
            ode_func,
            a_0,
            t_span,
            method='dopri5',
            atol=self.config.ode_atol,
            rtol=self.config.ode_rtol
        )
        
        # Return solution at t=1
        a_1 = solution[-1]
        
        return a_1
    
    def _sample_euler(
        self,
        state: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample action using Euler method (faster but less accurate).
        
        Args:
            state: State tensor, shape [B, state_dim]
            num_steps: Number of integration steps (uses config value if None)
        
        Returns:
            Sampled action, shape [B, action_dim * chunk_size]
        """
        if num_steps is None:
            num_steps = self.config.num_sampling_steps
        
        device = state.device
        batch_size = state.shape[0]
        action_dim_total = self.action_dim * self.chunk_size
        
        dt = 1.0 / num_steps
        
        # Start from noise
        a_t = torch.randn(batch_size, action_dim_total, device=device)
        
        # Euler integration
        for i in range(num_steps):
            t = torch.tensor(i * dt, device=device)
            v = self.velocity_model(a_t, t, state)
            a_t = a_t + v * dt
        
        return a_t
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Execute one training step.
        
        Args:
            batch: Dictionary containing 'state' and 'action'
        
        Returns:
            Loss value
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")
        
        state = batch['state']
        action = batch['action']
        
        # Forward pass
        output = self.forward(state=state, action=action)
        loss = output['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def set_optimizer(self, lr: Optional[float] = None):
        """Set up optimizer."""
        if lr is None:
            lr = self.config.learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
