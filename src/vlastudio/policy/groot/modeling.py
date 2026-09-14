"""
GR00T Model Adapter for ILStudio.

Wraps the GR00T model from third_party/lerobot for use with ILStudio's
training and inference pipeline.

Key Points:
  - Uses NVIDIA's GR00T-N1.5 model via lerobot integration
  - Eagle2 vision-language backbone
  - Flow matching diffusion action head
  - Supports embodiment tags for multi-robot training

Reference:
  - GR00T: https://huggingface.co/nvidia/GR00T-N1.5-3B
  - Isaac-GR00T: NVIDIA's robotics foundation model
"""

import sys
import os
from pathlib import Path

# Add third_party/lerobot to path for imports
LEROBOT_PATH = Path(__file__).resolve().parents[2] / "third_party" / "lerobot" / "src"
if str(LEROBOT_PATH) not in sys.path:
    sys.path.insert(0, str(LEROBOT_PATH))

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List
from collections import deque
from dataclasses import dataclass, field
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from loguru import logger


# =============================================================================
# Model Configuration
# =============================================================================

class GrootOFTConfig(PretrainedConfig):
    """
    Configuration for GR00T-OFT model adapter.
    
    Args:
        base_model_path: Path to pretrained GR00T model
        action_dim: Dimension of action space
        state_dim: Dimension of state space
        chunk_size: Action prediction horizon
        n_action_steps: Number of action steps to execute
        max_state_dim: Maximum state dimension (for padding)
        max_action_dim: Maximum action dimension (for padding)
        embodiment_tag: Tag for embodiment (e.g., 'new_embodiment')
        tune_llm: Whether to fine-tune LLM backbone
        tune_visual: Whether to fine-tune vision tower
        tune_projector: Whether to fine-tune projector
        tune_diffusion_model: Whether to fine-tune diffusion head
        use_bf16: Whether to use bfloat16
    """
    model_type = "groot_oft"

    def __init__(
        self,
        base_model_path: str = "nvidia/GR00T-N1.5-3B",
        action_dim: int = 7,
        state_dim: int = 7,
        chunk_size: int = 16,
        n_action_steps: int = 16,
        max_state_dim: int = 64,
        max_action_dim: int = 32,
        embodiment_tag: str = "new_embodiment",
        tokenizer_assets_repo: str = "lerobot/eagle2hg-processor-groot-n1p5",
        tune_llm: bool = False,
        tune_visual: bool = False,
        tune_projector: bool = True,
        tune_diffusion_model: bool = True,
        use_bf16: bool = True,
        image_size: tuple = (224, 224),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_path = base_model_path
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.embodiment_tag = embodiment_tag
        self.tokenizer_assets_repo = tokenizer_assets_repo
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.use_bf16 = use_bf16
        self.image_size = image_size


# =============================================================================
# Main Model
# =============================================================================

class GrootOFTForPolicy(PreTrainedModel):
    """
    GR00T-OFT model adapter for ILStudio.
    
    Wraps the GR00T model from lerobot for continuous action prediction.
    """
    config_class = GrootOFTConfig

    def __init__(self, config: GrootOFTConfig):
        super().__init__(config)
        self.config = config
        
        # Handle Flash Attention compatibility
        self._handle_flash_attention_compatibility()
        
        # Import GR00T components
        from lerobot.policies.groot.groot_n1 import GR00TN15
        
        # Load GR00T model
        logger.info(f"Loading GR00T from: {config.base_model_path}")
        self._groot_model = GR00TN15.from_pretrained(
            pretrained_model_name_or_path=config.base_model_path,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
        )
        
        # Set compute dtype
        if config.use_bf16:
            self._groot_model.compute_dtype = "bfloat16"
            self._groot_model.config.compute_dtype = "bfloat16"
        
        # Action queue for inference
        self._action_queue = deque([], maxlen=config.n_action_steps)
        
        # Processor (set later)
        self.processor = None
        
        logger.info(f"GR00T initialized with action_dim={config.action_dim}, chunk_size={config.chunk_size}")

    def _handle_flash_attention_compatibility(self) -> None:
        """Handle Flash Attention compatibility issues."""
        os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "0")
        
        try:
            import flash_attn
            logger.info(f"Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            logger.warning(f"Flash Attention not available: {e}")
        except Exception as e:
            if "undefined symbol" in str(e):
                logger.warning(f"Flash Attention compatibility issue: {e}")
            else:
                logger.warning(f"Flash Attention error: {e}")

    def set_processor(self, processor):
        """Set the data processor."""
        self.processor = processor

    def reset(self):
        """Reset action queue for new episode."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_input_embeddings(self):
        """Get input embeddings from backbone."""
        return self._groot_model.backbone.eagle_model.get_input_embeddings()

    def set_requires_grad(self, training_args):
        """Set trainable parameters."""
        # GR00T handles this internally via tune_* flags
        pass

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            batch: Dict containing:
                - eagle_* keys: Preprocessed Eagle inputs
                - state: Robot state
                - action: Target actions
                - state_mask, action_mask: Masks
                - embodiment_id: Embodiment identifier
        
        Returns:
            Dict with 'loss' and 'action_loss' keys.
        """
        # Filter batch for GR00T expected keys
        allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_")) and not k.startswith("next.")
        }
        
        device = next(self.parameters()).device
        
        # Forward with bf16 autocast
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(groot_inputs)
        
        loss = outputs.get("loss")
        
        return {
            'loss': loss,
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict action chunk for inference.
        
        Returns:
            Tensor of shape (B, n_action_steps, action_dim)
        """
        self.eval()
        
        # Filter batch - don't include action during inference
        allowed_base = {"state", "state_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_")) and not k.startswith("next.")
        }
        
        device = next(self.parameters()).device
        
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.get_action(groot_inputs)
        
        actions = outputs.get("action_pred")
        
        # Trim to original action dimension
        actions = actions[:, :, :self.config.action_dim]
        
        return actions

    @torch.no_grad()
    def generate(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, np.ndarray]:
        """Generate actions for inference."""
        actions = self.predict_action_chunk(batch)
        return {'pred_actions': actions.detach().cpu().numpy()}

    def select_action(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        Select action from observation batch.
        
        Uses action queue to return actions step by step.
        """
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        
        actions = self.predict_action_chunk(batch)
        return actions.cpu().numpy()
