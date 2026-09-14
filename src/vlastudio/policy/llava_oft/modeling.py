"""
LLaVA-OFT Model Implementation for ILStudio.

A lightweight implementation using LLaVA-OneVision backbone with action prediction
via L1 regression on the last token hidden states.

Key Points:
  - LLaVA-OneVision vision-language backbone
  - Extracts last token hidden states for action prediction
  - Continuous action prediction via L1 regression

Reference:
  - LLaVA-OneVision: https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf
  - UniAct implementation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Any
from PIL import Image
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

from loguru import logger


# =============================================================================
# MLP ResNet Action Head (L1 Regression)
# =============================================================================

class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([
            MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)
        ])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 3072,
        action_dim: int = 7,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=num_blocks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions_hidden_states: (B, chunk_len, hidden_dim)
        Returns:
            actions: (B, chunk_len, action_dim)
        """
        batch_size, chunk_len, hidden_dim = actions_hidden_states.shape
        x = actions_hidden_states.reshape(batch_size * chunk_len, hidden_dim)
        x = self.model(x)
        actions = x.view(batch_size, chunk_len, self.action_dim)
        return actions

    def forward(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.predict_action(actions_hidden_states)


# =============================================================================
# Model Configuration
# =============================================================================

class LlavaOFTConfig(PretrainedConfig):
    """
    Configuration for LLaVA-OFT model.
    
    Args:
        vlm_model_name_or_path: Path to LLaVA model
        action_dim: Dimension of action space
        state_dim: Dimension of state space
        chunk_size: Action prediction horizon (chunk length)
        action_hidden_dim: Hidden dimension for action head (auto-set from VLM)
        action_head_hidden_mult: Multiplier for action head hidden dimension
        action_head_num_blocks: Number of MLP ResNet blocks
        action_token: Special token for action prediction
    """
    model_type = "llava_oft"

    def __init__(
        self,
        vlm_model_name_or_path: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        action_dim: int = 7,
        state_dim: int = 7,
        chunk_size: int = 16,
        action_hidden_dim: int = None,  # Will be set from VLM hidden size
        action_head_hidden_mult: int = 2,
        action_head_num_blocks: int = 2,
        action_token: str = "🔍",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vlm_model_name_or_path = vlm_model_name_or_path
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.action_hidden_dim = action_hidden_dim
        self.action_head_hidden_mult = action_head_hidden_mult
        self.action_head_num_blocks = action_head_num_blocks
        self.action_token = action_token


# =============================================================================
# Main Model
# =============================================================================

class LlavaOFTForPolicy(PreTrainedModel):
    """
    LLaVA-OFT model for continuous action prediction.
    
    Uses LLaVA-OneVision as backbone and predicts actions via L1 regression
    on the last token hidden states.
    """
    config_class = LlavaOFTConfig

    def __init__(self, config: LlavaOFTConfig):
        super().__init__(config)
        self.config = config
        
        # Check flash attention availability
        use_flash_attn = self._check_flash_attn()
        
        # Load LLaVA-OneVision model
        logger.info(f"Loading LLaVA from: {config.vlm_model_name_or_path}")
        
        load_kwargs = {
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if use_flash_attn:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        else:
            logger.info("Using default attention (flash_attn not available)")
        
        self.vlm = LlavaOnevisionForConditionalGeneration.from_pretrained(
            config.vlm_model_name_or_path,
            **load_kwargs
        )
        
        # Get hidden size from VLM
        vlm_hidden_size = self.vlm.language_model.config.hidden_size
        
        if config.action_hidden_dim is None:
            config.action_hidden_dim = vlm_hidden_size
        
        # Build action head
        self.action_head = L1RegressionActionHead(
            input_dim=config.action_hidden_dim,
            hidden_dim=config.action_hidden_dim * config.action_head_hidden_mult,
            action_dim=config.action_dim,
            num_blocks=config.action_head_num_blocks,
        )
        
        # Action token setup
        self.action_token = config.action_token
        self.chunk_size = config.chunk_size
        
        # L1 loss
        self.l1_loss = nn.L1Loss()
        
        # Will be set after processor is loaded
        self.action_token_id = None
        self.multimodal_processor = None
        self.tokenizer = None
        
        logger.info(f"LlavaOFT initialized with hidden_size={vlm_hidden_size}")

    def _check_flash_attn(self) -> bool:
        """Check if flash attention is available."""
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    def set_processor(self, processor):
        """Set the multimodal processor.
        
        Note: We use last-N-embeddings approach for action prediction, so action token
        encoding is not critical. The action tokens in the prompt simply serve to
        extend the sequence length to provide positions for action prediction.
        """
        self.multimodal_processor = processor
        self.tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        
        # Check action token tokenization (informational only)
        test_tokenization = self.tokenizer(
            self.action_token, add_special_tokens=False
        )["input_ids"]
        
        logger.info(
            f"Action token '{self.action_token}' -> {len(test_tokenization)} token(s). "
            f"Using last-{self.chunk_size}-embeddings approach for action prediction."
        )

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_requires_grad(self, training_args):
        """Set trainable parameters based on training arguments."""
        # Freeze vision tower if specified
        if hasattr(training_args, 'freeze_vision_tower') and training_args.freeze_vision_tower:
            if hasattr(self.vlm, 'vision_tower'):
                for p in self.vlm.vision_tower.parameters():
                    p.requires_grad = False
            logger.info("Vision tower frozen")
        
        # Freeze language model if specified
        if hasattr(training_args, 'freeze_backbone') and training_args.freeze_backbone:
            if hasattr(self.vlm, 'language_model'):
                for p in self.vlm.language_model.parameters():
                    p.requires_grad = False
            logger.info("Language model backbone frozen")
        
        # Action head always trainable
        self.action_head.requires_grad_(True)

    def _get_last_n_embeddings(
        self,
        last_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract the last chunk_size embeddings from hidden states.
        
        This is the correct approach for decoder models like LLaVA:
        - Each position in the sequence has unique hidden states due to causal attention
        - The last chunk_size positions contain the most context-rich representations
        - Action tokens at the end of the prompt provide natural positions for action prediction
        
        Args:
            last_hidden: (B, L, H) - Last layer hidden states
            attention_mask: (B, L) - Optional attention mask
            
        Returns:
            action_queries: (B, chunk_size, H)
        """
        B, L, H = last_hidden.shape

        if attention_mask is not None:
            # Find the last valid position for each sample
            # Then take chunk_size positions ending at that position
            seq_lengths = attention_mask.sum(dim=1)  # (B,)
            
            # Build indices for gathering
            # For each sample, we want positions [seq_len - chunk_size, seq_len)
            action_queries_list = []
            for i in range(B):
                seq_len = seq_lengths[i].item()
                start_pos = max(0, seq_len - self.chunk_size)
                end_pos = seq_len
                
                # Get embeddings for this sample
                sample_embeddings = last_hidden[i, start_pos:end_pos, :]  # (actual_len, H)

                # Pad if needed (shouldn't happen normally)
                actual_len = sample_embeddings.shape[0]
                if actual_len < self.chunk_size:
                    # Repeat the last embedding to fill
                    padding = sample_embeddings[-1:].repeat(self.chunk_size - actual_len, 1)
                    sample_embeddings = torch.cat([sample_embeddings, padding], dim=0)
                
                action_queries_list.append(sample_embeddings)
            
            action_queries = torch.stack(action_queries_list, dim=0)  # (B, chunk_size, H)
        else:
            # No attention mask, simply take last chunk_size positions
            action_queries = last_hidden[:, -self.chunk_size:, :]
        
        return action_queries

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Returns:
            Dict with 'loss' and 'action_loss' keys.
        """
        # Forward through VLM to get hidden states
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                output_hidden_states=True,
                return_dict=True,
            )
            # Get last layer hidden states
            last_hidden = outputs.hidden_states[-1]  # (B, L, H)

        # Predict actions using last chunk_size embeddings
        # For decoder models like LLaVA, the last positions have the richest context
        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._get_last_n_embeddings(last_hidden, attention_mask)
            pred_actions = self.action_head(action_queries)

            # Compute loss if actions provided
            if actions is not None:
                # Take last chunk_size actions as target
                if actions.dim() == 3:
                    actions_target = actions[:, -self.chunk_size:, :]
                else:
                    actions_target = actions
                
                # Apply padding mask if provided
                if is_pad is not None:
                    is_pad_chunk = is_pad[:, -self.chunk_size:]
                    valid_mask = ~is_pad_chunk
                    if valid_mask.any():
                        action_loss = self.l1_loss(
                            pred_actions[valid_mask],
                            actions_target[valid_mask]
                        )
                    else:
                        action_loss = torch.tensor(0.0, device=pred_actions.device)
                else:
                    action_loss = self.l1_loss(pred_actions, actions_target)
            else:
                action_loss = torch.tensor(0.0, device=pred_actions.device)

        return {
            'loss': action_loss,
            'action_loss': action_loss,
            'pred_actions': pred_actions,
        }

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Generate actions for inference.
        
        Returns:
            Dict with 'pred_actions' as numpy array.
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._get_last_n_embeddings(last_hidden, attention_mask)
            pred_actions = self.action_head(action_queries)

        return {'pred_actions': pred_actions.detach().float().cpu().numpy()}

    def select_action(self, batch_obs: Dict[str, Any]) -> np.ndarray:
        """
        Select action from observation batch (inference interface).
        
        Args:
            batch_obs: Processed and collated batch observation
            
        Returns:
            Action predictions as numpy array
        """
        # Move batch to device
        for k, v in batch_obs.items():
            if isinstance(v, torch.Tensor):
                batch_obs[k] = v.to(self.device)
        
        result = self.generate(**batch_obs)
        return result['pred_actions']

