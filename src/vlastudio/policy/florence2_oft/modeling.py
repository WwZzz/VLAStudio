"""
Florence2-OFT Model Implementation for ILStudio.

A lightweight implementation using Florence2 encoder backbone with action token
prediction via L1 regression on hidden states.

Key Points:
  - Florence2 vision-language encoder backbone
  - Uses encoder-only architecture (decoder removed for efficiency)
  - Continuous action prediction via L1 regression

Reference:
  - Florence2: https://huggingface.co/microsoft/Florence-2-large
  - StarVLA Florence2 implementation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Any
from PIL import Image
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoProcessor, AutoModelForCausalLM

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
        input_dim: int = 1024,
        hidden_dim: int = 2048,
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

class Florence2OFTConfig(PretrainedConfig):
    """
    Configuration for Florence2-OFT model.
    
    Args:
        vlm_model_name_or_path: Path to Florence2 model
        action_dim: Dimension of action space
        state_dim: Dimension of state space
        chunk_size: Action prediction horizon (chunk length)
        action_hidden_dim: Hidden dimension for action head (auto-set from VLM)
        action_head_hidden_mult: Multiplier for action head hidden dimension
        action_head_num_blocks: Number of MLP ResNet blocks
        action_token: Special token for action prediction
        image_size: Image size for processing (can be int or [h, w] list)
    """
    model_type = "florence2_oft"

    def __init__(
        self,
        vlm_model_name_or_path: str = "microsoft/Florence-2-large",
        action_dim: int = 7,
        state_dim: int = 7,
        chunk_size: int = 16,
        action_hidden_dim: int = None,  # Will be set from VLM projection_dim
        action_head_hidden_mult: int = 2,
        action_head_num_blocks: int = 2,
        action_token: str = "@",
        image_size = None,  # Image size for processing (int or [h, w])
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
        self.image_size = image_size


# =============================================================================
# Main Model
# =============================================================================

class Florence2OFTForPolicy(PreTrainedModel):
    """
    Florence2-OFT model for continuous action prediction.
    
    Uses Florence2 encoder as backbone and predicts actions via L1 regression
    on encoder hidden states.
    """
    config_class = Florence2OFTConfig

    def __init__(self, config: Florence2OFTConfig):
        super().__init__(config)
        self.config = config
        
        # Load Florence2 model
        logger.info(f"Loading Florence2 from: {config.vlm_model_name_or_path}")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.vlm = AutoModelForCausalLM.from_pretrained(
            config.vlm_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager",  # Florence2 uses eager attention
        )
        
        # Get hidden size from VLM (Florence2 uses projection_dim)
        vlm_hidden_size = self.vlm.config.projection_dim
        self.vlm.config.hidden_size = vlm_hidden_size  # Align with other VLMs
        
        if config.action_hidden_dim is None:
            config.action_hidden_dim = vlm_hidden_size
        
        # Remove unused modules to save memory
        if hasattr(self.vlm, "decoder"):
            del self.vlm.decoder
            logger.info("Removed Florence2 decoder to save memory")
        if hasattr(self.vlm, "lm_head"):
            del self.vlm.lm_head
            logger.info("Removed Florence2 lm_head to save memory")
        
        # Build action head (kept in float32 for numerical stability)
        self.action_head = L1RegressionActionHead(
            input_dim=config.action_hidden_dim,
            hidden_dim=config.action_hidden_dim * config.action_head_hidden_mult,
            action_dim=config.action_dim,
            num_blocks=config.action_head_num_blocks,
        ).float()  # Keep action head in float32
        
        # Action token setup
        self.action_token = config.action_token
        self.chunk_size = config.chunk_size
        
        # L1 loss
        self.l1_loss = nn.L1Loss()
        
        # Will be set after processor is loaded
        self.action_token_id = None
        self.multimodal_processor = None
        self.tokenizer = None
        
        # Register action_head as float32 module (won't be converted by model.to())
        self._action_head_dtype = torch.float32
        
        # Cache VLM dtype to avoid next() in forward (which fails with DataParallel)
        self._vlm_dtype = torch_dtype
        
        logger.info(f"Florence2OFT initialized with hidden_size={vlm_hidden_size}")

    def set_processor(self, processor):
        """Set the multimodal processor and add action token as special token if needed."""
        self.multimodal_processor = processor
        # Florence2 processor has tokenizer attribute
        self.tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        
        # Check current tokenization
        action_token_only = self.tokenizer(self.action_token, add_special_tokens=False)["input_ids"]
        tokens_per_action_before = len(action_token_only)
        
        # Try to add action_token as a special token to force single-token tokenization
        if tokens_per_action_before != 1:
            # Check if tokenizer supports add_tokens
            if hasattr(self.tokenizer, 'add_tokens'):
                num_added = self.tokenizer.add_tokens([self.action_token], special_tokens=True)
                if num_added > 0:
                    logger.info(
                        f"Added action token '{self.action_token}' as special token. "
                        f"Tokenizer vocabulary size increased by {num_added}."
                    )
                    # Resize model embeddings if needed
                    # For Florence2, embeddings are in language_model.embed_tokens
                    if hasattr(self.vlm, 'resize_token_embeddings'):
                        try:
                            self.vlm.resize_token_embeddings(len(self.tokenizer))
                            logger.info("Resized VLM embeddings to match new vocabulary size")
                        except Exception as e:
                            logger.warning(
                                f"Failed to resize VLM embeddings: {e}. "
                                f"This may be fine if embeddings are handled differently."
                            )
                    elif hasattr(self.vlm, 'language_model') and hasattr(self.vlm.language_model, 'resize_token_embeddings'):
                        try:
                            self.vlm.language_model.resize_token_embeddings(len(self.tokenizer))
                            logger.info("Resized language_model embeddings to match new vocabulary size")
                        except Exception as e:
                            logger.warning(
                                f"Failed to resize language_model embeddings: {e}. "
                                f"This may be fine if embeddings are handled differently."
                            )
                else:
                    logger.warning(
                        f"Failed to add action token '{self.action_token}' as special token. "
                        f"It may already exist or tokenizer doesn't support adding tokens."
                    )
            else:
                logger.warning(
                    f"Tokenizer doesn't support add_tokens method. "
                    f"Action token '{self.action_token}' will remain as {tokens_per_action_before} token(s)."
                )
        
        # Verify tokenization after adding
        action_token_only_after = self.tokenizer(self.action_token, add_special_tokens=False)["input_ids"]
        tokens_per_action_after = len(action_token_only_after)
        
        logger.info(
            f"Action token '{self.action_token}' tokenization: "
            f"{tokens_per_action_before} token(s) before -> {tokens_per_action_after} token(s) after. "
            f"Will use last {self.chunk_size} hidden states for action prediction."
        )
        
        if tokens_per_action_after != 1:
            logger.warning(
                f"Action token '{self.action_token}' is still tokenized into {tokens_per_action_after} tokens. "
                f"This may cause misalignment. The model will use last {self.chunk_size} hidden states."
            )
        else:
            logger.info(f"Successfully configured action token '{self.action_token}' as single token.")
        
        logger.info("Florence2OFT processor set")

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_requires_grad(self, training_args):
        """Set trainable parameters based on training arguments."""
        # Freeze vision encoder if specified
        if hasattr(training_args, 'freeze_vision_tower') and training_args.freeze_vision_tower:
            if hasattr(self.vlm, 'vision_tower'):
                for p in self.vlm.vision_tower.parameters():
                    p.requires_grad = False
            elif hasattr(self.vlm, 'image_projection'):
                for p in self.vlm.image_projection.parameters():
                    p.requires_grad = False
            logger.info("Vision components frozen")
        
        # Freeze language encoder if specified
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
    ) -> torch.Tensor:
        """
        Extract the last chunk_size embeddings from hidden states.
        
        For Florence2 encoder, the merged sequence is [image_features, text_embeddings].
        Since action tokens are appended at the end of the prompt, the last chunk_size
        hidden states correspond to the action token positions.
        
        Args:
            last_hidden: (B, L, H) - Last layer hidden states
            
        Returns:
            action_queries: (B, chunk_size, H)
        """
        # Simply take the last chunk_size hidden states
        # Florence2 encoder output: [image_features, text_embeddings]
        # Action tokens are at the end of text, so last chunk_size positions
        return last_hidden[:, -self.chunk_size:, :]

    def forward_encoder(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through Florence2 encoder.
        
        Returns:
            last_hidden_state: (B, L, H)
        """
        # Use cached dtype to avoid next() which fails with DataParallel
        pixel_values = pixel_values.to(dtype=self._vlm_dtype)
        
        # Get image features
        valid_feats = self.vlm._encode_image(pixel_values)  # [B, N, D]
        
        # Get text embeddings
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)  # [B, L, D]
        
        # Merge image features and text embeddings
        B, L, D = inputs_embeds.shape
        image_features = valid_feats.view(B, -1, D)  # [B, N_view*N, D]
        
        merged_embeds, merged_attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features,
            inputs_embeds,
        )
        
        # Forward through encoder
        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=merged_attention_mask,
            inputs_embeds=merged_embeds,
        )
        
        return enc_out.last_hidden_state

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
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
        # Forward through encoder
        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden = self.forward_encoder(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )

        # Predict actions using last chunk_size embeddings
        with torch.autocast("cuda", enabled=False):
            # Convert to float32 for action head computation
            last_hidden_f32 = last_hidden.float()
            
            # Get last chunk_size embeddings (corresponding to action tokens)
            action_queries = self._get_last_n_embeddings(last_hidden_f32)
            
            pred_actions = self.action_head(action_queries)

            # Compute loss if actions provided
            if actions is not None:
                # Take last chunk_size actions as target
                if actions.dim() == 3:
                    actions_target = actions[:, -self.chunk_size:, :]
                else:
                    actions_target = actions
                
                # Ensure targets are float32
                actions_target = actions_target.float()
                
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
                        action_loss = torch.tensor(0.0, device=pred_actions.device, dtype=torch.float32)
                else:
                    action_loss = self.l1_loss(pred_actions, actions_target)
            else:
                action_loss = torch.tensor(0.0, device=pred_actions.device, dtype=torch.float32)

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
        states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Generate actions for inference.
        
        Returns:
            Dict with 'pred_actions' as numpy array.
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden = self.forward_encoder(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )

        with torch.autocast("cuda", enabled=False):
            last_hidden_f32 = last_hidden.float()
            action_queries = self._get_last_n_embeddings(last_hidden_f32)
            pred_actions = self.action_head(action_queries)

        return {'pred_actions': pred_actions.detach().cpu().numpy()}

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

