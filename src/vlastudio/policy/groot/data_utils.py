"""
Data utilities for GR00T model in ILStudio.

Handles:
  - Data processing: Convert ILStudio standard format to GR00T format
  - Data collation: Batch samples for training with Eagle encoding

GR00T expects a specific input format with Eagle-encoded visual features.
"""

import sys
from pathlib import Path

# Add third_party/lerobot to path for imports
LEROBOT_PATH = Path(__file__).resolve().parents[2] / "third_party" / "lerobot" / "src"
if str(LEROBOT_PATH) not in sys.path:
    sys.path.insert(0, str(LEROBOT_PATH))

import torch
import numpy as np
from PIL import Image
from typing import Dict, Sequence, Optional, List, Any
from dataclasses import dataclass, field
from einops import rearrange

from loguru import logger

# Import lerobot utilities
try:
    from lerobot.utils.constants import HF_LEROBOT_HOME
except ImportError:
    HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"


def _build_eagle_processor(tokenizer_assets_repo: str = "lerobot/eagle2hg-processor-groot-n1p5"):
    """Build Eagle processor for GR00T."""
    from transformers import AutoProcessor
    from lerobot.policies.groot.utils import ensure_eagle_cache_ready
    
    # Prepare cache directory
    vendor_dir = str(Path(__file__).resolve().parents[2] / "third_party" / "lerobot" / "src" / 
                     "lerobot" / "policies" / "groot" / "eagle2_hg_model")
    cache_dir = HF_LEROBOT_HOME / tokenizer_assets_repo
    
    try:
        ensure_eagle_cache_ready(vendor_dir, cache_dir, tokenizer_assets_repo)
    except Exception as exc:
        logger.warning(f"Failed to prepare Eagle cache: {exc}")
    
    proc = AutoProcessor.from_pretrained(str(cache_dir), trust_remote_code=True, use_fast=True)
    proc.tokenizer.padding_side = "left"
    return proc


class GrootDataProcessor:
    """
    Process ILStudio samples to GR00T model format.
    
    Converts standard ILStudio sample format to GR00T expected format
    with proper video/state/action packing.
    """
    
    def __init__(
        self,
        eagle_processor=None,
        chunk_size: int = 16,
        max_state_dim: int = 64,
        max_action_dim: int = 32,
        embodiment_tag: str = "new_embodiment",
        image_size: tuple = (224, 224),
        tokenizer_assets_repo: str = "lerobot/eagle2hg-processor-groot-n1p5",
    ):
        self.eagle_processor = eagle_processor
        self.chunk_size = min(chunk_size, 16)  # GR00T max action horizon is 16
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.embodiment_tag = embodiment_tag
        self.image_size = image_size
        self.tokenizer_assets_repo = tokenizer_assets_repo
        
        # Embodiment mapping (matches GR00T)
        self.embodiment_mapping = {
            "new_embodiment": 31,
            "oxe_droid": 17,
            "agibot_genie1": 26,
            "gr1": 24,
            "so100": 2,
            "unitree_g1": 3,
        }

    def _preprocess_image(self, image_tensor) -> np.ndarray:
        """Convert tensor to numpy HWC uint8 format."""
        if isinstance(image_tensor, Image.Image):
            return np.array(image_tensor)
        elif isinstance(image_tensor, np.ndarray):
            if image_tensor.ndim == 3 and image_tensor.shape[0] in [1, 3, 4]:
                # CHW -> HWC
                image_tensor = np.transpose(image_tensor, (1, 2, 0))
            if image_tensor.dtype != np.uint8:
                if image_tensor.max() <= 1.0:
                    image_tensor = (image_tensor * 255).astype(np.uint8)
                else:
                    image_tensor = image_tensor.astype(np.uint8)
            return image_tensor
        elif isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:
                image_tensor = image_tensor.squeeze(0)
            if image_tensor.shape[0] in [1, 3, 4]:
                # CHW -> HWC
                image_tensor = image_tensor.permute(1, 2, 0)
            image_np = image_tensor.cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            return image_np
        else:
            raise ValueError(f"Unsupported image type: {type(image_tensor)}")

    def _pad_to_dim(self, tensor: torch.Tensor, target_dim: int, dim: int = -1) -> torch.Tensor:
        """Pad tensor to target dimension."""
        current_dim = tensor.shape[dim]
        if current_dim >= target_dim:
            return tensor[..., :target_dim] if dim == -1 else tensor[:target_dim]
        
        pad_size = target_dim - current_dim
        if dim == -1:
            padding = torch.zeros(*tensor.shape[:-1], pad_size, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=-1)
        else:
            padding = torch.zeros(pad_size, *tensor.shape[1:], dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=0)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sample.
        """
        # === Extract and process images ===
        image_data = sample['image']
        if isinstance(image_data, torch.Tensor):
            if image_data.dim() == 4:  # (K, C, H, W) - multiple cameras
                images = [self._preprocess_image(image_data[i]) for i in range(image_data.shape[0])]
            else:
                images = [self._preprocess_image(image_data)]
        elif isinstance(image_data, list):
            images = [self._preprocess_image(img) for img in image_data]
        else:
            images = [self._preprocess_image(image_data)]
        
        # Stack images as video format: (1, V, H, W, C) where V = num cameras
        video = np.stack(images, axis=0)  # (V, H, W, C)
        video = np.expand_dims(video, axis=0)  # (1, V, H, W, C) - single timestep
        
        # === Get state ===
        orig_state_dim = 0
        state = sample.get('state', None)
        if state is not None:
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            elif not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            
            orig_state_dim = state.shape[-1]
            
            # Pad state (normalization handled by ILStudio)
            state = self._pad_to_dim(state, self.max_state_dim)
            state = state.unsqueeze(0).numpy()  # (1, state_dim) for state_horizon=1
        
        # === Get action ===
        orig_action_dim = 0
        orig_action_len = 0
        action = sample.get('action', None)
        if action is not None:
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()
            elif not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)
            
            # Ensure correct shape: (chunk_size, action_dim)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            
            orig_action_len = action.shape[0]
            orig_action_dim = action.shape[-1]
            
            # Truncate and pad (normalization handled by ILStudio)
            action = action[:self.chunk_size]
            action = self._pad_to_dim(action, self.max_action_dim, dim=-1)
            
            # Pad time dimension if needed
            if action.shape[0] < self.chunk_size:
                pad_len = self.chunk_size - action.shape[0]
                padding = torch.zeros(pad_len, action.shape[1], dtype=action.dtype)
                action = torch.cat([action, padding], dim=0)
            
            action = action.numpy()
        
        # === Get language instruction ===
        instruction = sample.get('raw_lang', 'Perform the task.')
        if not instruction:
            instruction = 'Perform the task.'
        
        # === Create masks ===
        state_mask = np.ones((1, self.max_state_dim), dtype=bool) if state is not None else None
        if state_mask is not None and orig_state_dim > 0:
            state_mask[0, orig_state_dim:] = False
        
        action_mask = np.ones((self.chunk_size, self.max_action_dim), dtype=bool) if action is not None else None
        if action_mask is not None:
            action_mask[min(orig_action_len, self.chunk_size):, :] = False
            action_mask[:, orig_action_dim:] = False
        
        # === Create embodiment ID ===
        # GR00T expects embodiment_id as scalar per sample
        embodiment_id = self.embodiment_mapping.get(self.embodiment_tag, 31)
        
        # Build output dict
        data_dict = {
            'video': video,  # (1, V, H, W, C) uint8
            'language': instruction,
            'state': state,  # (1, max_state_dim)
            'state_mask': state_mask,  # (1, max_state_dim)
            'action': action,  # (chunk_size, max_action_dim)
            'action_mask': action_mask,  # (chunk_size, max_action_dim)
            'embodiment_id': embodiment_id,  # scalar int
        }
        
        return data_dict


@dataclass
class GrootDataCollator:
    """
    Collate examples for GR00T training.
    
    This collator handles the Eagle encoding step and creates
    batched inputs for the GR00T model.
    """
    
    eagle_processor: Any = None
    max_state_dim: int = 64
    max_action_dim: int = 32
    chunk_size: int = 16
    dtype: torch.dtype = torch.bfloat16
    tokenizer_assets_repo: str = "lerobot/eagle2hg-processor-groot-n1p5"
    _proc: Any = field(default=None, init=False, repr=False)
    
    @property
    def proc(self):
        """Lazy load Eagle processor."""
        if self._proc is None:
            if self.eagle_processor is not None:
                self._proc = self.eagle_processor
            else:
                try:
                    self._proc = _build_eagle_processor(self.tokenizer_assets_repo)
                except Exception as e:
                    logger.warning(f"Failed to build Eagle processor: {e}")
        return self._proc
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of instances."""
        batch_size = len(instances)
        
        # === Collect data for Eagle encoding ===
        eagle_contents = []
        for inst in instances:
            video = inst['video']  # (1, V, H, W, C)
            lang = inst['language']
            
            # Flatten video to list of PIL Images
            t, v, h, w, c = video.shape
            flat = rearrange(video, "t v h w c -> (t v) h w c")
            images = [Image.fromarray(flat[i]) for i in range(t * v)]
            
            # Format language as string list representation (matches GR00T)
            lang_formatted = str([lang])
            text_content = [{"type": "text", "text": lang_formatted}]
            image_content = [{"type": "image", "image": img} for img in images]
            conv = [{"role": "user", "content": image_content + text_content}]
            
            # Process with Eagle processor
            if self.proc is not None:
                text_list = [self.proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)]
                img_inputs, vid_inputs = self.proc.process_vision_info(conv)
                eagle_contents.append({
                    "text_list": text_list,
                    "image_inputs": img_inputs,
                })
        
        # === Collate Eagle inputs ===
        batch = {}
        if eagle_contents and self.proc is not None:
            text_list = []
            image_inputs = []
            for content in eagle_contents:
                text_list += content["text_list"]
                image_inputs += content["image_inputs"]
            
            eagle_inputs = self.proc(
                text=text_list,
                images=image_inputs,
                images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},
                return_tensors="pt",
                padding=True,
            )
            
            # Prefix with eagle_
            for k, v in eagle_inputs.items():
                batch[f"eagle_{k}"] = v
        
        # === Stack state tensors ===
        states = [inst['state'] for inst in instances if inst.get('state') is not None]
        if states:
            state = torch.from_numpy(np.stack(states)).to(self.dtype)
            batch['state'] = state
        
        # === Stack state masks ===
        state_masks = [inst['state_mask'] for inst in instances if inst.get('state_mask') is not None]
        if state_masks:
            batch['state_mask'] = torch.from_numpy(np.stack(state_masks))
        
        # === Stack action tensors ===
        actions = [inst['action'] for inst in instances if inst.get('action') is not None]
        if actions:
            action = torch.from_numpy(np.stack(actions)).to(self.dtype)
            batch['action'] = action
        
        # === Stack action masks ===
        action_masks = [inst['action_mask'] for inst in instances if inst.get('action_mask') is not None]
        if action_masks:
            batch['action_mask'] = torch.from_numpy(np.stack(action_masks))
        
        # === Stack embodiment IDs ===
        # embodiment_id should be shape (B,) - a scalar per sample
        embodiment_ids = [inst['embodiment_id'] for inst in instances]
        batch['embodiment_id'] = torch.tensor(embodiment_ids, dtype=torch.long)
        
        return batch
