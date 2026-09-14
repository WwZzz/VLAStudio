"""
RoboFlamingo Policy for ILStudio.

A VLM-based robotics learning framework that uses OpenFlamingo as backbone
for learning language-conditioned robot skills from offline imitation datasets.

This module provides three main interfaces:
  - load_model: Load model and components
  - get_data_processor: Get data preprocessing function
  - get_data_collator: Get data collation function

Reference:
  - Paper: Vision-Language Foundation Models as Effective Robot Imitators
  - GitHub: https://github.com/RoboFlamingo/RoboFlamingo
"""

from .modeling import RoboFlamingoConfig, RoboFlamingoForPolicy
from .data_utils import RoboFlamingoDataProcessor, RoboFlamingoDataCollator

import torch
import os
from typing import Dict, Any, Optional
from pathlib import Path

from loguru import logger

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    logger.warning("open_clip not available. Install with: pip install open-clip-torch")

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available. Install with: pip install huggingface-hub")

from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# OpenFlamingo Model Registry with Auto-Download Support
# =============================================================================

# OpenFlamingo models on Hugging Face
OPENFLAMINGO_MODELS = {
    "openflamingo-3b": {
        "repo_id": "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
        "filename": "checkpoint.pt",
        "lang_encoder": "anas-awadalla/mpt-1b-redpajama-200b",
        "cross_attn_every_n_layers": 1,
        "vision_encoder": "ViT-L-14",
        "vision_pretrained": "openai",
    },
    "openflamingo-3b-instruct": {
        "repo_id": "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct",
        "filename": "checkpoint.pt",
        "lang_encoder": "anas-awadalla/mpt-1b-redpajama-200b-dolly",
        "cross_attn_every_n_layers": 1,
        "vision_encoder": "ViT-L-14",
        "vision_pretrained": "openai",
    },
    "openflamingo-4b": {
        "repo_id": "openflamingo/OpenFlamingo-4B-vitl-rpj3b",
        "filename": "checkpoint.pt",
        "lang_encoder": "togethercomputer/RedPajama-INCITE-Base-3B-v1",
        "cross_attn_every_n_layers": 2,
        "vision_encoder": "ViT-L-14",
        "vision_pretrained": "openai",
    },
    "openflamingo-4b-instruct": {
        "repo_id": "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct",
        "filename": "checkpoint.pt",
        "lang_encoder": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        "cross_attn_every_n_layers": 2,
        "vision_encoder": "ViT-L-14",
        "vision_pretrained": "openai",
    },
    "openflamingo-9b": {
        "repo_id": "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
        "filename": "checkpoint.pt",
        "lang_encoder": "anas-awadalla/mpt-7b",
        "cross_attn_every_n_layers": 4,
        "vision_encoder": "ViT-L-14",
        "vision_pretrained": "openai",
    },
}

# Alias for convenience
OPENFLAMINGO_ALIASES = {
    "3b": "openflamingo-3b-instruct",
    "3b-base": "openflamingo-3b",
    "4b": "openflamingo-4b-instruct",
    "4b-base": "openflamingo-4b",
    "9b": "openflamingo-9b",
    # Legacy names
    "mpt_3b": "openflamingo-3b",
    "mpt_dolly_3b": "openflamingo-3b-instruct",
    "mpt_4b": "openflamingo-4b-instruct",
    "mpt_base_4b": "openflamingo-4b",
    "mpt_9b": "openflamingo-9b",
}

# Pre-defined model configurations (legacy, for backward compatibility)
MODEL_CONFIGS = {
    "mpt_3b": {
        "lang_encoder_path": "anas-awadalla/mpt-1b-redpajama-200b",
        "tokenizer_path": "anas-awadalla/mpt-1b-redpajama-200b",
        "cross_attn_every_n_layers": 1,
    },
    "mpt_dolly_3b": {
        "lang_encoder_path": "anas-awadalla/mpt-1b-redpajama-200b-dolly",
        "tokenizer_path": "anas-awadalla/mpt-1b-redpajama-200b-dolly",
        "cross_attn_every_n_layers": 1,
    },
    "mpt_4b": {
        "lang_encoder_path": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        "tokenizer_path": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        "cross_attn_every_n_layers": 2,
    },
    "mpt_9b": {
        "lang_encoder_path": "anas-awadalla/mpt-7b",
        "tokenizer_path": "anas-awadalla/mpt-7b",
        "cross_attn_every_n_layers": 4,
    },
}


def get_openflamingo_checkpoint(
    model_name: str = "openflamingo-3b-instruct",
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> Dict[str, Any]:
    """
    Download and cache OpenFlamingo checkpoint from Hugging Face.
    
    Args:
        model_name: Model name or alias (e.g., "3b", "openflamingo-3b-instruct")
        cache_dir: Optional custom cache directory
        force_download: Force re-download even if cached
        
    Returns:
        Dict with:
            - checkpoint_path: Path to downloaded checkpoint
            - lang_encoder: Language encoder repo ID
            - cross_attn_every_n_layers: Cross-attention interval
            - vision_encoder: Vision encoder name
            - vision_pretrained: Vision encoder pretrained weights
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for auto-download. "
            "Install with: pip install huggingface-hub"
        )
    
    # Resolve alias
    if model_name in OPENFLAMINGO_ALIASES:
        model_name = OPENFLAMINGO_ALIASES[model_name]
    
    if model_name not in OPENFLAMINGO_MODELS:
        available = list(OPENFLAMINGO_MODELS.keys()) + list(OPENFLAMINGO_ALIASES.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {available}"
        )
    
    model_info = OPENFLAMINGO_MODELS[model_name]
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = os.environ.get(
            "OPENFLAMINGO_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "openflamingo")
        )
    
    logger.info(f"Downloading OpenFlamingo checkpoint: {model_name}")
    logger.info(f"  Repository: {model_info['repo_id']}")
    logger.info(f"  Cache directory: {cache_dir}")
    
    # Download checkpoint
    try:
        checkpoint_path = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["filename"],
            cache_dir=cache_dir,
            force_download=force_download,
        )
        logger.info(f"  Checkpoint downloaded to: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {e}")
        raise
    
    return {
        "checkpoint_path": checkpoint_path,
        "lang_encoder": model_info["lang_encoder"],
        "cross_attn_every_n_layers": model_info["cross_attn_every_n_layers"],
        "vision_encoder": model_info["vision_encoder"],
        "vision_pretrained": model_info["vision_pretrained"],
    }


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available OpenFlamingo models.
    
    Returns:
        Dict mapping model names to their configurations.
    """
    result = {}
    for name, config in OPENFLAMINGO_MODELS.items():
        result[name] = {
            "repo_id": config["repo_id"],
            "lang_encoder": config["lang_encoder"],
            "cross_attn_every_n_layers": config["cross_attn_every_n_layers"],
        }
    
    # Add aliases
    result["_aliases"] = OPENFLAMINGO_ALIASES
    
    return result


def _load_openflamingo_weights(model, checkpoint: Dict[str, torch.Tensor]):
    """
    Load OpenFlamingo checkpoint weights into RoboFlamingo model.
    
    Handles key mapping between OpenFlamingo and RoboFlamingo.
    """
    # Filter and remap keys
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    
    for key, value in checkpoint.items():
        # Map OpenFlamingo keys to RoboFlamingo keys
        new_key = key
        
        # Perceiver resampler mapping
        if key.startswith("perceiver."):
            new_key = key  # Direct mapping
        
        # Vision encoder mapping (usually frozen, skip)
        elif key.startswith("vision_encoder."):
            skipped_keys.append(key)
            continue
        
        # Language model / cross-attention mapping
        elif key.startswith("lang_encoder."):
            new_key = key  # Direct mapping
        
        # Check if key exists in model
        if new_key in model_state:
            if model_state[new_key].shape == value.shape:
                model_state[new_key] = value
                loaded_keys.append(new_key)
            else:
                logger.warning(
                    f"Shape mismatch for {new_key}: "
                    f"model={model_state[new_key].shape}, checkpoint={value.shape}"
                )
                skipped_keys.append(key)
        else:
            skipped_keys.append(key)
    
    # Load matched weights
    model.load_state_dict(model_state, strict=False)
    
    logger.info(f"Loaded {len(loaded_keys)} weights from OpenFlamingo checkpoint")
    if skipped_keys:
        logger.debug(f"Skipped {len(skipped_keys)} keys (not in model or shape mismatch)")


def _get_decoder_layers_attr_name(lang_encoder) -> str:
    """
    Get the attribute name for decoder layers in the language model.
    
    Different LLMs have different attribute names for their transformer blocks.
    """
    # Try to infer automatically first
    try:
        from open_flamingo.src.factory import _infer_decoder_layers_attr_name
        return _infer_decoder_layers_attr_name(lang_encoder)
    except (ImportError, ValueError):
        pass
    
    # Manual mapping for known models
    model_type = getattr(lang_encoder.config, "model_type", "").lower()
    
    # MPT models
    if model_type == "mpt" or hasattr(lang_encoder, "transformer"):
        if hasattr(lang_encoder.transformer, "blocks"):
            return "transformer.blocks"
        elif hasattr(lang_encoder.transformer, "h"):
            return "transformer.h"
    
    # GPT-NeoX / RedPajama models
    if model_type in ["gpt_neox", "gpt-neox"] or hasattr(lang_encoder, "gpt_neox"):
        return "gpt_neox.layers"
    
    # LLaMA models
    if model_type == "llama" or hasattr(lang_encoder, "model"):
        if hasattr(lang_encoder.model, "layers"):
            return "model.layers"
    
    # GPT-2 style models
    if hasattr(lang_encoder, "transformer") and hasattr(lang_encoder.transformer, "h"):
        return "transformer.h"
    
    # OPT models
    if model_type == "opt":
        return "model.decoder.layers"
    
    # Fallback: try common patterns
    for attr_name in [
        "transformer.blocks",
        "transformer.h", 
        "model.layers",
        "gpt_neox.layers",
        "model.decoder.layers",
    ]:
        parts = attr_name.split(".")
        obj = lang_encoder
        try:
            for part in parts:
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.ModuleList):
                return attr_name
        except AttributeError:
            continue
    
    raise ValueError(
        f"Could not infer decoder layers attribute name for model type: {model_type}. "
        "Please specify 'decoder_layers_attr_name' in the configuration."
    )


def _patch_mpt_decoder_for_flamingo(lang_encoder, decoder_layers_attr_name: str):
    """
    Patch MPT model's decoder to be compatible with open_flamingo.
    
    open_flamingo expects get_decoder().layers to return the transformer blocks,
    but MPT uses transformer.blocks instead.
    
    After init_flamingo is called, _get_decoder_layers() returns FlamingoLayer objects,
    so we need get_decoder().layers to return those same FlamingoLayer objects.
    """
    # Create a wrapper class that has a 'layers' attribute
    # This will dynamically get the layers from _get_decoder_layers()
    class DecoderWrapper:
        def __init__(self, encoder):
            self._encoder = encoder
        
        @property
        def layers(self):
            # Return the (possibly wrapped) decoder layers
            return self._encoder._get_decoder_layers()
    
    # Store the wrapper
    decoder_wrapper = DecoderWrapper(lang_encoder)
    
    # Patch get_decoder to return wrapper with layers attribute
    def patched_get_decoder():
        return decoder_wrapper
    
    lang_encoder.get_decoder = patched_get_decoder
    logger.info(f"Patched get_decoder() to return wrapper with dynamic 'layers' attribute")


def _init_flamingo_layers(lang_encoder, vis_dim, lang_dim, cross_attn_every_n_layers, decoder_layers_attr_name=None):
    """
    Initialize Flamingo cross-attention layers in language model.
    
    This is a simplified version - for full functionality, the open_flamingo
    package should be installed.
    """
    try:
        from open_flamingo.src.flamingo_lm import FlamingoLMMixin
        from open_flamingo.src.utils import extend_instance
        
        # Patch config for MPT models that use d_model instead of hidden_size
        # open_flamingo expects config.hidden_size to exist
        if hasattr(lang_encoder.config, 'd_model') and not hasattr(lang_encoder.config, 'hidden_size'):
            lang_encoder.config.hidden_size = lang_encoder.config.d_model
            logger.info(f"Patched config.hidden_size = {lang_encoder.config.d_model} (from d_model)")
        
        extend_instance(lang_encoder, FlamingoLMMixin)
        
        # Get decoder layers attribute name
        if decoder_layers_attr_name is None:
            decoder_layers_attr_name = _get_decoder_layers_attr_name(lang_encoder)
        
        logger.info(f"Using decoder layers attribute: {decoder_layers_attr_name}")
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        
        # Patch MPT model's get_decoder() to return object with 'layers' attribute
        # This is needed because open_flamingo's forward() calls self.get_decoder().layers
        _patch_mpt_decoder_for_flamingo(lang_encoder, decoder_layers_attr_name)
        
        # Try different init_flamingo signatures for compatibility
        # Different versions of open_flamingo have different parameters
        import inspect
        init_sig = inspect.signature(lang_encoder.init_flamingo)
        init_params = list(init_sig.parameters.keys())
        
        # Build kwargs based on available parameters
        kwargs = {
            "media_token_id": 0,
            "vis_hidden_size": vis_dim,
            "cross_attn_every_n_layers": cross_attn_every_n_layers,
        }
        
        # Add optional parameters if supported
        if "lang_hidden_size" in init_params:
            kwargs["lang_hidden_size"] = lang_dim
        if "gradient_checkpointing" in init_params:
            kwargs["gradient_checkpointing"] = False
        if "use_media_placement_augmentation" in init_params:
            kwargs["use_media_placement_augmentation"] = False
        if "residual" in init_params:
            kwargs["residual"] = False
        
        logger.info(f"Initializing Flamingo with params: {list(kwargs.keys())}")
        lang_encoder.init_flamingo(**kwargs)
        
        # Verify initialization
        if hasattr(lang_encoder, 'initialized_flamingo') and lang_encoder.initialized_flamingo:
            logger.info("Flamingo layers initialized successfully")
        else:
            # Manually set the flag if init succeeded but flag wasn't set
            lang_encoder.initialized_flamingo = True
            logger.info("Flamingo layers initialized (flag set manually)")
        
        return True
    except ImportError:
        logger.warning("open_flamingo not available. Cross-attention layers not initialized.")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize Flamingo layers: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def load_model(args) -> Dict[str, Any]:
    """
    Load RoboFlamingo model and components.
    
    Args:
        args: Configuration namespace with:
            - openflamingo_model: OpenFlamingo model name for auto-download
                Options: "3b", "3b-base", "4b", "4b-base", "9b",
                         "openflamingo-3b-instruct", etc.
            - clip_vision_encoder_path: CLIP ViT model name (default: "ViT-L-14")
            - clip_vision_encoder_pretrained: CLIP pretrained weights (default: "openai")
            - lang_encoder_path: Path to language model (overrides auto-download)
            - tokenizer_path: Path to tokenizer (overrides auto-download)
            - openflamingo_checkpoint: Path to checkpoint (overrides auto-download)
            - llm_name: LLM preset name for legacy compatibility
            - cache_dir: Cache directory for downloaded models
            - action_dim: Action dimension
            - state_dim: State dimension
            - window_size: Number of frames in window
            - decoder_type: Action decoder type
            - is_training: Whether loading for training
            - model_name_or_path: Path to checkpoint (for inference)
            
    Returns:
        Dict with 'model', 'tokenizer', 'image_processor' keys.
    """
    args.device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip is required. Install with: pip install open-clip-torch")
    
    # Check for inference mode
    if not getattr(args, 'is_training', True):
        return _load_for_inference(args)
    
    # === Training Mode ===
    
    # Check for auto-download mode
    openflamingo_model = getattr(args, 'openflamingo_model', None)
    openflamingo_checkpoint = getattr(args, 'openflamingo_checkpoint', None)
    cache_dir = getattr(args, 'cache_dir', None)
    
    # Auto-download if model name specified and no local checkpoint
    if openflamingo_model and not openflamingo_checkpoint:
        logger.info(f"Auto-downloading OpenFlamingo model: {openflamingo_model}")
        model_info = get_openflamingo_checkpoint(
            model_name=openflamingo_model,
            cache_dir=cache_dir,
            force_download=getattr(args, 'force_download', False),
        )
        openflamingo_checkpoint = model_info["checkpoint_path"]
        
        # Use model info for configuration if not explicitly set
        if not getattr(args, 'lang_encoder_path', None):
            args.lang_encoder_path = model_info["lang_encoder"]
        if not getattr(args, 'tokenizer_path', None):
            args.tokenizer_path = model_info["lang_encoder"]  # Usually same as lang_encoder
        if not getattr(args, 'cross_attn_every_n_layers', None):
            args.cross_attn_every_n_layers = model_info["cross_attn_every_n_layers"]
        if not getattr(args, 'clip_vision_encoder_path', None):
            args.clip_vision_encoder_path = model_info["vision_encoder"]
        if not getattr(args, 'clip_vision_encoder_pretrained', None):
            args.clip_vision_encoder_pretrained = model_info["vision_pretrained"]
    
    # Get LLM config from preset or args
    llm_name = getattr(args, 'llm_name', 'mpt_dolly_3b')
    if llm_name in MODEL_CONFIGS:
        llm_config = MODEL_CONFIGS[llm_name]
        lang_encoder_path = getattr(args, 'lang_encoder_path', None) or llm_config['lang_encoder_path']
        tokenizer_path = getattr(args, 'tokenizer_path', None) or llm_config['tokenizer_path']
        cross_attn_every_n_layers = getattr(args, 'cross_attn_every_n_layers', None) or llm_config['cross_attn_every_n_layers']
    else:
        lang_encoder_path = getattr(args, 'lang_encoder_path', None) or args.lang_encoder_path
        tokenizer_path = getattr(args, 'tokenizer_path', None) or lang_encoder_path
        cross_attn_every_n_layers = getattr(args, 'cross_attn_every_n_layers', 1)
    
    clip_vision_encoder_path = getattr(args, 'clip_vision_encoder_path', 'ViT-L-14')
    clip_vision_encoder_pretrained = getattr(args, 'clip_vision_encoder_pretrained', 'openai')
    
    # Load CLIP vision encoder
    logger.info(f"Loading CLIP vision encoder: {clip_vision_encoder_path}")
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,
        pretrained=clip_vision_encoder_pretrained,
    )
    vision_encoder.visual.output_tokens = True
    vis_dim = open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"]["width"]
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # Add special tokens
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|endofchunk|>", "<image>"]
    })
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    
    # Load language encoder
    logger.info(f"Loading language encoder from: {lang_encoder_path}")
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        trust_remote_code=True,
    )
    
    # Get language dimension
    if hasattr(lang_encoder.config, "d_model"):
        lang_dim = lang_encoder.config.d_model
    else:
        lang_dim = lang_encoder.config.hidden_size
    
    # Add embedding function for MPT if needed
    if "mpt-1b-redpajama" in lang_encoder_path:
        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings
        
        from open_flamingo.src.utils import extend_instance
        extend_instance(lang_encoder, EmbeddingFnMixin)
    
    # Resize token embeddings
    lang_encoder.resize_token_embeddings(len(tokenizer))
    
    # Initialize Flamingo cross-attention layers
    decoder_layers_attr_name = getattr(args, 'decoder_layers_attr_name', None)
    _init_flamingo_layers(lang_encoder, vis_dim, lang_dim, cross_attn_every_n_layers, decoder_layers_attr_name)
    
    # Create model config
    config = RoboFlamingoConfig(
        clip_vision_encoder_path=clip_vision_encoder_path,
        clip_vision_encoder_pretrained=clip_vision_encoder_pretrained,
        lang_encoder_path=lang_encoder_path,
        tokenizer_path=tokenizer_path,
        action_dim=getattr(args, 'action_dim', 6),
        state_dim=getattr(args, 'state_dim', 7),
        window_size=getattr(args, 'window_size', 12),
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        decoder_type=getattr(args, 'decoder_type', 'lstm'),
        decoder_hidden_size=getattr(args, 'decoder_hidden_size', 1024),
        decoder_num_layers=getattr(args, 'decoder_num_layers', 4),
        use_gripper=getattr(args, 'use_gripper', True),
        fusion_mode=getattr(args, 'fusion_mode', 'post'),
        use_state=getattr(args, 'use_state', False),
        multi_step_action=getattr(args, 'multi_step_action', 1),
        pooling=getattr(args, 'pooling', 'max'),
        freeze_vision=getattr(args, 'freeze_vision', True),
        freeze_embed=getattr(args, 'freeze_embed', False),
        llm_name=llm_name,
        sep_resampler=getattr(args, 'sep_resampler', False),
    )
    
    # Create model
    model = RoboFlamingoForPolicy(config)
    
    # Initialize modules
    model.initialize_modules(
        vision_encoder=vision_encoder,
        lang_encoder=lang_encoder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        vis_dim=vis_dim,
    )
    
    # Load OpenFlamingo checkpoint if provided
    if openflamingo_checkpoint:
        if os.path.exists(openflamingo_checkpoint):
            logger.info(f"Loading OpenFlamingo checkpoint from: {openflamingo_checkpoint}")
            checkpoint = torch.load(openflamingo_checkpoint, map_location='cpu')
            # Load perceiver and cross-attention weights
            _load_openflamingo_weights(model, checkpoint)
        else:
            logger.warning(f"Checkpoint not found: {openflamingo_checkpoint}")
    
    # Set requires_grad for training
    _set_requires_grad(model, args)
    
    # Move to device and dtype
    if getattr(args, 'bf16', False):
        model = model.to(dtype=torch.bfloat16, device=args.device)
    elif getattr(args, 'fp16', True):
        model = model.to(dtype=torch.float16, device=args.device)
    else:
        model = model.to(device=args.device)
    
    # Save config
    if hasattr(args, 'output_dir') and getattr(args, 'is_training', True):
        os.makedirs(args.output_dir, exist_ok=True)
        config.save_pretrained(args.output_dir)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"RoboFlamingo: {trainable_params:,} trainable / {total_params:,} total parameters")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'image_processor': image_processor,
    }


def _load_for_inference(args) -> Dict[str, Any]:
    """Load model for inference from checkpoint."""
    checkpoint_path = args.model_name_or_path
    
    # Load config
    config = RoboFlamingoConfig.from_pretrained(checkpoint_path)
    
    # Load components using config
    clip_vision_encoder_path = config.clip_vision_encoder_path
    clip_vision_encoder_pretrained = config.clip_vision_encoder_pretrained
    
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,
        pretrained=clip_vision_encoder_pretrained,
    )
    vision_encoder.visual.output_tokens = True
    vis_dim = open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"]["width"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|endofchunk|>", "<image>"]
    })
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    
    # Load language encoder
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        config.lang_encoder_path,
        trust_remote_code=True,
    )
    
    if hasattr(lang_encoder.config, "d_model"):
        lang_dim = lang_encoder.config.d_model
    else:
        lang_dim = lang_encoder.config.hidden_size
    
    lang_encoder.resize_token_embeddings(len(tokenizer))
    decoder_layers_attr_name = getattr(args, 'decoder_layers_attr_name', None)
    _init_flamingo_layers(lang_encoder, vis_dim, lang_dim, config.cross_attn_every_n_layers, decoder_layers_attr_name)
    
    # Create model
    model = RoboFlamingoForPolicy(config)
    model.initialize_modules(
        vision_encoder=vision_encoder,
        lang_encoder=lang_encoder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        vis_dim=vis_dim,
    )
    
    # Load checkpoint weights
    checkpoint_file = os.path.join(checkpoint_path, 'pytorch_model.bin')
    if os.path.exists(checkpoint_file):
        state_dict = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint from {checkpoint_file}")
    
    # Set up for inference
    model.data_processor = RoboFlamingoDataProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        window_size=config.window_size,
        use_gripper=config.use_gripper,
    )
    model.data_collator = RoboFlamingoDataCollator(
        tokenizer=tokenizer,
        window_size=config.window_size,
        use_gripper=config.use_gripper,
    )
    
    # Move to device
    model = model.to(device=args.device)
    model.eval()
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'image_processor': image_processor,
    }


def _set_requires_grad(model, args):
    """Set trainable parameters based on training arguments."""
    # Freeze all first
    model.requires_grad_(False)
    
    # Unfreeze perceiver
    if model.perceiver is not None:
        model.perceiver.requires_grad_(True)
    if model.perceiver_gripper is not None:
        model.perceiver_gripper.requires_grad_(True)
    
    # Unfreeze cross-attention layers
    if hasattr(model.lang_encoder, 'gated_cross_attn_layers'):
        model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    
    # Unfreeze embeddings if not frozen
    if not getattr(args, 'freeze_embed', False):
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
    
    # Unfreeze LM head
    if hasattr(model.lang_encoder, 'lm_head'):
        model.lang_encoder.lm_head.requires_grad_(True)
    
    # Unfreeze action decoder
    if model.action_decoder is not None:
        model.action_decoder.requires_grad_(True)
    
    # Unfreeze state FC if present
    if hasattr(model, 'state_fc') and model.state_fc is not None:
        model.state_fc.requires_grad_(True)
    
    # Optionally unfreeze vision encoder
    if not getattr(args, 'freeze_vision', True):
        model.vision_encoder.requires_grad_(True)


def get_data_processor(args, model_components: Dict[str, Any]):
    """
    Get data processor for RoboFlamingo.
    
    Args:
        args: Configuration namespace
        model_components: Dict from load_model
        
    Returns:
        RoboFlamingoDataProcessor instance
    """
    return RoboFlamingoDataProcessor(
        image_processor=model_components.get('image_processor'),
        tokenizer=model_components.get('tokenizer'),
        window_size=getattr(args, 'window_size', 12),
        use_gripper=getattr(args, 'use_gripper', True),
        rgb_pad=getattr(args, 'rgb_pad', 10),
        gripper_pad=getattr(args, 'gripper_pad', 4),
    )


def get_data_collator(args, model_components: Dict[str, Any]):
    """
    Get data collator for RoboFlamingo.
    
    Args:
        args: Configuration namespace
        model_components: Dict from load_model
        
    Returns:
        RoboFlamingoDataCollator instance
    """
    dtype = torch.float32
    if getattr(args, 'bf16', False):
        dtype = torch.bfloat16
    elif getattr(args, 'fp16', True):
        dtype = torch.float16
    
    return RoboFlamingoDataCollator(
        tokenizer=model_components.get('tokenizer'),
        window_size=getattr(args, 'window_size', 12),
        use_gripper=getattr(args, 'use_gripper', True),
        dtype=dtype,
    )


# Optional: Export Trainer if custom training logic needed
try:
    from vlastudio.policy.trainer import BaseTrainer
    from safetensors.torch import save_file, load_file
    
    class Trainer(BaseTrainer):
        """RoboFlamingo Trainer with custom loss computation."""
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            loss = outputs["loss"]
            
            # Log metrics
            logging_steps = self.args.logging_steps
            if (self.state.global_step % logging_steps == 0) and (self.state.global_step != 0):
                log_dict = {}
                if "action_loss" in outputs:
                    log_dict["action_loss"] = outputs["action_loss"].detach().cpu().item()
                if "gripper_loss" in outputs:
                    log_dict["gripper_loss"] = outputs["gripper_loss"].detach().cpu().item()
                if log_dict:
                    self.log(log_dict)
            
            return (loss, outputs) if return_outputs else loss
        
        def save_model(self, output_dir=None, _internal_call=False):
            """Save model."""
            output_dir = output_dir or self.args.output_dir
            super().save_model(output_dir, _internal_call)
            
            # Save config
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.config.save_pretrained(output_dir)
            
except ImportError:
    # BaseTrainer not available, skip Trainer export
    pass

