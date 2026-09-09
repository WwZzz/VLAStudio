from .act import ACTPolicy, ACTPolicyConfig
from .data_utils import data_collator, ACTDataProcessor
import json
import re
from pathlib import Path
import torch
from transformers import AutoConfig
from loguru import logger

# Lazy import Trainer to avoid segfault when transformers.trainer conflicts with TensorFlow/CUDA
# Trainer is only needed for training, not inference
def get_trainer_class():
    from .trainer import Trainer
    return Trainer

# For backwards compatibility, create a lazy proxy
class _LazyTrainer:
    _trainer_class = None
    
    def __new__(cls, *args, **kwargs):
        if cls._trainer_class is None:
            cls._trainer_class = get_trainer_class()
        return cls._trainer_class(*args, **kwargs)

Trainer = _LazyTrainer


def _load_policy_metadata(checkpoint_path):
    if not checkpoint_path:
        return {}
    meta_path = Path(checkpoint_path) / "policy_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"[ACT] Failed to read {meta_path}: {exc}")
        return {}


def _example_data_image_size(checkpoint_path):
    """Return (width, height) from example_data, or None if unavailable."""
    if not checkpoint_path:
        return None
    ckpt = Path(checkpoint_path)
    info_path = ckpt / "example_data" / "info.txt"
    if info_path.exists():
        text = info_path.read_text(encoding="utf-8")
        match = re.search(r"image: shape=torch\.Size\(\[2, 3, (\d+), (\d+)\]\)", text)
        if match:
            height, width = int(match.group(1)), int(match.group(2))
            return width, height
    cam_path = ckpt / "example_data" / "camera_0.png"
    if cam_path.exists():
        try:
            from PIL import Image

            with Image.open(cam_path) as img:
                return img.size
        except Exception as exc:
            logger.warning(f"[ACT] Failed to read {cam_path}: {exc}")
    return None


def _should_skip_act_resize(args, metadata, example_size):
    """Skip letterbox only for native-resolution training (example == raw_size)."""
    if args is not None and getattr(args, "act_no_resize", False):
        return True
    if metadata.get("inference_resize") is False:
        return True
    if metadata.get("inference_resize") is True:
        return False
    raw_size = metadata.get("raw_size")
    if raw_size and example_size and list(example_size) == list(raw_size):
        return True
    return False


def _make_act_data_processor(args, checkpoint_path=None):
    checkpoint_path = checkpoint_path or getattr(args, "model_name_or_path", None)
    metadata = _load_policy_metadata(checkpoint_path)
    example_size = _example_data_image_size(checkpoint_path)
    if _should_skip_act_resize(args, metadata, example_size):
        if example_size:
            logger.info(
                f"[ACT] inference resize disabled "
                f"(example_data={example_size[0]}x{example_size[1]})"
            )
        else:
            logger.info("[ACT] inference resize disabled (checkpoint metadata)")
        return None

    image_size = (
        getattr(args, "image_size", None)
        or metadata.get("image_size")
        or [256, 256]
    )
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
    return ACTDataProcessor(image_size=tuple(image_size), pad_value=0)


def load_model(args):
    if not args.is_training:
        model = ACTPolicy.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        device = getattr(args, "device", "cuda")
        model.to(device)
        model.data_processor = _make_act_data_processor(args, args.model_name_or_path)
        model.data_collator = data_collator
        if model.data_processor is None:
            logger.info("[ACT] data_processor=none (native camera resolution)")
        else:
            logger.info(
                f"[ACT] data_processor=resize_with_pad "
                f"image_size={model.data_processor.width}x{model.data_processor.height} "
                f"pad_value={model.data_processor.pad_value}"
            )
    else:
        model_args = getattr(args, 'model_args', {})
        config = ACTPolicyConfig(**model_args) 
        model = ACTPolicy(config=config)
    # model.to(dtype=torch.float32, device=args.device)
    return {'model': model}

def get_data_collator(args, model_components):
    return data_collator


def get_data_processor(args, model_components):
    model = model_components.get("model")
    if model is not None and getattr(model, "data_processor", None) is not None:
        return model.data_processor
    return _make_act_data_processor(args, getattr(args, "model_name_or_path", None))