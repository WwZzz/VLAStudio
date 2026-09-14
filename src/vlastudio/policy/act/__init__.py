from .act import ACTPolicy, ACTPolicyConfig
from .data_utils import data_collator
import torch
from transformers import AutoConfig

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

def load_model(args):
    if not args.is_training:
        model = ACTPolicy.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model.to('cuda')
        # Only set collator, no processor needed (samples already in correct format)
        model.data_collator = data_collator
    else:
        model_args = getattr(args, 'model_args', {})
        config = ACTPolicyConfig(**model_args) 
        model = ACTPolicy(config=config)
    # model.to(dtype=torch.float32, device=args.device)
    return {'model': model}

def get_data_collator(args, model_components):
    return data_collator