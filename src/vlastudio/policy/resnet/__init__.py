from .cnnmlp import CNNMLPPolicy, CNNMLPPolicyConfig
from .data_utils import data_collator
import torch
from transformers import AutoConfig
from .trainer import Trainer

def load_model(args):
    if not args.is_training:
        model = CNNMLPPolicy.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model.to('cuda')
        # Only set collator, no processor needed (samples already in correct format)
        model.data_collator = data_collator
    else:
        model_args = getattr(args, 'model_args', {})
        config = CNNMLPPolicyConfig(camera_names=args.camera_names, state_dim = args.state_dim, action_dim=args.action_dim, chunk_size=args.chunk_size, **model_args) 
        model = CNNMLPPolicy(config=config)
    return {'model': model}

def get_data_collator(args, model_components):
    return data_collator


