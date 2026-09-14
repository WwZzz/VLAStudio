import os
import json
from .data_utils import SmolVLAProcess, SmolVLADataCollator
from .modeling import SmolVLAPolicy, SmolVLAPolicyConfig
from transformers import AutoTokenizer, AutoConfig, AutoModelForImageTextToText
from .trainer import Trainer 

def load_model(args):
    if not args.is_training:
        config = SmolVLAPolicyConfig.from_pretrained(args.model_name_or_path)
        if config.load_vlm_weights:
            vlm = AutoModelForImageTextToText.from_pretrained(
                config.vlm_model_name,
                device_map=config.device,
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            if config.num_vlm_layers>0:
                vlm.model.text_model.layers = vlm.model.text_model.layers[:config.num_vlm_layers]
            config.load_vlm_weights = False
        else:
            vlm = None
        model = SmolVLAPolicy.from_pretrained(args.model_name_or_path, config=config)
        # if vlm is not None:
        #     model.model.vlm_with_expert.vlm.load_state_dict(vlm.state_dict())
        tokenizer = AutoTokenizer.from_pretrained(model.config.vlm_model_name)
        data_processor = SmolVLAProcess()
        data_collator = SmolVLADataCollator(tokenizer=tokenizer, max_state_dim=model.config.max_state_dim, max_action_dim=model.config.max_action_dim, resize_imgs_with_padding=model.config.resize_imgs_with_padding, max_length=model.config.tokenizer_max_length, padding=model.config.pad_language_to)
        model.data_processor = data_processor
        model.data_collator = data_collator
        model.tokenizer = tokenizer
    else:
        model_args = getattr(args, 'model_args', {})
        config = SmolVLAPolicyConfig(**model_args) 
        model = SmolVLAPolicy(config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.vlm_model_name)
    model.to('cuda')
    return {'model': model, 'tokenizer': tokenizer}

def get_data_processor(args, model_components):
    return SmolVLAProcess()

def get_data_collator(args, model_components):
    return SmolVLADataCollator(tokenizer=model_components['tokenizer'], max_state_dim=args.max_state_dim, max_action_dim=args.max_action_dim, resize_imgs_with_padding=args.resize_imgs_with_padding, max_length=args.tokenizer_max_length, padding=args.pad_language_to)