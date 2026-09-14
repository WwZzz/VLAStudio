from .modeling import OpenConfig, OpenPolicy
from .data_utils import OpenVLAProcessor, OpenVLACollator
from .trainer import Trainer
from peft import LoraConfig, get_peft_model, PeftModel

def load_model(args):
    """Load OpenVLA model components."""
    if not args.is_training:
        config = OpenConfig.from_pretrained(args.model_name_or_path)
        if config.training_mode == "lora":  
            base_model = OpenPolicy(config)
            model = PeftModel.from_pretrained(base_model, args.model_name_or_path)
        else:
            model = OpenPolicy.from_pretrained(args.model_name_or_path, config=config)
        model.to('cuda')
    else:
        config = OpenConfig(
            training_mode=getattr(args, 'training_mode', 'lora'),
            lora_r=getattr(args, 'lora_r', 16),
            lora_alpha=getattr(args, 'lora_alpha', 32),
            lora_dropout=getattr(args, 'lora_dropout', 0.1),
            use_quantization=getattr(args, 'use_quantization', False),
            max_length=getattr(args, 'max_length', 2048),
            state_dim=getattr(args, 'state_dim', 14),
            action_dim=getattr(args, 'action_dim', 14),
            camera_names=getattr(args, 'camera_names', ['primary']),
            pretrained_weight_path=getattr(args, 'pretrained_weight_path', 'openvla/openvla-7b'),
        )
        model = OpenPolicy(config)
        if config.training_mode == "lora":
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=min(config.lora_alpha, 16),
                lora_dropout=config.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
    
    # Initialize data_processor and data_collator for inference
    if not args.is_training:
        image_transform = model.processor.image_processor.apply_transform
        model.data_processor = OpenVLAProcessor(
            tokenizer=model.tokenizer,
            image_transform=image_transform
        )
        model.data_collator = OpenVLACollator(model.tokenizer)
    
    return {
        'model': model,
        'tokenizer': model.tokenizer,
        'processor': model.processor
    }

def get_data_collator(args, model_components):
    """Get data collator for OpenVLA."""
    tokenizer = model_components['tokenizer']
    return OpenVLACollator(tokenizer)

def get_data_processor(args, model_components):
    """Get data processor for OpenVLA."""
    tokenizer = model_components['tokenizer']
    image_transform = model_components['processor'].image_processor.apply_transform
    return OpenVLAProcessor(
        tokenizer=tokenizer,
        image_transform=image_transform
    )

