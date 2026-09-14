from .modeling import OctoConfig, OctoPolicy
from .data_utils import OctoDataProcessor, OctoCollator

def load_model(args):
    if not args.is_training:
        model = OctoPolicy.from_pretrained(args.model_name_or_path)
        model.data_processor =  OctoDataProcessor(model.text_processor, model.config.use_wrist, model.config.image_size)
        model.data_collator = OctoCollator()
    else:
        model_args = getattr(args, 'model_args', {})
        config = OctoConfig(**model_args) 
        model = OctoPolicy(config=config)
    model.to('cuda')
    return {"model": model, 'text_processor': model.text_processor}

def get_data_processor(args, model_components):
    return OctoDataProcessor(model_components['text_processor'], model_components['model'].config.use_wrist, model_components['model'].config.image_size, model_components['model'].config.wrist_image_size)

def get_data_collator(args, model_components):
    return OctoCollator()