from .diffusion_policy import DiffusionPolicyConfig, DiffusionPolicyModel
from .utils import data_collator
import torch
from .trainer import Trainer
import torchvision.transforms as transforms

def load_model(args):
    if args.is_training:
        # Use unified image_size for all cameras
        image_sizes = [args.image_size for _ in args.camera_names]
        config = DiffusionPolicyConfig(
            camera_names=args.camera_names, 
            image_sizes=image_sizes,
            observation_horizon=1, 
            action_dim=args.action_dim, 
            state_dim = args.state_dim, 
            chunk_size = args.chunk_size,
            clip_sample = args.model_args.get('clip_sample', True),
            clip_sample_range = args.model_args.get('clip_sample_range', 1.0),
        ) 
        model = DiffusionPolicyModel(config=config)
    else:
        model = DiffusionPolicyModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model_args = getattr(args, 'model_args', {})
        # ema
        if model_args.get('using_ema', False):
            model.ema.copy_to(model.parameters()) # using ema for testing
        model.config.num_inference_timesteps = model_args.get('num_inference_steps', 10)
        model.to('cuda')
        # Only set collator, no processor needed (samples already in correct format)
        model.data_collator = data_collator
    # model.to(dtype=torch.float32, device=args.device)
    return {'model': model}

def get_data_collator(args, model_components):
    return data_collator

def data_collator_wrapper(args, model_components):
    """Wrapper for data_collator that matches the expected signature."""
    return data_collator

def StateToQposDataCollator(dataset, args, model_components):
    """Data collator wrapper that maps 'state' to 'qpos' for diffusion policy compatibility."""
    def wrapper(instances):
        # First apply the original data collator
        batch = data_collator(instances)
        # The data collator already maps state to qpos, so we just return it
        return batch
    return wrapper

class DataTransform:
    def __init__(self, img_size, ratio=0.95):
        self.transformations = transforms.Compose([
            transforms.RandomCrop(size=[int(img_size[1] * ratio), int(img_size[0] * ratio)]),
            transforms.Resize((img_size[1], img_size[0]), antialias=True),
            transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
            transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) #, hue=0.08)
        ])
    
    def __call__(self, sample):
        # if 'image' in sample and sample['image'] is not None:
        sample['image'] = sample['image']/255.0
        return sample
    

# def get_data_processor(args, model_components):
#     img_size = eval(args.image_size)
#     return DataTransform(img_size, 0.95)

def StateToQposProcessor(dataset, args, model_components):
    """Data processor that maps 'state' to 'qpos' for diffusion policy compatibility."""
    class Processor:
        def __call__(self, sample):
            # Map 'state' to 'qpos' for diffusion policy compatibility
            if 'state' in sample:
                sample['qpos'] = sample['state']
                # Keep the original 'state' field for the data collator
                # sample['state'] = sample['state']  # Already exists
            return sample
    return Processor()

