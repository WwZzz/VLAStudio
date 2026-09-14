import os
import warnings
import torch.nn as nn
import torch
import numpy as np
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from .policy import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from collections import OrderedDict
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
import torchvision.transforms as transforms

class DiffusionPolicyConfig(PretrainedConfig):
    """
    Configuration class for the DiffusionPolicy model.
    This class manages all hyperparameters related to the model's architecture.
    """
    def __init__(
        self,
        action_dim=7,               # Action dimensionality
        state_dim=7,                # State dimensionality
        camera_names=['primary'],             # List of names for input cameras
        image_sizes=['(256, 256)'],
        observation_horizon=1,      # Number of timesteps in observation
        chunk_size=64,       # Number of timesteps in prediction horizon
        num_inference_timesteps=10,  # Timesteps for inference in noise scheduler
        ema_power=0.75,                # Power for EMA updates
        feature_dimension=64,     # Feature dimensionality for camera embeddings, default is 64
        num_kp=32,                # Number of keypoints extracted using SpatialSoftmax, e.g., default is 32
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        # variance_type='fixed_small', # Yilun's paper uses fixed_small_log instead, but easy to cause Nan
        clip_sample=True, # required when predict_epsilon=False
        clip_sample_range=1.0,
        prediction_type='epsilon', # or sample
        **kwargs
    ):
        super().__init__(**kwargs)
        self.camera_names = camera_names
        self.observation_horizon = observation_horizon
        self.prediction_horizon = chunk_size
        self.num_inference_timesteps = num_inference_timesteps
        self.ema_power = ema_power
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.feature_dimension = feature_dimension
        self.num_kp = num_kp
        self.image_sizes = image_sizes
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        # self.variance_type = variance_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.prediction_type = prediction_type

# Converted model
class DiffusionPolicyModel(PreTrainedModel):
    # Default configuration class
    config_class = DiffusionPolicyConfig

    def __init__(self, config):
        super().__init__(config)
        self.camera_names = config.camera_names
        self.observation_horizon = config.observation_horizon
        self.prediction_horizon = config.prediction_horizon
        self.num_kp = config.num_kp
        self.feature_dimension = config.feature_dimension
        self.ac_dim = config.action_dim
        self.obs_dim = self.feature_dimension * len(self.camera_names) + config.state_dim

        # Build model structure
        backbones = []
        pools = []
        linears = []
        for cam_id, cam_name in enumerate(self.camera_names):
            img_size = self.config.image_sizes[cam_id]
            if isinstance(img_size, str): img_size = eval(img_size)
            # Suppress torchvision pretrained parameter deprecation warning from robomimic
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
                backbones.append(ResNet18Conv(input_channel=3, pretrained=True, input_coord_conv=False))
            feat_shape = list(backbones[-1](torch.rand((1, 3, img_size[1], img_size[0])))[0].shape)
            # feat_shape = [512, 8, 8]
            pools.append(SpatialSoftmax(input_shape=feat_shape, num_kp=self.num_kp, temperature=1.0,
                                        learnable_temperature=False, noise_std=0.0))
            linears.append(nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
        
        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)
        backbones = replace_bn_with_gn(backbones)

        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim * self.observation_horizon,
        )

        self.nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'noise_pred_net': noise_pred_net,
            })
        }).float().cuda()

        # Use EMA
        self.ema = EMAModel(self.nets.parameters(), power=config.ema_power)
        # self.ema = EMAModel(model=self.nets, power=config.ema_power)

        # Initialize noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            beta_schedule=self.config.beta_schedule,
            # variance_type=self.config.variance_type,
            clip_sample=self.config.clip_sample,
            clip_sample_range=self.config.clip_sample_range,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type=self.config.prediction_type,
        )

        self.init_weights()

    def init_weights(self):
        """Initialize weights for all modules"""
        for module in self.nets.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        During training, input `qpos`, `image`, and `actions`, returns loss.
        During inference, only input `qpos`, `image`, returns denoised action sequence.
        """
        # device = nets['policy']['linears'][0].weight.device
        # qpos = qpos.to(device)
        # image = image.to(device)
        # actions = actions.to(device)
        if image is not None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normalize(image)
        B = qpos.shape[0]
        nets = self.nets

        # Camera feature extraction
        all_features = []
        for cam_id in range(len(self.camera_names)):
            cam_image = image[:, cam_id]
            cam_features = nets['policy']['backbones'][cam_id](cam_image)
            pool_features = nets['policy']['pools'][cam_id](cam_features)
            pool_features = torch.flatten(pool_features, start_dim=1)
            out_features = nets['policy']['linears'][cam_id](pool_features)
            all_features.append(out_features)
        if len(all_features)>0:
            obs_cond = torch.cat(all_features + [qpos.squeeze(1)], dim=1)
        else:
            obs_cond = qpos.squeeze(1)

        if actions is not None:  # Training mode
            # Add noise
            noise = torch.randn(actions.shape, device=obs_cond.device)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=obs_cond.device
            ).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

            # Calculate L2 loss
            all_l2 = nn.functional.mse_loss(noise_pred, noise, reduction='none')
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            return {'loss': loss}
        else:  # Inference mode
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            noisy_action = torch.randn((B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action

            self.noise_scheduler.set_timesteps(self.config.num_inference_timesteps)
            for k in self.noise_scheduler.timesteps:
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
            return naction
        
    def select_action(self, batch_obs):
        """
        Inference action from processed batch observation.
        
        Args:
            batch_obs: Processed and collated batch from meta2obs
        
        Returns:
            Action predictions
        """
        # Move tensors to device and get data
        device = next(self.parameters()).device
        image = batch_obs['image'].to(device)
        state = batch_obs['qpos'].to(device)
        # Forward pass
        action = self.forward(state, image, None, None)
        return action
    
    def save_pretrained(self, save_directory, state_dict=None, *args, **kwargs):
        """
        Save model weights and configuration to specified directory.
        
        Args:
            save_directory (str): Directory path to save the model.
            state_dict (dict, optional): Model weights. If None, get from `self.state_dict()`.
            *args, **kwargs: Other parameters passed by `Trainer` for compatibility.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration file to {save_directory}/config.json
        self.config.save_pretrained(save_directory)

        # If state_dict not provided, get from model
        if state_dict is None:
            state_dict = self.state_dict()

        # Save model weights to {save_directory}/pytorch_model.bin
        model_save_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(state_dict, model_save_path)

        # If EMA is also in use, we save the EMA averaged weights
        if self.ema is not None:
            ema_save_path = os.path.join(save_directory, "ema_model.bin")
            torch.save(self.ema.state_dict(), ema_save_path)
            
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load pretrained model and its configuration.
        """
        # 1. Load configuration file
        config = DiffusionPolicyConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # 2. Initialize model
        model = cls(config)

        # 3. Load model weights
        model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        # 4. Load EMA weights (if exists)
        ema_path = os.path.join(pretrained_model_name_or_path, "ema_model.bin")
        if os.path.isfile(ema_path) and model.ema is not None:
            ema_state_dict = torch.load(ema_path, map_location="cpu")
            model.ema.load_state_dict(ema_state_dict)

        return model

def replace_submodules(
        root_module, 
        predicate, 
        func):
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module
