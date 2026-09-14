import json
import logging
import os

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .action_hub import build_action_space
from .soft_transformer import SoftPromptedTransformer

ACTION = "action"
OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"


class XVLAConfig(PretrainedConfig):
    model_type = "xvla"

    def __init__(
        self,
        pretrained_weight_path=None,
        base_vlm_model_name_or_path="microsoft/Florence-2-large",
        n_obs_steps=1,
        chunk_size=32,
        n_action_steps=32,
        dtype="float32",
        florence_config=None,
        tokenizer_name="facebook/bart-large",
        tokenizer_max_length=64,
        tokenizer_padding_side="right",
        pad_language_to="max_length",
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_domains=30,
        len_soft_prompts=32,
        dim_time=32,
        max_len_seq=512,
        use_hetero_proj=False,
        action_mode="ee6d",
        num_denoising_steps=10,
        use_proprio=True,
        max_state_dim=32,
        max_action_dim=20,
        domain_feature_key=None,
        resize_imgs_with_padding=None,
        num_image_views=None,
        empty_cameras=0,
        freeze_vision_encoder=False,
        freeze_language_encoder=False,
        train_policy_transformer=True,
        train_soft_prompts=True,
        optimizer_lr=1e-4,
        optimizer_betas=(0.9, 0.99),
        optimizer_eps=1e-8,
        optimizer_weight_decay=0.0,
        optimizer_grad_clip_norm=10.0,
        optimizer_soft_prompt_lr_scale=1.0,
        optimizer_soft_prompt_warmup_lr_scale=None,
        scheduler_warmup_steps=1000,
        scheduler_decay_steps=30000,
        scheduler_decay_lr=2.5e-6,
        camera_names=None,
        state_dim=0,
        action_dim=0,
        device="cuda",
        lora_module=None,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_bias="none",
        lora_modules_to_save=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_weight_path = pretrained_weight_path
        self.base_vlm_model_name_or_path = base_vlm_model_name_or_path
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.dtype = dtype
        self.florence_config = florence_config or {}
        self.tokenizer_name = tokenizer_name
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer_padding_side = tokenizer_padding_side
        self.pad_language_to = pad_language_to
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_domains = num_domains
        self.len_soft_prompts = len_soft_prompts
        self.dim_time = dim_time
        self.max_len_seq = max_len_seq
        self.use_hetero_proj = use_hetero_proj
        self.action_mode = action_mode
        self.num_denoising_steps = num_denoising_steps
        self.use_proprio = use_proprio
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.domain_feature_key = domain_feature_key
        self.resize_imgs_with_padding = tuple(resize_imgs_with_padding) if resize_imgs_with_padding else None
        self.num_image_views = num_image_views
        self.empty_cameras = empty_cameras
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_language_encoder = freeze_language_encoder
        self.train_policy_transformer = train_policy_transformer
        self.train_soft_prompts = train_soft_prompts
        self.optimizer_lr = optimizer_lr
        self.optimizer_betas = tuple(optimizer_betas)
        self.optimizer_eps = optimizer_eps
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_grad_clip_norm = optimizer_grad_clip_norm
        self.optimizer_soft_prompt_lr_scale = optimizer_soft_prompt_lr_scale
        self.optimizer_soft_prompt_warmup_lr_scale = optimizer_soft_prompt_warmup_lr_scale
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.scheduler_decay_steps = scheduler_decay_steps
        self.scheduler_decay_lr = scheduler_decay_lr
        self.camera_names = list(camera_names or ["primary"])
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.lora_module = list(lora_module) if lora_module else []
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.lora_modules_to_save = list(lora_modules_to_save) if lora_modules_to_save else []

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        if os.path.isdir(str(pretrained_name_or_path)):
            config_path = os.path.join(str(pretrained_name_or_path), "config.json")
        else:
            config_path = hf_hub_download(str(pretrained_name_or_path), "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("type", None)
        data.update(kwargs)
        return cls(**data)


class XVLAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size = config.chunk_size
        self.use_proprio = config.use_proprio
        if config.action_mode.lower() == "auto":
            real_dim = config.action_dim if config.action_dim > 0 else config.max_action_dim
            self.action_space = build_action_space("auto", real_dim=real_dim, max_dim=config.max_action_dim)
        else:
            self.action_space = build_action_space(config.action_mode.lower())
        self.dim_action = self.action_space.dim_action
        self.dim_proprio = config.max_state_dim if config.use_proprio else 0
        self.vlm = self._build_florence_model(config)
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head
        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim`.")
        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=self.dim_action,
            dim_propio=self.dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )
        self._apply_freezing()
        self._apply_dtype()

    def _build_florence_model(self, config):
        base_cfg = AutoConfig.from_pretrained(config.base_vlm_model_name_or_path, trust_remote_code=True)
        if config.florence_config:
            florence_cfg = type(base_cfg).from_dict(config.florence_config)
        else:
            florence_cfg = base_cfg
        model_ref = base_cfg.auto_map["AutoModelForCausalLM"]
        model_cls = get_class_from_dynamic_module(model_ref, config.base_vlm_model_name_or_path)
        return model_cls(florence_cfg)

    def _get_target_dtype(self):
        return torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32

    def _apply_dtype(self):
        self.to(dtype=self._get_target_dtype())

    def _apply_freezing(self):
        if self.config.freeze_vision_encoder and hasattr(self.vlm, "vision_tower"):
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = False
        if self.config.freeze_language_encoder and hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "encoder"):
                for param in lm.model.encoder.parameters():
                    param.requires_grad = False
            if hasattr(lm, "model") and hasattr(lm.model, "shared"):
                for param in lm.model.shared.parameters():
                    param.requires_grad = False
        if not self.config.train_policy_transformer:
            for name, param in self.transformer.named_parameters():
                if "soft_prompt" not in name:
                    param.requires_grad = False
        if not self.config.train_soft_prompts and hasattr(self.transformer, "soft_prompt_hub"):
            for param in self.transformer.soft_prompt_hub.parameters():
                param.requires_grad = False

    def forward_vlm(self, input_ids, pixel_values, image_mask):
        batch_size, num_views = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(dtype=torch.bool)
        flat_images = pixel_values.flatten(0, 1)
        valid_images = flat_images[flat_mask]
        valid_feats = self.vlm._encode_image(valid_images)
        tokens_per_view, hidden_dim = valid_feats.shape[1:]
        image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],
            inputs_embeds,
        )
        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]
        aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    def forward(self, input_ids, image_input, image_mask, domain_id, proprio, action):
        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        action = action.to(dtype=target_dtype)
        enc = self.forward_vlm(input_ids, image_input, image_mask)
        batch_size = input_ids.shape[0]
        t = (
            torch.rand(1, device=input_ids.device, dtype=target_dtype)
            + torch.arange(batch_size, device=input_ids.device, dtype=target_dtype) / batch_size
        ) % (1 - 1e-5)
        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)
        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )
        return self.action_space.compute_loss(pred_action, action)

    @torch.no_grad()
    def generate_actions(self, input_ids, image_input, image_mask, domain_id, proprio, steps):
        self.eval()
        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        enc = self.forward_vlm(input_ids, image_input, image_mask)
        batch_size = input_ids.shape[0]
        action_dim = self.dim_action
        x1 = torch.randn(batch_size, self.chunk_size, action_dim, device=proprio.device, dtype=target_dtype)
        action = torch.zeros_like(x1)
        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((batch_size,), i / steps, device=proprio.device, dtype=target_dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        return self.action_space.postprocess(action)


class XVLAPolicy(PreTrainedModel):
    config_class = XVLAConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.model = XVLAModel(config)
        self.reset()

    def reset(self):
        pass

    def get_optim_params(self):
        return dict(filter(lambda kv: kv[1].requires_grad, self.named_parameters()))

    def _policy_device(self):
        return next(self.parameters()).device

    def _prepare_state(self, batch, batch_size, device):
        if not self.config.use_proprio or OBS_STATE not in batch:
            return torch.zeros(batch_size, 0, device=device)
        state = batch[OBS_STATE]
        if state.ndim > 2:
            state = state[:, -1, :]
        return pad_vector(state, self.model.dim_proprio)

    def _prepare_images(self, batch):
        present_img_keys = [f"{OBS_IMAGES}.{name}" for name in self.config.camera_names if f"{OBS_IMAGES}.{name}" in batch]
        if len(present_img_keys) == 0:
            present_img_keys = [key for key in batch if key.startswith(f"{OBS_IMAGES}.")]
        if len(present_img_keys) == 0:
            raise ValueError(f"All image features are missing from the batch. Batch keys: {list(batch.keys())}")
        images = []
        masks = []
        for key in present_img_keys:
            img = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding)
            images.append(img)
            masks.append(torch.ones(img.size(0), dtype=torch.bool, device=img.device))
        stacked_imgs = torch.stack(images, dim=1)
        stacked_masks = torch.stack(masks, dim=1)
        total_views = self.config.num_image_views or stacked_imgs.size(1)
        total_views = max(total_views, stacked_imgs.size(1))
        num_pad = total_views - stacked_imgs.size(1)
        if num_pad > 0:
            pad_shape = (stacked_imgs.size(0), num_pad, *stacked_imgs.shape[2:])
            stacked_imgs = torch.cat([stacked_imgs, stacked_imgs.new_zeros(pad_shape)], dim=1)
            stacked_masks = torch.cat([stacked_masks, stacked_masks.new_zeros((stacked_masks.size(0), num_pad))], dim=1)
        return stacked_imgs, stacked_masks

    def _get_domain_id(self, batch, batch_size, device):
        candidate = None
        if self.config.domain_feature_key and self.config.domain_feature_key in batch:
            candidate = batch[self.config.domain_feature_key]
        elif "domain_id" in batch:
            candidate = batch["domain_id"]
        if candidate is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if not isinstance(candidate, torch.Tensor):
            candidate = torch.as_tensor(candidate, device=device)
        else:
            candidate = candidate.to(device=device)
        if candidate.ndim == 0:
            candidate = candidate.expand(batch_size)
        if candidate.ndim > 1:
            candidate = candidate.view(candidate.shape[0], -1)[:, 0]
        if candidate.shape[0] != batch_size:
            candidate = candidate.expand(batch_size)
        return candidate.to(dtype=torch.long)

    def _prepare_action_targets(self, batch):
        if ACTION not in batch:
            raise ValueError("Batch is missing action targets required for training.")
        actions = batch[ACTION]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = pad_tensor_along_dim(actions, self.config.chunk_size, dim=1)
        if actions.shape[-1] != self.model.dim_action:
            actions = pad_vector(actions, self.model.dim_action)
        return actions

    def _build_model_inputs(self, batch):
        device = self._policy_device()
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        batch_size = input_ids.shape[0]
        images, image_mask = self._prepare_images(batch)
        domain_id = self._get_domain_id(batch, batch_size, device)
        proprio = self._prepare_state(batch, batch_size, device)
        return {
            "input_ids": input_ids.to(device=device),
            "image_input": images.to(device=device),
            "image_mask": image_mask.to(device=device),
            "domain_id": domain_id,
            "proprio": proprio.to(device=device),
        }

    def forward(self, batch=None, **kwargs):
        if batch is None:
            batch = kwargs
        inputs = self._build_model_inputs(batch)
        targets = self._prepare_action_targets(batch).to(device=self._policy_device())
        losses = self.model(action=targets, **inputs)
        total_loss = sum(losses.values())
        log_dict = {k: v.detach().item() for k, v in losses.items()}
        log_dict["loss"] = total_loss.detach().item()
        return total_loss, log_dict

    def _get_action_chunk(self, batch):
        inputs = self._build_model_inputs(batch)
        return self.model.generate_actions(**inputs, steps=self.config.num_denoising_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch=None, noise=None, **kwargs):
        if batch is None:
            batch = kwargs
        self.eval()
        return self._get_action_chunk(batch)

    @torch.no_grad()
    def select_action(self, batch=None, noise=None, **kwargs):
        if batch is None:
            batch = kwargs
        self.eval()
        actions = self._get_action_chunk(batch)
        n = min(actions.shape[1], self.config.n_action_steps)
        return actions[:, :n, :]

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, config=None, revision=None, **kwargs):
        if config is None:
            config = XVLAConfig.from_pretrained(pretrained_name_or_path)
        instance = cls(config)
        model_id = str(pretrained_name_or_path)
        if os.path.isdir(model_id):
            model_file = os.path.join(model_id, "model.safetensors")
        else:
            try:
                model_file = hf_hub_download(repo_id=model_id, filename="model.safetensors", revision=revision)
            except HfHubHTTPError as e:
                raise FileNotFoundError(f"model.safetensors not found on the Hub at {model_id}") from e
        logging.info("Loading XVLA checkpoint from %s", model_file)
        state_dict = load_file(model_file)
        encoder_key = "model.vlm.language_model.model.encoder.embed_tokens.weight"
        shared_key = "model.vlm.language_model.model.shared.weight"
        # Florence/BART-style checkpoints may store only one of tied embed weights.
        if encoder_key in state_dict and shared_key not in state_dict:
            state_dict[shared_key] = state_dict[encoder_key]
        if shared_key in state_dict and encoder_key not in state_dict:
            state_dict[encoder_key] = state_dict[shared_key]
        instance.load_state_dict(state_dict, strict=True)
        instance.model._apply_dtype()
        instance.to(config.device)
        instance.eval()
        return instance


def resize_with_pad(img, height, width, pad_value=0.0):
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")
    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


def pad_vector(vector, new_dim):
    if vector.shape[-1] == new_dim:
        return vector
    if new_dim == 0:
        shape = list(vector.shape)
        shape[-1] = 0
        return vector.new_zeros(*shape)
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = vector.new_zeros(*shape)
    length = min(current_dim, new_dim)
    new_vector[..., :length] = vector[..., :length]
    return new_vector


def pad_tensor_along_dim(tensor, target_len, dim=1):
    current_len = tensor.size(dim)
    if current_len == target_len:
        return tensor
    if current_len > target_len:
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(0, target_len)
        return tensor[tuple(slices)]
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_len - current_len
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=dim)
