import os

import torch
from transformers import AutoTokenizer

from .data_utils import XVLADataCollator, XVLADataProcessor
from .modeling import XVLAConfig, XVLAPolicy
from .trainer import Trainer


def find_all_linear_names(model, lora_module=None):
    cls = torch.nn.Linear
    lora_module = lora_module or []
    names = set()
    for name, module in model.named_modules():
        if any(k in name for k in lora_module) and isinstance(module, cls):
            names.add(name)
    return list(names)


def _get_xvla_config(model):
    """Get XVLAConfig regardless of whether model is wrapped in PeftModel."""
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            return model.get_base_model().config
    except ImportError:
        pass
    return model.config


def _get_pretrained_source(args):
    for key in ("model_name_or_path", "pretrained_weight_path", "pretrained_model_name_or_path"):
        value = getattr(args, key, None)
        if value:
            return value
    model_args = getattr(args, "model_args", {}) or {}
    for key in ("pretrained_weight_path", "pretrained_model_name_or_path", "model_name_or_path"):
        value = model_args.get(key)
        if value:
            return value
    return None


def _to_image_size(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, int):
        return (value, value)
    raise TypeError(f"Unsupported image_size value: {value!r}")


def _apply_runtime_overrides(config, args):
    overrides = {
        "device": getattr(args, "device", None) or "cuda",
        "n_obs_steps": getattr(args, "n_obs_steps", None),
        "chunk_size": getattr(args, "chunk_size", None),
        "n_action_steps": getattr(args, "n_action_steps", None),
        "dtype": getattr(args, "dtype", None),
        "tokenizer_name": getattr(args, "tokenizer_name", None),
        "tokenizer_max_length": getattr(args, "tokenizer_max_length", None),
        "tokenizer_padding_side": getattr(args, "tokenizer_padding_side", None),
        "pad_language_to": getattr(args, "pad_language_to", None),
        "action_mode": getattr(args, "action_mode", None),
        "num_denoising_steps": getattr(args, "num_denoising_steps", None),
        "use_proprio": getattr(args, "use_proprio", None),
        "max_state_dim": getattr(args, "max_state_dim", None),
        "max_action_dim": getattr(args, "max_action_dim", None),
        "domain_feature_key": getattr(args, "domain_feature_key", None),
        "num_image_views": getattr(args, "num_image_views", None),
        "empty_cameras": getattr(args, "empty_cameras", None),
        "freeze_vision_encoder": getattr(args, "freeze_vision_encoder", None),
        "freeze_language_encoder": getattr(args, "freeze_language_encoder", None),
        "train_policy_transformer": getattr(args, "train_policy_transformer", None),
        "train_soft_prompts": getattr(args, "train_soft_prompts", None),
        "optimizer_lr": getattr(args, "optimizer_lr", None),
        "optimizer_betas": getattr(args, "optimizer_betas", None),
        "optimizer_eps": getattr(args, "optimizer_eps", None),
        "optimizer_weight_decay": getattr(args, "optimizer_weight_decay", None),
        "optimizer_grad_clip_norm": getattr(args, "optimizer_grad_clip_norm", None),
        "optimizer_soft_prompt_lr_scale": getattr(args, "optimizer_soft_prompt_lr_scale", None),
        "optimizer_soft_prompt_warmup_lr_scale": getattr(args, "optimizer_soft_prompt_warmup_lr_scale", None),
        "scheduler_warmup_steps": getattr(args, "scheduler_warmup_steps", None),
        "scheduler_decay_steps": getattr(args, "scheduler_decay_steps", None),
        "scheduler_decay_lr": getattr(args, "scheduler_decay_lr", None),
        "state_dim": getattr(args, "state_dim", None),
        "action_dim": getattr(args, "action_dim", None),
        "camera_names": getattr(args, "camera_names", None),
        "base_vlm_model_name_or_path": getattr(args, "base_vlm_model_name_or_path", None),
        "pretrained_weight_path": getattr(args, "pretrained_weight_path", None),
        "lora_module": getattr(args, "lora_module", None),
        "lora_r": getattr(args, "lora_r", None),
        "lora_alpha": getattr(args, "lora_alpha", None),
        "lora_dropout": getattr(args, "lora_dropout", None),
        "lora_bias": getattr(args, "lora_bias", None),
        "lora_modules_to_save": getattr(args, "lora_modules_to_save", None),
    }
    resize_size = _to_image_size(
        getattr(args, "resize_imgs_with_padding", None) or getattr(args, "image_size", None)
    )
    if resize_size is not None:
        overrides["resize_imgs_with_padding"] = resize_size

    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    if getattr(config, "n_action_steps", None) is None:
        config.n_action_steps = config.chunk_size

    return _resolve_lora_training_config(config)


def _build_config_from_args(args):
    model_args = dict(getattr(args, "model_args", {}) or {})
    config = XVLAConfig(**model_args)
    _apply_runtime_overrides(config, args)
    return config


def _normalize_modules_to_save(value):
    if value is None:
        return []
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "false", "[]"}:
            return []
        return [value]
    if isinstance(value, tuple):
        return list(value)
    return list(value)


def _lora_targets_transformer(lora_module):
    modules = lora_module or []
    return any("transformer" in str(name).lower() for name in modules)


def _resolve_lora_training_config(config):
    config.lora_modules_to_save = _normalize_modules_to_save(config.lora_modules_to_save)
    config.lora_targets_transformer = _lora_targets_transformer(config.lora_module)
    if config.lora_targets_transformer:
        config.train_policy_transformer = False
        config.lora_modules_to_save = [
            kw
            for kw in config.lora_modules_to_save
            if not (str(kw).startswith("model.transformer") and "soft_prompt" not in str(kw))
        ]
    return config


def _load_pretrained_config(model_name_or_path):
    return XVLAConfig.from_pretrained(model_name_or_path)


def _maybe_apply_lora(model, config):
    if not config.lora_module:
        return model

    from peft import LoraConfig, get_peft_model

    target_modules = find_all_linear_names(model, config.lora_module)
    if not target_modules:
        raise ValueError(
            f"No linear layers matched lora_module={config.lora_module}. "
            f"Available modules: {[n for n, _ in model.named_modules()][:30]}..."
        )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
    )

    if config.dtype == "bfloat16":
        model.to(torch.bfloat16)

    model = get_peft_model(model, lora_config)

    trainable_keywords = list(config.lora_modules_to_save)
    if config.train_policy_transformer and not getattr(config, "lora_targets_transformer", False):
        trainable_keywords.append("model.transformer")
    if config.train_soft_prompts:
        trainable_keywords.extend(["soft_prompt_hub", "soft_prompt"])
    trainable_keywords = list(dict.fromkeys(trainable_keywords))
    for name, param in model.named_parameters():
        if "lora" in name or "adapter" in name:
            continue
        if any(kw in name for kw in trainable_keywords):
            param.requires_grad = True

    # print("\nXVLA LoRA trainable parameters:")
    # model.print_trainable_parameters()
    return model


def load_model(args):
    pretrained_source = _get_pretrained_source(args)

    if getattr(args, "is_training", True):
        if pretrained_source:
            config = _load_pretrained_config(pretrained_source)
            _apply_runtime_overrides(config, args)
            model = XVLAPolicy.from_pretrained(pretrained_source, config=config)
        else:
            config = _build_config_from_args(args)
            model = XVLAPolicy(config=config)

        model = _maybe_apply_lora(model, config)
    else:
        checkpoint_path = args.model_name_or_path
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        is_peft_checkpoint = os.path.exists(adapter_config_path)

        if is_peft_checkpoint:
            from peft import PeftModel

            config = _load_pretrained_config(checkpoint_path)
            _apply_runtime_overrides(config, args)
            base_src = config.pretrained_weight_path
            if not base_src:
                raise ValueError(
                    "PEFT checkpoint requires pretrained_weight_path in config.json "
                    "to locate the base XVLA model."
                )
            base_model = XVLAPolicy.from_pretrained(base_src, config=config)
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            model = model.merge_and_unload()
        else:
            config = _load_pretrained_config(checkpoint_path)
            _apply_runtime_overrides(config, args)
            model = XVLAPolicy.from_pretrained(checkpoint_path, config=config)

    cfg = _get_xvla_config(model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    data_processor = XVLADataProcessor(default_domain_id=getattr(args, "domain_id", 0))
    data_collator = XVLADataCollator(
        tokenizer=tokenizer,
        config=cfg,
        camera_names=cfg.camera_names,
        default_domain_id=getattr(args, "domain_id", 0),
    )

    base = model
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            base = model.get_base_model()
    except ImportError:
        pass
    base.data_processor = data_processor
    base.data_collator = data_collator
    base.tokenizer = tokenizer

    model.to(cfg.device)

    return {
        "model": model,
        "tokenizer": tokenizer,
    }


def get_data_processor(args, model_components):
    return XVLADataProcessor(default_domain_id=getattr(args, "domain_id", 0))


def get_data_collator(args, model_components):
    model = model_components["model"]
    tokenizer = model_components["tokenizer"]
    cfg = _get_xvla_config(model)
    return XVLADataCollator(
        tokenizer=tokenizer,
        config=cfg,
        camera_names=cfg.camera_names,
        default_domain_id=getattr(args, "domain_id", 0),
    )
