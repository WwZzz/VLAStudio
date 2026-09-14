import torch

ACTION = "action"
OBS_IMAGES = "observation.images"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_STATE = "observation.state"
IMAGENET_STATS = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def _to_tensor(value, dtype=None):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    tensor = torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _ensure_image_tensor(image):
    image = _to_tensor(image)
    if image is None:
        return None
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image with shape (K, C, H, W), got {tuple(image.shape)}")
    return image


def _pad_last_dim(tensor, target_dim):
    if tensor.shape[-1] == target_dim:
        return tensor
    output_shape = list(tensor.shape)
    output_shape[-1] = target_dim
    padded = tensor.new_zeros(output_shape)
    length = min(tensor.shape[-1], target_dim)
    padded[..., :length] = tensor[..., :length]
    return padded


def _pad_time_dim(tensor, target_len):
    if tensor.shape[0] == target_len:
        return tensor
    if tensor.shape[0] > target_len:
        return tensor[:target_len]
    pad_shape = list(tensor.shape)
    pad_shape[0] = target_len - tensor.shape[0]
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)


class XVLADataProcessor:
    def __init__(self, default_domain_id=0):
        self.default_domain_id = default_domain_id

    def __call__(self, sample):
        reasoning = sample.get("reasoning") or {}
        domain_id = reasoning.get("domain_id", sample.get("domain_id", self.default_domain_id))

        processed = {
            "raw_lang": sample.get("raw_lang", "") or "",
            "domain_id": int(domain_id),
        }

        image = _ensure_image_tensor(sample.get("image"))
        if image is not None:
            processed["image"] = image

        state = _to_tensor(sample.get("state"), dtype=torch.float32)
        if state is not None:
            processed["state"] = state

        action = _to_tensor(sample.get("action"), dtype=torch.float32)
        if action is not None:
            processed["action"] = action

        is_pad = _to_tensor(sample.get("is_pad"))
        if is_pad is not None:
            processed["is_pad"] = is_pad.bool()

        return processed


class XVLADataCollator:
    def __init__(self, tokenizer, config, camera_names=None, default_domain_id=0):
        self.tokenizer = tokenizer
        self.config = config
        self.default_domain_id = default_domain_id
        self.camera_names = camera_names or list(getattr(config, "camera_names", ["primary"]))
        self.real_state_dim = int(getattr(config, "state_dim", 0) or 0)
        self.real_action_dim = int(getattr(config, "action_dim", 0) or 0)
        mean = torch.tensor(IMAGENET_STATS["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STATS["std"], dtype=torch.float32).view(1, 3, 1, 1)
        self.image_mean = mean
        self.image_std = std

    def _normalize_image(self, image):
        image = image.float()
        if image.max().item() > 1.0:
            image = image / 255.0
        mean = self.image_mean.to(device=image.device, dtype=image.dtype)
        std = self.image_std.to(device=image.device, dtype=image.dtype)
        return (image - mean) / std

    def _stack_states(self, instances):
        if self.real_state_dim <= 0:
            return None
        states = []
        for instance in instances:
            state = _to_tensor(instance.get("state"), dtype=torch.float32)
            if state is None:
                state = torch.zeros(self.real_state_dim, dtype=torch.float32)
            states.append(_pad_last_dim(state, self.real_state_dim))
        return torch.stack(states, dim=0)

    def _stack_actions(self, instances):
        actions = []
        masks = []
        for instance in instances:
            action = instance.get("action")
            if action is None:
                return None, None
            action = _to_tensor(action, dtype=torch.float32)
            if action.ndim == 1:
                action = action.unsqueeze(0)
            action = _pad_time_dim(action, self.config.chunk_size)
            if self.real_action_dim > 0:
                action = _pad_last_dim(action, self.real_action_dim)
            actions.append(action)

            is_pad = instance.get("is_pad")
            if is_pad is None:
                mask = torch.zeros(self.config.chunk_size, dtype=torch.bool)
            else:
                mask = _to_tensor(is_pad).bool()
                if mask.ndim > 1:
                    mask = mask[..., 0]
                mask = _pad_time_dim(mask, self.config.chunk_size)
            masks.append(mask)

        return torch.stack(actions, dim=0), torch.stack(masks, dim=0)

    def _stack_images(self, instances):
        image_tensors = [_ensure_image_tensor(instance.get("image")) for instance in instances]
        if any(image is None for image in image_tensors):
            raise ValueError("XVLA requires image observations in every sample.")

        reference = image_tensors[0]
        view_tensors = {}
        for view_idx, camera_name in enumerate(self.camera_names):
            per_view = []
            for image in image_tensors:
                if image.shape[0] > view_idx:
                    camera_image = image[view_idx]
                else:
                    camera_image = torch.zeros_like(reference[0])
                per_view.append(camera_image)
            stacked = torch.stack(per_view, dim=0)
            view_tensors[f"{OBS_IMAGES}.{camera_name}"] = self._normalize_image(stacked)
        return view_tensors

    def __call__(self, instances):
        languages = [instance.get("raw_lang", "") for instance in instances]
        tokenized = self.tokenizer(
            languages,
            max_length=self.config.tokenizer_max_length,
            truncation=True,
            padding=self.config.pad_language_to,
            padding_side=self.config.tokenizer_padding_side,
            return_tensors="pt",
        )

        batch = {
            OBS_LANGUAGE_TOKENS: tokenized.input_ids,
            OBS_LANGUAGE_ATTENTION_MASK: tokenized.attention_mask.bool(),
            "domain_id": torch.tensor(
                [int(instance.get("domain_id", self.default_domain_id)) for instance in instances],
                dtype=torch.long,
            ),
        }

        batch.update(self._stack_images(instances))

        states = self._stack_states(instances)
        if states is not None:
            batch[OBS_STATE] = states

        actions, is_pad = self._stack_actions(instances)
        if actions is not None:
            batch[ACTION] = actions
            batch["is_pad"] = is_pad

        return batch
