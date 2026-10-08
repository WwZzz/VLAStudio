"""FastWAM Base inference adapter using the official model and normalization."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def create_runtime(spec, device="cuda"):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from fastwam.utils.config_resolvers import register_default_resolvers
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.utils.pytorch_utils import set_global_seed

    root = Path(os.path.expandvars(spec["upstream_root"])).expanduser().resolve()
    register_default_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        cfg = compose(config_name="sim_libero", overrides=["task=libero_uncond_2cam224_1e-4"])
    cfg.model.redirect_common_files = False
    cfg.seed = int(spec.get("seed", 42))
    set_global_seed(cfg.seed, get_worker_init_fn=False)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
    # Require both experts and proprioception weights; reject partial legacy loads.
    payload = torch.load(Path(spec["weights"]).expanduser(), map_location="cpu", weights_only=True, mmap=True)
    model.mot.load_state_dict(payload["mot"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    del payload
    model = model.to(device).eval()
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(spec["dataset_stats"]))
    return model, processor


def prepare_inputs(sample, processor, device, dtype):
    """MetaObs images are uint8 KCHW, already rotated to the training orientation."""
    images = sample["image"]
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if images.shape != (2, 3, 256, 256) or images.dtype != np.uint8:
        raise ValueError(f"Expected two uint8 256px RGB cameras, got {images.shape}/{images.dtype}")
    cameras = []
    for image in images:
        image = Image.fromarray(image.transpose(1, 2, 0))
        cameras.append(np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR)))
    rgb = np.concatenate(cameras, axis=1)
    # Match the official order: convert to BF16 before scaling to [-1, 1].
    image = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    image = image * (2.0 / 255.0) - 1.0
    state = torch.as_tensor(sample["state"], dtype=torch.float32).cpu().reshape(1, 8)
    state_batch = processor.action_state_transform({"state": {"default": state}})
    proprio = processor.normalizer.forward(state_batch)["state"]["default"]
    return image, proprio


class FastWAMPolicy(torch.nn.Module):
    def __init__(self, spec, device="cuda"):
        super().__init__()
        self.spec = spec
        self.model, self.processor = create_runtime(spec, device)
        self.ctrl_space, self.ctrl_type = "ee", "delta"
        self.chunk_size, self.action_dim = 32, 7

    def meta2obs(self, samples):
        return samples

    @torch.inference_mode()
    def select_action(self, samples):
        chunks = []
        for sample in samples:
            image, proprio = prepare_inputs(sample, self.processor, self.model.device, self.model.torch_dtype)
            prediction = self.model.infer_action(
                prompt="A video recorded from a robot's point of view executing the following instruction: " + sample["raw_lang"],
                input_image=image, proprio=proprio, action_horizon=32,
                num_inference_steps=int(self.spec.get("num_inference_steps", 10)),
                sigma_shift=float(self.spec.get("sigma_shift", 5.0)),
                text_cfg_scale=1.0, negative_prompt="", seed=int(self.spec.get("seed", 42)),
                rand_device="cpu", tiled=False,
                compile_action_infer=bool(self.spec.get("compile_action_infer", False)),
            )
            action = prediction["action"].to(device="cpu", dtype=torch.float32)
            if action.ndim == 2:
                action = action.unsqueeze(0)
            action = self.processor.normalizer.normalizers["action"]["default"].backward(action).numpy()[0]
            action[..., -1] = np.sign(-(action[..., -1] * 2 - 1))
            if action.shape != (32, 7) or not np.isfinite(action).all():
                raise ValueError(f"Invalid FastWAM action chunk: {action.shape}")
            chunks.append(action)
        return np.stack(chunks)


def load_model(args):
    bundle = Path(args.model_name_or_path).expanduser()
    spec = json.loads((bundle / "fastwam.json").read_text())
    policy = FastWAMPolicy(spec, device=getattr(args, "device", "cuda"))
    return {"model": policy, "config": None}
