"""Run inside a managed worker; synthetic inputs test execution, not policy quality."""
import argparse
import json
import os
from pathlib import Path


def step(model, inputs):
    import torch
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    loss = model(**inputs)["loss"].mean()
    assert loss.is_cuda and torch.isfinite(loss), loss
    loss.backward()
    gradients = [p for p in parameters if p.grad is not None]
    assert gradients and all(torch.isfinite(p.grad).all() for p in gradients)
    changed = next(p for p in gradients if p.grad.abs().max() > 0)
    before = changed.detach().clone()
    optimizer.step()
    assert not torch.equal(before, changed), "Optimizer did not update parameters"
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def mlp():
    import torch
    from policy.mlp import MLPPolicy, MLPPolicyConfig
    model = MLPPolicy(MLPPolicyConfig(state_dim=2, action_dim=2, hidden_dim=32)).cuda()
    state = torch.randn(2, 2, device="cuda")
    loss = step(model, dict(state=state, action=torch.randn(2, 1, 2, device="cuda"),
                           is_pad=torch.zeros(2, 1, dtype=torch.bool, device="cuda")))
    model.eval()
    with torch.no_grad():
        prediction = model(state=state)
    if isinstance(prediction, dict):
        prediction = prediction["action"]
    assert prediction.is_cuda and torch.isfinite(prediction).all()
    return dict(loss=loss, shape=list(prediction.shape))


def act():
    import torch
    from policy.act import ACTPolicy, ACTPolicyConfig
    model = ACTPolicy(ACTPolicyConfig(state_dim=2, action_dim=2, chunk_size=2,
        camera_names=["primary"], hidden_dim=64, dim_feedforward=128,
        enc_layers=1, dec_layers=1, nheads=4)).cuda()
    inputs = dict(qpos=torch.randn(2, 2, device="cuda"),
                  image=torch.rand(2, 1, 3, 64, 64, device="cuda"))
    loss = step(model, dict(**inputs, actions=torch.randn(2, 2, 2, device="cuda"),
                           is_pad=torch.zeros(2, 2, dtype=torch.bool, device="cuda")))
    model.eval()
    with torch.no_grad():
        prediction = model(**inputs)
    assert prediction.is_cuda and torch.isfinite(prediction).all()
    return dict(loss=loss, shape=list(prediction.shape))


def openpi():
    import torch
    from types import SimpleNamespace
    from policy.openpi import load_model
    from openpi.models.model import Observation
    model = load_model(SimpleNamespace(is_training=True, model_args={
        "chunk_size": 2, "max_token_len": 8, "action_dim": 2,
        "freeze_vision_tower": True, "lora_r": 2, "lora_alpha": 2,
    }))["model"]
    observation = dict(
        image={name: torch.rand(1, 3, 224, 224, device="cuda") * 2 - 1
               for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        image_mask={name: torch.ones(1, dtype=torch.bool, device="cuda")
                    for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        state=torch.zeros(1, 32, device="cuda"),
        tokenized_prompt=torch.ones(1, 8, dtype=torch.long, device="cuda"),
        tokenized_prompt_mask=torch.ones(1, 8, dtype=torch.bool, device="cuda"),
    )
    loss = step(model, dict(observation=observation,
                           actions=torch.randn(1, 2, 32, device="cuda")))
    model.eval()
    with torch.no_grad():
        prediction = model.get_base_model().model.sample_actions(
            "cuda", Observation.from_dict(observation), num_steps=1)
    assert prediction.is_cuda and torch.isfinite(prediction).all()
    return dict(loss=loss, shape=list(prediction.shape), weights="random initialization")


def openvla():
    import gc
    import torch
    from types import SimpleNamespace
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from policy.openvla import load_model
    from policy.openvla.modeling import (OpenVLAConfig,
        OpenVLAForActionPrediction, PrismaticImageProcessor, PrismaticProcessor)
    # Use the actual save/load and multimodal model path without downloading 7B weights.
    # Small language/vision dimensions keep this integration fixture inexpensive.
    config = OpenVLAConfig(text_config=dict(vocab_size=32064, hidden_size=128,
        intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, pad_token_id=0), pad_token_id=0,
        norm_stats={"smoke": {"action": {"q01": [-1., -1.], "q99": [1., 1.]}}})
    config.timm_model_ids = ["vit_tiny_patch16_224"]
    config.timm_override_act_layers = [None]
    checkpoint = Path("synthetic-openvla-checkpoint").resolve()
    base = OpenVLAForActionPrediction(config)
    base.save_pretrained(checkpoint)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {"<pad>": 0, "<unk>": 1, **{f"t{i}": i for i in range(2, 32064)}}, unk_token="<unk>")),
        pad_token="<pad>", unk_token="<unk>")
    processor = PrismaticProcessor(PrismaticImageProcessor(interpolations=["bicubic"]), tokenizer)
    processor.save_pretrained(checkpoint)
    del base
    gc.collect()
    model = load_model(SimpleNamespace(is_training=True, pretrained_weight_path=str(checkpoint),
                                      training_mode="full", action_dim=2))["model"].cuda()
    ids = torch.tensor([[2, 3, 4, 5]], device="cuda")
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  pixel_values=torch.rand(1, 3, 224, 224, device="cuda"))
    loss = step(model, dict(**inputs, labels=ids.clone()))
    model.eval()
    inference_devices = []
    handle = model.model.register_forward_hook(
        lambda module, args, output: inference_devices.append(output.logits.is_cuda))
    with torch.no_grad():
        prediction = model.select_action(inputs)
    handle.remove()
    import numpy as np
    assert inference_devices and all(inference_devices)
    assert prediction.shape == (1, 1, 2) and np.isfinite(prediction).all()
    return dict(loss=loss, shape=list(prediction.shape),
                weights="random initialization; reduced architecture; not a 7B checkpoint")


def train_mlp():
    import runpy
    import sys
    import torch
    import yaml
    from vlastudio.paths import legacy_root
    examples = Path(__file__).resolve().parents[2] / "examples/extensions"
    if not examples.is_dir():
        examples = Path(__file__).resolve().parent
    task = yaml.safe_load((examples / "task_mlp.yaml").read_text())
    task["datasets"][0]["type"] = str(examples / "toy_dataset.py") + ":ToyDataset"
    training = yaml.safe_load((examples / "training_mlp.yaml").read_text())
    training["use_cpu"] = False
    Path("gpu-task.yaml").write_text(yaml.safe_dump(task))
    Path("gpu-training.yaml").write_text(yaml.safe_dump(training))
    destination = Path("GPU checkpoints with spaces").resolve()
    previous = sys.argv
    try:
        sys.argv = ["train.py", "-p", str(examples / "policy_mlp.yaml"),
                    "-t", "gpu-task.yaml", "-c", "gpu-training.yaml", "-o", str(destination)]
        runpy.run_path(str(legacy_root() / "train.py"), run_name="__main__")
    finally:
        sys.argv = previous
    state = json.loads((destination / "trainer_state.json").read_text())
    assert state["global_step"] == 2 and torch.cuda.max_memory_allocated() > 0
    assert (destination / "model.safetensors").is_file()
    assert (destination / "policy_metadata.json").is_file()
    return dict(steps=state["global_step"], checkpoint=str(destination))


def run(command, argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("mlp", "act", "openpi", "openvla", "train_mlp"), required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import torch
    assert torch.cuda.is_available(), "GPU validation must not fall back to CPU"
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    result = globals()[args.case]()
    torch.cuda.synchronize()
    result.update(case=args.case, torch=torch.__version__, cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(),
                  peak_memory_bytes=torch.cuda.max_memory_allocated(), passed=True)
    Path(args.result).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0
