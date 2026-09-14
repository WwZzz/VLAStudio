"""Train ACT on ALOHA transfer-cube data, then evaluate the saved policy.

Normal mode uses the built-in dataset alias and downloads its data on first use.
Use --smoke for a two-step train plus a short real MuJoCo rollout with generated
ALOHA-format data. The smoke run validates execution, not policy quality.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import vlastudio as vla


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="act", help="Built-in name or policy YAML")
    parser.add_argument("--dataset", default="sim_transfer_cube_scripted",
                        help="Built-in task alias or task YAML")
    parser.add_argument("--training-config", default="default")
    parser.add_argument("--env", default="aloha_transfer", help="Built-in env alias or YAML")
    parser.add_argument("--runtime-cache", type=Path, default=Path("~/.cache/vlastudio").expanduser())
    parser.add_argument("--data-cache", type=Path, default=Path("~/.cache/vlastudio/data").expanduser())
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/act_aloha_transfer"))
    parser.add_argument("--eval-output-dir", type=Path, default=Path("results/act_aloha_transfer"))
    parser.add_argument("--checkpoint", type=Path,
                        help="Skip training and evaluate this existing checkpoint")
    parser.add_argument("--max-steps", type=int, help="Override training.max_steps")
    parser.add_argument("--num-rollout", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 is sequential; parallel evaluation also requires tianshou")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true",
                        help="Generate tiny data, train 2 steps and evaluate 5 simulator steps")
    return parser.parse_args()


def make_smoke_task(data_cache: Path) -> Path:
    """Create a task config whose synthetic Dataset loads inside the ACT worker."""
    import yaml

    implementation = Path(__file__).resolve().parent / "_support" / "aloha_smoke_dataset.py"
    task = {
        "name": "sim_transfer_cube_scripted_smoke",
        "datasets": [{
            "type": f"{implementation}:AlohaSmokeDataset",
            "name": "sim_transfer_cube_scripted_smoke",
            "args": {
                "size": 8,
                "chunk_size": 50,
            },
        }],
        "meta": {
            "action_dim": 14,
            "state_dim": 14,
            "image_size": [64, 64],
            "action_normalize": "zscore",
            "state_normalize": "zscore",
        },
    }
    task_path = data_cache.resolve() / "smoke_task.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    return task_path


def main():
    args = parse_args()
    if args.device.startswith("cuda"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    runtime_cache = args.runtime_cache.resolve()
    data_cache = args.data_cache.resolve()
    dataset_config = make_smoke_task(data_cache) if args.smoke else args.dataset

    dataset = vla.load_dataset(dataset_config, cache_dir=data_cache)
    policy = vla.load_policy(
        args.policy,
        checkpoint=args.checkpoint,
        cache_dir=runtime_cache,
    )

    if args.checkpoint is None:
        overrides = {}
        if args.max_steps is not None:
            overrides["training.max_steps"] = args.max_steps
        if args.smoke:
            overrides.update({
                "training.max_steps": 2,
                "training.per_device_train_batch_size": 2,
                "training.dataloader_num_workers": 0,
                "training.report_to": "none",
                "training.save_strategy": "no",
            })
        trained = vla.train(
            policy,
            dataset,
            args.training_config,
            output_dir=args.output_dir.resolve(),
            overrides=overrides,
        )
        print(f"Training complete: {trained.checkpoint}")
    else:
        print(f"Using checkpoint: {policy.checkpoint}")

    bench = vla.load_env(args.env)
    env_overrides = {"env.max_timesteps": 5} if args.smoke else None
    evaluated = bench.evaluate(
        policy,
        output_dir=args.eval_output_dir.resolve(),
        num_rollout=1 if args.smoke else args.num_rollout,
        batch_size=0 if args.smoke else args.batch_size,
        device=args.device,
        overrides=env_overrides,
    )
    print(f"Evaluation artifacts: {evaluated.output_dir}")
    print(f"Metrics: {evaluated.metrics}")


if __name__ == "__main__":
    main()
