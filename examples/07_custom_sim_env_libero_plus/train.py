"""Train a two-camera SmolVLA on the original Object demonstrations."""
import argparse
import os
from pathlib import Path
import vlastudio as vla

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="Directory containing libero_object/*.hdf5")
    parser.add_argument("--output-dir", default="checkpoints/example07_smolvla")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.data_root:
        os.environ["LIBERO_DATASET_ROOT"] = str(Path(args.data_root).expanduser().resolve())
    dataset = vla.load_dataset(HERE / "task_object.yaml")
    policy = vla.load_policy(HERE / "smolvla.yaml")
    vla.train(policy, dataset, "smolvla", output_dir=args.output_dir,
              overrides={"training.max_steps": args.steps,
                         "training.per_device_train_batch_size": args.batch_size,
                         "training.dataloader_num_workers": 4,
                         "training.save_steps": args.steps,
                         "training.save_total_limit": 2})


if __name__ == "__main__":
    main()
