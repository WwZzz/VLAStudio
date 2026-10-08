"""Generate a YAML evaluation list from the official Plus task catalog."""
import argparse
import json
import os
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("LIBERO_ROOT"))
    parser.add_argument("--suite", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"], default="libero_object")
    parser.add_argument("--category", help="Exact category name from task_classification.json")
    parser.add_argument("--limit", type=int, help="Optional number of tasks; omitted means all matching tasks")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.root:
        parser.error("Source libero-plus/env.sh or pass --root /path/to/LIBERO-plus")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    catalog = Path(args.root).expanduser() / "libero/libero/benchmark/task_classification.json"
    rows = json.loads(catalog.read_text(encoding="utf-8"))[args.suite]
    categories = sorted({row["category"] for row in rows})
    if args.category and args.category not in categories:
        parser.error(f"Unknown category. Choose from: {categories}")
    selected = [row for row in rows if not args.category or row["category"] == args.category]
    if args.limit:
        selected = selected[:args.limit]
    horizon = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}[args.suite]
    adapter = str(Path(__file__).with_name("libero_plus_env.py").resolve()) + ":LiberoPlusEnv"
    configs = []
    for index, row in enumerate(selected):
        name = f"{args.suite}_{index}"
        configs.append(dict(type=adapter, name=name, args=dict(
            task=name, task_name=row["name"], perturbation=row["category"],
            require_plus=True, max_timesteps=horizon, ctrl_space="ee", ctrl_type="delta",
            image_size=[256, 256], use_wrist=True, image_transform="rotate180",
            seed=0, num_steps_wait=10)))
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(configs, sort_keys=False), encoding="utf-8")
    print(f"Wrote {len(configs)} tasks to {output}; one rollout per task is the upstream Plus protocol.")


if __name__ == "__main__":
    main()
