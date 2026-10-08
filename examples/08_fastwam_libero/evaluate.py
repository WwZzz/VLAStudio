"""Evaluate the released FastWAM Base checkpoint through VLAStudio."""
import argparse
import json
import os
from pathlib import Path
import vlastudio as vla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", "spatial", "object", "goal", "10"], default="all")
    parser.add_argument("--checkpoint", default=os.path.expandvars("${VLASTUDIO_CACHE}/fastwam/bundle") if os.getenv("VLASTUDIO_CACHE") else str(Path.home()/".cache/vlastudio/fastwam/bundle"))
    parser.add_argument("--output-dir", default="results/example08_fastwam")
    parser.add_argument("--num-rollouts", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.num_rollouts <= 50:
        parser.error("--num-rollouts must be between 1 and 50 official initial states")
    config = Path(__file__).resolve().parent / "config"
    policy = vla.load_policy(str(config/"fastwam.yaml"), checkpoint=args.checkpoint)
    for suite in (["spatial", "object", "goal", "10"] if args.suite == "all" else [args.suite]):
        result = vla.load_env(str(config/f"libero_{suite}.yaml")).evaluate(
            policy, output_dir=str(Path(args.output_dir)/f"libero_{suite}"),
            num_rollout=args.num_rollouts, batch_size=0,
            action_manager=str(config/"action_manager.yaml"), overrides={"seed": 42},
        )
        tasks = [value for key, value in result.metrics.items()
                 if Path(key).name.startswith(f"libero_{suite}_") and "total_success" in value]
        if len(tasks) != 10 or any(t["total"] != args.num_rollouts for t in tasks):
            raise RuntimeError("Incomplete suite results; refusing to report a comparison")
        success, total = sum(t["total_success"] for t in tasks), sum(t["total"] for t in tasks)
        paper = {"spatial": 98.2, "object": 100.0, "goal": 97.0, "10": 95.2}[suite]
        summary = {"suite": suite, "success": success, "total": total,
                   "success_rate_percent": 100 * success / total, "paper_percent": paper,
                   "full_protocol": args.num_rollouts == 50,
                   "delta_percentage_points": 100 * success / total - paper if args.num_rollouts == 50 else None}
        (result.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
