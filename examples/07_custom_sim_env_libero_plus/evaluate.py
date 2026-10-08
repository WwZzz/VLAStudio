"""Evaluate a local checkpoint or a remote policy using the same YAML adapter."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import vlastudio as vla

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--server", help="host:port, http(s)://host:port, or shm://name")
    parser.add_argument("--env", default=str(HERE / "env_object.yaml"))
    parser.add_argument("--output-dir", default="results/example07_plus")
    parser.add_argument("--num-rollout", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()
    policy = (vla.connect_policy(args.server) if args.server else
              vla.load_policy(HERE / "smolvla.yaml", checkpoint=args.checkpoint))
    result = vla.load_env(args.env).evaluate(
        policy, output_dir=args.output_dir, num_rollout=args.num_rollout,
        batch_size=args.batch_size, action_manager=HERE / "action_manager.yaml")
    source = os.environ.get("LIBERO_ROOT")
    revision = None
    if source:
        revision = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    protocol = dict(checkpoint=args.checkpoint, server=args.server, env_config=Path(args.env).read_text(),
                    action_manager=(HERE / "action_manager.yaml").read_text(),
                    simulator_root=source, simulator_revision=revision,
                    num_rollout=args.num_rollout, batch_size=args.batch_size, python=sys.executable)
    (result.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for name, metrics in result.metrics.items():
        if "success_rate" in metrics:
            print(f"{name}: {metrics['total_success']}/{metrics['total']} ({metrics['success_rate']:.1%})")


if __name__ == "__main__":
    main()
