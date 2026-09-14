"""Train from user configs with automatically isolated policy dependencies."""
import argparse
from pathlib import Path
import vlastudio as vla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True, help='Policy config name or YAML path')
    parser.add_argument('--task', required=True, help='Task/dataset config name or YAML path')
    parser.add_argument('--training-config', default='default')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--data-cache-dir', default=None)
    parser.add_argument('--env', help='Optional simulation environment config')
    parser.add_argument('--eval-runtime', help='Complete runtime manifest for policy + simulator')
    args = parser.parse_args()

    dataset = vla.load_dataset(args.task, cache_dir=args.data_cache_dir)
    policy = vla.load_policy(args.policy, cache_dir=args.cache_dir)
    result = vla.train(policy, dataset, args.training_config, output_dir=args.output_dir)
    print(f'Checkpoint: {result.checkpoint}')

    if args.env:
        bench = vla.load_env(args.env, runtime_manifest=args.eval_runtime)
        evaluation = bench.evaluate(policy, output_dir=Path(args.output_dir) / 'evaluation')
        print(evaluation.metrics)


if __name__ == '__main__':
    main()
