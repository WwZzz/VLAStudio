"""Two real CPU training steps; run from any working directory in a checkout."""
import argparse
from pathlib import Path
import tempfile
import yaml
import vlastudio as vla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parents[1] / 'extensions'
    # The legacy dataset loader resolves file references against the caller's cwd.
    task = yaml.safe_load((fixtures / 'task_mlp.yaml').read_text())
    task['datasets'][0]['type'] = str(fixtures / 'toy_dataset.py') + ':ToyDataset'
    with tempfile.TemporaryDirectory(prefix='vlastudio-example-') as directory:
        task_path = Path(directory) / 'task.yaml'
        task_path.write_text(yaml.safe_dump(task), encoding='utf-8')
        dataset = vla.load_dataset(task_path)
        policy = vla.load_policy(fixtures / 'policy_mlp.yaml', cache_dir=args.cache_dir)
        result = vla.train(
            policy, dataset, fixtures / 'training_mlp.yaml', output_dir=args.output_dir,
            overrides={'training.use_cpu': not args.gpu},
        )
        print(f'Saved checkpoint: {result.checkpoint}')


if __name__ == '__main__':
    main()
