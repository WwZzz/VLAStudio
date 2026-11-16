"""Run TCP on fixed official Bench2Drive routes and preserve each native score."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml


def main(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=default_config)
    parser.add_argument('--model', default='official')
    parser.add_argument('--output', default='results/example10/tcp-official')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-routes', type=int)
    parser.add_argument('--route-ids', help='Optional comma-separated official route IDs')
    args = parser.parse_args()
    from . import Bench2DriveEnv
    from vlastudio.policy.tcp_b2d import load_model
    from .install import B2D_REVISION, TCP_REVISION, source
    from vlastudio.data_utils.datasets.bench2drive import B2D_DATA_REVISION
    from vlastudio.utils.assets import sha256
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text())
    else:
        config = {'ctrl_space': 'vehicle', 'ctrl_type': 'control', 'seed': 0}
    if args.max_routes is not None:
        if args.max_routes < 1:
            parser.error('max-routes must be positive')
        config['max_routes'] = args.max_routes
    if args.route_ids:
        config['route_ids'] = args.route_ids.split(',')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    checkpoint = Path(args.model)/'model.safetensors'
    model_digest = sha256(checkpoint) if checkpoint.is_file() else 'e6573ff1f8ea910b9a53eddfb68f69cac469bf5bfa253a516578f6126110b4fe' if args.model == 'official' else None
    if model_digest is None:
        raise FileNotFoundError('Evaluation requires official weights or a local saved TCP checkpoint')
    planner_type = json.loads((checkpoint.parent/'config.json').read_text())['planner_type'] if checkpoint.is_file() else 'merge_ctrl_traj'
    package = Path(__file__).resolve().parents[2]
    adapter_digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py'))+[package/'data_utils/datasets/bench2drive.py', package/'utils/assets.py', package/'policy/tcp_b2d/__init__.py', package/'policy/tcp_b2d/upstream.py']:
        adapter_digest.update(str(path.relative_to(package)).encode())
        adapter_digest.update(path.read_bytes())
    manifest = {'model_sha256': model_digest, 'config_sha256': config_digest,
                'planner_type': planner_type,
                'adapter_sha256': adapter_digest.hexdigest(),
                'model_config_sha256': sha256(checkpoint.parent/'config.json') if checkpoint.is_file() else hashlib.sha256(b'official:merge_ctrl_traj').hexdigest(),
                'benchmark_revision': B2D_REVISION, 'model_code_revision': TCP_REVISION,
                'training_data_revision': B2D_DATA_REVISION,
                'routes_sha256': sha256(Path(config.get('routes') or source()/'leaderboard/data/bench2drive_0.0.4_val.xml'))}
    manifest_path = output/'run.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Output contains a different evaluation; choose a new directory')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    env = Bench2DriveEnv(SimpleNamespace(**config))
    expected = [row['route_id'] for row in env.catalog]
    journal = output/'episodes.jsonl'
    records = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    completed = [row['source_route_id'] for row in records]
    if len(set(completed)) != len(completed) or completed != expected[:len(completed)]:
        env.close()
        raise ValueError('Journal is not a unique completed prefix of the selected official routes')
    if completed:
        env.close()
        remaining = expected[len(completed):]
        if remaining:
            config = {**config, 'route_ids': remaining}
            config.pop('max_routes', None)
            env = Bench2DriveEnv(SimpleNamespace(**config))
    model = load_model(SimpleNamespace(is_training=False, model_name_or_path=args.model,
        download=True, planner_type='merge_ctrl_traj'))['model'].to(args.device).eval()
    started = time.monotonic()
    try:
        for route in expected[len(completed):]:
            model.reset()
            observation = asdict(env.reset())
            while True:
                inputs = {'state': observation['state'][None], 'image': observation['image'][None],
                          'raw_lang': [observation['raw_lang']],
                          'timestamp': np.asarray([observation['timestep']])}
                action = model.select_action(inputs)[0, 0].cpu().numpy()
                observation, _, done, information = env.step(action)
                if done:
                    if information['source_route_id'] != route:
                        raise RuntimeError('Native route identity differs from selected route')
                    with journal.open('a') as stream:
                        stream.write(json.dumps(information)+'\n')
                        stream.flush()
                    records.append(information)
                    print(json.dumps({'completed': len(records), 'total': len(expected),
                        'route': route, 'status': information['status'],
                        'success_rate_so_far': np.mean([r['success'] for r in records]),
                        'driving_score_so_far': np.mean([r['driving_score'] for r in records]),
                        'elapsed_seconds': time.monotonic()-started}), flush=True)
                    break
    finally:
        env.close()
    if len(records) != len(expected):
        raise RuntimeError('Incomplete evaluation; no aggregate acceptance result was written')
    official_routes = source()/'leaderboard/data/bench2drive_0.0.4_val.xml'
    official_ids = [route.attrib['id'] for route in ET.parse(official_routes).getroot().findall('route')]
    full_split = expected == official_ids and manifest['routes_sha256'] == sha256(official_routes)
    result = {**manifest, 'routes': len(records), 'complete_official_val_split': full_split,
              'success_rate': float(np.mean([r['success'] for r in records])),
              'driving_score': float(np.mean([r['driving_score'] for r in records])),
              'route_completion': float(np.mean([r['route_completion'] for r in records])),
              'elapsed_seconds': time.monotonic()-started}
    (output/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
