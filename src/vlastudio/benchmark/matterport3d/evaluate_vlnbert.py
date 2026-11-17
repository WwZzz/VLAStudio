"""Full-split, single-run R2R evaluation through VLAStudio + real MatterSim.

Paper R2R scoring uses the final viewpoint, including at the 15-action budget;
it does not require an explicit STOP. The generic environment's stricter STOP
metric is retained as explicit_stop_success for comparison.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from vlastudio.benchmark.base import MetaPolicy
from vlastudio.benchmark.matterport3d import MatterportNavigationEnv
from vlastudio.benchmark.navigation import spl
from vlastudio.benchmark.utils import organize_obs
from vlastudio.data_utils.normalize import Identity
from vlastudio.policy.recurrent_vlnbert import RecurrentVLNBERT


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main(default_config=None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=default_config)
    p.add_argument('--model', default='official')
    p.add_argument('--output', default='results/example09_vlnbert_official')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    from vlastudio.utils.assets import resolve_checkpoint
    from vlastudio.policy.recurrent_vlnbert.pretrained import official_checkpoint
    a.model = str(resolve_checkpoint(a.model, official=official_checkpoint))
    torch.set_num_threads(2)
    cfg = yaml.safe_load(Path(a.config).read_text())
    if cfg.get('auto_download', False) and not Path(cfg['episodes']).is_file():
        from vlastudio.data_utils.datasets.navigation import r2r_annotations
        split = Path(cfg['episodes']).stem.removeprefix('R2R_')
        cfg['episodes'] = str(r2r_annotations(split))
    if cfg.get('max_timesteps') != 15 or cfg.get('candidate_view_alignment') != 'panoramic':
        raise ValueError('Use the published 15-decision panoramic R2R protocol')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    signature = {'model_sha256': digest(Path(a.model)/'policy.pt'), 'config_sha256': digest(a.config),
        'episodes_sha256': digest(cfg['episodes']), 'scoring': 'R2R final viewpoint within 3m; greedy 15 decisions'}
    manifest = out/'run.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Existing output belongs to another benchmark or checkpoint')
    manifest.write_text(json.dumps(signature, indent=2)+'\n')
    journal = out/'episodes.jsonl'
    records = [json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
    completed = {x['episode_id'] for x in records}
    if len(completed) != len(records):
        raise ValueError('Duplicate episode IDs in existing journal')
    envs = []
    started = time.time()
    try:
        envs = [MatterportNavigationEnv(SimpleNamespace(**cfg)) for _ in range(a.batch_size)]
        episodes = envs[0].episodes
        model = RecurrentVLNBERT.from_pretrained(a.model, a.device)
        normalizer = Identity(ctrl_space='navigation', ctrl_type='candidate_viewpoint')
        policy = MetaPolicy(model, state_normalizer=normalizer, action_normalizer=normalizer,
                            ctrl_space='navigation', ctrl_type='candidate_viewpoint')
        pending = [i for i, e in enumerate(episodes) if e['episode_id'] not in completed]
        with journal.open('a', buffering=1) as stream:
            for offset in range(0, len(pending), a.batch_size):
                indices = pending[offset:offset+a.batch_size]
                obs, traces, actions = [], [], []
                for env, index in zip(envs, indices):
                    env.index = index
                    obs.append(asdict(env.reset()))
                    traces.append([env.env.getState()[0].location.viewpointId])
                    actions.append([])
                active_envs = envs[:len(indices)]
                done = [False]*len(indices)
                metrics = [{} for _ in indices]
                policy.reset()
                for _ in range(15):
                    selected = policy.inference(organize_obs(obs))[0]
                    for slot, (env, action) in enumerate(zip(active_envs, selected)):
                        if done[slot]:
                            continue
                        value = int(action['action'][0])
                        obs[slot], _, done[slot], metrics[slot] = env.step(value)
                        actions[slot].append(value)
                        if value:
                            traces[slot].append(env.env.getState()[0].location.viewpointId)
                    if all(done):
                        break
                for index, env, info, trace, action_trace in zip(indices, active_envs, metrics, traces, actions):
                    success = info['distance_to_goal'] < 3.
                    oracle = min(env.distances[x] for x in trace) < 3.
                    record = {**info, 'episode_index': index, 'scan': env.episode['scan'],
                        'explicit_stop_success': info.get('explicit_stop_success', info['success']), 'success': bool(success),
                        'spl': spl(success, env.shortest, env.travelled),
                        'oracle_success': oracle, 'trajectory': trace, 'actions': action_trace}
                    stream.write(json.dumps(record)+'\n')
                    records.append(record)
                if len(records) % 32 < len(indices):
                    print(json.dumps({'completed': len(records), 'total': len(episodes),
                        'success_rate_so_far': float(np.mean([x['success'] for x in records])),
                        'spl_so_far': float(np.mean([x['spl'] for x in records])),
                        'elapsed_seconds': time.time()-started}), flush=True)
        if len(records) != len(episodes) or {x['episode_id'] for x in records} != {e['episode_id'] for e in episodes}:
            raise RuntimeError('Complete split membership audit failed')
        summary = {**signature, 'episodes': len(records), 'scenes': len({e['scan'] for e in episodes}),
            'success_rate': float(np.mean([x['success'] for x in records])),
            'spl': float(np.mean([x['spl'] for x in records])),
            'navigation_error': float(np.mean([x['distance_to_goal'] for x in records])),
            'path_length': float(np.mean([x['path_length'] for x in records])),
            'oracle_success': float(np.mean([x['oracle_success'] for x in records])),
            'elapsed_seconds': time.time()-started}
        (out/'evaluation.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary), flush=True)
    finally:
        for env in envs:
            env.close()


if __name__ == '__main__':
    main()
