"""Evaluate the learned CMA policy on a complete, fixed VLN-CE split.

Native environments run in spawned processes; policy batching uses the same
MetaObs -> MetaPolicy path as eval_sim. Journals can resume only with identical
checkpoint/config hashes. No teacher, goal sensor, or reference route is queried
by the policy. Simulator metrics remain the scoring authority.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
import signal
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import yaml


def worker(pipe, config):
    os.environ.setdefault('HABITAT_SIM_LOG', 'quiet')
    os.environ.setdefault('MAGNUM_LOG', 'quiet')
    import torch
    torch.set_num_threads(1)
    from vlastudio.benchmark.habitat import HabitatLabEnv
    env = None
    try:
        env = HabitatLabEnv(SimpleNamespace(**config))
        while True:
            command, value = pipe.recv()
            if command == 'close':
                break
            if command == 'catalog':
                pipe.send(('ok', [(str(e.episode_id), e.scene_id) for e in env.env.episodes]))
                continue
            if command == 'reset':
                env.index = value
                obs = env.reset()
                previous_position = np.array(env.env.sim.get_agent_state().position)
                travelled = 0.
                oracle_success = env.env.get_metrics()['distance_to_goal'] < 3.
                pipe.send(('ok', asdict(obs)))
            elif command == 'step':
                obs, reward, done, info = env.step(value)
                position = np.array(env.env.sim.get_agent_state().position)
                travelled += float(np.linalg.norm(position - previous_position))
                previous_position = position
                oracle_success |= info['distance_to_goal'] < 3.
                info.update(path_length=travelled, oracle_success=bool(oracle_success),
                            steps=env.steps, stopped=value == 0)
                pipe.send(('ok', (obs, reward, done, info)))
            else:
                raise ValueError('Unknown worker command')
    except BaseException as error:
        import traceback
        pipe.send(('error', traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        pipe.close()


def receive(pipe):
    status, value = pipe.recv()
    if status != 'ok':
        raise RuntimeError(value)
    return value


def main(default_config=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=default_config)
    parser.add_argument('--model', default='official')
    parser.add_argument('--episodes', type=int, default=1839)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/example09_cma_official')
    args = parser.parse_args()
    from vlastudio.utils.assets import resolve_checkpoint
    from vlastudio.policy.vlnce_cma.pretrained import official_checkpoint
    args.model = str(resolve_checkpoint(args.model, official=official_checkpoint))
    if args.episodes < 1 or args.batch_size < 1:
        parser.error('Episode count and batch size must be positive')
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    signature = {'model_sha256': hashlib.sha256((model_path / 'policy.pt').read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(), 'episodes': args.episodes}
    manifest = output / 'run.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Existing journal belongs to another checkpoint or evaluation protocol')
    manifest.write_text(json.dumps(signature, indent=2) + '\n')
    journal = output / 'episodes.jsonl'
    records = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    completed = {row['episode_index'] for row in records}
    if len(completed) != len(records):
        raise ValueError('Duplicate episode indices in journal')
    context = mp.get_context('spawn')
    count = min(args.batch_size, args.episodes - len(records))
    pipes, processes, indices, observations, traces = [], [], [], [], []
    started = time.time()
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    try:
        for _ in range(count):
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(child, config))
            process.start()
            child.close()
            pipes.append(parent)
            processes.append(process)
            traces.append([])
        if pipes:
            pipes[0].send(('catalog', None))
            catalog = receive(pipes[0])
            if len(catalog) != args.episodes:
                raise ValueError('Requested count must equal the complete native split')
            if len({episode for episode, _ in catalog}) != args.episodes:
                raise ValueError('Native dataset contains duplicate episode IDs')
            # Keep each worker in one scene where possible; policy resets still
            # occur per episode. This changes scheduling, never split membership.
            groups = defaultdict(list)
            for index, (_, scene) in enumerate(catalog):
                if index not in completed:
                    groups[scene].append(index)
            assignments = [[] for _ in pipes]
            for group in sorted(groups.values(), key=len, reverse=True):
                slot = min(range(count), key=lambda i: len(assignments[i]))
                assignments[slot].extend(group)
            remaining = [iter(group) for group in assignments]
            for slot, pipe in enumerate(pipes):
                index = next(remaining[slot], None)
                indices.append(index)
                # Idle slots still need a structurally valid observation.
                pipe.send(('reset', index if index is not None else 0))
        observations = [receive(pipe) for pipe in pipes]
        # Initialize CUDA only after spawning native simulator workers.
        import torch
        torch.set_num_threads(2)
        from vlastudio.benchmark.base import MetaPolicy
        from vlastudio.benchmark.utils import organize_obs
        from vlastudio.data_utils.normalize import Identity
        from vlastudio.policy.vlnce_cma import CMAPolicy
        model = CMAPolicy.from_pretrained(model_path, args.device)
        normalizer = Identity(ctrl_space='navigation', ctrl_type='discrete')
        policy = MetaPolicy(model, state_normalizer=normalizer, action_normalizer=normalizer,
                            ctrl_space='navigation', ctrl_type='discrete')
        with journal.open('a', buffering=1) as stream:
            while any(index is not None for index in indices):
                actions = policy.inference(organize_obs(observations))[0]
                for slot, (pipe, action) in enumerate(zip(pipes, actions)):
                    if indices[slot] is not None:
                        value = int(action['action'][0])
                        traces[slot].append(value)
                        pipe.send(('step', value))
                for slot, pipe in enumerate(pipes):
                    if indices[slot] is None:
                        continue
                    obs, _, done, info = receive(pipe)
                    observations[slot] = obs
                    if not done:
                        continue
                    record = {'episode_index': indices[slot], **info, 'actions': traces[slot]}
                    stream.write(json.dumps(record) + '\n')
                    records.append(record)
                    next_index = None if stop_requested else next(remaining[slot], None)
                    indices[slot], traces[slot] = next_index, []
                    if next_index is not None:
                        pipe.send(('reset', next_index))
                        observations[slot] = receive(pipe)
                    if len(records) % 20 == 0:
                        print(json.dumps({'completed': len(records), 'total': args.episodes,
                            'success_rate_so_far': float(np.mean([row['success'] for row in records])),
                            'spl_so_far': float(np.mean([row['spl'] for row in records])),
                            'elapsed_seconds': time.time() - started}), flush=True)
        if stop_requested:
            print(json.dumps({'interrupted': True, 'completed': len(records), 'resume_output': str(output)}), flush=True)
            return
        if len(records) != args.episodes or len({row['episode_id'] for row in records}) != args.episodes:
            raise RuntimeError('Full split did not produce exactly one result per episode')
        summary = {**signature, 'successes': sum(row['success'] for row in records), 'total': len(records),
            'success_rate': float(np.mean([row['success'] for row in records])),
            'spl': float(np.mean([row['spl'] for row in records])),
            'navigation_error': float(np.mean([row['distance_to_goal'] for row in records])),
            'oracle_success': float(np.mean([row['oracle_success'] for row in records])),
            'path_length': float(np.mean([row['path_length'] for row in records])),
            'elapsed_seconds': time.time() - started}
        (output / 'evaluation.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
    finally:
        for pipe, process in zip(pipes, processes):
            if process.is_alive():
                try:
                    pipe.send(('close', None))
                except (BrokenPipeError, EOFError):
                    pass
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for pipe in pipes:
            pipe.close()


if __name__ == '__main__':
    main()
