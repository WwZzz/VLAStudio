"""Parse the driving task text inside the policy, preserving route geometry."""
import json
import numpy as np
import torch

from vlastudio.data_utils.datasets.bench2drive import COMMANDS, TARGET_PREFIX


def encode_task(state, text):
    speed = np.asarray(state, dtype=np.float32)
    if speed.shape != (1,) or not np.isfinite(speed).all():
        raise ValueError('TCP observation state must contain one finite speed in m/s')
    if not isinstance(text, str):
        raise ValueError('TCP requires the complete raw_lang driving task')
    command, separator, goal = text.partition('\n')
    if command.strip().lower() not in COMMANDS:
        raise ValueError(f'TCP supports only the six canonical driving instructions: {COMMANDS}')
    if not separator or not goal.startswith(TARGET_PREFIX):
        raise ValueError('TCP task text must declare an ego-frame local waypoint in meters with forward/right axes')
    try:
        point = np.asarray(json.loads(goal[len(TARGET_PREFIX):]), dtype=np.float32)
    except (ValueError, TypeError) as error:
        raise ValueError('TCP task text must contain a numeric two-dimensional route waypoint') from error
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('TCP requires a finite two-dimensional route waypoint in task text')
    index = COMMANDS.index(command.strip().lower())
    return np.concatenate([speed, point, np.eye(6, dtype=np.float32)[index]])


class TCPDataProcessor:
    def __call__(self, sample):
        # Keep public state intact; only this policy consumes the private vector.
        reasoning = sample.get('reasoning')
        supervision = reasoning.get('supervision', {}) if isinstance(reasoning, dict) else {}
        keys = ('waypoints', 'action_index', 'future_action_index', 'expert_features', 'expert_value', 'future_features')
        targets = {key: supervision[key] for key in keys if key in supervision}
        return {**sample, **targets, 'tcp_state': encode_task(sample['state'], sample.get('raw_lang'))}


def network_state(state, raw_lang):
    """Adapt direct model calls that bypass the per-sample data processor."""
    if state.ndim != 2 or state.shape[1] != 1:
        raise ValueError('TCP observation state must be [batch, 1]: speed in m/s')
    texts = [raw_lang] if isinstance(raw_lang, str) else raw_lang
    if texts is None or len(texts) != len(state):
        raise ValueError('TCP requires one raw_lang entry per observation')
    rows = [encode_task(row, text) for row, text in zip(state.detach().cpu().numpy(), texts)]
    return torch.as_tensor(np.stack(rows), device=state.device, dtype=torch.float32)
