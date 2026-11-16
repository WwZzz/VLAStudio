"""Convert an author-published legacy CMA checkpoint to tensor/plain-data format.

Uses PyTorch's restricted weights-only loader, with the legacy Habitat config
mapped to a known data-only dictionary class. No unrestricted pickle load.
"""

import copy
import hashlib
import json
from pathlib import Path
import torch
from yacs.config import CfgNode
from vlastudio.utils.assets import fetch
from vlastudio.core.paths import cache_root as shared_cache


CMA_SHA = '06278fdbb52ad8c453b404dacb079fb353252056314b1c567a381df77e6827a3'


class LegacyConfig(CfgNode):
    pass


LegacyConfig.__module__ = 'habitat.config.default'
LegacyConfig.__name__ = 'Config'


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unexpected config value: {type(value).__name__}')


def convert_checkpoint(source, output, expected_sha256):
    from types import SimpleNamespace
    args = SimpleNamespace(source=str(source), output=str(output), sha256=expected_sha256)
    source = Path(args.source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != args.sha256:
        raise ValueError('Source checkpoint hash mismatch')
    torch.serialization.add_safe_globals([LegacyConfig, set])
    try:
        checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    finally:
        torch.serialization.clear_safe_globals()
    config = plain(checkpoint['config'])
    model_config = copy.deepcopy(config.get('MODEL', config.get('VLN')))
    if model_config is None:
        raise ValueError('Missing model configuration')
    if 'RGB_ENCODER' not in model_config:
        model_config['RGB_ENCODER'] = model_config.pop('VISUAL_ENCODER')
    for field in ['RGB_ENCODER', 'DEPTH_ENCODER']:
        model_config[field]['trainable'] = False
    model_config['DEPTH_ENCODER']['ddppo_checkpoint'] = 'NONE'
    model_config['INSTRUCTION_ENCODER'].update(sensor_uuid='instruction',
        use_pretrained_embeddings=False, final_state_only=False)
    for field in ['normalize_rgb', 'ablate_instruction', 'ablate_rgb', 'ablate_depth']:
        model_config.setdefault(field, False)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': checkpoint['state_dict'], 'model_config': model_config,
                'source_sha256': digest}, output / 'policy.pt')
    (output / 'source_config.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'config.json').write_text(json.dumps({'ctrl_space': 'navigation', 'ctrl_type': 'discrete',
        'chunk_size': 1, 'state_dim': 0, 'action_dim': 1}))
    (output / 'policy_metadata.json').write_text(json.dumps({'policy_module': 'policy.vlnce_cma', 'policy_name': 'vlnce_cma'}))
    (output / 'normalize.json').write_text(json.dumps({'datasets': [{'dataset_id': 'r2r_vlnce',
        'ctrl_space': 'navigation', 'ctrl_type': 'discrete'}], 'state': {'r2r_vlnce': 'identity'},
        'action': {'r2r_vlnce': 'identity'}}))
    print(f'Converted {len(checkpoint["state_dict"])} tensors; original checkpoint SHA256={digest}')


def official_checkpoint(download=True):
    from filelock import FileLock
    cache = shared_cache() / "navigation"
    root = cache / "models/vlnce_cma"
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root) + ".lock"):
        required = ['policy.pt', 'config.json', 'policy_metadata.json', 'normalize.json']
        if all((root / file).is_file() for file in required):
            return root
        source = fetch('https://drive.usercontent.google.com/download?id=1o9PgBT38BH9pw_7V1QB3XUkY8auJqGKw&export=download&confirm=t',
                       cache/'downloads/cma-pm-da-aug.pth', CMA_SHA, download)
        convert_checkpoint(source, root, CMA_SHA)
    return root
