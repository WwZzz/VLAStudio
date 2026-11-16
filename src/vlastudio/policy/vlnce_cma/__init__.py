"""Official VLN-CE CMA policy using RGB, normalized depth, and instruction tokens.

The recurrent state is owned by the policy. Goal sensors, reference paths and
expert actions are never policy inputs. See vendor/PROVENANCE.json.
"""
from pathlib import Path
import copy
import json

import numpy as np
import torch
from torch import nn
from gym import spaces
from yacs.config import CfgNode

from .vendor.cma import CMANet


class CategoricalHead(nn.Module):
    def __init__(self, hidden_size, action_count):
        super().__init__()
        self.linear = nn.Linear(hidden_size, action_count)


class CMAPolicy(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = copy.deepcopy(model_config)
        observation_space = spaces.Dict({
            'rgb': spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
            'depth': spaces.Box(0., 1., shape=(256, 256, 1), dtype=np.float32),
            'instruction': spaces.Box(0, 2503, shape=(200,), dtype=np.int64)})
        self.net = CMANet(observation_space, CfgNode(model_config), num_actions=4)
        self.net.instruction_encoder.embedding_layer.weight.requires_grad_(
            bool(model_config['INSTRUCTION_ENCODER'].get('fine_tune_embeddings', False)))
        self.action_distribution = CategoricalHead(self.net.output_size, 4)
        from .tokenizer import InstructionTokenizer
        self.instruction_tokenizer = InstructionTokenizer()
        self.reset()

    def reset(self):
        self.hidden = None
        self.prev_actions = None

    def initial_state(self, batch_size):
        return torch.zeros(batch_size, self.net.num_recurrent_layers,
                           self.net.output_size, device=next(self.parameters()).device)

    def forward(self, observations=None, hidden=None, previous_actions=None, masks=None, episodes=None, return_loss=True):
        if episodes is not None:
            from vlastudio.policy.navigation_training import cma_loss
            return cma_loss(self, episodes)
        features, hidden = self.net(observations, hidden, previous_actions, masks)
        return self.action_distribution.linear(features), hidden

    def train(self, mode=True):
        super().train(mode)
        self.net.rgb_encoder.cnn.eval()
        self.net.depth_encoder.visual_encoder.eval()
        return self

    def save_pretrained(self, path, state_dict=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': state_dict if state_dict is not None else self.state_dict(),
                    'model_config': self.model_config}, path/'policy.pt')
        (path/'config.json').write_text(json.dumps({'ctrl_space': 'navigation', 'ctrl_type': 'discrete',
                                                  'chunk_size': 1, 'state_dim': 0, 'action_dim': 1}))
        (path/'policy_metadata.json').write_text(json.dumps({'policy_module': 'policy.vlnce_cma', 'policy_name': 'vlnce_cma'}))
        self.instruction_tokenizer.save(path)

    def select_action(self, obs):
        device = next(self.parameters()).device
        rgb = torch.as_tensor(obs['image'], device=device)
        depth = torch.as_tensor(obs['depth'], device=device, dtype=torch.float32)
        if rgb.ndim != 5 or tuple(rgb.shape[1:]) != (1, 3, 224, 224):
            raise ValueError('CMA requires one 224x224 RGB camera')
        if depth.ndim != 4 or tuple(depth.shape[1:]) != (1, 256, 256):
            raise ValueError('CMA requires one normalized 256x256 depth camera')
        if not torch.isfinite(depth).all() or depth.min() < 0 or depth.max() > 1:
            raise ValueError('CMA depth must be finite and normalized to [0,1] using 0-10m')
        tokens = torch.tensor([self.instruction_tokenizer(text) for text in obs['raw_lang']], device=device)
        masks = (torch.as_tensor(obs['timestamp'], device=device).reshape(-1, 1) > 0)
        if self.hidden is None or self.hidden.shape[0] != rgb.shape[0]:
            self.hidden = self.initial_state(rgb.shape[0])
            self.prev_actions = torch.zeros(rgb.shape[0], 1, device=device, dtype=torch.long)
        observations = {'rgb': rgb[:, 0].permute(0, 2, 3, 1),
                        'depth': depth[:, 0, :, :, None], 'instruction': tokens}
        with torch.inference_mode():
            logits, self.hidden = self(observations, self.hidden, self.prev_actions, masks)
            self.prev_actions = logits.argmax(-1, keepdim=True)
        return self.prev_actions.float().reshape(-1, 1, 1)

    @classmethod
    def from_pretrained(cls, path, device='cpu', download=True):
        from vlastudio.utils.assets import resolve_checkpoint
        from vlastudio.policy.vlnce_cma.pretrained import official_checkpoint
        path = resolve_checkpoint(path, official=official_checkpoint, download=download)
        checkpoint = torch.load(path / 'policy.pt' if path.is_dir() else path,
                                map_location='cpu', weights_only=True)
        model = cls(checkpoint['model_config'])
        model.instruction_tokenizer.download = download
        vocabulary = path/'instruction_vocab.json' if path.is_dir() else path.with_name('instruction_vocab.json')
        if vocabulary.exists():
            model.instruction_tokenizer.vocabulary = json.loads(vocabulary.read_text())
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        return model.to(device).eval()


def load_model(args):
    if getattr(args, 'is_training', False):
        reference = getattr(args, 'pretrained_weight_path', None)
        if reference:
            model = CMAPolicy.from_pretrained(reference, download=getattr(args, 'download', True))
        else:
            config = json.loads(Path(__file__).with_name('default_config.json').read_text())
            model = CMAPolicy(config)
    else:
        model = CMAPolicy.from_pretrained(args.model_name_or_path, getattr(args, 'device', 'cpu'), getattr(args, 'download', True))
    model.instruction_tokenizer.download = getattr(args, 'download', True)
    if getattr(args, 'vocabulary', None):
        from .tokenizer import InstructionTokenizer
        model.instruction_tokenizer = InstructionTokenizer(json.loads(Path(args.vocabulary).read_text()),
                                                          getattr(args, 'download', True))
    return {'model': model}


def get_data_processor(args, model_components):
    if not getattr(args, 'is_training', False):
        return None
    from vlastudio.policy.navigation_processing import NavigationProcessor
    return NavigationProcessor(model_components['model'], 'cma')


def get_data_collator(args, model_components):
    if not getattr(args, 'is_training', False):
        return None
    from vlastudio.policy.navigation_training import collate_trajectories
    return collate_trajectories


from vlastudio.policy.navigation_training import Trainer
