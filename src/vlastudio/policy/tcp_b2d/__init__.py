"""TCP driving baseline using standard VLAStudio training and HF checkpoints."""
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel


class TCPConfig(PretrainedConfig):
    model_type = 'tcp_b2d'

    def __init__(self, planner_type='merge_ctrl_traj', download=True, freeze_batch_norm=True, **kwargs):
        super().__init__(**kwargs)
        if planner_type not in ['only_traj', 'only_ctrl', 'merge_ctrl_traj']:
            raise ValueError('planner_type must be only_traj, only_ctrl or merge_ctrl_traj')
        self.planner_type, self.download = planner_type, download
        self.freeze_batch_norm = freeze_batch_norm
        self.ctrl_space, self.ctrl_type = 'vehicle', 'control'
        self.state_dim, self.action_dim, self.chunk_size = 1, 4, 1
        self.network_state_dim = 9
        self.action_order = ['steer', 'throttle', 'brake', 'reverse']


class TCPPolicy(PreTrainedModel):
    config_class = TCPConfig

    def __init__(self, config):
        super().__init__(config)
        from .upstream import network
        self.navigator = network(config.download)
        self.controllers = {}
        from .inputs import TCPDataProcessor
        self.data_processor = TCPDataProcessor()

    def reset(self):
        self.controllers.clear()

    def save_pretrained(self, save_directory, *args, **kwargs):
        super().save_pretrained(save_directory, *args, **kwargs)
        target = Path(save_directory)
        metadata_path = target/'policy_metadata.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        metadata.update(policy_module='vlastudio.policy.tcp_b2d', policy_name='tcp_b2d')
        metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
        normalization = target/'normalize.json'
        if not normalization.exists():
            normalization.write_text(json.dumps({'version': '2.0', 'datasets': [{'dataset_id': 'bench2drive',
                'ctrl_space': 'vehicle', 'ctrl_type': 'control'}],
                'state': {'bench2drive': 'identity'}, 'action': {'bench2drive': 'identity'}}, indent=2)+'\n')

    def train(self, mode=True):
        super().train(mode)
        if mode and self.config.freeze_batch_norm:
            for module in self.navigator.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def predict(self, image, state, raw_lang=None):
        from .inputs import network_state
        return self._predict(image, network_state(state, raw_lang))

    def _predict(self, image, state):
        if image.ndim == 5:
            image = image[:, 0]
        if tuple(image.shape[1:]) != (3, 256, 900) or state.shape[-1] != 9:
            raise ValueError('TCP requires a 3x256x900 panorama and [speed, target_xy, command_one_hot] state')
        image = image.float()/255.
        mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
        std = image.new_tensor([.229, .224, .225])[None, :, None, None]
        measured = state.float().clone()
        measured[:, 0] /= 12.
        return self.navigator((image-mean)/std, measured, measured[:, 1:3])

    def forward(self, image, state, raw_lang=None, tcp_state=None, waypoints=None, action_index=None, future_action_index=None,
                expert_features=None, expert_value=None, future_features=None, return_loss=True):
        prediction = self._predict(image, tcp_state) if tcp_state is not None else self.predict(image, state, raw_lang)
        if waypoints is None:
            return prediction
        loss = F.l1_loss(prediction['pred_wp'], waypoints.float())
        loss = loss+.05*F.l1_loss(prediction['pred_speed'], state[:, :1].float()/12.)
        loss = loss+F.cross_entropy(prediction['action_index'], action_index.long())
        loss = loss+.001*(F.mse_loss(prediction['pred_value_traj'], expert_value.float()[:, None])+
                          F.mse_loss(prediction['pred_value_ctrl'], expert_value.float()[:, None]))
        loss = loss+.05*(F.mse_loss(prediction['pred_features_traj'], expert_features.float())+
                         F.mse_loss(prediction['pred_features_ctrl'], expert_features.float()))
        for index in range(4):
            loss = loss+F.cross_entropy(prediction['future_action_index'][index], future_action_index[:, index].long())/4
            loss = loss+.05*F.mse_loss(prediction['future_feature'][index], future_features[:, index].float())/4
        return {'loss': loss}

    @torch.inference_mode()
    def select_action(self, obs):
        from .upstream import modules
        from .inputs import network_state
        device = self.device
        state = torch.as_tensor(obs['state'], device=device, dtype=torch.float32)
        state = (torch.as_tensor(obs['tcp_state'], device=device, dtype=torch.float32)
                 if 'tcp_state' in obs else network_state(state, obs.get('raw_lang')))
        image = torch.as_tensor(obs['image'], device=device)
        prediction = self._predict(image, state)
        resets = torch.as_tensor(obs['timestamp']).reshape(-1).cpu().tolist()
        module, _ = modules(self.config.download)
        controls = []
        for index in range(len(state)):
            if index not in self.controllers or resets[index] == 0:
                cfg = self.navigator.config
                self.controllers[index] = (module.PIDController(cfg.turn_KP, cfg.turn_KI, cfg.turn_KD, cfg.turn_n),
                                           module.PIDController(cfg.speed_KP, cfg.speed_KI, cfg.speed_KD, cfg.speed_n))
            self.navigator.turn_controller, self.navigator.speed_controller = self.controllers[index]
            planner = self.config.planner_type
            if planner != 'only_ctrl':
                reverse = False
                steer, throttle, brake, _ = self.navigator.control_pid(
                    prediction['pred_wp'][index:index+1].clone(), state[index:index+1, 0], state[index:index+1, 1:3].clone())
            if planner != 'only_traj':
                action = prediction['action_index'][index].argmax().item()
                ctrl_throttle, ctrl_steer, ctrl_brake, ctrl_reverse = module.Discrete_Actions_DICT[action]
                if planner == 'merge_ctrl_traj':
                    steer = .5*steer+.5*ctrl_steer
                    throttle = .5*throttle+.5*ctrl_throttle
                    brake = max(brake, ctrl_brake)
                else:
                    throttle, steer, brake, reverse = ctrl_throttle, ctrl_steer, ctrl_brake, ctrl_reverse
            speed_threshold = 1. if abs(steer) > .07 else 1.5
            throttle = np.clip(throttle, 0., .05 if float(state[index, 0]) > speed_threshold else .5)
            brake = float(brake > 0)
            if brake > .5:
                throttle = 0.
            controls.append([float(np.clip(steer, -1., 1.)), float(throttle), brake, float(reverse)])
        return torch.tensor(controls, device=device, dtype=torch.float32)[:, None]


TCP_WEIGHTS_REVISION = '140aea58c02185e787c3df39d7ad79e910967b8b'


def official_model(config):
    from vlastudio.utils.assets import hf_asset, sha256
    from vlastudio.core.paths import cache_root
    source = hf_asset('rethinklab/Bench2DriveZoo', 'tcp_b2d.ckpt', TCP_WEIGHTS_REVISION, download=config.download, cache_dir=cache_root()/'driving/huggingface')
    if sha256(source) != 'e6573ff1f8ea910b9a53eddfb68f69cac469bf5bfa253a516578f6126110b4fe':
        raise ValueError('Official TCP checkpoint checksum mismatch')
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    weights = {key.removeprefix('model.'): value for key, value in checkpoint['state_dict'].items()}
    model = TCPPolicy(config)
    model.navigator.load_state_dict(weights, strict=True)
    return model


def load_model(args):
    if getattr(args, 'is_training', False):
        reference = getattr(args, 'pretrained_weight_path', None)
    else:
        reference = args.model_name_or_path
    config = TCPConfig(planner_type=getattr(args, 'planner_type', 'merge_ctrl_traj'), download=getattr(args, 'download', True),
                       freeze_batch_norm=getattr(args, 'freeze_batch_norm', True))
    if reference == 'official':
        model = official_model(config)
    elif reference:
        model = TCPPolicy.from_pretrained(reference)
    else:
        model = TCPPolicy(config)
    if not getattr(args, 'is_training', False):
        model = model.to(getattr(args, 'device', 'cpu')).eval()
    return {'model': model}


def get_data_processor(args, model_components):
    from .inputs import TCPDataProcessor
    return TCPDataProcessor()


def get_data_collator(args, model_components):
    if not getattr(args, 'is_training', False):
        return None
    keys = ['image', 'state', 'tcp_state', 'waypoints', 'action_index', 'future_action_index',
            'expert_features', 'expert_value', 'future_features']
    def collate(samples):
        from .inputs import TCPDataProcessor
        processor = TCPDataProcessor()
        samples = [row if 'tcp_state' in row else processor(row) for row in samples]
        batch = {key: torch.as_tensor(np.stack([row[key] for row in samples])) for key in keys}
        batch['raw_lang'] = [row['raw_lang'] for row in samples]
        return batch
    return collate


def trainer_class():
    from transformers import Trainer as HFTrainer
    from vlastudio.policy.trainer import BaseTrainer
    class SupervisedDrivingTrainer(BaseTrainer):
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
            return HFTrainer.evaluate(self, eval_dataset, ignore_keys, metric_key_prefix)
    return SupervisedDrivingTrainer


class Trainer:
    def __new__(cls, *args, **kwargs):
        return trainer_class()(*args, **kwargs)
