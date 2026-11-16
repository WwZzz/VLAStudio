"""PREVALENT Recurrent VLN-BERT with the published R2R input/action protocol."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .vendor.model import VLNBERT


def angle_features(heading, elevation):
    return np.tile([np.sin(heading), np.cos(heading), np.sin(elevation), np.cos(elevation)], 32).astype(np.float32)


class RecurrentVLNBERT(nn.Module):
    def __init__(self, feature_store, vocabulary):
        super().__init__()
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer(vocab_file=str(vocabulary), do_lower_case=True)
        self.vocabulary = Path(vocabulary)
        self.feature_store = Path(feature_store) if feature_store else None
        self.download = True
        self._feature_data = None
        self.navigator = VLNBERT(feature_size=2176)
        self.reset()

    def reset(self):
        self.language = None
        self.language_mask = None
        self.hidden = None

    def encode_instructions(self, instructions):
        encoded = []
        for text in instructions:
            tokens = self.tokenizer.tokenize(text)[:78]
            tokens = ['[CLS]']+tokens+['[SEP]']
            tokens += ['[PAD]']*(80-len(tokens))
            encoded.append(self.tokenizer.convert_tokens_to_ids(tokens))
        ids = torch.tensor(encoded, device=next(self.parameters()).device, dtype=torch.long)
        mask = ids != 0
        hidden, language = self.navigator('language', sentence=ids, attention_mask=mask.long(), lang_mask=mask.long())
        return hidden, language, mask

    def candidate_inputs(self, navigation, states, panoramas=None):
        if panoramas is None:
            if self._feature_data is None:
                from vlastudio.data_utils.datasets.navigation_features import PanoramaFeatureStore
                self._feature_data = PanoramaFeatureStore(self.feature_store, self.download)
            panoramas = [self._feature_data.read(row['scan_id'], row['viewpoint_id']) for row in navigation]
        if isinstance(panoramas, torch.Tensor):
            panoramas = panoramas.detach().cpu().numpy()
        device = next(self.parameters()).device
        lengths = [len(row['candidates'])+1 for row in navigation]
        features = np.zeros((len(navigation), max(lengths), 2176), dtype=np.float32)
        actions = []
        for i, row in enumerate(navigation):
            if row.get('candidate_view_alignment') != 'panoramic':
                raise ValueError('Recurrent VLN-BERT requires candidate_view_alignment=panoramic')
            # Only the panorama at the currently observed viewpoint is read.
            panorama = panoramas[i]
            for j, candidate in enumerate(row['candidates']):
                features[i, j, :2048] = panorama[candidate['source_view_index']]
                features[i, j, 2048:] = angle_features(candidate['relative_heading'], candidate['elevation'])
            actions.append([c['action_id'] for c in row['candidates']]+[0])
        state_array = np.asarray(states.detach().cpu() if isinstance(states, torch.Tensor) else states)
        heading_features = np.stack([angle_features(float(s[0]), float(s[1])) for s in state_array])
        mask = torch.arange(max(lengths), device=device)[None] < torch.tensor(lengths, device=device)[:, None]
        return torch.from_numpy(features).to(device), torch.from_numpy(heading_features).to(device), mask, actions

    def navigate(self, language, hidden, language_mask, navigation, states, panoramas=None):
        candidates, angles, valid, action_ids = self.candidate_inputs(navigation, states, panoramas)
        sentence = torch.cat([hidden[:, None], language[:, 1:]], dim=1)
        self.navigator.vln_bert.config.directions = candidates.shape[1]
        hidden, logits = self.navigator('visual', sentence=sentence,
            attention_mask=torch.cat([language_mask, valid], dim=1).long(),
            lang_mask=language_mask.long(), vis_mask=valid.long(), action_feats=angles, cand_feats=candidates)
        return hidden, logits.masked_fill(~valid, -torch.inf), action_ids

    @torch.inference_mode()
    def select_action(self, obs):
        metadata = obs.get('reasoning')
        rows = [row['navigation'] for row in metadata] if metadata else obs['navigation']
        panoramas = obs.get('image') if metadata and all(row.get('image_type') == 'features' for row in metadata) else None
        reset = torch.as_tensor(obs['timestamp']).reshape(-1) == 0
        device = next(self.parameters()).device
        if self.hidden is None or self.hidden.shape[0] != len(rows):
            self.hidden, self.language, self.language_mask = self.encode_instructions(obs['raw_lang'])
        elif reset.any():
            indices = reset.nonzero().flatten().tolist()
            hidden, language, mask = self.encode_instructions([obs['raw_lang'][i] for i in indices])
            self.hidden[indices], self.language[indices], self.language_mask[indices] = hidden, language, mask
        self.hidden, logits, action_ids = self.navigate(self.language, self.hidden,
            self.language_mask, rows, obs['state'], panoramas)
        selected = logits.argmax(-1).tolist()
        return torch.tensor([ids[index] for ids, index in zip(action_ids, selected)],
                            device=device, dtype=torch.float32).reshape(-1, 1, 1)

    @classmethod
    def from_pretrained(cls, path, device='cpu', feature_store=None, download=True):
        from vlastudio.utils.assets import resolve_checkpoint
        from vlastudio.policy.recurrent_vlnbert.pretrained import official_checkpoint
        path = resolve_checkpoint(path, official=official_checkpoint, download=download)
        config = json.loads((path/'config.json').read_text())
        model = cls(feature_store or config.get('feature_store'), path/'vocab.txt')
        model.download = download
        checkpoint = torch.load(path/'policy.pt', map_location='cpu', weights_only=True)
        model.navigator.load_state_dict(checkpoint['state_dict'], strict=True)
        return model.to(device).eval()

    def forward(self, episodes, return_loss=True):
        from vlastudio.policy.navigation_training import vlnbert_loss
        return vlnbert_loss(self, episodes)

    def save_pretrained(self, path, state_dict=None):
        import shutil
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {k.removeprefix('navigator.'): v for k, v in state_dict.items()} if state_dict is not None else self.navigator.state_dict()
        torch.save({'state_dict': state}, path/'policy.pt')
        (path/'config.json').write_text(json.dumps({'ctrl_space': 'navigation', 'ctrl_type': 'candidate_viewpoint',
            'chunk_size': 1, 'state_dim': 3, 'action_dim': 1,
            'feature_store': str(self.feature_store.resolve()) if self.feature_store else None}))
        (path/'policy_metadata.json').write_text(json.dumps({'policy_module': 'policy.recurrent_vlnbert', 'policy_name': 'recurrent_vlnbert'}))
        if self.vocabulary.resolve() != (path/'vocab.txt').resolve():
            shutil.copyfile(self.vocabulary, path/'vocab.txt')


def load_model(args):
    if getattr(args, 'is_training', False):
        reference = getattr(args, 'pretrained_weight_path', None)
        if reference:
            model = RecurrentVLNBERT.from_pretrained(reference, feature_store=getattr(args, 'feature_store', None), download=getattr(args, 'download', True))
        else:
            vocabulary = getattr(args, 'vocabulary', None)
            if not vocabulary or not Path(vocabulary).is_file():
                raise ValueError('Training without pretrained weights requires a local vocabulary path')
            model = RecurrentVLNBERT(getattr(args, 'feature_store', None), vocabulary)
            model.download = getattr(args, 'download', True)
        return {'model': model}
    return {'model': RecurrentVLNBERT.from_pretrained(args.model_name_or_path,
        getattr(args, 'device', 'cpu'), getattr(args, 'feature_store', None), getattr(args, 'download', True))}


def get_data_processor(args, model_components):
    if not getattr(args, 'is_training', False):
        return None
    from vlastudio.policy.navigation_processing import NavigationProcessor
    return NavigationProcessor(model_components['model'], 'vlnbert')


def get_data_collator(args, model_components):
    if not getattr(args, 'is_training', False):
        return None
    from vlastudio.policy.navigation_training import collate_trajectories
    return collate_trajectories


from vlastudio.policy.navigation_training import Trainer
