"""Shared V1.5 configuration, reproducibility and checkpoint I/O."""
import json
from pathlib import Path

import numpy as np
import torch

from protocast.config import normalize_saved_config


def dump_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config(config):
    data = config['data']
    return dict(config['model'], height=data['height'], width=data['width'],
                input_steps=data['input_steps'], output_steps=data['output_steps'],
                channels=len(data['variables']),
                target_indices=[data['variables'].index(v) for v in data['targets']])


def save_checkpoint(path, payload):
    temporary = Path(str(path) + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    # Paths are local experiment artifacts, containing optimizer/RNG metadata.
    saved = torch.load(path, map_location=device, weights_only=False)
    saved['config'] = normalize_saved_config(saved['config'])
    normalization = saved['normalization']
    if 'benchmark' not in normalization:
        normalization.pop('table', None)
        normalization['benchmark'] = saved['config']['benchmark']
    return saved
