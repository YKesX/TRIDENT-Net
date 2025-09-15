"""Dataset and transforms tests for TRIDENT-Net data stack."""

import torch
import numpy as np
import yaml
import sys

sys.path.append('.')

from trident.data.synthetic import generate_synthetic_batch
from trident.data.collate import pad_tracks_collate
from trident.data.transforms import AlbuStereoClip


def test_synthetic_batch_shapes():
    batch = generate_synthetic_batch(B=2, T=16, H=720, W=1280)
    assert batch['rgb'].shape == (2, 3, 16, 720, 1280)
    assert batch['ir'].shape == (2, 1, 16, 720, 1280)
    assert batch['kin'].shape == (2, 3, 9)
    assert 'labels' in batch and set(batch['labels'].keys()) == {'hit', 'kill'}


def test_collate_padding_variable_T():
    a = {
        'rgb': torch.rand(3, 10, 720, 1280),
        'ir': torch.rand(1, 10, 720, 1280),
        'kin': torch.zeros(3, 9),
        'labels': {'hit': torch.tensor([1.0]), 'kill': torch.tensor([0.0])},
    }
    b = {
        'rgb': torch.rand(3, 12, 720, 1280),
        'ir': torch.rand(1, 12, 720, 1280),
        'kin': torch.zeros(3, 9),
        'labels': {'hit': torch.tensor([0.0]), 'kill': torch.tensor([0.0])},
    }
    collated = pad_tracks_collate([a, b])
    assert collated['rgb'].shape == (2, 3, 12, 720, 1280)
    assert collated['ir'].shape == (2, 1, 12, 720, 1280)
    assert collated['kin'].shape == (2, 3, 9)
    assert collated['labels']['hit'].shape == (2, 1)
    assert collated['labels']['kill'].shape == (2, 1)


def test_transforms_temporal_and_spatial():
    # Create simple sequences
    T = 8
    rgb_np = (np.random.rand(T, 720, 1280, 3) * 255).astype(np.uint8)
    ir_np = (np.random.rand(T, 720, 1280, 1) * 255).astype(np.uint8)
    tr = AlbuStereoClip(
        rgb_ops=[{"name": "HorizontalFlip", "p": 0.5}],
        ir_ops=[{"name": "GaussNoise", "p": 0.1, "var_limit": (10.0, 20.0)}],
        temporal={"jitter_frames": 1, "dropout_frames_p": 0.05},
        image_size={"h": 720, "w": 1280},
    )
    rgb_t, ir_t = tr(rgb_np, ir_np)
    assert rgb_t.shape == (3, tr._temporal_indices(T).__len__(), 720, 1280)
    assert ir_t.shape[0] == 1 and ir_t.shape[2:] == (720, 1280)


def test_config_native_resolution():
    with open('tasks.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    isz = cfg.get('preprocess', {}).get('image_size', {})
    assert isz.get('h') == 720 and isz.get('w') == 1280
