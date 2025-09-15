"""
TRIDENT-Net data package.

Provides dataset, transforms, video ring buffer, collate, and synthetic helpers.

This package is domain-neutral and adheres to strict shape contracts:
- RGB: Float[3, T, 720, 1280]
- IR:  Float[1, T, 720, 1280]
- Kin: Float[3, 9]

Exports
-------
- VideoJsonlDataset: JSONL-backed paired RGB/IR dataset with timing windows.
- AlbuStereoClip: Synchronized Albumentations transforms for stereo clips.
- VideoRing: CPU ring buffer-based video loader.
- pad_tracks_collate: Collate that pads variable-T and stacks labels.
- generate_synthetic_batch, synthetic_jsonl: Synthetic utilities.
"""

# Silence Albumentations update check to reduce noisy warnings in tests/CI
import os as _os
_os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from .dataset import VideoJsonlDataset, create_data_loaders
from .transforms import AlbuStereoClip
from .video_ring import VideoRing
from .collate import pad_tracks_collate
from .synthetic import generate_synthetic_batch, synthetic_jsonl, SyntheticVideoJsonl

__all__ = [
    "VideoJsonlDataset",
    "create_data_loaders",
    "AlbuStereoClip",
    "VideoRing",
    "pad_tracks_collate",
    "generate_synthetic_batch",
    "synthetic_jsonl",
    "SyntheticVideoJsonl",
]
