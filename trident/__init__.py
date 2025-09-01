"""
TRIDENT-Net: Modular Multimodal Fusion System

A PyTorch-based system for multimodal sensor fusion with explainable AI capabilities.
Supports visible/EO, radar, and thermal/IR modalities.

Author: Yağızhan Keskin
"""

__version__ = "0.1.0"
__author__ = "Yağızhan Keskin"

from . import common, data, i_models, r_models, t_models, fusion_guard, runtime
from . import trident_i, trident_t, trident_r

__all__ = [
    "common",
    "data", 
    "i_models",
    "r_models",
    "t_models",
    "trident_i",
    "trident_t", 
    "trident_r",
    "fusion_guard",
    "runtime",
]