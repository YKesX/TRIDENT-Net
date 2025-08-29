"""TRIDENT-I: EO/Visible processing modules."""

from .i1_frag_cnn import FragCNN
from .i2_therm_att_v import ThermAttentionV
from .i3_dual_vision import DualVisionNet

__all__ = [
    "FragCNN",
    "ThermAttentionV", 
    "DualVisionNet",
]