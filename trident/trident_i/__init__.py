"""TRIDENT-I: Visible/EO sensor processing modules."""

from .frag3d import Frag3D
from .flashnet_v import FlashNetV
from .dualvision import DualVision

# New modules
from .videox3d import VideoFrag3Dv2
from .dualvision_v2 import DualVisionV2

__all__ = [
    "Frag3D", 
    "FlashNetV", 
    "DualVision",
    "VideoFrag3Dv2",
    "DualVisionV2"
]