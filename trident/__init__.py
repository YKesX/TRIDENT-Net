"""
TRIDENT modular multimodal fusion system.

A PyTorch-based framework for multimodal data fusion across visual, radar, and thermal modalities.
"""

__version__ = "0.1.0"
__author__ = "TRIDENT Team"

from .common.types import EventToken, FeatureVec, OutcomeEstimate
from .common.types import BranchModule, FusionModule, GuardModule

__all__ = [
    "EventToken",
    "FeatureVec", 
    "OutcomeEstimate",
    "BranchModule",
    "FusionModule", 
    "GuardModule",
]