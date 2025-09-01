"""Fusion and guard modules for TRIDENT-Net."""

from .f1_late_svm import LateFusionSVM
from .f2_cross_attention import CrossAttentionFusion
from .f3_fuzzy_rules import FuzzyRuleOverlay
from .s_spoof_shield import SpoofShield

__all__ = ["LateFusionSVM", "CrossAttentionFusion", "FuzzyRuleOverlay", "SpoofShield"]