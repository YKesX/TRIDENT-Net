"""TRIDENT-T: IR/Thermal processing modules."""

from .t1_plume_net import PlumeNet
from .t2_cooling_curve import CoolingCurve

__all__ = [
    "PlumeNet",
    "CoolingCurve",
]