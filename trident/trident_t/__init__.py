"""TRIDENT-T: Thermal/IR sensor processing modules."""

from .plumedet_lite import PlumeDetLite
from .coolcurve3 import CoolCurve3

# New modules
from .ir_dettrack_v2 import PlumeDetXL

__all__ = [
    "PlumeDetLite", 
    "CoolCurve3",
    "PlumeDetXL"
]