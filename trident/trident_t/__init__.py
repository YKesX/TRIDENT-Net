"""TRIDENT-T: Thermal/IR sensor processing modules."""

from .plumedet_lite import PlumeDetLite
from .coolcurve3 import CoolCurve3

__all__ = ["PlumeDetLite", "CoolCurve3"]