"""
Core types and contracts for TRIDENT-Net multimodal fusion system.

Author: Yağızhan Keskin
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EventToken:
    """
    Represents a discrete event detected by a sensor module.
    
    Args:
        type: Event type identifier (e.g., "rgb_activity", "ir_hotspot", "radar_spike") 
        score: Event confidence/strength score in [0,1]
        t_ms: Event time in milliseconds (integer)
        meta: Additional metadata dictionary with optional fields
    """
    type: str
    score: float
    t_ms: int
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVec:
    """
    Standardized feature vector from branch modules.
    
    Args:
        z: Feature tensor of shape (B, D) where B=batch, D=feature_dim
    """
    z: torch.Tensor
    
    def __post_init__(self) -> None:
        """Validate feature tensor shape."""
        if len(self.z.shape) != 2:
            raise ValueError(f"Feature tensor must be 2D (B, D), got {self.z.shape}")


@dataclass  
class OutcomeEstimate:
    """
    Final outcome prediction with uncertainty and explanations.
    
    Args:
        p_outcome: Outcome probability in [0,1], shape (B,) or (B,1)
        binary_outcome: Binary prediction {0,1}, shape (B,) or (B,1)
        uncertainty: Optional uncertainty estimate, shape (B,) or (B,1)
        explanation: Optional explanation dictionary with attention maps, text rationale
    """
    p_outcome: torch.Tensor
    binary_outcome: torch.Tensor
    uncertainty: Optional[torch.Tensor] = None
    explanation: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate tensor shapes and ranges."""
        # Ensure tensors are proper shape
        if len(self.p_outcome.shape) > 2:
            raise ValueError(f"p_outcome must be 1D or 2D, got {self.p_outcome.shape}")
        if len(self.binary_outcome.shape) > 2:
            raise ValueError(f"binary_outcome must be 1D or 2D, got {self.binary_outcome.shape}")
            
        # Validate probability range
        if not torch.all((self.p_outcome >= 0) & (self.p_outcome <= 1)):
            raise ValueError("p_outcome must be in [0,1]")
            
        # Validate binary values
        if not torch.all((self.binary_outcome == 0) | (self.binary_outcome == 1)):
            raise ValueError("binary_outcome must be in {0,1}")


class BranchModule(nn.Module, ABC):
    """
    Abstract base class for sensor-specific processing modules.
    
    Each branch processes one modality (visible, radar, thermal) and outputs:
    - Standardized feature vector  
    - List of detected events
    """
    
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.out_dim = out_dim
    
    @abstractmethod
    def forward(self, **inputs: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Process sensor inputs and extract features + events.
        
        Args:
            **inputs: Named tensor inputs specific to each modality
            
        Returns:
            tuple: (FeatureVec with shape (B, out_dim), list of EventTokens)
        """
        pass


class FusionModule(nn.Module, ABC):
    """
    Abstract base class for multimodal fusion modules.
    
    Combines features from multiple sensor branches and produces outcome estimates.
    """
    
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.out_dim = out_dim
    
    @abstractmethod
    def forward(
        self,
        z_r: Optional[FeatureVec] = None,
        z_i: Optional[FeatureVec] = None, 
        z_t: Optional[FeatureVec] = None,
        events: Optional[List[EventToken]] = None,
    ) -> Union[OutcomeEstimate, Dict[str, torch.Tensor]]:
        """
        Fuse multimodal features and produce outcome estimate.
        
        Args:
            z_r: Radar features, shape (B, D)
            z_i: Visible/EO features, shape (B, D)  
            z_t: Thermal/IR features, shape (B, D)
            events: List of events from all modalities
            
        Returns:
            OutcomeEstimate or dict with intermediate outputs
        """
        pass


class GuardModule(nn.Module, ABC):
    """
    Abstract base class for guard/safety modules.
    
    Validates fusion outputs against consistency checks and priors.
    """
    
    @abstractmethod
    def forward(
        self,
        estimate: OutcomeEstimate,
        events: List[EventToken],
        geom: Dict[str, Any],
        priors: Dict[str, Any],
    ) -> OutcomeEstimate:
        """
        Apply safety checks and modify outcome estimate if needed.
        
        Args:
            estimate: Initial outcome estimate from fusion
            events: All detected events  
            geom: Geometric/spatial context
            priors: Prior knowledge and constraints
            
        Returns:
            Modified outcome estimate with guard explanations
        """
        pass