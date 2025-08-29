"""
Core data types and base classes for the TRIDENT system.

This module defines the fundamental contracts and data structures used
throughout the multimodal fusion pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EventToken:
    """
    Represents a detected event from any modality with temporal bounds.
    
    Attributes:
        type: Event type identifier (e.g., "flash", "rcs_drop", "hotspot")
        value: Associated value (scalar, array, or tensor)
        t_start: Event start time in seconds
        t_end: Event end time in seconds  
        quality: Confidence/quality score [0,1]
        meta: Additional metadata dictionary
    """
    type: str
    value: Union[float, np.ndarray, torch.Tensor]
    t_start: float
    t_end: float
    quality: float
    meta: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate event token constraints."""
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(f"Quality must be in [0,1], got {self.quality}")
        if self.t_start > self.t_end:
            raise ValueError(f"Start time {self.t_start} > end time {self.t_end}")


@dataclass
class FeatureVec:
    """
    Feature vector representation from any modality branch.
    
    Attributes:
        z: Feature tensor of shape (B, D) where B=batch_size, D=feature_dim
    """
    z: torch.Tensor
    
    def __post_init__(self) -> None:
        """Validate feature vector dimensions."""
        if self.z.dim() != 2:
            raise ValueError(f"Feature tensor must be 2D (B,D), got shape {self.z.shape}")


@dataclass  
class OutcomeEstimate:
    """
    System output containing probability estimate and explanation.
    
    Attributes:
        p_outcome: Probability tensor [0,1] of shape (B,)
        binary_outcome: Binary prediction tensor {0,1} of shape (B,)
        uncertainty: Optional uncertainty estimate
        explanation: Optional explanation dictionary with attention maps, text, etc.
    """
    p_outcome: torch.Tensor
    binary_outcome: torch.Tensor
    uncertainty: Optional[torch.Tensor] = None
    explanation: Optional[dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate outcome estimate constraints."""
        if self.p_outcome.dim() != 1:
            raise ValueError(f"p_outcome must be 1D, got shape {self.p_outcome.shape}")
        if self.binary_outcome.dim() != 1:
            raise ValueError(f"binary_outcome must be 1D, got shape {self.binary_outcome.shape}")
        if self.p_outcome.shape != self.binary_outcome.shape:
            raise ValueError("p_outcome and binary_outcome must have same shape")
        
        # Check value ranges
        if not torch.all((self.p_outcome >= 0) & (self.p_outcome <= 1)):
            raise ValueError("p_outcome values must be in [0,1]")
        if not torch.all((self.binary_outcome == 0) | (self.binary_outcome == 1)):
            raise ValueError("binary_outcome values must be in {0,1}")


class BranchModule(nn.Module, ABC):
    """
    Abstract base class for modality-specific processing branches.
    
    Each branch processes inputs from one modality (visual, radar, thermal)
    and outputs both feature vectors and detected events.
    """
    
    @abstractmethod
    def forward(self, **inputs: Any) -> tuple[FeatureVec, list[EventToken]]:
        """
        Process modality inputs and extract features + events.
        
        Args:
            **inputs: Modality-specific input tensors
            
        Returns:
            Tuple of (feature_vector, list_of_events)
        """
        pass


class FusionModule(nn.Module):
    """
    Base class for multimodal fusion modules.
    
    Combines feature vectors from multiple modalities and generates
    outcome estimates.
    """
    
    def forward(
        self,
        z_r: Optional[FeatureVec] = None,
        z_i: Optional[FeatureVec] = None, 
        z_t: Optional[FeatureVec] = None,
        events: Optional[list[EventToken]] = None
    ) -> Union[OutcomeEstimate, dict[str, Any]]:
        """
        Fuse multimodal features and generate outcome estimate.
        
        Args:
            z_r: Radar feature vector
            z_i: Visual/EO feature vector
            z_t: Thermal/IR feature vector
            events: List of detected events from all modalities
            
        Returns:
            OutcomeEstimate or dictionary with intermediate results
        """
        raise NotImplementedError("Subclasses must implement forward method")


class GuardModule(nn.Module):
    """
    Base class for guard/verification modules.
    
    Applies consistency checks and spoofing protection to outcome estimates.
    """
    
    def forward(
        self,
        estimate: OutcomeEstimate,
        events: list[EventToken],
        geom: dict[str, Any],
        priors: dict[str, Any]
    ) -> OutcomeEstimate:
        """
        Apply guard logic to validate and potentially modify outcome estimate.
        
        Args:
            estimate: Initial outcome estimate from fusion
            events: All detected events
            geom: Geometric/spatial metadata
            priors: Prior knowledge and constraints
            
        Returns:
            Modified or validated outcome estimate
        """
        raise NotImplementedError("Subclasses must implement forward method")