"""
SpoofShield - Guard module for plausibility and consistency checks

Provides soft gating and timing/consistency validation as specified in tasks.yml.

Author: Yağızhan Keskin
"""

from typing import Dict, List, Any, Optional, Tuple
import logging

import torch
import torch.nn as nn
import numpy as np

from ..common.types import GuardModule, EventToken


class SpoofShield(GuardModule):
    """
    Guard module for plausibility checks and spoofing detection.
    
    Implements soft gating and consistency checks as specified in tasks.yml:
    - Cross-modal timing/physics plausibility checks
    - Gates final probabilities 
    - Outputs spoof risk and rationale
    """
    
    def __init__(
        self,
        consistency_dt_ms: float = 80.0,
        tau_bounds: Tuple[float, float] = (0.05, 0.6),
        require_postframe_evidence: bool = True,
        class_prior_strength: float = 0.5,
    ):
        super().__init__()
        
        self.consistency_dt_ms = consistency_dt_ms
        self.tau_bounds = tau_bounds
        self.require_postframe_evidence = require_postframe_evidence
        self.class_prior_strength = class_prior_strength
        
        # Learnable gating parameters
        self.gate_weights = nn.Parameter(torch.ones(3))  # [temporal, spatial, class]
        self.confidence_threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(
        self,
        p_hit: torch.Tensor,
        p_kill: torch.Tensor,
        events: List[EventToken],
        geom: Dict[str, Any],
        priors: Dict[str, Any],
        class_id: Optional[torch.Tensor] = None,
        class_conf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """
        Apply guard checks and gating.
        
        Args:
            p_hit: Hit probabilities (B, 1)
            p_kill: Kill probabilities (B, 1)
            events: List of detected events
            geom: Geometry information (bearing, elevation)
            priors: Prior information
            class_id: Optional class IDs (B,)
            class_conf: Optional class confidence (B, 1)
            
        Returns:
            Tuple of:
            - p_hit_masked: Gated hit probabilities
            - p_kill_masked: Gated kill probabilities  
            - spoof_risk: Spoofing risk scores
            - gates: Gate activation information
            - explanation: Structured explanation
        """
        batch_size = p_hit.shape[0]
        device = p_hit.device
        
        # Initialize outputs
        spoof_risk = torch.zeros(batch_size, 1, device=device)
        gates = {"temporal": [], "spatial": [], "class": []}
        explanations = []
        
        # Process each sample in batch
        for i in range(batch_size):
            sample_events = [e for e in events if getattr(e, 'batch_idx', 0) == i]
            sample_geom = geom if isinstance(geom, dict) else geom[i] if isinstance(geom, list) else {}
            sample_priors = priors if isinstance(priors, dict) else priors[i] if isinstance(priors, list) else {}
            
            # Temporal consistency check
            temporal_gate, temporal_risk, temporal_reason = self._check_temporal_consistency(
                sample_events, self.consistency_dt_ms
            )
            
            # Spatial plausibility check
            spatial_gate, spatial_risk, spatial_reason = self._check_spatial_plausibility(
                sample_events, sample_geom
            )
            
            # Class prior check
            class_gate, class_risk, class_reason = self._check_class_priors(
                p_hit[i].item(), p_kill[i].item(),
                class_id[i].item() if class_id is not None else None,
                class_conf[i].item() if class_conf is not None else None,
                sample_priors
            )
            
            # Store gate information
            gates["temporal"].append(temporal_gate)
            gates["spatial"].append(spatial_gate)  
            gates["class"].append(class_gate)
            
            # Compute overall spoofing risk
            risks = torch.tensor([temporal_risk, spatial_risk, class_risk], device=device)
            weights = torch.softmax(self.gate_weights, dim=0)
            sample_risk = torch.sum(weights * risks)
            spoof_risk[i, 0] = sample_risk
            
            # Create explanation
            explanation = {
                "temporal": {"gate": temporal_gate, "risk": temporal_risk, "reason": temporal_reason},
                "spatial": {"gate": spatial_gate, "risk": spatial_risk, "reason": spatial_reason},
                "class": {"gate": class_gate, "risk": class_risk, "reason": class_reason},
                "overall_risk": sample_risk.item(),
            }
            explanations.append(explanation)
        
        # Apply soft gating
        temporal_gates = torch.tensor(gates["temporal"], device=device).float().unsqueeze(1)
        spatial_gates = torch.tensor(gates["spatial"], device=device).float().unsqueeze(1)
        class_gates = torch.tensor(gates["class"], device=device).float().unsqueeze(1)
        
        # Combine gates (soft multiplication)
        overall_gate = temporal_gates * spatial_gates * class_gates
        
        # Apply gating to probabilities
        p_hit_masked = p_hit * overall_gate
        p_kill_masked = p_kill * overall_gate
        
        return p_hit_masked, p_kill_masked, spoof_risk, gates, {"explanations": explanations}
    
    def _check_temporal_consistency(
        self,
        events: List[EventToken],
        max_dt_ms: float
    ) -> Tuple[float, float, str]:
        """
        Check temporal consistency of events.
        
        Returns:
            Tuple of (gate_value, risk_score, reason)
        """
        if len(events) < 2:
            return 1.0, 0.0, "insufficient_events"
        
        # Check timing consistency between events
        timestamps = [e.t_ms for e in events if hasattr(e, 't_ms')]
        if len(timestamps) < 2:
            return 0.8, 0.2, "missing_timestamps"
        
        # Check for temporal clustering
        timestamps.sort()
        max_gap = max(timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1))
        
        if max_gap > max_dt_ms:
            gate = 0.5
            risk = 0.6
            reason = f"large_time_gap_{max_gap:.1f}ms"
        else:
            gate = 1.0
            risk = 0.1
            reason = "temporal_consistent"
        
        return gate, risk, reason
    
    def _check_spatial_plausibility(
        self,
        events: List[EventToken],
        geom: Dict[str, Any]
    ) -> Tuple[float, float, str]:
        """
        Check spatial plausibility of events.
        
        Returns:
            Tuple of (gate_value, risk_score, reason)
        """
        if not geom:
            return 0.9, 0.1, "no_geometry"
        
        bearing = geom.get("bearing", 0.0)
        elevation = geom.get("elevation", 0.0)
        
        # Basic plausibility checks
        if abs(elevation) > np.pi/3:  # > 60 degrees
            return 0.3, 0.7, "extreme_elevation"
        
        # Check event spatial consistency
        spatial_events = [e for e in events if hasattr(e, 'meta') and 'spatial' in e.meta]
        if len(spatial_events) > 1:
            # Check for spatial clustering (simplified)
            positions = [e.meta['spatial'] for e in spatial_events]
            if len(set(positions)) > len(positions) // 2:  # Too scattered
                return 0.6, 0.4, "spatially_scattered"
        
        return 1.0, 0.1, "spatially_consistent"
    
    def _check_class_priors(
        self,
        p_hit: float,
        p_kill: float,
        class_id: Optional[int],
        class_conf: Optional[float],
        priors: Dict[str, Any]
    ) -> Tuple[float, float, str]:
        """
        Check consistency with class priors.
        
        Returns:
            Tuple of (gate_value, risk_score, reason)
        """
        if class_id is None:
            return 0.95, 0.05, "no_class_info"
        
        baseline_prob = priors.get("baseline_prob", 0.1)
        max_prob = max(p_hit, p_kill)
        
        # Check if prediction is reasonable given class
        if class_conf is not None and class_conf < 0.5:
            return 0.7, 0.3, "low_class_confidence"
        
        # Check against baseline
        if max_prob > baseline_prob * 10:  # Much higher than expected
            gate = 0.6 if class_conf and class_conf > 0.8 else 0.4
            risk = 0.5
            reason = "above_baseline"
        elif max_prob < baseline_prob * 0.1:  # Much lower than expected
            gate = 0.9
            risk = 0.1
            reason = "below_baseline"
        else:
            gate = 1.0
            risk = 0.05
            reason = "consistent_with_priors"
        
        return gate, risk, reason
    
    def _check_postframe_evidence(
        self,
        events: List[EventToken],
        require_postframe: bool = True
    ) -> Tuple[float, float, str]:
        """
        Check for post-frame evidence requirement.
        
        Returns:
            Tuple of (gate_value, risk_score, reason)
        """
        if not require_postframe:
            return 1.0, 0.0, "not_required"
        
        # Look for events after trigger time (t_ms > 0)
        postframe_events = [e for e in events if hasattr(e, 't_ms') and e.t_ms > 0]
        
        if not postframe_events:
            return 0.5, 0.5, "no_postframe_evidence"
        else:
            return 1.0, 0.1, "postframe_evidence_present"
    
    def calibrate(
        self,
        validation_data: List[Dict[str, Any]],
        target_false_positive_rate: float = 0.05
    ):
        """
        Calibrate guard thresholds based on validation data.
        
        Args:
            validation_data: List of validation samples
            target_false_positive_rate: Target FPR for calibration
        """
        logging.info("Calibrating SpoofShield thresholds...")
        
        # This would implement threshold calibration in production
        # For now, just log the calibration request
        logging.info(f"Would calibrate with {len(validation_data)} samples for FPR={target_false_positive_rate}")
        
    def get_explanation(self, explanation_dict: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation from guard results.
        
        Args:
            explanation_dict: Explanation dictionary from forward pass
            
        Returns:
            Human-readable explanation string
        """
        if "explanations" not in explanation_dict:
            return "No explanation available"
        
        explanations = explanation_dict["explanations"]
        if not explanations:
            return "No explanations generated"
        
        # Take first explanation (assuming batch size 1 for simplicity)
        exp = explanations[0]
        
        parts = []
        overall_risk = exp.get("overall_risk", 0.0)
        
        parts.append(f"Overall spoofing risk: {overall_risk:.3f}")
        
        for check_type in ["temporal", "spatial", "class"]:
            if check_type in exp:
                info = exp[check_type]
                gate = info.get("gate", 1.0)
                risk = info.get("risk", 0.0)
                reason = info.get("reason", "unknown")
                
                parts.append(f"{check_type.title()}: gate={gate:.2f}, risk={risk:.2f} ({reason})")
        
        return "; ".join(parts)