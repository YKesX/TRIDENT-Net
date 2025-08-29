"""
TRIDENT-S: SpoofShield (guard module for consistency checks)

Author: Yağızhan Keskin
"""

from typing import Dict, List, Optional, Any, Tuple
import math

import torch
import torch.nn as nn
import numpy as np

from ..common.types import GuardModule, OutcomeEstimate, EventToken


class ConsistencyChecker:
    """Check temporal and physical consistency of events."""
    
    def __init__(self, consistency_dt_ms: float = 80.0):
        self.consistency_dt_ms = consistency_dt_ms
    
    def check_temporal_consistency(self, events: List[EventToken]) -> Dict[str, Any]:
        """Check if events occur in physically plausible time windows."""
        if len(events) < 2:
            return {"consistent": True, "violations": []}
        
        violations = []
        
        # Group events by type
        event_groups = {}
        for event in events:
            if event.type not in event_groups:
                event_groups[event.type] = []
            event_groups[event.type].append(event)
        
        # Check for temporal anomalies
        for event_type, event_list in event_groups.items():
            if len(event_list) > 1:
                # Check for events too close together
                for i in range(len(event_list) - 1):
                    e1, e2 = event_list[i], event_list[i + 1]
                    dt_ms = abs(e2.t_start - e1.t_start) * 1000
                    
                    if dt_ms < self.consistency_dt_ms:
                        violations.append({
                            "type": "temporal_clustering",
                            "events": [e1.type, e2.type],
                            "dt_ms": dt_ms,
                            "threshold_ms": self.consistency_dt_ms,
                        })
        
        return {
            "consistent": len(violations) == 0,
            "violations": violations,
            "total_events": len(events),
            "event_types": len(event_groups),
        }
    
    def check_physics_consistency(
        self, 
        events: List[EventToken], 
        geom: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if events are physically plausible given geometry."""
        violations = []
        
        range_m = geom.get("range_m", 1000.0)
        
        # Check radar events
        radar_events = [e for e in events if "radar" in e.type or "rcs" in e.type]
        for event in radar_events:
            # Check RCS drop magnitude
            if "rcs_drop" in event.type:
                if isinstance(event.value, (int, float)):
                    rcs_drop_db = event.value
                elif hasattr(event.value, 'item'):
                    rcs_drop_db = event.value.item()
                else:
                    continue
                
                # Physical limit check
                if rcs_drop_db > 50.0:  # Unrealistic RCS drop
                    violations.append({
                        "type": "excessive_rcs_drop",
                        "event": event.type,
                        "value": rcs_drop_db,
                        "limit": 50.0,
                    })
            
            # Check velocity changes
            if "velocity" in event.type:
                if isinstance(event.value, (int, float)):
                    velocity = event.value
                elif hasattr(event.value, 'item'):
                    velocity = event.value.item()
                else:
                    continue
                
                # Check for physically impossible velocities
                max_velocity_mps = 100.0  # Conservative limit
                if abs(velocity) > max_velocity_mps:
                    violations.append({
                        "type": "excessive_velocity",
                        "event": event.type,
                        "value": velocity,
                        "limit": max_velocity_mps,
                    })
        
        # Check thermal events
        thermal_events = [e for e in events if "thermal" in e.type or "cooling" in e.type]
        for event in thermal_events:
            if "cooling_tau" in event.type:
                if isinstance(event.value, (int, float)):
                    tau = event.value
                elif hasattr(event.value, 'item'):
                    tau = event.value.item()
                else:
                    continue
                
                # Check cooling time constant bounds
                if tau < 0.01 or tau > 10.0:  # Unrealistic cooling rates
                    violations.append({
                        "type": "unrealistic_cooling_tau",
                        "event": event.type,
                        "value": tau,
                        "bounds": [0.01, 10.0],
                    })
        
        return {
            "consistent": len(violations) == 0,
            "violations": violations,
            "geometry": geom,
        }


class SensorHealthChecker:
    """Check sensor health and data quality."""
    
    def __init__(self):
        pass
    
    def check_data_quality(self, events: List[EventToken]) -> Dict[str, Any]:
        """Check overall data quality from events."""
        if not events:
            return {
                "healthy": False,
                "reason": "no_events",
                "quality_score": 0.0,
            }
        
        # Compute average quality
        qualities = [e.quality for e in events]
        avg_quality = sum(qualities) / len(qualities)
        min_quality = min(qualities)
        
        # Check for sensor degradation indicators
        degradation_events = [
            e for e in events 
            if "jamming" in e.type or "decorrelation" in e.type or "anomaly" in e.type
        ]
        
        degradation_score = len(degradation_events) / len(events)
        
        # Overall health assessment
        quality_threshold = 0.6
        degradation_threshold = 0.3
        
        healthy = (
            avg_quality >= quality_threshold and 
            degradation_score < degradation_threshold and
            min_quality > 0.1
        )
        
        return {
            "healthy": healthy,
            "quality_score": avg_quality,
            "min_quality": min_quality,
            "degradation_score": degradation_score,
            "degradation_events": len(degradation_events),
            "total_events": len(events),
        }


class PlausibilityChecker:
    """Check overall plausibility of outcome estimate."""
    
    def __init__(self, min_rcs_drop_db: float = 7.0, tau_bounds: Tuple[float, float] = (0.05, 0.6)):
        self.min_rcs_drop_db = min_rcs_drop_db
        self.tau_bounds = tau_bounds
    
    def check_outcome_plausibility(
        self, 
        estimate: OutcomeEstimate, 
        events: List[EventToken],
        priors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if outcome estimate is plausible given events and priors."""
        
        p_outcome = estimate.p_outcome
        if hasattr(p_outcome, 'item'):
            p_val = p_outcome.item()
        else:
            p_val = float(p_outcome)
        
        baseline_prob = priors.get("baseline_prob", 0.1)
        
        # Check if high probability is supported by evidence
        high_prob_threshold = 0.8
        support_events = []
        
        if p_val > high_prob_threshold:
            # Look for supporting evidence
            for event in events:
                if "debris" in event.type or "change" in event.type:
                    support_events.append(event.type)
                elif "rcs_drop" in event.type:
                    if isinstance(event.value, (int, float)):
                        rcs_drop = event.value
                    elif hasattr(event.value, 'item'):
                        rcs_drop = event.value.item()
                    else:
                        continue
                    
                    if rcs_drop >= self.min_rcs_drop_db:
                        support_events.append(event.type)
                elif "thermal_debris" in event.type or "cooling" in event.type:
                    support_events.append(event.type)
        
        # Check probability deviation from baseline
        prob_ratio = p_val / (baseline_prob + 1e-6)
        
        plausible = True
        reasons = []
        
        if p_val > high_prob_threshold and len(support_events) == 0:
            plausible = False
            reasons.append("high_probability_without_evidence")
        
        if prob_ratio > 10.0 and len(support_events) < 2:
            plausible = False
            reasons.append("extreme_probability_increase_insufficient_support")
        
        if p_val < 0.01 and len(events) > 5:
            plausible = False
            reasons.append("many_events_but_very_low_probability")
        
        return {
            "plausible": plausible,
            "reasons": reasons,
            "support_events": support_events,
            "probability_ratio": prob_ratio,
            "baseline_probability": baseline_prob,
        }


class SpoofShield(GuardModule):
    """
    SpoofShield: Guard module for consistency and plausibility checks.
    
    Validates fusion outputs against physical constraints, temporal consistency,
    and sensor health indicators to detect potential spoofing or errors.
    """
    
    def __init__(
        self,
        consistency_dt_ms: float = 80.0,
        min_rcs_drop_db: float = 7.0,
        tau_bounds: Tuple[float, float] = (0.05, 0.6),
        confidence_threshold: float = 0.8,
    ):
        super().__init__()
        
        self.consistency_dt_ms = consistency_dt_ms
        self.min_rcs_drop_db = min_rcs_drop_db
        self.tau_bounds = tau_bounds
        self.confidence_threshold = confidence_threshold
        
        # Initialize checkers
        self.consistency_checker = ConsistencyChecker(consistency_dt_ms)
        self.sensor_checker = SensorHealthChecker()
        self.plausibility_checker = PlausibilityChecker(min_rcs_drop_db, tau_bounds)
        
        # Learnable confidence adjustment
        self.confidence_adjustment = nn.Parameter(torch.tensor(0.0))
        
    def forward(
        self,
        estimate: OutcomeEstimate,
        events: List[EventToken],
        geom: Dict[str, Any],
        priors: Dict[str, Any],
    ) -> OutcomeEstimate:
        """
        Apply guard checks and modify estimate if needed.
        
        Args:
            estimate: Initial outcome estimate
            events: List of event tokens
            geom: Geometric context
            priors: Prior knowledge
            
        Returns:
            Modified outcome estimate with guard explanations
        """
        # Run all checks
        temporal_check = self.consistency_checker.check_temporal_consistency(events)
        physics_check = self.consistency_checker.check_physics_consistency(events, geom)
        sensor_check = self.sensor_checker.check_data_quality(events)
        plausibility_check = self.plausibility_checker.check_outcome_plausibility(
            estimate, events, priors
        )
        
        # Compute risk scores
        temporal_risk = 0.0 if temporal_check["consistent"] else len(temporal_check["violations"]) * 0.2
        physics_risk = 0.0 if physics_check["consistent"] else len(physics_check["violations"]) * 0.3
        sensor_risk = 0.0 if sensor_check["healthy"] else (1.0 - sensor_check["quality_score"])
        plausibility_risk = 0.0 if plausibility_check["plausible"] else len(plausibility_check["reasons"]) * 0.4
        
        # Combined risk score
        total_risk = temporal_risk + physics_risk + sensor_risk + plausibility_risk
        spoof_risk = torch.sigmoid(torch.tensor(total_risk)).item()
        
        # Apply guard modifications
        p_original = estimate.p_outcome
        
        if hasattr(p_original, 'clone'):
            p_masked = p_original.clone()
        else:
            p_masked = torch.tensor(p_original)
        
        # Apply risk-based adjustments
        if spoof_risk > 0.5:
            # High risk - reduce confidence
            risk_factor = torch.tensor(1.0 - spoof_risk * 0.5)
            p_masked = p_masked * risk_factor
        
        if not sensor_check["healthy"]:
            # Sensor issues - further reduction
            health_factor = torch.tensor(sensor_check["quality_score"])
            p_masked = p_masked * health_factor
        
        # Apply learnable adjustment
        adjustment = torch.sigmoid(self.confidence_adjustment)
        p_masked = p_masked * adjustment + p_original * (1 - adjustment)
        
        # Clamp to valid range
        p_masked = torch.clamp(p_masked, 0.0, 1.0)
        
        # Update binary outcome
        binary_masked = (p_masked > 0.5).long()
        
        # Create comprehensive explanation
        guard_explanation = {
            "guard_type": "spoof_shield",
            "spoof_risk": spoof_risk,
            "checks": {
                "temporal": temporal_check,
                "physics": physics_check,
                "sensor_health": sensor_check,
                "plausibility": plausibility_check,
            },
            "risk_scores": {
                "temporal_risk": temporal_risk,
                "physics_risk": physics_risk,
                "sensor_risk": sensor_risk,
                "plausibility_risk": plausibility_risk,
                "total_risk": total_risk,
            },
            "adjustments": {
                "original_probability": p_original.tolist() if hasattr(p_original, 'tolist') else [float(p_original)],
                "masked_probability": p_masked.tolist() if hasattr(p_masked, 'tolist') else [float(p_masked)],
                "confidence_adjustment": adjustment.item(),
            },
            "gates": {
                "temporal_gate": temporal_check["consistent"],
                "physics_gate": physics_check["consistent"],
                "sensor_gate": sensor_check["healthy"],
                "plausibility_gate": plausibility_check["plausible"],
            }
        }
        
        # Combine with original explanation
        if estimate.explanation:
            combined_explanation = estimate.explanation.copy()
            combined_explanation["guard"] = guard_explanation
        else:
            combined_explanation = {"guard": guard_explanation}
        
        return OutcomeEstimate(
            p_outcome=p_masked,
            binary_outcome=binary_masked,
            uncertainty=estimate.uncertainty,
            explanation=combined_explanation,
        )
    
    def get_guard_status(self) -> Dict[str, Any]:
        """Get current guard configuration and status."""
        return {
            "consistency_dt_ms": self.consistency_dt_ms,
            "min_rcs_drop_db": self.min_rcs_drop_db,
            "tau_bounds": self.tau_bounds,
            "confidence_threshold": self.confidence_threshold,
            "confidence_adjustment": self.confidence_adjustment.item(),
        }
    
    def update_thresholds(self, **kwargs) -> None:
        """Update guard thresholds."""
        if "consistency_dt_ms" in kwargs:
            self.consistency_dt_ms = kwargs["consistency_dt_ms"]
            self.consistency_checker.consistency_dt_ms = kwargs["consistency_dt_ms"]
        
        if "min_rcs_drop_db" in kwargs:
            self.min_rcs_drop_db = kwargs["min_rcs_drop_db"]
            self.plausibility_checker.min_rcs_drop_db = kwargs["min_rcs_drop_db"]
        
        if "tau_bounds" in kwargs:
            self.tau_bounds = kwargs["tau_bounds"]
            self.plausibility_checker.tau_bounds = kwargs["tau_bounds"]
        
        if "confidence_threshold" in kwargs:
            self.confidence_threshold = kwargs["confidence_threshold"]


def create_spoof_shield(config: dict) -> SpoofShield:
    """Factory function to create SpoofShield from config."""
    return SpoofShield(
        consistency_dt_ms=config.get("consistency_dt_ms", 80.0),
        min_rcs_drop_db=config.get("min_rcs_drop_db", 7.0),
        tau_bounds=config.get("tau_bounds", [0.05, 0.6]),
        confidence_threshold=config.get("confidence_threshold", 0.8),
    )