"""
Template-based explanation generator for TRIDENT-Net.

Provides fast one-liner explanations using predefined templates.

Author: Yağızhan Keskin
"""

from typing import Dict, List, Any, Optional
import json
import random
from pathlib import Path


class Templater:
    """
    Fast template-based explanation generator.
    
    Provides one-liner explanations with <20ms latency as specified in tasks.yml.
    """
    
    def __init__(
        self,
        templates_path: Optional[str] = None,
        latency_ms_budget: int = 20,
    ):
        self.latency_ms_budget = latency_ms_budget
        self.templates = self._load_templates(templates_path)
    
    def _load_templates(self, templates_path: Optional[str]) -> Dict[str, List[str]]:
        """Load explanation templates."""
        if templates_path and Path(templates_path).exists():
            try:
                with open(templates_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Default templates
        return {
            "high_confidence": [
                "High confidence detection: {confidence:.2f}",
                "Strong signal detected with {confidence:.2f} confidence",
                "Clear positive indication ({confidence:.2f})",
            ],
            "medium_confidence": [
                "Moderate confidence: {confidence:.2f}",
                "Uncertain detection ({confidence:.2f})",
                "Weak signal detected: {confidence:.2f}",
            ],
            "low_confidence": [
                "Low confidence: {confidence:.2f}",
                "Minimal detection signal ({confidence:.2f})",
                "Uncertain: {confidence:.2f}",
            ],
            "multi_sensor": [
                "Multi-sensor agreement: {modalities}",
                "Correlated signals from {modalities}",
                "{modalities} sensors confirm detection",
            ],
            "guard_warning": [
                "Guard warning: {reason}",
                "Plausibility check failed: {reason}",
                "Warning: {reason}",
            ],
        }
    
    def generate_oneliner(
        self,
        p_hit: float,
        p_kill: float,
        events: List[Dict[str, Any]],
        attention_maps: Optional[Dict[str, Any]] = None,
        spoof_risk: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a one-liner explanation.
        
        Args:
            p_hit: Hit probability
            p_kill: Kill probability  
            events: List of detected events
            attention_maps: Optional attention information
            spoof_risk: Optional spoofing risk score
            
        Returns:
            str: One-liner explanation
        """
        # Determine primary outcome
        max_prob = max(p_hit, p_kill)
        outcome_type = "hit" if p_hit > p_kill else "kill"
        
        # Check for guard warnings
        if spoof_risk is not None and spoof_risk > 0.5:
            template = random.choice(self.templates["guard_warning"])
            return template.format(reason="high spoofing risk")
        
        # Determine confidence level
        if max_prob > 0.7:
            template_key = "high_confidence"
        elif max_prob > 0.4:
            template_key = "medium_confidence"
        else:
            template_key = "low_confidence"
        
        # Check for multi-sensor agreement
        if events and len(events) > 1:
            event_types = set(event.get("type", "unknown") for event in events)
            if len(event_types) > 1:
                template = random.choice(self.templates["multi_sensor"])
                modalities = ", ".join(sorted(event_types))
                return template.format(modalities=modalities)
        
        # Default confidence-based template
        template = random.choice(self.templates[template_key])
        return template.format(confidence=max_prob, outcome=outcome_type)
    
    def explain_attention(
        self,
        attention_maps: Dict[str, Any],
        top_k: int = 3,
    ) -> str:
        """
        Generate explanation from attention maps.
        
        Args:
            attention_maps: Attention weight information
            top_k: Number of top attended regions to mention
            
        Returns:
            str: Attention-based explanation
        """
        if not attention_maps:
            return "No attention information available"
        
        # Extract top attended regions (simplified)
        explanations = []
        
        for modality, attn_data in attention_maps.items():
            if isinstance(attn_data, dict) and "top_regions" in attn_data:
                regions = attn_data["top_regions"][:top_k]
                if regions:
                    explanations.append(f"{modality}: {', '.join(regions)}")
        
        if explanations:
            return f"Key attention: {'; '.join(explanations)}"
        else:
            return "Distributed attention pattern"
    
    def explain_events(
        self,
        events: List[Dict[str, Any]],
        max_events: int = 3,
    ) -> str:
        """
        Generate explanation from detected events.
        
        Args:
            events: List of event dictionaries
            max_events: Maximum number of events to mention
            
        Returns:
            str: Event-based explanation
        """
        if not events:
            return "No significant events detected"
        
        # Sort events by score if available
        sorted_events = sorted(
            events,
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:max_events]
        
        event_descriptions = []
        for event in sorted_events:
            event_type = event.get("type", "unknown")
            score = event.get("score", 0)
            if score > 0:
                event_descriptions.append(f"{event_type} ({score:.2f})")
            else:
                event_descriptions.append(event_type)
        
        return f"Events: {', '.join(event_descriptions)}"
    
    def generate_detailed_explanation(
        self,
        p_hit: float,
        p_kill: float,
        events: List[Dict[str, Any]],
        attention_maps: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate a detailed explanation with multiple components.
        
        Returns:
            Dict with different explanation components
        """
        return {
            "oneliner": self.generate_oneliner(p_hit, p_kill, events, attention_maps, **kwargs),
            "attention": self.explain_attention(attention_maps or {}),
            "events": self.explain_events(events),
            "confidence": f"Hit: {p_hit:.2f}, Kill: {p_kill:.2f}",
        }