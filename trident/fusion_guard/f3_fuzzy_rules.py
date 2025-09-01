"""
TRIDENT-F3: Fuzzy Rule Overlay (rule-based reasoning)

Author: Yağızhan Keskin
"""

from typing import Dict, List, Optional, Any, Union
import yaml
import re
from pathlib import Path

import torch
import torch.nn as nn

from ..common.types import FusionModule, FeatureVec, OutcomeEstimate, EventToken


class FuzzyMembership:
    """Fuzzy membership function implementations."""
    
    @staticmethod
    def triangular(x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function."""
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)
    
    @staticmethod
    def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
        """Trapezoidal membership function."""
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - c)
    
    @staticmethod
    def gaussian(x: float, mean: float, std: float) -> float:
        """Gaussian membership function."""
        import math
        return math.exp(-0.5 * ((x - mean) / std) ** 2)


class FuzzyRule:
    """Single fuzzy rule with conditions and consequences."""
    
    def __init__(self, rule_dict: Dict[str, Any]):
        self.name = rule_dict.get("name", "unnamed_rule")
        self.description = rule_dict.get("description", "")
        self.conditions = rule_dict.get("conditions", [])
        self.action = rule_dict.get("action", {})
        self.priority = rule_dict.get("priority", 1.0)
        self.enabled = rule_dict.get("enabled", True)
    
    def evaluate_conditions(
        self, 
        events: List[EventToken], 
        p_outcome: float
    ) -> float:
        """
        Evaluate rule conditions and return activation strength.
        
        Args:
            events: List of event tokens
            p_outcome: Current outcome probability
            
        Returns:
            Activation strength in [0, 1]
        """
        if not self.enabled or not self.conditions:
            return 0.0
        
        # Convert events to lookup dict
        event_dict = {}
        for event in events:
            event_type = event.type
            if event_type not in event_dict:
                event_dict[event_type] = []
            
            # Convert value to float
            if isinstance(event.value, torch.Tensor):
                value = event.value.item()
            else:
                value = float(event.value)
            
            event_dict[event_type].append({
                'value': value,
                'quality': event.quality,
                'duration': event.t_end - event.t_start,
                'meta': event.meta,
            })
        
        # Evaluate each condition
        condition_activations = []
        
        for condition in self.conditions:
            activation = self._evaluate_single_condition(condition, event_dict, p_outcome)
            condition_activations.append(activation)
        
        # Combine conditions (using AND - minimum)
        if condition_activations:
            return min(condition_activations)
        else:
            return 0.0
    
    def _evaluate_single_condition(
        self, 
        condition: Dict[str, Any], 
        event_dict: Dict[str, List[Dict]], 
        p_outcome: float
    ) -> float:
        """Evaluate a single condition."""
        condition_type = condition.get("type", "")
        
        if condition_type == "event_exists":
            return self._eval_event_exists(condition, event_dict)
        elif condition_type == "event_value":
            return self._eval_event_value(condition, event_dict)
        elif condition_type == "event_count":
            return self._eval_event_count(condition, event_dict)
        elif condition_type == "probability_range":
            return self._eval_probability_range(condition, p_outcome)
        elif condition_type == "event_combination":
            return self._eval_event_combination(condition, event_dict)
        else:
            return 0.0
    
    def _eval_event_exists(self, condition: Dict, event_dict: Dict) -> float:
        """Check if specific event type exists."""
        event_type = condition.get("event_type", "")
        min_quality = condition.get("min_quality", 0.0)
        
        if event_type in event_dict:
            # Check if any event meets quality threshold
            for event_data in event_dict[event_type]:
                if event_data['quality'] >= min_quality:
                    return 1.0
        
        return 0.0
    
    def _eval_event_value(self, condition: Dict, event_dict: Dict) -> float:
        """Evaluate fuzzy membership of event value."""
        event_type = condition.get("event_type", "")
        membership = condition.get("membership", {})
        
        if event_type not in event_dict:
            return 0.0
        
        # Use highest quality event of this type
        best_event = max(event_dict[event_type], key=lambda x: x['quality'])
        value = best_event['value']
        
        # Apply fuzzy membership function
        func_type = membership.get("function", "triangular")
        params = membership.get("parameters", [])
        
        if func_type == "triangular" and len(params) >= 3:
            return FuzzyMembership.triangular(value, *params[:3])
        elif func_type == "trapezoidal" and len(params) >= 4:
            return FuzzyMembership.trapezoidal(value, *params[:4])
        elif func_type == "gaussian" and len(params) >= 2:
            return FuzzyMembership.gaussian(value, *params[:2])
        else:
            return 0.0
    
    def _eval_event_count(self, condition: Dict, event_dict: Dict) -> float:
        """Evaluate based on number of events."""
        event_type = condition.get("event_type", "")
        min_count = condition.get("min_count", 1)
        max_count = condition.get("max_count", float('inf'))
        
        count = len(event_dict.get(event_type, []))
        
        if min_count <= count <= max_count:
            return 1.0
        else:
            return 0.0
    
    def _eval_probability_range(self, condition: Dict, p_outcome: float) -> float:
        """Evaluate based on current outcome probability."""
        min_prob = condition.get("min_prob", 0.0)
        max_prob = condition.get("max_prob", 1.0)
        
        if min_prob <= p_outcome <= max_prob:
            return 1.0
        else:
            return 0.0
    
    def _eval_event_combination(self, condition: Dict, event_dict: Dict) -> float:
        """Evaluate combination of multiple events."""
        required_events = condition.get("required_events", [])
        operation = condition.get("operation", "and")  # and, or
        
        activations = []
        for event_type in required_events:
            if event_type in event_dict:
                activations.append(1.0)
            else:
                activations.append(0.0)
        
        if operation == "and":
            return min(activations) if activations else 0.0
        elif operation == "or":
            return max(activations) if activations else 0.0
        else:
            return 0.0
    
    def apply_action(self, p_outcome: float, activation: float) -> Tuple[float, str]:
        """
        Apply rule action based on activation strength.
        
        Args:
            p_outcome: Current outcome probability
            activation: Rule activation strength
            
        Returns:
            tuple: (modified_probability, rationale)
        """
        if activation <= 0.0:
            return p_outcome, ""
        
        action_type = self.action.get("type", "")
        rationale = self.action.get("rationale", f"Rule {self.name} applied")
        
        if action_type == "multiply":
            factor = self.action.get("factor", 1.0)
            modified_prob = p_outcome * (1 + activation * (factor - 1))
            return torch.clamp(torch.tensor(modified_prob), 0, 1).item(), rationale
            
        elif action_type == "add":
            delta = self.action.get("delta", 0.0)
            modified_prob = p_outcome + activation * delta
            return torch.clamp(torch.tensor(modified_prob), 0, 1).item(), rationale
            
        elif action_type == "set":
            target_prob = self.action.get("target", p_outcome)
            modified_prob = p_outcome + activation * (target_prob - p_outcome)
            return torch.clamp(torch.tensor(modified_prob), 0, 1).item(), rationale
            
        elif action_type == "threshold":
            threshold = self.action.get("threshold", 0.5)
            new_value = self.action.get("value", 0.0)
            if p_outcome >= threshold:
                return new_value, rationale
            else:
                return p_outcome, ""
        
        return p_outcome, ""


class FuzzyRuleOverlay(FusionModule):
    """
    Fuzzy Rule Overlay for post-processing fusion outputs.
    
    Applies fuzzy logic rules to modify outcome probabilities
    based on event patterns and domain knowledge.
    """
    
    def __init__(self, ruleset_path: Optional[str] = None):
        super().__init__(out_dim=1)  # Output is modified probability
        
        self.rules: List[FuzzyRule] = []
        self.ruleset_path = ruleset_path
        
        if ruleset_path:
            self.load_rules(ruleset_path)
    
    def load_rules(self, ruleset_path: str) -> None:
        """Load fuzzy rules from YAML file."""
        try:
            with open(ruleset_path, 'r') as f:
                ruleset_data = yaml.safe_load(f)
            
            self.rules = []
            for rule_data in ruleset_data.get("rules", []):
                rule = FuzzyRule(rule_data)
                self.rules.append(rule)
            
            print(f"Loaded {len(self.rules)} fuzzy rules from {ruleset_path}")
            
        except Exception as e:
            print(f"Warning: Could not load rules from {ruleset_path}: {e}")
            self.rules = []
    
    def add_rule(self, rule_dict: Dict[str, Any]) -> None:
        """Add a single rule."""
        rule = FuzzyRule(rule_dict)
        self.rules.append(rule)
    
    def forward(
        self,
        z_r: Optional[FeatureVec] = None,
        z_i: Optional[FeatureVec] = None,
        z_t: Optional[FeatureVec] = None,
        events: Optional[List[EventToken]] = None,
        p_outcome_in: Optional[torch.Tensor] = None,
    ) -> OutcomeEstimate:
        """
        Apply fuzzy rules to modify outcome probability.
        
        Args:
            z_r, z_i, z_t: Feature vectors (not used by rules)
            events: Event tokens to evaluate
            p_outcome_in: Input probability to modify
            
        Returns:
            OutcomeEstimate with rule-modified probability
        """
        if p_outcome_in is None:
            raise ValueError("Input probability p_outcome_in is required")
        
        if events is None:
            events = []
        
        batch_size = p_outcome_in.shape[0]
        
        # Process each sample in batch
        modified_probs = []
        rationales = []
        
        for b in range(batch_size):
            p_in = p_outcome_in[b].item()
            
            # Apply rules sequentially
            p_modified = p_in
            applied_rules = []
            
            # Sort rules by priority (higher priority first)
            sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)
            
            for rule in sorted_rules:
                activation = rule.evaluate_conditions(events, p_modified)
                
                if activation > 0.0:
                    p_new, rationale = rule.apply_action(p_modified, activation)
                    
                    if abs(p_new - p_modified) > 1e-6:  # Significant change
                        applied_rules.append({
                            'rule': rule.name,
                            'activation': activation,
                            'before': p_modified,
                            'after': p_new,
                            'rationale': rationale,
                        })
                        p_modified = p_new
            
            modified_probs.append(p_modified)
            
            # Create rationale text
            if applied_rules:
                rationale_text = "; ".join([
                    f"{r['rule']}({r['activation']:.2f}): {r['rationale']}"
                    for r in applied_rules
                ])
            else:
                rationale_text = "No rules activated"
            
            rationales.append(rationale_text)
        
        # Convert back to tensors
        device = p_outcome_in.device
        p_ruled = torch.tensor(modified_probs, device=device, dtype=torch.float32)
        binary_outcome = (p_ruled > 0.5).long()
        
        # Create explanation
        explanation = {
            "fusion_type": "fuzzy_rules",
            "rules_applied": rationales,
            "input_probability": p_outcome_in.tolist(),
            "output_probability": p_ruled.tolist(),
            "total_rules": len(self.rules),
            "active_rules": sum(1 for r in self.rules if r.enabled),
        }
        
        return OutcomeEstimate(
            p_outcome=p_ruled,
            binary_outcome=binary_outcome,
            explanation=explanation,
        )
    
    def get_rules_summary(self) -> Dict[str, Any]:
        """Get summary of loaded rules."""
        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules if r.enabled),
            "rule_names": [r.name for r in self.rules],
            "rule_priorities": [r.priority for r in self.rules],
        }
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable a specific rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                return True
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable a specific rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                return True
        return False


def create_fuzzy_rule_overlay(config: dict) -> FuzzyRuleOverlay:
    """Factory function to create FuzzyRuleOverlay from config."""
    return FuzzyRuleOverlay(
        ruleset_path=config.get("ruleset_path", None),
    )


def create_default_ruleset() -> Dict[str, Any]:
    """Create a default set of fuzzy rules for demonstration."""
    return {
        "rules": [
            {
                "name": "high_confidence_events",
                "description": "Boost probability when multiple high-quality events detected",
                "priority": 2.0,
                "conditions": [
                    {
                        "type": "event_count",
                        "event_type": "debris_detected",
                        "min_count": 2
                    }
                ],
                "action": {
                    "type": "multiply",
                    "factor": 1.3,
                    "rationale": "Multiple debris detections increase confidence"
                }
            },
            {
                "name": "thermal_signature_boost",
                "description": "Boost when thermal events support visible detection",
                "priority": 1.5,
                "conditions": [
                    {
                        "type": "event_combination",
                        "required_events": ["debris_detected", "thermal_debris"],
                        "operation": "and"
                    }
                ],
                "action": {
                    "type": "add",
                    "delta": 0.2,
                    "rationale": "Thermal signature confirms visible detection"
                }
            },
            {
                "name": "low_confidence_suppress",
                "description": "Suppress when only low-quality events",
                "priority": 1.0,
                "conditions": [
                    {
                        "type": "probability_range",
                        "min_prob": 0.3,
                        "max_prob": 0.7
                    }
                ],
                "action": {
                    "type": "multiply",
                    "factor": 0.8,
                    "rationale": "Medium confidence events need verification"
                }
            },
        ]
    }