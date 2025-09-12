"""
Test Phase 6: Guard require_postframe evidence functionality.
"""

import pytest
import torch
import sys
sys.path.append('.')

from trident.fusion_guard.spoof_shield import SpoofShield
from trident.common.types import EventToken


def test_require_postframe_evidence():
    """Test SpoofShield require_postframe evidence functionality."""
    print("🛡️ Testing SpoofShield postframe evidence requirement...")
    
    # Test case 1: require_postframe=True, no postframe events
    shield = SpoofShield(require_postframe_evidence=True)
    
    # Create events without postframe (all negative t_ms)
    events_no_postframe = [
        EventToken(type="rgb_activity", score=0.8, t_ms=-500, meta={"batch_idx": 0}),
        EventToken(type="ir_detection", score=0.7, t_ms=-200, meta={"batch_idx": 0}),
    ]
    
    p_hit = torch.tensor([[0.9]])
    p_kill = torch.tensor([[0.8]])
    geom = {"bearing": 45.0, "elevation": 20.0}
    priors = {"class_hit_rate": 0.6}
    
    p_hit_masked, p_kill_masked, spoof_risk, gates, explanations = shield(
        p_hit, p_kill, events_no_postframe, geom, priors
    )
    
    # Check that temporal gate is reduced due to no postframe evidence
    assert gates["temporal"][0] < 1.0, "Temporal gate should be reduced when no postframe evidence"
    assert explanations["explanations"][0]["temporal"]["postframe_reason"] == "no_postframe_evidence"
    print(f"✓ No postframe evidence detected, temporal gate: {gates['temporal'][0]:.3f}")
    
    # Test case 2: require_postframe=True, with postframe events (close timing)
    events_with_postframe = [
        EventToken(type="rgb_activity", score=0.8, t_ms=-50, meta={"batch_idx": 0}),  # close to trigger
        EventToken(type="ir_detection", score=0.7, t_ms=20, meta={"batch_idx": 0}),  # postframe, close timing
        EventToken(type="debris_signature", score=0.6, t_ms=40, meta={"batch_idx": 0}),  # postframe, close timing
    ]
    
    p_hit_masked2, p_kill_masked2, spoof_risk2, gates2, explanations2 = shield(
        p_hit, p_kill, events_with_postframe, geom, priors
    )
    
    # Check that temporal gate is higher with postframe evidence
    assert gates2["temporal"][0] > gates["temporal"][0], "Temporal gate should be higher with postframe evidence"
    assert explanations2["explanations"][0]["temporal"]["postframe_reason"] == "postframe_evidence_present"
    print(f"✓ Postframe evidence found, temporal gate: {gates2['temporal'][0]:.3f}")
    
    # Test case 3: require_postframe=False (disabled)
    shield_disabled = SpoofShield(require_postframe_evidence=False)
    
    p_hit_masked3, p_kill_masked3, spoof_risk3, gates3, explanations3 = shield_disabled(
        p_hit, p_kill, events_no_postframe, geom, priors
    )
    
    # Check that postframe requirement is disabled
    assert explanations3["explanations"][0]["temporal"]["postframe_reason"] == "not_required"
    print(f"✓ Postframe requirement disabled, reason: {explanations3['explanations'][0]['temporal']['postframe_reason']}")
    
    print("🛡️ SpoofShield postframe evidence tests passed!")


def test_postframe_evidence_integration():
    """Test that postframe evidence properly integrates into rationale."""
    print("🔗 Testing postframe evidence integration...")
    
    shield = SpoofShield(require_postframe_evidence=True)
    
    # Events with mixed timing
    events = [
        EventToken(type="trigger", score=1.0, t_ms=0, meta={"batch_idx": 0}),  # at trigger
        EventToken(type="pre_activity", score=0.5, t_ms=-300, meta={"batch_idx": 0}),  # before
        EventToken(type="post_debris", score=0.8, t_ms=150, meta={"batch_idx": 0}),  # after
    ]
    
    p_hit = torch.tensor([[0.85]])
    p_kill = torch.tensor([[0.75]]) 
    geom = {"bearing": 30.0, "elevation": 15.0}
    priors = {"class_hit_rate": 0.7}
    
    _, _, _, gates, explanations = shield(
        p_hit, p_kill, events, geom, priors
    )
    
    # Verify integration
    explanation = explanations["explanations"][0]
    assert "postframe_gate" in explanation["temporal"]
    assert "postframe_reason" in explanation["temporal"]
    assert explanation["temporal"]["postframe_reason"] == "postframe_evidence_present"
    
    # Check that combined reason includes both temporal and postframe info
    combined_reason = explanation["temporal"]["reason"]
    assert "postframe_evidence_present" in combined_reason
    
    print(f"✓ Postframe evidence integrated into rationale: {combined_reason}")
    print("🔗 Postframe evidence integration tests passed!")


if __name__ == "__main__":
    test_require_postframe_evidence()
    test_postframe_evidence_integration()
    print("\n✅ All Phase 6 (postframe evidence) tests passed!")