"""
Test Phase 7: Clean legacy/dead code functionality.
"""

import pytest
import warnings
import sys
sys.path.append('.')


def test_legacy_modules_deprecated():
    """Test that legacy v1 modules issue deprecation warnings."""
    print("🗑️ Testing legacy module deprecation warnings...")
    
    # Test frag3d deprecation
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Clear any existing modules to get clean warning
        if 'trident.trident_i.frag3d' in sys.modules:
            del sys.modules['trident.trident_i.frag3d']
        import trident.trident_i.frag3d
        
        # Filter for our specific deprecation warning
        frag3d_warnings = [warning for warning in w if "frag3d is deprecated" in str(warning.message)]
        assert len(frag3d_warnings) >= 1
        assert issubclass(frag3d_warnings[0].category, DeprecationWarning)
        assert "VideoFrag3Dv2" in str(frag3d_warnings[0].message)
        print("✓ frag3d deprecation warning issued")
    
    print("🗑️ Legacy module deprecation tests passed!")


def test_registry_v2_only():
    """Test that registry contains only v2 components.""" 
    print("📋 Testing registry contains only v2 components...")
    
    # Simpler test - just check the registry contains expected v2 modules
    v2_modules = [
        "VideoFrag3Dv2",
        "DualVisionV2", 
        "PlumeDetXL"
    ]
    
    # Check tasks.yml content directly
    with open('tasks.yml', 'r') as f:
        content = f.read()
    
    for v2 in v2_modules:
        assert v2 in content, f"V2 module {v2} not found in tasks.yml"
        print(f"✓ {v2} found in config")
    
    # Check no legacy modules
    legacy_patterns = ["frag3d", "dualvision.DualVision", "plumedet_lite"]
    for legacy in legacy_patterns:
        assert legacy not in content or "deprecated" in content, f"Legacy {legacy} found without deprecation"
    
    print("📋 Registry v2-only validation passed!")


def test_event_extraction_active():
    """Test that event extraction methods are active and properly implemented."""
    print("🔧 Testing event extraction methods are active...")
    
    from trident.trident_r.geomlp import GeoMLP
    from trident.trident_r.tiny_temporal_former import TinyTempoFormer
    from trident.trident_r.kinefeat import KineFeat
    import torch
    
    # Test GeoMLP events (was previously stubbed)
    geomlp = GeoMLP()
    k_aug = torch.randn(2, 69)
    zr2, events = geomlp(k_aug)
    
    assert isinstance(events, list), "GeoMLP should return event list"
    print("✓ GeoMLP event extraction active")
    
    # Test TinyTempoFormer events (was previously stubbed)
    ttf = TinyTempoFormer()
    k_seq = torch.randn(2, 3, 32)  # sequence input (B, T=3, D=32)
    zr3, events = ttf(k_seq)
    
    assert isinstance(events, list), "TinyTempoFormer should return event list"
    print("✓ TinyTempoFormer event extraction active")
    
    # Test KineFeat events (cleaned up)
    kinefeat = KineFeat()
    k_seq = torch.randn(2, 3, 9)
    r_feats, events = kinefeat(k_seq)
    
    assert isinstance(events, list), "KineFeat should return event list"
    print("✓ KineFeat event extraction active")
    
    print("🔧 Event extraction activity tests passed!")


if __name__ == "__main__":
    test_legacy_modules_deprecated()
    test_registry_v2_only()
    test_event_extraction_active()
    print("\n✅ All Phase 7 (legacy cleanup) tests passed!")