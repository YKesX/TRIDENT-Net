"""
Test EventToken consistency across all modules in TRIDENT-Net.

Tests that all modules emit EventTokens with the unified schema.

Author: Yağızhan Keskin
"""

import torch
import pytest
import sys
sys.path.append('.')

from trident.common.types import EventToken
from trident.trident_r.geomlp import GeoMLP
from trident.trident_r.tiny_temporal_former import TinyTempoFormer
from trident.trident_t.coolcurve3 import CoolCurve3


def test_event_token_schema():
    """Test EventToken has correct schema fields."""
    event = EventToken(
        type="test_event",
        score=0.8,
        t_ms=1000,
        meta={"test": "value"}
    )
    
    # Verify required fields exist
    assert hasattr(event, 'type')
    assert hasattr(event, 'score')
    assert hasattr(event, 't_ms')
    assert hasattr(event, 'meta')
    
    # Verify types
    assert isinstance(event.type, str)
    assert isinstance(event.score, float)
    assert isinstance(event.t_ms, int)
    assert isinstance(event.meta, dict)
    
    print("✅ EventToken schema test passed")


def test_geomlp_events():
    """Test GeoMLP produces EventTokens with correct schema."""
    model = GeoMLP()
    
    # Create test input
    k_aug = torch.randn(2, 69)  # Batch size 2, 69 features
    
    # Forward pass
    zr2, events = model(k_aug)
    
    # Check output shape
    assert zr2.shape == (2, 192), f"Expected zr2 shape (2, 192), got {zr2.shape}"
    
    # Check events are EventTokens
    assert isinstance(events, list), "Events should be a list"
    
    for event in events:
        assert isinstance(event, EventToken), f"Event should be EventToken, got {type(event)}"
        assert hasattr(event, 'type'), "EventToken should have 'type' field"
        assert hasattr(event, 'score'), "EventToken should have 'score' field"
        assert hasattr(event, 't_ms'), "EventToken should have 't_ms' field"
        assert hasattr(event, 'meta'), "EventToken should have 'meta' field"
        
        # Verify field types
        assert isinstance(event.type, str), "Event type should be string"
        assert isinstance(event.score, float), "Event score should be float"
        assert isinstance(event.t_ms, int), "Event t_ms should be int"
        assert isinstance(event.meta, dict), "Event meta should be dict"
        
        # Verify score range
        assert 0.0 <= event.score <= 1.0, f"Event score should be in [0,1], got {event.score}"
        
        # Verify meta has required fields
        assert 'source' in event.meta, "Event meta should have 'source' field"
        assert event.meta['source'] == 'geomlp', "Event source should be 'geomlp'"
    
    print(f"✅ GeoMLP events test passed - generated {len(events)} events")


def test_tinytempformer_events():
    """Test TinyTempoFormer produces EventTokens with correct schema."""
    model = TinyTempoFormer()
    
    # Create test input
    k_tokens = torch.randn(2, 3, 32)  # Batch size 2, 3 tokens, 32 dims each
    
    # Forward pass
    zr3, events = model(k_tokens)
    
    # Check output shape
    assert zr3.shape == (2, 192), f"Expected zr3 shape (2, 192), got {zr3.shape}"
    
    # Check events are EventTokens
    assert isinstance(events, list), "Events should be a list"
    
    for event in events:
        assert isinstance(event, EventToken), f"Event should be EventToken, got {type(event)}"
        assert hasattr(event, 'type'), "EventToken should have 'type' field"
        assert hasattr(event, 'score'), "EventToken should have 'score' field"
        assert hasattr(event, 't_ms'), "EventToken should have 't_ms' field"
        assert hasattr(event, 'meta'), "EventToken should have 'meta' field"
        
        # Verify field types
        assert isinstance(event.type, str), "Event type should be string"
        assert isinstance(event.score, float), "Event score should be float"
        assert isinstance(event.t_ms, int), "Event t_ms should be int"
        assert isinstance(event.meta, dict), "Event meta should be dict"
        
        # Verify score range
        assert 0.0 <= event.score <= 1.0, f"Event score should be in [0,1], got {event.score}"
        
        # Verify meta has required fields
        assert 'source' in event.meta, "Event meta should have 'source' field"
        assert event.meta['source'] == 'tinytempformer', "Event source should be 'tinytempformer'"
    
    print(f"✅ TinyTempoFormer events test passed - generated {len(events)} events")


def test_coolcurve3_events():
    """Test CoolCurve3 produces EventTokens with correct schema."""
    model = CoolCurve3()
    
    # Create test input
    batch_size = 2
    max_tracks = 10
    curves = torch.randn(batch_size, max_tracks, 3)  # Intensity curves
    areas = torch.randn(batch_size, max_tracks, 3)   # Area curves
    pad_mask = torch.ones(batch_size, max_tracks)    # All tracks valid
    
    # Forward pass
    tau_hat, debris_vs_flare, zt, events = model(curves, areas, pad_mask)
    
    # Check output shapes
    assert tau_hat.shape == (batch_size, 1), f"Expected tau_hat shape ({batch_size}, 1), got {tau_hat.shape}"
    assert debris_vs_flare.shape == (batch_size, 2), f"Expected debris_vs_flare shape ({batch_size}, 2), got {debris_vs_flare.shape}"
    assert zt.shape == (batch_size, 256), f"Expected zt shape ({batch_size}, 256), got {zt.shape}"
    
    # Check events are EventTokens
    assert isinstance(events, list), "Events should be a list"
    
    for event in events:
        assert isinstance(event, EventToken), f"Event should be EventToken, got {type(event)}"
        assert hasattr(event, 'type'), "EventToken should have 'type' field"
        assert hasattr(event, 'score'), "EventToken should have 'score' field"
        assert hasattr(event, 't_ms'), "EventToken should have 't_ms' field"
        assert hasattr(event, 'meta'), "EventToken should have 'meta' field"
        
        # Verify field types
        assert isinstance(event.type, str), "Event type should be string"
        assert isinstance(event.score, float), "Event score should be float"
        assert isinstance(event.t_ms, int), "Event t_ms should be int"
        assert isinstance(event.meta, dict), "Event meta should be dict"
        
        # Verify score range
        assert 0.0 <= event.score <= 1.0, f"Event score should be in [0,1], got {event.score}"
        
        # Verify meta has required fields
        assert 'source' in event.meta, "Event meta should have 'source' field"
        assert event.meta['source'] == 'coolcurve3', "Event source should be 'coolcurve3'"
        
        # Verify event types
        assert event.type in ['debris_cooling', 'flare_cooling', 'thermal_decay'], f"Unknown event type: {event.type}"
    
    print(f"✅ CoolCurve3 events test passed - generated {len(events)} events")


def test_event_token_fields_consistency():
    """Test that all EventTokens across modules use consistent field names."""
    # Test with different modules that use EventTokens
    geomlp = GeoMLP()
    tempformer = TinyTempoFormer()
    coolcurve = CoolCurve3()
    
    # Generate test inputs
    k_aug = torch.randn(1, 69)
    k_tokens = torch.randn(1, 3, 32)
    curves = torch.randn(1, 10, 3)
    areas = torch.randn(1, 10, 3)
    pad_mask = torch.ones(1, 10)
    
    # Collect all events
    all_events = []
    
    _, events_geomlp = geomlp(k_aug)
    all_events.extend(events_geomlp)
    
    _, events_tempformer = tempformer(k_tokens)
    all_events.extend(events_tempformer)
    
    _, _, _, events_coolcurve = coolcurve(curves, areas, pad_mask)
    all_events.extend(events_coolcurve)
    
    # Check field consistency across all events
    required_fields = {'type', 'score', 't_ms', 'meta'}
    required_meta_fields = {'source'}
    
    for i, event in enumerate(all_events):
        # Check required fields exist
        for field in required_fields:
            assert hasattr(event, field), f"Event {i} missing required field '{field}'"
        
        # Check meta fields exist
        for field in required_meta_fields:
            assert field in event.meta, f"Event {i} missing required meta field '{field}'"
        
        # Check no old field names are used
        old_fields = {'event_type', 'confidence', 'timestamp', 'metadata'}
        for old_field in old_fields:
            assert not hasattr(event, old_field), f"Event {i} uses deprecated field '{old_field}'"
            assert old_field not in event.meta, f"Event {i} meta uses deprecated field '{old_field}'"
    
    print(f"✅ Event field consistency test passed - checked {len(all_events)} events")


if __name__ == "__main__":
    test_event_token_schema()
    test_geomlp_events()
    test_tinytempformer_events()
    test_coolcurve3_events()
    test_event_token_fields_consistency()
    print("✅ All EventToken tests passed!")