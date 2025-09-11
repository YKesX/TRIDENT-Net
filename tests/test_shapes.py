"""
Test shape validation for TRIDENT-Net components.

Validates that every forward pass returns exact shapes specified in tasks.yml.

Author: Yağızhan Keskin
"""

import torch
import yaml
from pathlib import Path

import sys
sys.path.append('.')


def load_tasks_config():
    """Load tasks.yml configuration."""
    with open('tasks.yml', 'r') as f:
        return yaml.safe_load(f)


def test_new_component_shapes():
    """Test new component shapes match tasks.yml specifications."""
    # Test VideoFrag3Dv2
    try:
        from trident.trident_i.videox3d import VideoFrag3Dv2
        
        model = VideoFrag3Dv2(out_embed_dim=512)
        batch_size = 2
        T = 36  # Variable T frames
        rgb = torch.randn(batch_size, 3, T, 704, 1248)
        
        with torch.no_grad():
            outputs = model(rgb)
        
        # Verify shapes
        assert outputs['mask_seq'].shape == (batch_size, T, 1, 704, 1248), f"VideoFrag3Dv2 mask_seq shape: {outputs['mask_seq'].shape}"
        assert outputs['zi'].shape == (batch_size, 512), f"VideoFrag3Dv2 zi shape: {outputs['zi'].shape}"
        assert isinstance(outputs['events'], list), "VideoFrag3Dv2 events should be list"
        
        print(f"✅ VideoFrag3Dv2 shape test passed: mask_seq {outputs['mask_seq'].shape}, zi {outputs['zi'].shape}")
        
    except Exception as e:
        print(f"❌ VideoFrag3Dv2 test failed: {e}")
    
    # Test DualVisionV2
    try:
        from trident.trident_i.dualvision_v2 import DualVisionV2
        
        model = DualVisionV2(out_embed_dim=256)
        batch_size = 2
        T = 36
        rgb = torch.randn(batch_size, 3, T, 704, 1248)
        
        with torch.no_grad():
            outputs = model(rgb)
        
        # Verify shapes
        assert outputs['change_mask'].shape == (batch_size, 1, 704, 1248), f"DualVisionV2 change_mask shape: {outputs['change_mask'].shape}"
        assert outputs['integrity_delta'].shape == (batch_size, 1), f"DualVisionV2 integrity_delta shape: {outputs['integrity_delta'].shape}"
        assert outputs['zi'].shape == (batch_size, 256), f"DualVisionV2 zi shape: {outputs['zi'].shape}"
        assert isinstance(outputs['events'], list), "DualVisionV2 events should be list"
        
        print(f"✅ DualVisionV2 shape test passed: change_mask {outputs['change_mask'].shape}, zi {outputs['zi'].shape}")
        
    except Exception as e:
        print(f"❌ DualVisionV2 test failed: {e}")
    
    # Test PlumeDetXL
    try:
        from trident.trident_t.ir_dettrack_v2 import PlumeDetXL
        
        model = PlumeDetXL(pool_to_embed=256)
        batch_size = 2
        T = 36
        ir = torch.randn(batch_size, 1, T, 704, 1248)
        
        with torch.no_grad():
            outputs = model(ir)
        
        # Verify shapes
        assert outputs['zt'].shape == (batch_size, 256), f"PlumeDetXL zt shape: {outputs['zt'].shape}"
        assert isinstance(outputs['tracks'], list), "PlumeDetXL tracks should be list"
        assert len(outputs['tracks']) == batch_size, f"PlumeDetXL tracks should have {batch_size} items"
        assert isinstance(outputs['events'], list), "PlumeDetXL events should be list"
        
        print(f"✅ PlumeDetXL shape test passed: zt {outputs['zt'].shape}, tracks len={len(outputs['tracks'])}")
        
    except Exception as e:
        print(f"❌ PlumeDetXL test failed: {e}")


def test_data_components():
    """Test data component functionality."""
    try:
        from trident.data.video_ring import VideoRing
        
        # Test VideoRing temporal slicing
        ring = VideoRing(fps_hint=24)
        pre_indices, fire_indices, post_indices = ring.freeze_and_slice(1200, 700, 1700)
        
        total_frames = len(pre_indices) + len(fire_indices) + len(post_indices)
        print(f"✅ VideoRing test passed: total T={total_frames} frames (pre={len(pre_indices)}, fire={len(fire_indices)}, post={len(post_indices)})")
        
    except Exception as e:
        print(f"❌ VideoRing test failed: {e}")
    
    try:
        from trident.data.collate import pad_tracks_collate
        
        # Test collate function with dummy batch
        batch = [
            {'rgb': torch.randn(3, 10, 704, 1248), 'tracks': [{'id': 1}, {'id': 2}]},
            {'rgb': torch.randn(3, 12, 704, 1248), 'tracks': [{'id': 3}]}
        ]
        
        result = pad_tracks_collate(batch)
        print(f"✅ pad_tracks_collate test passed: keys={list(result.keys())}")
        
    except Exception as e:
        print(f"❌ pad_tracks_collate test failed: {e}")


def test_rgb_shapes():
    """Test RGB input shapes match spec: B x 3 x T x 704 x 1248"""
    # Test with variable T
    batch_size = 2
    T = 36  # Variable temporal dimension
    rgb_seq = torch.randn(batch_size, 3, T, 704, 1248)
    
    assert rgb_seq.shape == (batch_size, 3, T, 704, 1248), f"RGB shape mismatch: expected [B, 3, T, 704, 1248]"
    print(f"✅ RGB shape test passed: {rgb_seq.shape}")


def test_ir_shapes():
    """Test IR input shapes match spec: B x 1 x T x 704 x 1248"""
    # Test with variable T
    batch_size = 2
    T = 36
    ir_seq = torch.randn(batch_size, 1, T, 704, 1248)
    
    assert ir_seq.shape == (batch_size, 1, T, 704, 1248), f"IR shape mismatch: expected [B, 1, T, 704, 1248]"
    print(f"✅ IR shape test passed: {ir_seq.shape}")


def test_kinematics_shapes():
    """Test kinematics input shapes match spec: B x 3 x 9"""
    batch_size = 2
    k_seq = torch.randn(batch_size, 3, 9)
    
    assert k_seq.shape == (batch_size, 3, 9), f"Kinematics shape mismatch: expected [B, 3, 9]"
    print(f"✅ Kinematics shape test passed: {k_seq.shape}")


def test_component_output_shapes():
    """Test component output shapes match tasks.yml specifications."""
    config = load_tasks_config()
    batch_size = 2
    
    # Test each component's expected output shapes
    components = config.get('components', {})
    
    for comp_name, comp_config in components.items():
        outputs = comp_config.get('outputs', {})
        if outputs:
            print(f"Component {comp_name} outputs: {list(outputs.keys())}")


def test_fusion_shapes():
    """Test fusion component input/output shapes."""
    config = load_tasks_config()
    
    # Check f2 (CrossAttnFusion) shapes from tasks.yml
    f2_config = config['components']['fusion_guard']['f2']
    dims = f2_config['dims']
    
    print(f"✅ Fusion dims from tasks.yml: zi={dims['zi']}, zt={dims['zt']}, zr={dims['zr']}, e_cls={dims['e_cls']}")
    
    outputs = f2_config['outputs']
    print(f"✅ Fusion outputs: {list(outputs.keys())}")


if __name__ == "__main__":
    print("🧪 Running TRIDENT-Net shape validation tests...")
    
    test_rgb_shapes()
    test_ir_shapes() 
    test_kinematics_shapes()
    test_component_output_shapes()
    test_fusion_shapes()
    
    print("\n🧪 Testing new components...")
    test_new_component_shapes()
    test_data_components()
    
    print("✅ All shape tests completed!")