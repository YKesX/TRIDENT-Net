#!/usr/bin/env python3
"""
Test for dtype mismatch fix in CrossAttnFusion.

This test verifies that the dtype mismatch error in mixed precision training
has been resolved by ensuring input tensors are converted to match model dtype.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

# Create stub modules for missing dependencies during testing
sys.modules['cv2'] = type(sys)('cv2')
sys.modules['timm'] = type(sys)('timm')
sys.modules['timm'].create_model = lambda *args, **kwargs: nn.Linear(10, 10)

from trident.fusion_guard.cross_attn_fusion import CrossAttnFusion


def test_dtype_mismatch_fix():
    """Test that the dtype mismatch between Float and Half is resolved."""
    print("🧪 Testing dtype mismatch fix in CrossAttnFusion...")
    
    # Create model
    model = CrossAttnFusion(d_model=512, n_layers=2, n_heads=8)
    
    # Create test inputs (typical data pipeline outputs in float32)
    batch_size = 1
    zi = torch.randn(batch_size, 768, dtype=torch.float32)
    zt = torch.randn(batch_size, 512, dtype=torch.float32) 
    zr = torch.randn(batch_size, 384, dtype=torch.float32)
    class_ids = torch.tensor([1])
    
    # Test 1: Normal operation (float32 model and inputs)
    model.eval()
    with torch.no_grad():
        outputs = model(zi, zt, zr, class_ids=class_ids)
    assert len(outputs) == 5, "Expected 5 outputs from CrossAttnFusion"
    assert outputs[0].dtype == torch.float32, "Expected float32 output for float32 model"
    print("✅ Test 1 passed: Float32 model with float32 inputs")
    
    # Test 2: Mixed precision scenario (half model, float32 inputs)
    # This is the scenario that was failing before the fix
    model_half = model.half()
    model_half.eval()
    with torch.no_grad():
        outputs = model_half(zi, zt, zr, class_ids=class_ids)
    assert len(outputs) == 5, "Expected 5 outputs from CrossAttnFusion"
    assert outputs[0].dtype == torch.float16, "Expected float16 output for half precision model"
    print("✅ Test 2 passed: Half precision model with float32 inputs")
    
    # Test 3: Autocast scenario
    model.eval()
    with torch.autocast(device_type="cpu", dtype=torch.float16, enabled=True):
        with torch.no_grad():
            outputs = model(zi, zt, zr, class_ids=class_ids)
    assert len(outputs) == 5, "Expected 5 outputs from CrossAttnFusion"
    print("✅ Test 3 passed: Autocast scenario")
    
    # Test 4: Test with kwargs (trainer-style call)
    batch = {
        'zi': zi, 'zt': zt, 'zr': zr,
        'class_ids': class_ids,
        'events': None,
        'labels': {}
    }
    model_half.eval()
    with torch.no_grad():
        outputs = model_half(**batch)
    assert len(outputs) == 5, "Expected 5 outputs from CrossAttnFusion"
    print("✅ Test 4 passed: Trainer-style kwargs call")
    
    print("🎉 All dtype mismatch fix tests passed!")


def test_edge_cases():
    """Test edge cases for the dtype fix."""
    print("\n🧪 Testing edge cases...")
    
    model = CrossAttnFusion(d_model=512, n_layers=1, n_heads=4)
    model = model.half()
    
    batch_size = 1
    zi = torch.randn(batch_size, 768, dtype=torch.float32)
    zt = torch.randn(batch_size, 512, dtype=torch.float32)
    zr = torch.randn(batch_size, 384, dtype=torch.float32)
    
    # Test with None class_ids
    model.eval()
    with torch.no_grad():
        outputs = model(zi, zt, zr, class_ids=None)
    assert len(outputs) == 5, "Expected 5 outputs with None class_ids"
    print("✅ Edge case 1 passed: None class_ids")
    
    # Test with precomputed class embeddings (float32)
    class_emb = torch.randn(batch_size, 32, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(zi, zt, zr, class_ids=class_emb)
    assert len(outputs) == 5, "Expected 5 outputs with precomputed embeddings"
    print("✅ Edge case 2 passed: Precomputed class embeddings")
    
    print("🎉 All edge case tests passed!")


if __name__ == "__main__":
    test_dtype_mismatch_fix()
    test_edge_cases()
    print("\n✅ All tests passed! The dtype mismatch issue has been resolved.")