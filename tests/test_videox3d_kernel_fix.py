#!/usr/bin/env python3
"""
Test for VideoFrag3Dv2 3D pooling kernel size fix.

This test verifies that the fix for the kernel size > input size error works correctly.
The original error occurred when temporal dimension T=1 and pooling tried to use kernel_size=(2,2,2).
"""

import sys
import os
import torch

# Add the repository root to the path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trident.trident_i.videox3d import VideoFrag3Dv2


def test_kernel_size_fix():
    """Test that VideoFrag3Dv2 handles small temporal dimensions without kernel size errors."""
    
    model = VideoFrag3Dv2(
        in_channels=3,
        base_channels=32, 
        depth=3,
        temporal_kernel=3,
        temporal_stride=2,
        out_embed_dim=512
    )
    
    # Test the original problematic case: T=1 with large spatial dims
    # This would previously fail with: "Kernel size can't be greater than actual input size"
    rgb_chunk = torch.randn(1, 3, 1, 363, 643)
    
    # This should not raise an exception
    with torch.no_grad():
        output = model(rgb_chunk)
    
    # Verify output structure
    assert 'mask_seq' in output
    assert 'zi' in output  
    assert 'events' in output
    
    # Verify output shapes
    assert output['mask_seq'].shape == (1, 1, 1, 363, 643)
    assert output['zi'].shape == (1, 512)
    assert isinstance(output['events'], list)
    
    print("✅ Kernel size fix test passed")


def test_adaptive_pooling_behavior():
    """Test that adaptive pooling creates appropriate kernel sizes."""
    
    model = VideoFrag3Dv2()
    
    # Test cases: (T, expected_temporal_kernel, expected_temporal_stride)
    test_cases = [
        (1, 1, 1),  # T=1 should use kernel/stride=1
        (2, 2, 2),  # T=2 should use kernel/stride=2  
        (4, 2, 2),  # T=4 should use kernel/stride=2
        (8, 2, 2),  # T=8 should use kernel/stride=2
    ]
    
    for T, expected_kernel, expected_stride in test_cases:
        pool = model._create_adaptive_pool(channels=32, input_shape=(T, 100, 100))
        
        actual_kernel = pool.kernel_size[0]  # temporal dimension
        actual_stride = pool.stride[0]       # temporal dimension
        
        assert actual_kernel == expected_kernel, f"T={T}: expected kernel {expected_kernel}, got {actual_kernel}"
        assert actual_stride == expected_stride, f"T={T}: expected stride {expected_stride}, got {actual_stride}"
    
    print("✅ Adaptive pooling behavior test passed")


def test_various_temporal_sizes():
    """Test VideoFrag3Dv2 with various temporal sizes to ensure robustness."""
    
    model = VideoFrag3Dv2()
    
    temporal_sizes = [1, 2, 3, 4, 8, 16]
    
    for T in temporal_sizes:
        rgb = torch.randn(1, 3, T, 64, 64)
        
        # Should not raise any exceptions
        with torch.no_grad():
            output = model(rgb)
        
        # Verify output shapes make sense
        assert output['mask_seq'].shape == (1, T, 1, 64, 64)
        assert output['zi'].shape == (1, 512)
    
    print("✅ Various temporal sizes test passed")


if __name__ == "__main__":
    print("🧪 Running VideoFrag3Dv2 kernel size fix tests...")
    
    test_kernel_size_fix()
    test_adaptive_pooling_behavior() 
    test_various_temporal_sizes()
    
    print("🎉 All tests passed! The 3D pooling kernel size fix is working correctly.")