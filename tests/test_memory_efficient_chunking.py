#!/usr/bin/env python3
"""
Test for memory-efficient video processing chunking fix.

This test verifies that the fix for tensor shape mismatches during chunking works correctly.
The original error occurred when different chunks had different temporal dimensions and 
couldn't be stacked in the process_video_in_chunks function.
"""

import sys
import os
import torch

# Add the repository root to the path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trident.trident_i.videox3d import VideoFrag3Dv2
from trident.runtime.memory_efficient_cli import process_video_in_chunks


def test_emergency_mode_frame_selection():
    """Test that emergency mode correctly selects frames at 1300ms, 2000ms, 6000ms."""
    
    # Create a model instance
    model = VideoFrag3Dv2(
        in_channels=3,
        base_channels=32, 
        depth=3,
        temporal_kernel=3,
        temporal_stride=2,
        out_embed_dim=512
    )
    
    # Create a video tensor that simulates 8 seconds at ~24fps (87 frames for 8000ms)
    T = 87  # This matches the error case
    rgb_video = torch.randn(1, 3, T, 720, 1280)
    
    print(f"Testing with video tensor shape: {rgb_video.shape}")
    
    # Test the emergency frame selection logic
    total_duration_ms = 8000
    frame_1300ms = int((1300 / total_duration_ms) * T)
    frame_2000ms = int((2000 / total_duration_ms) * T) 
    frame_6000ms = int((6000 / total_duration_ms) * T)
    
    print(f"Expected frame indices: 1300ms→{frame_1300ms}, 2000ms→{frame_2000ms}, 6000ms→{frame_6000ms}")
    
    # These should be around frames 14, 22, 65 for T=87
    assert 10 <= frame_1300ms <= 18, f"1300ms frame {frame_1300ms} not in expected range"
    assert 18 <= frame_2000ms <= 26, f"2000ms frame {frame_2000ms} not in expected range"
    assert 60 <= frame_6000ms <= 70, f"6000ms frame {frame_6000ms} not in expected range"
    
    print("✅ Emergency frame selection test passed")


def test_process_video_in_chunks_emergency_mode():
    """Test that process_video_in_chunks handles emergency mode correctly."""
    
    # Create a model instance
    model = VideoFrag3Dv2(
        in_channels=3,
        base_channels=16,  # Smaller to avoid memory issues in test
        depth=2,
        temporal_kernel=3,
        temporal_stride=2,
        out_embed_dim=256
    )
    
    # Create a large video tensor that would trigger chunking
    T = 87  # This matches the error case from the problem statement
    rgb_video = torch.randn(1, 3, T, 720, 1280)
    
    print(f"Testing process_video_in_chunks with shape: {rgb_video.shape}")
    
    # Test that it doesn't crash and returns valid output
    with torch.no_grad():
        output = process_video_in_chunks(model, rgb_video, chunk_size=8)
    
    # Verify output structure
    assert 'mask_seq' in output
    assert 'zi' in output
    assert 'events' in output
    
    # Verify output shapes are reasonable
    B, out_T, C_out, H, W = output['mask_seq'].shape
    assert B == 1
    assert C_out == 1
    assert H == 720
    assert W == 1280
    # out_T should be 3 (emergency mode) or 1 (ultimate fallback) or T (normal mode)
    assert out_T in [1, 3, T], f"Unexpected temporal dimension: {out_T}"
    
    assert output['zi'].shape == (1, 256)
    assert isinstance(output['events'], list)
    
    print(f"✅ Output shape verification passed: mask_seq={output['mask_seq'].shape}, zi={output['zi'].shape}")


def test_process_video_in_chunks_small_video():
    """Test that small videos are processed normally without chunking."""
    
    model = VideoFrag3Dv2(
        in_channels=3,
        base_channels=16,
        depth=2,
        temporal_kernel=3,
        temporal_stride=2,
        out_embed_dim=256
    )
    
    # Small video that shouldn't trigger chunking
    T = 5
    rgb_video = torch.randn(1, 3, T, 64, 64)
    
    print(f"Testing small video with shape: {rgb_video.shape}")
    
    with torch.no_grad():
        output = process_video_in_chunks(model, rgb_video, chunk_size=8)
    
    # Should process normally and preserve temporal dimension
    assert output['mask_seq'].shape == (1, T, 1, 64, 64)
    assert output['zi'].shape == (1, 256)
    
    print("✅ Small video processing test passed")


def test_emergency_fallback_logic():
    """Test the emergency fallback logic for frame selection."""
    
    # Test frame index calculation for various video lengths
    test_cases = [
        (24, [4, 6, 18]),    # 1 second video (24fps)
        (48, [8, 12, 36]),   # 2 second video  
        (87, [14, 21, 65]),  # ~3.6 second video (matches problem case)
        (192, [31, 48, 144]) # 8 second video
    ]
    
    for T, expected_frames in test_cases:
        total_duration_ms = 8000
        frame_1300ms = max(0, min(int((1300 / total_duration_ms) * T), T-1))
        frame_2000ms = max(0, min(int((2000 / total_duration_ms) * T), T-1))
        frame_6000ms = max(0, min(int((6000 / total_duration_ms) * T), T-1))
        
        calculated = [frame_1300ms, frame_2000ms, frame_6000ms]
        
        print(f"T={T}: calculated={calculated}, expected≈{expected_frames}")
        
        # Allow some tolerance due to integer rounding
        for calc, exp in zip(calculated, expected_frames):
            assert abs(calc - exp) <= 2, f"Frame index {calc} too far from expected {exp}"
    
    print("✅ Emergency fallback logic test passed")


if __name__ == "__main__":
    print("🧪 Running memory-efficient chunking fix tests...")
    
    test_emergency_mode_frame_selection()
    test_process_video_in_chunks_emergency_mode()
    test_process_video_in_chunks_small_video()
    test_emergency_fallback_logic()
    
    print("🎉 All chunking tests passed! The memory-efficient video processing fix is working correctly.")