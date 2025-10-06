#!/usr/bin/env python3
"""
Simple test for the chunking fix logic without heavy dependencies.
"""

import torch

def test_emergency_frame_selection():
    """Test that emergency mode correctly calculates frame indices."""
    
    # Test frame index calculation for the problem case
    T = 87  # From the error message
    total_duration_ms = 8000
    
    # Calculate frame indices for specific timestamps  
    frame_1300ms = int((1300 / total_duration_ms) * T)
    frame_2000ms = int((2000 / total_duration_ms) * T) 
    frame_6000ms = int((6000 / total_duration_ms) * T)
    
    # Ensure indices are within bounds
    frame_1300ms = max(0, min(frame_1300ms, T-1))
    frame_2000ms = max(0, min(frame_2000ms, T-1))
    frame_6000ms = max(0, min(frame_6000ms, T-1))
    
    print(f"For T={T} frames (8000ms video):")
    print(f"  1300ms → frame {frame_1300ms}")
    print(f"  2000ms → frame {frame_2000ms}")
    print(f"  6000ms → frame {frame_6000ms}")
    
    # Verify reasonable frame indices
    assert 0 <= frame_1300ms < T
    assert 0 <= frame_2000ms < T  
    assert 0 <= frame_6000ms < T
    assert frame_1300ms < frame_2000ms < frame_6000ms
    
    # Ensure unique indices
    selected_frames = list(dict.fromkeys([frame_1300ms, frame_2000ms, frame_6000ms]))
    print(f"  Selected frames: {selected_frames}")
    
    assert len(selected_frames) >= 2, "Should have at least 2 unique frames"
    
    print("✅ Emergency frame selection test passed")


def test_tensor_slicing():
    """Test that tensor slicing with selected frames works correctly."""
    
    # Create a mock video tensor similar to the error case
    B, C, T, H, W = 1, 3, 87, 720, 1280
    video_tensor = torch.randn(B, C, T, H, W)
    
    print(f"Original tensor shape: {video_tensor.shape}")
    
    # Apply emergency frame selection
    total_duration_ms = 8000
    frame_1300ms = max(0, min(int((1300 / total_duration_ms) * T), T-1))
    frame_2000ms = max(0, min(int((2000 / total_duration_ms) * T), T-1))
    frame_6000ms = max(0, min(int((6000 / total_duration_ms) * T), T-1))
    
    selected_frames = list(dict.fromkeys([frame_1300ms, frame_2000ms, frame_6000ms]))
    
    # This should not raise any errors
    emergency_tensor = video_tensor[:, :, selected_frames]
    
    print(f"Emergency tensor shape: {emergency_tensor.shape}")
    
    # Verify shape
    expected_shape = (B, C, len(selected_frames), H, W)
    assert emergency_tensor.shape == expected_shape, f"Expected {expected_shape}, got {emergency_tensor.shape}"
    
    print("✅ Tensor slicing test passed")


def test_frame_calculation_edge_cases():
    """Test frame calculation for edge cases."""
    
    test_cases = [
        (1, "Very short video"),
        (3, "3-frame video"),  
        (24, "1-second video"),
        (48, "2-second video"),
        (87, "Problem case"),
        (192, "8-second video")
    ]
    
    for T, description in test_cases:
        total_duration_ms = 8000
        
        frame_1300ms = max(0, min(int((1300 / total_duration_ms) * T), T-1))
        frame_2000ms = max(0, min(int((2000 / total_duration_ms) * T), T-1))
        frame_6000ms = max(0, min(int((6000 / total_duration_ms) * T), T-1))
        
        selected_frames = list(dict.fromkeys([frame_1300ms, frame_2000ms, frame_6000ms]))
        
        # Ensure we have at least one frame
        if not selected_frames:
            selected_frames = [0]
        
        # Add more frames if needed and available
        while len(selected_frames) < 3 and len(selected_frames) < T:
            for i in range(T):
                if i not in selected_frames:
                    selected_frames.append(i)
                    if len(selected_frames) >= 3:
                        break
        
        print(f"{description} (T={T}): frames {selected_frames}")
        
        # Verify all frames are valid
        for frame in selected_frames:
            assert 0 <= frame < T, f"Invalid frame index {frame} for T={T}"
    
    print("✅ Edge cases test passed")


if __name__ == "__main__":
    print("🧪 Testing chunking fix logic...")
    
    test_emergency_frame_selection()
    test_tensor_slicing()
    test_frame_calculation_edge_cases()
    
    print("🎉 All chunking fix tests passed!")