#!/usr/bin/env python3
"""
Demonstration of the original error and how the fix resolves it.
"""

import torch


def original_broken_chunking(video_tensor, chunk_size=8, overlap=2):
    """
    This is the ORIGINAL BROKEN implementation that caused the error.
    DO NOT USE - This is for demonstration only.
    """
    B, C, T, H, W = video_tensor.shape
    
    print(f"🔧 Processing large video (T={T}) in chunks of {chunk_size} frames")
    
    # Process in overlapping chunks and average results
    chunk_outputs = []
    for start_idx in range(0, T, chunk_size - overlap):
        end_idx = min(start_idx + chunk_size, T)
        if end_idx - start_idx < 3:  # Skip chunks that are too small
            break
            
        chunk = video_tensor[:, :, start_idx:end_idx]
        chunk_T = chunk.shape[2]
        
        # Mock model output - this shows the problem
        mock_output = {
            'mask_seq': torch.randn(B, chunk_T, 1, H, W),  # Different T values!
            'zi': torch.randn(B, 512)
        }
        chunk_outputs.append(mock_output)
        
        print(f"  Chunk {len(chunk_outputs)}: frames {start_idx}-{end_idx-1} → mask_seq shape {mock_output['mask_seq'].shape}")
    
    # This is where the error occurs!
    print(f"\n❌ Attempting to stack {len(chunk_outputs)} chunks with different temporal dimensions:")
    try:
        for key in chunk_outputs[0].keys():
            if key != 'zi':  # zi is always the same shape
                shapes = [out[key].shape for out in chunk_outputs]
                print(f"  {key} shapes: {shapes}")
                
                # This will fail!
                stacked = torch.stack([out[key] for out in chunk_outputs])
                print(f"  ✅ Successfully stacked {key}")
    except RuntimeError as e:
        print(f"  💥 ERROR: {e}")
        return None
    
    return "Success (unexpected)"


def fixed_emergency_mode(video_tensor):
    """
    This is the FIXED implementation that resolves the error.
    """
    B, C, T, H, W = video_tensor.shape
    
    print(f"\n🚨 Emergency mode: Processing specific frames at 1300ms, 2000ms, and 6000ms")
    total_duration_ms = 8000  # All videos are 8000ms
    
    # Calculate frame indices for specific timestamps
    frame_1300ms = int((1300 / total_duration_ms) * T)
    frame_2000ms = int((2000 / total_duration_ms) * T) 
    frame_6000ms = int((6000 / total_duration_ms) * T)
    
    # Ensure indices are within bounds
    frame_1300ms = max(0, min(frame_1300ms, T-1))
    frame_2000ms = max(0, min(frame_2000ms, T-1))
    frame_6000ms = max(0, min(frame_6000ms, T-1))
    
    selected_frames = [frame_1300ms, frame_2000ms, frame_6000ms]
    emergency_tensor = video_tensor[:, :, selected_frames]
    
    print(f"  Selected frames: {selected_frames}")
    print(f"  Emergency tensor shape: {emergency_tensor.shape}")
    
    # Mock model output with consistent temporal dimension
    mock_output = {
        'mask_seq': torch.randn(B, len(selected_frames), 1, H, W),
        'zi': torch.randn(B, 512)
    }
    
    print(f"  ✅ Output mask_seq shape: {mock_output['mask_seq'].shape}")
    print(f"  ✅ Memory usage reduced: {T} frames → {len(selected_frames)} frames")
    
    return mock_output


def demonstrate_fix():
    """Demonstrate the original problem and the fix."""
    
    print("=" * 80)
    print("DEMONSTRATION: Video Chunking Tensor Shape Mismatch Fix")
    print("=" * 80)
    
    # Create the exact problematic case from the error message
    rgb_tensor = torch.randn(1, 3, 87, 720, 1280)
    print(f"\nInput video tensor: {rgb_tensor.shape}")
    print(f"This represents an 8-second video with 87 frames at ~24fps")
    
    print(f"\n{'='*50}")
    print("ORIGINAL BROKEN IMPLEMENTATION")
    print(f"{'='*50}")
    
    # Show the original broken approach
    result = original_broken_chunking(rgb_tensor, chunk_size=8, overlap=2)
    
    print(f"\n{'='*50}")
    print("FIXED EMERGENCY MODE IMPLEMENTATION")  
    print(f"{'='*50}")
    
    # Show the fixed approach
    result = fixed_emergency_mode(rgb_tensor)
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print("✅ Original error: 'stack expects each tensor to be equal size'")
    print("✅ Root cause: Different chunks had different temporal dimensions")
    print("✅ Solution: Emergency mode samples specific frames at 1300ms, 2000ms, 6000ms")
    print("✅ Result: Consistent tensor shapes, reduced memory usage, no crashes")
    print("✅ All videos are 8s, frames taken from specified timestamps as requested")


if __name__ == "__main__":
    demonstrate_fix()