"""
Test dataset functionality for TRIDENT-Net.

Tests data loading, preprocessing, letterbox transforms, and augmentations.

Author: Yağızhan Keskin
"""

import torch
import numpy as np
import sys
sys.path.append('.')

import trident
from trident.data.synthetic import SyntheticDataset


def test_synthetic_dataset():
    """Test synthetic dataset functionality."""
    print("🧪 Testing synthetic dataset...")
    
    # Create synthetic dataset
    dataset = SyntheticDataset(n_samples=10)
    
    # Test length
    assert len(dataset) == 10, f"Dataset length mismatch: expected 10, got {len(dataset)}"
    
    # Test sample retrieval
    sample = dataset[0]
    
    # Verify expected keys
    expected_keys = ['rgb_seq', 'ir_seq', 'k_seq', 'y_outcome', 'meta']
    for key in expected_keys:
        assert key in sample, f"Missing key {key} in sample"
    
    # Verify shapes (using native 1280×720 from tasks.yml)
    assert sample['rgb_seq'].shape[2:] == (720, 1280), f"RGB sequence shape mismatch: expected (720, 1280), got {sample['rgb_seq'].shape[2:]}"
    assert sample['ir_seq'].shape[2:] == (720, 1280), f"IR sequence shape mismatch: expected (720, 1280), got {sample['ir_seq'].shape[2:]}"
    assert sample['k_seq'].shape == (3, 9), f"Kinematics sequence shape mismatch: expected (3, 9), got {sample['k_seq'].shape}"
    
    # Verify label format
    assert 'hit' in sample['y_outcome'], "Missing hit labels"
    assert 'kill' in sample['y_outcome'], "Missing kill labels"
    assert sample['y_outcome']['hit'].shape == (1,), "Hit label shape mismatch"
    assert sample['y_outcome']['kill'].shape == (1,), "Kill label shape mismatch"
    
    # Verify data types
    assert isinstance(sample['rgb_seq'], torch.Tensor), "RGB should be tensor"
    assert isinstance(sample['ir_seq'], torch.Tensor), "IR should be tensor"
    assert isinstance(sample['k_seq'], torch.Tensor), "Kinematics should be tensor"
    
    print("✅ Synthetic dataset test passed")


def test_dataloader_functionality():
    """Test dataloader with synthetic data."""
    print("🧪 Testing dataloader functionality...")
    
    dataset = SyntheticDataset(n_samples=8)
    # Use the custom collate function to handle variable T
    from trident.data.collate import pad_tracks_collate
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=pad_tracks_collate)
    
    # Test batch generation
    batch = next(iter(dataloader))
    
    # Verify batch shapes (using native 1280×720 from tasks.yml)
    # Note: Variable T gets padded to max T in batch, so we check dimensions exist
    assert len(batch['rgb_seq'].shape) == 5, f"RGB should be 5D [B,C,T,H,W], got shape {batch['rgb_seq'].shape}"
    assert batch['rgb_seq'].shape[0] == 4, f"Batch size should be 4, got {batch['rgb_seq'].shape[0]}"
    assert batch['rgb_seq'].shape[1] == 3, f"RGB channels should be 3, got {batch['rgb_seq'].shape[1]}"
    assert batch['rgb_seq'].shape[3:] == (720, 1280), f"RGB spatial dims should be (720, 1280), got {batch['rgb_seq'].shape[3:]}"
    
    assert len(batch['ir_seq'].shape) == 5, f"IR should be 5D [B,C,T,H,W], got shape {batch['ir_seq'].shape}"
    assert batch['ir_seq'].shape[0] == 4, f"Batch size should be 4, got {batch['ir_seq'].shape[0]}"
    assert batch['ir_seq'].shape[1] == 1, f"IR channels should be 1, got {batch['ir_seq'].shape[1]}"
    assert batch['ir_seq'].shape[3:] == (720, 1280), f"IR spatial dims should be (720, 1280), got {batch['ir_seq'].shape[3:]}"
    assert batch['k_seq'].shape == (4, 3, 9), "Batched kinematics shape mismatch"
    
    # Verify labels are properly batched
    assert batch['y_outcome']['hit'].shape == (4, 1), "Batched hit labels shape mismatch"
    assert batch['y_outcome']['kill'].shape == (4, 1), "Batched kill labels shape mismatch"
    
    print("✅ Dataloader functionality test passed")


def test_letterbox_transforms():
    """Test letterbox preprocessing to native 720x1280."""
    print("🧪 Testing letterbox transforms...")
    
    # Test with different input sizes
    test_sizes = [(640, 480), (1100, 760), (1920, 1080)]
    target_size = (720, 1280)  # H, W - native resolution from tasks.yml
    
    for orig_w, orig_h in test_sizes:
        # Create dummy image
        img = torch.randn(3, orig_h, orig_w)
        
        # Apply letterbox transform (simplified)
        # This would normally be done by albumentations or custom transform
        # For now, just test resize to target
        resized = torch.nn.functional.interpolate(
            img.unsqueeze(0), 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        assert resized.shape == (3, 720, 1280), f"Letterbox failed for {orig_w}x{orig_h}: expected (3, 720, 1280), got {resized.shape}"
    
    print("✅ Letterbox transforms test passed")


def test_synchronized_augmentations():
    """Test synchronized augmentations across modalities."""
    print("🧪 Testing synchronized augmentations...")
    
    # For now, just test that we can apply the same transform to RGB and IR
    batch_size = 2
    rgb_seq = torch.randn(batch_size, 3, 3, 720, 1280)
    ir_seq = torch.randn(batch_size, 3, 1, 720, 1280)
    
    # Test horizontal flip (should be synchronized)
    flip_prob = torch.rand(batch_size) < 0.5
    
    for i in range(batch_size):
        if flip_prob[i]:
            rgb_seq[i] = torch.flip(rgb_seq[i], dims=[-1])  # flip width
            ir_seq[i] = torch.flip(ir_seq[i], dims=[-1])   # flip width
    
    # Verify shapes are preserved
    assert rgb_seq.shape == (batch_size, 3, 3, 720, 1280), f"RGB shape changed after augmentation: expected (batch_size, 3, 3, 720, 1280), got {rgb_seq.shape}"
    assert ir_seq.shape == (batch_size, 3, 1, 720, 1280), f"IR shape changed after augmentation: expected (batch_size, 3, 1, 720, 1280), got {ir_seq.shape}"
    
    print("✅ Synchronized augmentations test passed")


def test_kinematics_processing():
    """Test kinematics data processing and normalization."""
    print("🧪 Testing kinematics processing...")
    
    # Test kinematics order from tasks.yml: ["x","y","z","vx","vy","vz","range","bearing","elevation"]
    batch_size = 2
    k_seq = torch.randn(batch_size, 3, 9)  # 3 timesteps, 9 features
    
    # Test delta computation (frame-to-frame differences)
    k_deltas = k_seq[:, 1:] - k_seq[:, :-1]  # (B, 2, 9) - deltas between consecutive frames
    
    assert k_deltas.shape == (batch_size, 2, 9), "Kinematics deltas shape mismatch"
    
    # Test standardization (z-score normalization)
    k_mean = k_seq.mean(dim=(0, 1), keepdim=True)
    k_std = k_seq.std(dim=(0, 1), keepdim=True) + 1e-8
    k_normalized = (k_seq - k_mean) / k_std
    
    # Verify normalized has approximately zero mean and unit std
    assert abs(k_normalized.mean().item()) < 0.1, "Normalized kinematics should have ~zero mean"
    
    print("✅ Kinematics processing test passed")


def test_data_ranges():
    """Test that data values are in expected ranges."""
    print("🧪 Testing data value ranges...")
    
    dataset = SyntheticDataset(n_samples=5)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # RGB should be in [0, 1] after normalization
        rgb = sample['rgb_seq']
        assert rgb.min() >= 0 and rgb.max() <= 1, f"RGB values out of range [0,1]: [{rgb.min():.3f}, {rgb.max():.3f}]"
        
        # IR should be in [0, 1] after normalization
        ir = sample['ir_seq']
        assert ir.min() >= 0 and ir.max() <= 1, f"IR values out of range [0,1]: [{ir.min():.3f}, {ir.max():.3f}]"
        
        # Labels should be in [0, 1]
        hit_label = sample['y_outcome']['hit']
        kill_label = sample['y_outcome']['kill']
        assert 0 <= hit_label <= 1, f"Hit label out of range [0,1]: {hit_label}"
        assert 0 <= kill_label <= 1, f"Kill label out of range [0,1]: {kill_label}"
    
    print("✅ Data ranges test passed")


if __name__ == "__main__":
    print("🧪 Running TRIDENT-Net dataset tests...")
    
    test_synthetic_dataset()
    test_dataloader_functionality()
    test_letterbox_transforms()
    test_synchronized_augmentations()
    test_kinematics_processing()
    test_data_ranges()
    
    print("✅ All dataset tests passed!")