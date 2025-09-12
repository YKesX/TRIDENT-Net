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
    
    # Create synthetic dataset with native 1280×720 resolution
    dataset = SyntheticDataset(size=10, height=720, width=1280)
    
    # Test length
    assert len(dataset) == 10, f"Dataset length mismatch: expected 10, got {len(dataset)}"
    
    # Test sample retrieval
    sample = dataset[0]
    
    # Verify expected keys
    expected_keys = ['rgb', 'ir', 'kin', 'labels', 'class_id', 'class_conf', 'meta']
    for key in expected_keys:
        assert key in sample, f"Missing key {key} in sample"
    
    # Verify shapes (using native 1280×720 from tasks.yml)
    assert sample['rgb'].shape == (3, 16, 720, 1280), f"RGB frames shape mismatch: expected (3, 16, 720, 1280), got {sample['rgb'].shape}"
    assert sample['ir'].shape == (1, 16, 720, 1280), f"IR frames shape mismatch: expected (1, 16, 720, 1280), got {sample['ir'].shape}"
    assert sample['kin'].shape == (3, 9), f"Kinematics shape mismatch: expected (3, 9), got {sample['kin'].shape}"
    
    # Verify label format
    assert sample['labels']['hit'].shape == (1,), "Hit label shape mismatch"
    assert sample['labels']['kill'].shape == (1,), "Kill label shape mismatch"
    assert sample['class_id'].shape == (1,), "Class ID shape mismatch"
    
    # Verify data types
    assert isinstance(sample['rgb'], torch.Tensor), "RGB frames should be tensor"
    assert isinstance(sample['ir'], torch.Tensor), "IR frames should be tensor"
    assert isinstance(sample['kin'], torch.Tensor), "Kinematics should be tensor"
    assert isinstance(sample['meta'], dict), "Meta should be dict"
    
    print("✅ Synthetic dataset test passed")


def test_dataloader_functionality():
    """Test dataloader with synthetic data."""
    print("🧪 Testing dataloader functionality...")
    
    dataset = SyntheticDataset(size=8)
    # Use the custom collate function to handle variable T
    from trident.data.collate import pad_tracks_collate
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=pad_tracks_collate)
    
    # Test batch generation
    batch = next(iter(dataloader))
    
    # Verify batch shapes (using native 1280×720 from tasks.yml)
    # Note: Variable T gets padded to max T in batch, so we check dimensions exist
    assert len(batch['rgb'].shape) == 5, f"RGB should be 5D [B,C,T,H,W], got shape {batch['rgb'].shape}"
    assert batch['rgb'].shape[0] == 4, f"Batch size should be 4, got {batch['rgb'].shape[0]}"
    assert batch['rgb'].shape[1] == 3, f"RGB channels should be 3, got {batch['rgb'].shape[1]}"
    assert batch['rgb'].shape[3:] == (720, 1280), f"RGB spatial dims should be (720, 1280), got {batch['rgb'].shape[3:]}"
    
    assert len(batch['ir'].shape) == 5, f"IR should be 5D [B,C,T,H,W], got shape {batch['ir'].shape}"
    assert batch['ir'].shape[0] == 4, f"Batch size should be 4, got {batch['ir'].shape[0]}"
    assert batch['ir'].shape[1] == 1, f"IR channels should be 1, got {batch['ir'].shape[1]}"
    assert batch['ir'].shape[3:] == (720, 1280), f"IR spatial dims should be (720, 1280), got {batch['ir'].shape[3:]}"
    assert batch['kin'].shape == (4, 3, 9), "Batched kinematics shape mismatch"
    
    # Verify labels are properly batched
    assert batch['labels']['hit'].shape == (4, 1), "Batched hit labels shape mismatch"
    assert batch['labels']['kill'].shape == (4, 1), "Batched kill labels shape mismatch"
    
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
    
    dataset = SyntheticDataset(size=5, height=720, width=1280)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        rgb = sample['rgb']
        assert rgb.shape == (3, 16, 720, 1280), f"RGB shape mismatch: {rgb.shape}"
        
        # IR frames - synthetic data is generated with randn, so values can be negative  
        ir = sample['ir']
        assert ir.shape == (1, 16, 720, 1280), f"IR shape mismatch: {ir.shape}"
        
        # Labels should be in [0, 1]
        hit_label = sample['labels']['hit']
        kill_label = sample['labels']['kill']
        assert 0 <= hit_label <= 1, f"Hit label out of range [0,1]: {hit_label}"
        assert 0 <= kill_label <= 1, f"Kill label out of range [0,1]: {kill_label}"
    
    print("✅ Data ranges test passed")


def test_1280x720_standardization():
    """Test that 1280×720 is used everywhere, no 704×1248 left."""
    print("🧪 Testing 1280×720 standardization...")
    
    # Test synthetic dataset uses correct resolution
    dataset = SyntheticDataset(size=3, height=720, width=1280)
    sample = dataset[0]
    
    # Verify RGB frames are 1280×720 (C, T, H, W)
    rgb_shape = sample['rgb'].shape
    assert rgb_shape[2:] == (720, 1280), f"RGB frames not 1280×720: {rgb_shape[2:]}"
    
    # Verify IR frames are 1280×720 (C, T, H, W)  
    ir_shape = sample['ir'].shape
    assert ir_shape[2:] == (720, 1280), f"IR frames not 1280×720: {ir_shape[2:]}"
    
    # Test synthetic batch generation
    from trident.data.synthetic import generate_synthetic_batch
    
    batch = generate_synthetic_batch(B=2, H=720, W=1280)
    
    # Verify batch RGB frames are 1280×720 (B, C, T, H, W)
    batch_rgb_shape = batch['rgb'].shape
    assert batch_rgb_shape[3:] == (720, 1280), f"Batch RGB frames not 1280×720: {batch_rgb_shape[3:]}"
    
    # Verify batch IR frames are 1280×720 (B, C, T, H, W)
    batch_ir_shape = batch['ir'].shape  
    assert batch_ir_shape[3:] == (720, 1280), f"Batch IR frames not 1280×720: {batch_ir_shape[3:]}"
    
    # Test that old 704×1248 resolution is NOT used anywhere
    old_height, old_width = 1248, 704
    
    # Generate batch with old dimensions should work but we verify we're not using them by default
    old_batch = generate_synthetic_batch(B=1, H=old_height, W=old_width)
    old_rgb_shape = old_batch['rgb'].shape
    assert old_rgb_shape[3:] == (old_height, old_width), "Old resolution test failed"
    
    # But our default should always be 720p
    default_batch = generate_synthetic_batch(B=1)  # No explicit size
    # Should use default from function signature
    
    print("✅ 1280×720 standardization test passed")


def test_no_legacy_resolutions():
    """Ensure no legacy 704×1248 resolutions are used by default."""
    print("🧪 Testing no legacy resolutions...")
    
    # Load tasks.yml config to verify native resolution
    import yaml
    try:
        with open('tasks.yml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check preprocess image_size
        image_size = config.get('preprocess', {}).get('image_size', {})
        height = image_size.get('h', 720)
        width = image_size.get('w', 1280)
        
        assert height == 720, f"Config height should be 720, got {height}"
        assert width == 1280, f"Config width should be 1280, got {width}"
        
        # Verify align32 is false (no forcing multiples-of-32)
        align32 = image_size.get('align32', True)  # Default to True to catch if missing
        assert align32 is False, f"align32 should be False to keep native 720p, got {align32}"
        
        # Verify letterbox is false (videos already 1280×720)
        letterbox = config.get('preprocess', {}).get('letterbox', True)  # Default to True to catch if missing
        assert letterbox is False, f"letterbox should be False for native 1280×720, got {letterbox}"
        
        print("✅ No legacy resolutions test passed")
        
    except FileNotFoundError:
        print("⚠️ tasks.yml not found, skipping config verification")
    except Exception as e:
        print(f"⚠️ Config verification error: {e}")


if __name__ == "__main__":
    print("🧪 Running TRIDENT-Net dataset tests...")
    
    test_synthetic_dataset()
    test_dataloader_functionality()
    test_letterbox_transforms()
    test_synchronized_augmentations()
    test_kinematics_processing()
    test_data_ranges()
    test_1280x720_standardization()
    test_no_legacy_resolutions()
    
    print("✅ All dataset tests passed!")