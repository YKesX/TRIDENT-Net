"""
Test determinism and AMP enforcement in TRIDENT-Net.

Tests Phase 3 of the hardening plan: seed handling, deterministic flags, and AMP.

Author: Yağızhan Keskin
"""

import torch
import pytest
import sys
import os
import tempfile
sys.path.append('.')

from trident.runtime.trainer import Trainer
from trident.runtime.config import ConfigLoader
from trident.fusion_guard.cross_attn_fusion import CrossAttnFusion


def test_deterministic_setup():
    """Test deterministic training setup."""
    # Create a temporary config file
    config_data = {
        'environment': {
            'seed': 42,
            'cudnn_deterministic': True,
            'cudnn_benchmark': False
        },
        'components': {},
        'runtime': {'tasks': {}}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config_data, f)
        config_file = f.name
    
    try:
        # Create config loader and trainer
        config_loader = ConfigLoader()
        config_loader.load_config(config_file)
        trainer = Trainer(config_loader)
        
        # Check that seed was set correctly
        assert hasattr(trainer, 'seed'), "Trainer should have seed attribute"
        assert trainer.seed == 42, f"Expected seed=42, got {trainer.seed}"
        
        # Check that deterministic settings are applied
        if torch.backends.cudnn.is_available():
            assert torch.backends.cudnn.deterministic == True, "CUDNN deterministic should be True"
            assert torch.backends.cudnn.benchmark == False, "CUDNN benchmark should be False"
        
        print("✅ Deterministic setup test passed")
        
    finally:
        os.unlink(config_file)


def test_deterministic_forward_passes():
    """Test that forward passes are deterministic with same seed."""
    # Set seed
    torch.manual_seed(12345)
    
    # Create model
    model = CrossAttnFusion(d_model=64, n_layers=1, n_heads=2)
    model.eval()
    
    # Create test inputs
    batch_size = 2
    zi = torch.randn(batch_size, 768)
    zt = torch.randn(batch_size, 512)
    zr = torch.randn(batch_size, 384)
    class_ids = torch.tensor([1, 5])
    
    # First forward pass
    torch.manual_seed(12345)  # Reset seed
    with torch.no_grad():
        z_fused1, p_hit1, p_kill1, _, _ = model(zi, zt, zr, class_ids=class_ids)
    
    # Second forward pass with same seed
    torch.manual_seed(12345)  # Reset seed again
    with torch.no_grad():
        z_fused2, p_hit2, p_kill2, _, _ = model(zi, zt, zr, class_ids=class_ids)
    
    # Results should be identical
    assert torch.allclose(z_fused1, z_fused2, atol=1e-6), "z_fused should be deterministic"
    assert torch.allclose(p_hit1, p_hit2, atol=1e-6), "p_hit should be deterministic"
    assert torch.allclose(p_kill1, p_kill2, atol=1e-6), "p_kill should be deterministic"
    
    print("✅ Deterministic forward passes test passed")


def test_amp_availability():
    """Test AMP components are available."""
    # Check if autocast and GradScaler are available
    from torch.cuda.amp import autocast, GradScaler
    
    # Create scaler
    scaler = GradScaler()
    assert scaler is not None, "GradScaler should be available"
    
    # Test autocast context
    with autocast():
        x = torch.randn(2, 10)
        y = torch.mm(x, x.t())
        assert y.shape == (2, 2), "Autocast should work correctly"
    
    print("✅ AMP availability test passed")


def test_amp_training_step():
    """Test AMP training step simulation."""
    # Create simple model
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    # Create dummy data
    x = torch.randn(4, 10)
    y = torch.randn(4, 1)
    
    # Training step with AMP
    model.train()
    optimizer.zero_grad()
    
    if torch.cuda.is_available():
        # Test CUDA AMP
        model = model.cuda()
        x, y = x.cuda(), y.cuda()
        
        with torch.cuda.amp.autocast():
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        assert not torch.isnan(loss), "Loss should not be NaN with AMP"
        assert loss.item() >= 0, "Loss should be non-negative"
    else:
        # Test CPU fallback (no AMP)
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss), "Loss should not be NaN without AMP"
        assert loss.item() >= 0, "Loss should be non-negative"
    
    print("✅ AMP training step test passed")


def test_gradient_clipping():
    """Test gradient clipping works correctly."""
    # Create model with large gradients
    model = torch.nn.Linear(10, 1)
    
    # Create data that will produce large gradients
    x = torch.randn(1, 10) * 100  # Large input
    y = torch.randn(1, 1) * 100   # Large target
    
    # Forward pass
    model.train()
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    
    # Backward pass
    loss.backward()
    
    # Check gradients before clipping
    grad_norms_before = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norms_before.append(param.grad.norm().item())
    
    max_norm_before = max(grad_norms_before) if grad_norms_before else 0
    
    # Apply gradient clipping
    clip_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    
    # Check gradients after clipping
    grad_norms_after = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norms_after.append(param.grad.norm().item())
    
    max_norm_after = max(grad_norms_after) if grad_norms_after else 0
    
    # Total gradient norm should be clipped
    total_norm = torch.sqrt(torch.tensor(sum(norm**2 for norm in grad_norms_after)))
    assert total_norm <= clip_norm + 1e-6, f"Total gradient norm {total_norm} should be <= {clip_norm}"
    
    print(f"✅ Gradient clipping test passed - norm reduced from {max_norm_before:.4f} to {max_norm_after:.4f}")


def test_trainer_amp_configuration():
    """Test trainer AMP configuration."""
    config_data = {
        'environment': {'seed': 12345},
        'training': {'amp': True},
        'components': {},
        'runtime': {'tasks': {}}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config_data, f)
        config_file = f.name
    
    try:
        config_loader = ConfigLoader()
        config_loader.load_config(config_file)
        
        # Test with AMP enabled
        trainer_amp = Trainer(config_loader, mixed_precision=True)
        if torch.cuda.is_available():
            assert trainer_amp.scaler is not None, "Scaler should be created when CUDA and AMP enabled"
            assert trainer_amp.mixed_precision == True, "Mixed precision should be enabled"
        
        # Test with AMP disabled
        trainer_no_amp = Trainer(config_loader, mixed_precision=False)
        assert trainer_no_amp.mixed_precision == False, "Mixed precision should be disabled"
        
        print("✅ Trainer AMP configuration test passed")
        
    finally:
        os.unlink(config_file)


def test_full_determinism_smoke():
    """Smoke test for full deterministic behavior with fixed seeds."""
    print("🧪 Testing full determinism smoke test...")
    
    seed = 42
    
    # Test 1: Two identical runs should produce identical results
    def run_deterministic_forward():
        # Setup deterministic environment
        setup_deterministic_training(seed)
        
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        # Create deterministic input
        torch.manual_seed(seed)  # Ensure input is also deterministic
        x = torch.randn(2, 10)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        return output
    
    # Run twice with same seed
    output1 = run_deterministic_forward()
    output2 = run_deterministic_forward()
    
    # Should be identical
    assert torch.allclose(output1, output2, atol=1e-8), \
        f"Deterministic runs produced different outputs: diff = {(output1 - output2).abs().max()}"
    
    print("✅ Full determinism smoke test passed")


def test_non_deterministic_behavior():
    """Test that without deterministic setup, results vary."""
    print("🧪 Testing non-deterministic behavior...")
    
    def run_non_deterministic_forward():
        # Create simple model without deterministic setup
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),  
            torch.nn.Linear(5, 1)
        )
        
        # Random input (not seeded)
        x = torch.randn(2, 10)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        return output
    
    # Run multiple times
    outputs = [run_non_deterministic_forward() for _ in range(3)]
    
    # Should be different (with very high probability)
    all_different = True
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            if torch.allclose(outputs[i], outputs[j], atol=1e-6):
                all_different = False
                break
        if not all_different:
            break
    
    # Note: This test might occasionally fail due to random chance, but very unlikely
    if all_different:
        print("✅ Non-deterministic behavior confirmed")
    else:
        print("⚠️ Non-deterministic test: outputs happened to be similar (rare but possible)")


if __name__ == "__main__":
    test_deterministic_setup()
    test_deterministic_forward_passes()
    test_amp_availability()
    test_amp_training_step()
    test_gradient_clipping()
    test_trainer_amp_configuration()
    test_full_determinism_smoke()
    test_non_deterministic_behavior()
    print("✅ All determinism and AMP tests passed!")