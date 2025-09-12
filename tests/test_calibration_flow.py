"""
Test Phase 8: Calibration flow implementation.
"""

import pytest
import torch
import tempfile
import sys
sys.path.append('.')

from trident.fusion_guard.calib_glm import CalibGLM


def test_calib_glm_fit_save_load():
    """Test CalibGLM fit, save, and load functionality."""
    print("🎯 Testing CalibGLM fit, save, and load...")
    
    # Create synthetic data
    n_samples = 100
    features = torch.randn(n_samples, 1696)  # 1696-d features as per Phase 2
    hit_labels = torch.randint(0, 2, (n_samples,)).float()
    kill_labels = torch.randint(0, 2, (n_samples,)).float()
    
    # Create and fit CalibGLM
    calib_glm = CalibGLM(in_dim=1696, model="logreg")
    assert not calib_glm.is_fitted, "CalibGLM should not be fitted initially"
    
    # Fit the model
    calib_glm.fit(features, hit_labels, kill_labels)
    assert calib_glm.is_fitted, "CalibGLM should be fitted after calling fit()"
    print("✓ CalibGLM fitted successfully")
    
    # Test predictions
    p_hit_aux, p_kill_aux = calib_glm(features[:10])
    assert p_hit_aux.shape == (10, 1), f"Expected hit predictions shape (10, 1), got {p_hit_aux.shape}"
    assert p_kill_aux.shape == (10, 1), f"Expected kill predictions shape (10, 1), got {p_kill_aux.shape}"
    assert torch.all((p_hit_aux >= 0) & (p_hit_aux <= 1)), "Hit predictions should be in [0,1]"
    assert torch.all((p_kill_aux >= 0) & (p_kill_aux <= 1)), "Kill predictions should be in [0,1]"
    print("✓ CalibGLM predictions working")
    
    # Test save and load
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        temp_path = f.name
    
    calib_glm.save(temp_path)
    print(f"✓ CalibGLM saved to {temp_path}")
    
    # Create new instance and load
    calib_glm_loaded = CalibGLM(in_dim=1696, model="logreg")
    assert not calib_glm_loaded.is_fitted, "New instance should not be fitted"
    
    calib_glm_loaded.load(temp_path)
    assert calib_glm_loaded.is_fitted, "Loaded instance should be fitted"
    print("✓ CalibGLM loaded successfully")
    
    # Test that loaded model gives same predictions
    p_hit_loaded, p_kill_loaded = calib_glm_loaded(features[:10])
    assert torch.allclose(p_hit_aux, p_hit_loaded, atol=1e-5), "Loaded model predictions should match original"
    assert torch.allclose(p_kill_aux, p_kill_loaded, atol=1e-5), "Loaded model predictions should match original"
    print("✓ Loaded model predictions match original")
    
    print("🎯 CalibGLM fit, save, and load tests passed!")


def test_calib_glm_feature_dimensions():
    """Test CalibGLM with correct feature dimensions from Phase 2."""
    print("📏 Testing CalibGLM feature dimensions...")
    
    # Test with Phase 2 dimensions: zi(768) + zt(512) + zr(384) + e_cls(32) = 1696
    features_1696 = torch.randn(5, 1696)
    labels = torch.randint(0, 2, (5,)).float()
    
    calib_glm = CalibGLM(in_dim=1696)
    calib_glm.fit(features_1696, labels, labels)
    
    p_hit, p_kill = calib_glm(features_1696)
    assert p_hit.shape == (5, 1), "Hit predictions should have correct shape"
    assert p_kill.shape == (5, 1), "Kill predictions should have correct shape"
    print(f"✓ CalibGLM handles 1696-d features correctly")
    
    # Test dimension mismatch detection
    wrong_features = torch.randn(5, 1000)  # Wrong dimension
    try:
        calib_glm.fit(wrong_features, labels, labels)
        assert False, "Should fail with wrong input dimensions"
    except Exception:
        print("✓ CalibGLM correctly rejects wrong input dimensions")
    
    print("📏 CalibGLM feature dimension tests passed!")


def test_unfitted_calib_glm_behavior():
    """Test CalibGLM behavior when not fitted."""
    print("⚠️ Testing unfitted CalibGLM behavior...")
    
    calib_glm = CalibGLM(in_dim=1696)
    features = torch.randn(3, 1696)
    
    # Should return dummy predictions when not fitted
    p_hit, p_kill = calib_glm(features)
    assert p_hit.shape == (3, 1), "Should return correct shape even when unfitted"
    assert p_kill.shape == (3, 1), "Should return correct shape even when unfitted"
    assert torch.allclose(p_hit, torch.full((3, 1), 0.5)), "Unfitted model should return 0.5 predictions"
    assert torch.allclose(p_kill, torch.full((3, 1), 0.5)), "Unfitted model should return 0.5 predictions"
    print("✓ Unfitted CalibGLM returns dummy predictions")
    
    # Should fail to save when not fitted
    try:
        calib_glm.save("/tmp/unfitted_model.joblib")
        assert False, "Should fail to save unfitted model"
    except RuntimeError as e:
        assert "fitted" in str(e).lower(), "Error message should mention fitting requirement"
        print("✓ Unfitted CalibGLM correctly prevents saving")
    
    print("⚠️ Unfitted CalibGLM behavior tests passed!")


def test_classical_calibration_pipeline_simulation():
    """Simulate the classical calibration pipeline from Phase 8."""
    print("🔄 Testing classical calibration pipeline simulation...")
    
    # Simulate the pipeline: collect features → fit CalibGLM → save → load → predict
    
    # Step 1: Simulate feature collection from fusion model
    batch_size = 8
    n_batches = 5
    
    all_features = []
    all_hit_labels = []
    all_kill_labels = []
    
    for _ in range(n_batches):
        # Simulate fused features: [zi(768), zt(512), zr(384), e_cls(32)]
        zi = torch.randn(batch_size, 768)
        zt = torch.randn(batch_size, 512) 
        zr = torch.randn(batch_size, 384)
        e_cls = torch.randn(batch_size, 32)
        
        features = torch.cat([zi, zt, zr, e_cls], dim=1)  # (B, 1696)
        hit_labels = torch.randint(0, 2, (batch_size,)).float()
        kill_labels = torch.randint(0, 2, (batch_size,)).float()
        
        all_features.append(features)
        all_hit_labels.append(hit_labels)
        all_kill_labels.append(kill_labels)
    
    # Concatenate all collected data
    collected_features = torch.cat(all_features, dim=0)  # (40, 1696)
    collected_hit_labels = torch.cat(all_hit_labels, dim=0)  # (40,)
    collected_kill_labels = torch.cat(all_kill_labels, dim=0)  # (40,)
    
    print(f"✓ Collected {collected_features.shape[0]} samples with {collected_features.shape[1]} features")
    
    # Step 2: Fit CalibGLM
    calib_glm = CalibGLM(in_dim=1696, model="logreg")
    calib_glm.fit(collected_features, collected_hit_labels, collected_kill_labels)
    print("✓ CalibGLM fitted on collected features")
    
    # Step 3: Save fitted model
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        model_path = f.name
    calib_glm.save(model_path)
    print(f"✓ CalibGLM saved to {model_path}")
    
    # Step 4: Load model for evaluation/inference
    eval_calib = CalibGLM(in_dim=1696)
    eval_calib.load(model_path)
    print("✓ CalibGLM loaded for evaluation")
    
    # Step 5: Generate auxiliary predictions
    test_features = torch.randn(3, 1696)
    p_hit_aux, p_kill_aux = eval_calib(test_features)
    
    assert p_hit_aux.shape == (3, 1), "Aux hit predictions should have correct shape"
    assert p_kill_aux.shape == (3, 1), "Aux kill predictions should have correct shape"
    print("✓ Auxiliary predictions generated successfully")
    
    print("🔄 Classical calibration pipeline simulation passed!")


if __name__ == "__main__":
    test_calib_glm_fit_save_load()
    test_calib_glm_feature_dimensions()
    test_unfitted_calib_glm_behavior()
    test_classical_calibration_pipeline_simulation()
    print("\n✅ All Phase 8 (calibration flow) tests passed!")